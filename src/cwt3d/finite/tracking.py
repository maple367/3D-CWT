"""Same-grid sweep correspondence, not global mode identity or frequency sorting.

Use area-weighted normalized complex center-field overlap |<u,v>| (amplitude,
not squared power). Hungarian matching maximizes eligible match count first,
then summed scores. No interpolation across grids, no eigenvalue sign edits.
Near/exact degeneracies are connected components of complex eigenvalue
proximity. Equal-dimensional full-rank clusters can match as subspaces via
principal overlaps; their individual identities remain explicitly ambiguous.
No rotation/mixing is returned, even within a near-degenerate cluster.
Global assignment alternatives within a bounded score roundoff budget are
also ambiguous, independently of eigenvalue proximity; only forced edges
are reported as definite single-mode or cluster matches.
"""
from dataclasses import dataclass
from numbers import Real
from math import fsum
import numpy as np
from scipy.optimize import linear_sum_assignment

from .fields import FiniteMode
from .grid import _snapshot


@dataclass(frozen=True, slots=True, init=False, eq=False)
class ClusterMatch:
    old_indices: tuple
    new_indices: tuple
    _principal: bytes
    score: float

    @property
    def principal_overlaps(self):
        return np.ndarray((len(self.old_indices),),dtype=float,buffer=self._principal)


@dataclass(frozen=True, slots=True, init=False, eq=False)
class ModeTracking:
    """assignment[i]=-1 if no unambiguous single-mode match.

    matched contains only definite (old,new) pairs. Cluster-covered or
    assignment-ambiguous indices are NOT called lost/new in unmatched_old/new.
    ``ambiguous`` / ``ambiguous_new`` combine eigenvalue-cluster ambiguity
    with global-assignment ambiguity. ``assignment_ambiguous_old/new`` isolate
    the latter, including nonunique cluster correspondence. An ambiguous
    vertex may be matched in one optimum and unmatched in another; this is
    not evidence of physical disappearance. ``clusters`` contains only forced
    subspace correspondences, not arbitrary choices between tied candidates.
    scores is per-old assigned amplitude overlap (NaN otherwise), overlap is
    the full pair score matrix. aligned_vectors[i] is a fresh read-only free
    vector of the new match, with only one global phase changed, or None.
    For cluster matches only principal overlaps are provided, never an
    arbitrary basis rotation pretending to preserve individual eigenpairs.
    """
    matched: tuple
    unmatched_old: tuple
    unmatched_new: tuple
    ambiguous: tuple
    ambiguous_new: tuple
    assignment_ambiguous_old: tuple
    assignment_ambiguous_new: tuple
    clusters: tuple
    _assignment: bytes
    _scores: bytes
    _overlap: bytes
    _shape: tuple
    _aligned: tuple

    @property
    def assignment(self):
        return np.ndarray((self._shape[0],),dtype=np.int64,buffer=self._assignment)

    @property
    def scores(self):
        return np.ndarray((self._shape[0],),dtype=float,buffer=self._scores)

    @property
    def overlap(self):
        return np.ndarray(self._shape,dtype=float,buffer=self._overlap)

    @property
    def aligned_vectors(self):
        return tuple(None if entry is None else np.ndarray((entry[0],),dtype=complex,buffer=entry[1])
                     for entry in self._aligned)


def _clusters(modes,atol,rtol):
    remaining = set(range(len(modes)))
    groups = []
    while remaining:
        group = {min(remaining)}
        remaining -= group
        frontier = list(group)
        while frontier:
            i = frontier.pop()
            li = modes[i].eigenvalue
            neighbors = {j for j in remaining if abs(li-modes[j].eigenvalue)
                         <= atol+rtol*max(abs(li),abs(modes[j].eigenvalue))}
            group |= neighbors
            remaining -= neighbors
            frontier.extend(sorted(neighbors))
        groups.append(tuple(sorted(group)))
    return tuple(groups)


def _assign(scores,eligible,*,required_old=None,required_new=None):
    n,m = scores.shape
    if not n or not m:
        return ()
    # Dummy rows/columns permit real unmatched entries; eligibility is enforced
    # BEFORE optimization, not greedy rejection after an unconstrained solve.
    weights = np.zeros((n+m,n+m))
    weights[:n,:m] = np.where(eligible,min(n,m)+1+scores,-1.)
    # A bonus larger than the entire ordinary objective forces this vertex
    # to participate if possible. Used only to probe unmatched alternatives.
    bonus = (n+m+1)**2
    if required_old is not None:
        weights[required_old,:m] += eligible[required_old,:]*bonus
    if required_new is not None:
        weights[:n,required_new] += eligible[:,required_new]*bonus
    rows,cols = linear_sum_assignment(weights,maximize=True)
    return tuple((int(i),int(j)) for i,j in zip(rows,cols) if i<n and j<m and eligible[i,j])


def _eligible(scores,threshold):
    """Bounded absolute amplitude roundoff budget, shared by both match paths."""
    return scores >= threshold-32*np.finfo(float).eps


def _analyze_assignment(scores,eligible):
    """Return only forced edges, and vertices with nonunique correspondence.

    Compare cardinality exactly, then score sums within 64 eps per matched
    edge. Forbid each selected edge and solve again; this tests GLOBAL optima,
    not row-local ties. Also force each originally unmatched vertex in turn:
    one forbidden-edge witness alone need not visit all rectangular choices.
    At most 1+k+(n-k)+(m-k) padded Hungarian solves; no enumeration.
    """
    n,m = scores.shape
    best = _assign(scores,eligible)
    base = set(best)
    value = fsum(float(scores[i,j]) for i,j in best)
    tol = 64*np.finfo(float).eps*max(1,len(best))
    ao,an = [False]*n,[False]*m
    forced = []

    def equivalent(alternative):
        return (len(alternative) == len(best) and
                abs(fsum(float(scores[i,j]) for i,j in alternative)-value) <= tol)

    def mark(alternative):
        for i,j in base.symmetric_difference(alternative):
            ao[i] = True
            an[j] = True

    for i,j in best:
        allowed = eligible.copy()
        allowed[i,j] = False
        alternative = _assign(scores,allowed)
        if equivalent(alternative):
            mark(alternative)
        else:
            forced.append((i,j))
    used_o,used_n = {i for i,j in best},{j for i,j in best}
    for i in range(n):
        if i not in used_o and eligible[i,:].any():
            alternative = _assign(scores,eligible,required_old=i)
            if any(a == i for a,b in alternative) and equivalent(alternative):
                mark(alternative)
    for j in range(m):
        if j not in used_n and eligible[:,j].any():
            alternative = _assign(scores,eligible,required_new=j)
            if any(b == j for a,b in alternative) and equivalent(alternative):
                mark(alternative)
    return tuple(forced),tuple(ao),tuple(an)


def _basis(u,indices):
    x = u[:,indices]
    q,s,_ = np.linalg.svd(x,full_matrices=False)
    # Rank deficiency cannot certify an entire cluster's physical subspace.
    if len(s) != len(indices) or s[-1] <= np.finfo(float).eps*max(x.shape)*s[0]:
        return None
    return q


def track_modes(old,new,*,min_overlap=.7,degeneracy_atol=1e-8,degeneracy_rtol=1e-6):
    """Match two finite-mode sets on exactly the same physical staggered grid.

    A subset can disappear or enter; no unreturned mode is fabricated. Chained
    proximity clustering is conservative and may mark a broad cluster
    ambiguous. Subspace rescue needs equal returned cluster dimensions and
    minimum principal overlap >= min_overlap. Truncated/unequal clusters or
    defective eigenbases are not certified; ambiguous unmatched entries are
    reported rather than asserted to be physical mode disappearance.

    Eligibility allows only 32 float64 eps absolute amplitude roundoff below
    min_overlap (single and subspace alike). Assignment uniqueness compares
    maximum cardinality exactly and sums within 64 eps*max(1,cardinality).
    These fixed dimensionless budgets are not frequency/physical tolerances.
    """
    old,new = tuple(old),tuple(new)
    for name,value in (('min_overlap',min_overlap),('degeneracy_atol',degeneracy_atol),('degeneracy_rtol',degeneracy_rtol)):
        if isinstance(value,(bool,np.bool_)) or not isinstance(value,Real) or not np.isfinite(value) or value < 0:
            raise ValueError(f'{name} must be finite and nonnegative')
    if min_overlap > 1:
        raise ValueError('min_overlap must be <= 1')
    all_modes = old+new
    if any(not isinstance(m,FiniteMode) for m in all_modes):
        raise TypeError('old/new must contain FiniteMode objects')
    n,m = len(old),len(new)
    if all_modes:
        g = all_modes[0].grid
        if any(mode.grid != g for mode in all_modes):
            raise ValueError('same grid and physical domain required; cross-grid matching unsupported')
        area_root = np.sqrt(g.hx_um*g.hy_um)
        def matrix(modes):
            x = np.column_stack([mode.center_vector*area_root for mode in modes]) if modes else np.empty((g.n_dof,0),complex)
            # Normalize independently, not assuming a caller's amplitude scale.
            return x/np.linalg.norm(x,axis=0) if modes else x
        u,v = matrix(old),matrix(new)
        complex_overlap = u.conj().T@v
    else:
        u,v = np.empty((0,0),complex),np.empty((0,0),complex)
        complex_overlap = np.empty((0,0),complex)
    scores = np.clip(abs(complex_overlap),0,1)
    og,ng = _clusters(old,degeneracy_atol,degeneracy_rtol),_clusters(new,degeneracy_atol,degeneracy_rtol)
    multi_o = [group for group in og if len(group)>1]
    multi_n = [group for group in ng if len(group)>1]
    amb_o = tuple(any(i in group for group in multi_o) for i in range(n))
    amb_n = tuple(any(j in group for group in multi_n) for j in range(m))
    cscores = np.zeros((len(multi_o),len(multi_n)))
    principal = {}
    for i,oi in enumerate(multi_o):
        qo = _basis(u,oi)
        for j,nj in enumerate(multi_n):
            if len(oi) != len(nj) or qo is None:
                continue
            qn = _basis(v,nj)
            if qn is None:
                continue
            s = np.clip(np.linalg.svd(qo.conj().T@qn,compute_uv=False),0,1)
            principal[i,j] = s
            cscores[i,j] = min(s)
    cvalid = np.zeros_like(cscores,dtype=bool)
    for key in principal:
        cvalid[key] = _eligible(cscores[key],min_overlap)
    clusters = []
    covered_o,covered_n = set(),set()
    cpairs,ca_o,ca_n = _analyze_assignment(cscores,cvalid)
    assign_o,assign_n = [False]*n,[False]*m
    for group,amb in zip(multi_o,ca_o):
        if amb:
            for i in group: assign_o[i] = True
            covered_o.update(group)
    for group,amb in zip(multi_n,ca_n):
        if amb:
            for j in group: assign_n[j] = True
            covered_n.update(group)
    for i,j in cpairs:
        result = object.__new__(ClusterMatch)
        for name,value in dict(old_indices=multi_o[i],new_indices=multi_n[j],
                               _principal=principal[i,j].tobytes(),score=float(cscores[i,j])).items():
            object.__setattr__(result,name,value)
        clusters.append(result)
        covered_o.update(multi_o[i]); covered_n.update(multi_n[j])
    eligible = _eligible(scores,min_overlap)
    for i,amb in enumerate(amb_o):
        if amb: eligible[i,:] = False
    for j,amb in enumerate(amb_n):
        if amb: eligible[:,j] = False
    pairs,sa_o,sa_n = _analyze_assignment(scores,eligible)
    assign_o = tuple(a or b for a,b in zip(assign_o,sa_o))
    assign_n = tuple(a or b for a,b in zip(assign_n,sa_n))
    amb_o = tuple(a or b for a,b in zip(amb_o,assign_o))
    amb_n = tuple(a or b for a,b in zip(amb_n,assign_n))
    covered_o.update(i for i,amb in enumerate(assign_o) if amb)
    covered_n.update(j for j,amb in enumerate(assign_n) if amb)
    assignment = np.full(n,-1,dtype=np.int64)
    selected = np.full(n,np.nan)
    aligned = [None]*n
    for i,j in pairs:
        assignment[i] = j
        selected[i] = scores[i,j]
        overlap = complex_overlap[i,j]
        phase = np.conj(overlap/abs(overlap)) if overlap else 1.
        vector = new[j].free_vector*phase
        aligned[i] = (vector.size,vector.tobytes())
        covered_o.add(i); covered_n.add(j)
    result = object.__new__(ModeTracking)
    entries = dict(matched=pairs,unmatched_old=tuple(i for i in range(n) if i not in covered_o),
                   unmatched_new=tuple(j for j in range(m) if j not in covered_n),
                   ambiguous=amb_o,ambiguous_new=amb_n,clusters=tuple(clusters),
                   assignment_ambiguous_old=assign_o,assignment_ambiguous_new=assign_n,
                   _assignment=assignment.tobytes(),_scores=selected.tobytes(),
                   _overlap=scores.tobytes(),_shape=(n,m),_aligned=tuple(aligned))
    for name,value in entries.items():
        object.__setattr__(result,name,value)
    return result
