"""Targeted general complex sparse pencils without an SPD mass assumption.

For A v = lambda B v, apply T x = (A-sigma B)^(-1) B x using one
sparse LU, then lambda = sigma + 1/mu. No explicit inverse, dense fallback,
second shift, or Hermitian approximation is used. Only a target subset is
computed; distance sorting is not mode tracking. Small residuals do not
certify eigenvalue accuracy for non-normal/ill-conditioned pencils.
"""
from dataclasses import dataclass
from numbers import Integral, Real, Complex
from time import perf_counter
from types import MappingProxyType

import numpy as np
from scipy.sparse import csr_matrix, issparse
from scipy.sparse.linalg import LinearOperator, splu, eigs, ArpackNoConvergence

from .assembly import FinitePencil


class FactorizationError(RuntimeError):
    """A-sigma B could not be factored; no automatic regularization is applied."""


class EigenSolveError(RuntimeError):
    """Incomplete/invalid solve. ``partial`` contains only residual-gated pairs."""

    def __init__(self, message, partial):
        super().__init__(message)
        self.partial = partial


@dataclass(frozen=True, slots=True, init=False, eq=False)
class EigenResult:
    """Independent immutable snapshots and read-only scalar diagnostics.

    Eigenvectors are columns in the input free-DOF ordering. ``alpha`` is
    signed Im(lambda), and ``power_loss`` is 2*alpha: the threshold gain
    compensation convention lambda=delta+i*alpha, NOT a passive beta sign.
    Generic pencils inherit only this algebraic labeling, not optical units.

    Metadata is also accessible as scalar attributes: requested, returned,
    converged, status, sigma, k, tolerance, residual_tolerance, seed, ncv,
    maxiter, arpack_converged, candidate_count, discarded_transform,
    discarded_residual, discarded_numeric_range (a residual-discard subset),
    lu_seconds, iteration_seconds, lu_nnz, linear_solves.
    No mutable input, sparse matrix, ndarray header or RNG is retained.
    """
    _arrays: tuple
    _metadata: object

    def __init__(self, values, vectors, residuals, metadata):
        entries = []
        for a in (values, vectors, residuals, values.real, values.imag, 2*values.imag):
            a = np.asarray(a)
            entries.append((a.shape, a.dtype.str, a.tobytes()))
        object.__setattr__(self, '_arrays', tuple(entries))
        object.__setattr__(self, '_metadata', MappingProxyType(dict(metadata)))

    def _get(self, index):
        shape, dtype, data = self._arrays[index]
        return np.ndarray(shape, dtype=dtype, buffer=data)

    @property
    def metadata(self):
        return self._metadata

    def __getattr__(self, name):
        try:
            return self._metadata[name]
        except KeyError:
            raise AttributeError(name) from None

    @property
    def eigenvalues(self):
        return self._get(0)

    @property
    def eigenvectors(self):
        return self._get(1)

    @property
    def residuals(self):
        return self._get(2)

    @property
    def delta(self):
        return self._get(3)

    @property
    def alpha(self):
        return self._get(4)

    @property
    def power_loss(self):
        return self._get(5)


def _integer(value, name, low, high=None):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
            or value < low or (high is not None and value > high)):
        raise ValueError(f'{name} must be an integer in [{low}, {high}]')
    return int(value)


def _real(value, name, positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f'{name} must be a finite real number')
    value = float(value)
    if not np.isfinite(value) or (value <= 0 if positive else value < 0):
        raise ValueError(f'{name} must be finite and {"positive" if positive else "nonnegative"}')
    return value


def _matrix(value, name):
    if not issparse(value):
        raise TypeError(f'{name} must be sparse; dense inputs/fallback are not supported')
    if len(value.shape) != 2 or value.shape[0] != value.shape[1]:
        raise ValueError(f'{name} must be square')
    if value.dtype.kind not in 'biufc':
        raise ValueError(f'{name} must contain finite numeric coefficients')
    with np.errstate(over='ignore', invalid='ignore'):
        matrix = csr_matrix(value, dtype=np.complex128, copy=True)
        matrix.sum_duplicates()
    if not np.all(np.isfinite(matrix.data)):
        raise ValueError(f'{name} must contain finite numeric coefficients')
    matrix.sort_indices()
    return matrix


def _start_vector(v0, n, seed):
    if v0 is None:
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(n)+1j*rng.standard_normal(n)
    else:
        value = np.asarray(v0)
        if value.shape != (n,) or value.dtype.kind not in 'biufc':
            raise ValueError(f'v0 must be a finite numeric vector with shape ({n},)')
        with np.errstate(over='ignore', invalid='ignore'):
            v = np.array(value, dtype=np.complex128, copy=True)
    if not np.all(np.isfinite(v)):
        raise ValueError('v0 must be finite and nonzero')
    scale = max(np.max(abs(v.real)), np.max(abs(v.imag)))
    if scale == 0:
        raise ValueError('v0 must be finite and nonzero')
    # Divide real components directly: NumPy complex division may form 1/scale
    # and overflow/underflow unnecessarily for subnormal/huge finite vectors.
    with np.errstate(under='ignore'):
        v.real /= scale
        v.imag /= scale
    v /= np.linalg.norm(v)
    return v


def _component_scale(data):
    return max(np.max(abs(data.real), initial=0.),
               np.max(abs(data.imag), initial=0.))


def _scaled_components(data, scale):
    """Divide real components directly; reject loss of nonzero coefficients."""
    result = np.empty_like(data, dtype=np.complex128)
    with np.errstate(under='ignore'):
        result.real = data.real/scale
        result.imag = data.imag/scale
    if (np.any((data.real != 0) & (result.real == 0))
            or np.any((data.imag != 0) & (result.imag == 0))):
        raise ValueError('numeric range exhausted during residual scaling')
    return result


def _safe_norm(data):
    """Euclidean/Frobenius norm: scale BEFORE squaring real components."""
    scale = _component_scale(data)
    if not scale:
        return 0.
    with np.errstate(over='ignore', invalid='ignore', under='ignore'):
        # Losing negligible components here is harmless: the maximum is 1.
        real, imag = data.real/scale, data.imag/scale
        return float(scale*np.sqrt(np.sum(real*real)+np.sum(imag*imag)))


def _frobenius(matrix):
    """O(nnz) norm with scaled accumulation, no domain-sized dense array."""
    norm = _safe_norm(matrix.data)
    if not np.isfinite(norm):
        raise ValueError('matrix Frobenius norm exceeds finite numeric range')
    return norm


def _residual_pencil(a, b):
    """Common scaling only for residual evaluation; physical LU is untouched."""
    scale = max(_component_scale(a.data), _component_scale(b.data))
    if not scale:
        return a, b  # the original zero shift matrix will fail LU
    aa, bb = a.copy(), b.copy()
    aa.data = _scaled_components(a.data, scale)
    bb.data = _scaled_components(b.data, scale)
    return aa, bb


class _ResidualRangeError(ValueError):
    """Residual arithmetic cannot certify this candidate in float64."""


def _check_products(left, right):
    """Guard all four real products BEFORE complex multiplication.

    frexp mantissas lie in [1/2, 1), so exponent sum >= -1020
    guarantees a normal product (>= 2**-1022). Conservatively reject
    smaller bounds even if the actual product would still be normal.
    Zero factors are exact and need no range budget. No trial product,
    dense matrix, or dependency on SciPy's floating-point flags is used.
    """
    left, right = np.asarray(left), np.asarray(right)
    for x in (left.real, left.imag):
        for y in (right.real, right.imag):
            nonzero = (x != 0) & (y != 0)
            exponents = np.frexp(x)[1] + np.frexp(y)[1]
            if np.any(nonzero & (exponents < -1020)):
                raise _ResidualRangeError('numeric range exhausted in residual product')


def _check_normal(data):
    """Reject subnormal nonzero components at residual operation boundaries."""
    data = np.asarray(data)
    for part in (data.real, data.imag):
        if np.any((part != 0) & (abs(part) < np.finfo(float).tiny)):
            raise _ResidualRangeError('numeric range exhausted in residual component')


def _residual(a, b, a_norm, b_norm, value, vector):
    """Original-pencil normwise residual; never a transformed residual.

    ||Av-lambda Bv|| / ((||A||F + |lambda| ||B||F) ||v||2).
    Frobenius matrix norms are accumulated sparsely. Unlike action-relative
    normalization, this remains meaningful at a zero eigenvalue when A!=0.
    For exactly A=0, the solver first certifies the known all-zero spectrum;
    the exact (0,v) pair has residual zero, not a roundoff/roundoff ratio.
    Reject common nullspaces, nonfinite arithmetic, and zero vectors rather
    than blessing a spurious physical zero mode.
    The matrices and their Frobenius norms share one coefficient scale;
    vector amplitude is removed before matvecs. These factors cancel exactly
    in the stated original-pencil ratio; LU still uses the physical pencil.
    All norms scale real/imaginary components before squaring. No absolute
    floor is added. Sparse product exponents are checked explicitly because
    SciPy kernels do not reliably signal underflow to np.errstate. Products
    near the subnormal range are conservatively rejected; this is not an
    arbitrary-range or correctly-rounded residual certificate.
    """
    if not np.all(np.isfinite(vector)) or not np.any(vector):
        return np.inf
    try:
        vector = _scaled_components(vector, _component_scale(vector))
    except ValueError as exc:
        raise _ResidualRangeError('numeric range exhausted in residual scaling') from exc
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise', under='raise'):
            _check_products(a.data, vector[a.indices])
            _check_products(b.data, vector[b.indices])
            av, bv = a@vector, b@vector
            _check_normal(av)
            _check_normal(bv)
            if not np.all(np.isfinite(av)) or not np.all(np.isfinite(bv)) or not np.any(bv):
                return np.inf
            # Real/imag scaling avoids overflow even for large finite complex lambda.
            scale = max(1., abs(value.real), abs(value.imag))
            scaled_av = _scaled_components(av, scale)
            scaled_value = _scaled_components(np.asarray(value), scale)
            _check_normal(scaled_av)
            _check_normal(scaled_value)
            _check_products(scaled_value, bv)
            difference = scaled_av-scaled_value*bv
            _check_normal(difference)
            numerator = _safe_norm(difference)
            _check_products(abs(scaled_value), b_norm)
            term_a, term_b = a_norm/scale, abs(scaled_value)*b_norm
            _check_normal(np.asarray([term_a, term_b]))
            norm_scale = max(term_a, term_b)
            if norm_scale == 0:
                # Only the certified exactly-zero spectrum has a defined 0/0 here.
                return 0. if a_norm == 0 and value == 0 and numerator == 0 else np.inf
            denominator = (term_a/norm_scale+term_b/norm_scale)*_safe_norm(vector)
            residual = (numerator/norm_scale)/denominator
            if residual == 0 and np.any(difference):
                return np.inf  # nonzero residual must not underflow into acceptance
    except (FloatingPointError, ValueError) as exc:
        raise _ResidualRangeError('numeric range exhausted in residual arithmetic') from exc
    return float(residual) if np.isfinite(residual) and np.isfinite(denominator) else np.inf


def solve_generalized(A, B, *, sigma=0j, k=None, tol=1e-10,
                      residual_tol=1e-8, seed=0, ncv=None, maxiter=None,
                      v0=None, allow_partial=False):
    """Solve a finite near-shift subset of a general sparse complex pencil.

    Require square finite sparse A,B with n>=3, 1<=k<n-1; k=None uses
    min(6,n-2). Complex ARPACK requires k+2<=ncv<=n (stricter than ncv>k).
    tol is ARPACK's nonnegative tolerance (0 means machine precision), while
    residual_tol is an independent positive original-pencil acceptance gate.
    seed uses a local Generator; v0 is copied, complex-capable and normalized.

    A singular B is permitted if the shift matrix is factorizable and the
    targeted subset is finite; zero/nonfinite mu and nonfinite lambda are
    discarded and counted in discarded_transform, not mapped to lambda=0.
    This is not a complete solver for arbitrary singular/defective pencils.

    Exactly zero A is special: successful LU of -sigma B requires nonzero
    sigma and nonsingular B, so all finite eigenvalues are mathematically 0.
    Require |1+sigma*mu| <= 32*eps (dimensionless roundoff consistency with
    T=-I/sigma), then return exact zero and evaluate its original residual.
    This conservative certificate is independent of tol/residual_tol, never
    applies to nonzero A, and does not introduce a shift-sized residual floor.
    Singular/zero B in this case fails LU, rather than inventing eigenvalues.

    On incomplete ARPACK convergence or residual/transform rejection, raise
    EigenSolveError(partial=result) unless allow_partial=True and at least one
    valid pair survived. No available pair ALWAYS raises. Partial results
    explicitly have converged=False. Exact LU failure raises FactorizationError
    without retrying/perturbing sigma; near-singular factors are not rejected
    by an arbitrary pivot threshold. Timings exclude assembly and validation.
    """
    a, b = _matrix(A, 'A'), _matrix(B, 'B')
    if a.shape != b.shape or a.shape[0] < 3:
        raise ValueError('A and B must have the same square shape with n >= 3')
    n = a.shape[0]
    k = min(6,n-2) if k is None else _integer(k,'k',1,n-2)
    if isinstance(sigma, (bool, np.bool_)) or not isinstance(sigma, Complex):
        raise ValueError('sigma must be a finite complex scalar')
    sigma = complex(sigma)
    if not np.isfinite(sigma):
        raise ValueError('sigma must be finite')
    tol = _real(tol,'tol')
    residual_tol = _real(residual_tol,'residual_tol',positive=True)
    seed = None if seed is None else _integer(seed,'seed',0)
    ncv = min(n,max(2*k+1,20)) if ncv is None else _integer(ncv,'ncv',k+2,n)
    maxiter = 10*n if maxiter is None else _integer(maxiter,'maxiter',1)
    if not isinstance(allow_partial, (bool, np.bool_)):
        raise ValueError('allow_partial must be boolean')
    start = _start_vector(v0,n,seed)
    a_norm, b_norm = _frobenius(a), _frobenius(b)
    residual_a, residual_b = _residual_pencil(a, b)
    residual_a_norm, residual_b_norm = _frobenius(residual_a), _frobenius(residual_b)
    with np.errstate(over='ignore', invalid='ignore'):
        shifted = (a-sigma*b).tocsc()
    if not np.all(np.isfinite(shifted.data)):
        raise ValueError('shift matrix coefficients must be finite')
    begin = perf_counter()
    try:
        lu = splu(shifted)
    except RuntimeError as exc:
        raise FactorizationError(f'cannot factor A-sigma B at shift {sigma!r}: {exc}') from exc
    lu_seconds = perf_counter()-begin
    linear_solves = 0

    def apply(x):
        nonlocal linear_solves
        linear_solves += 1
        return lu.solve(b@x)

    operator = LinearOperator((n,n), matvec=apply, dtype=np.complex128)
    begin = perf_counter()
    arpack_converged = True
    failure = None
    try:
        mu, vectors = eigs(operator,k=k,which='LM',tol=tol,v0=start,
                           ncv=ncv,maxiter=maxiter)
    except ArpackNoConvergence as exc:
        arpack_converged = False
        failure = exc
        mu, vectors = exc.eigenvalues, exc.eigenvectors
    iteration_seconds = perf_counter()-begin
    # SciPy supplies empty arrays when no Ritz vectors have converged.
    mu = np.asarray(mu,dtype=np.complex128)
    vectors = np.asarray(vectors,dtype=np.complex128)
    values, valid_vectors, residuals = [], [], []
    discarded_transform = discarded_residual = discarded_numeric_range = 0
    # Test stored coefficients, not a tolerance/norm that could hide tiny A.
    exact_zero_a = not np.any(a.data)
    for i, transformed in enumerate(mu):
        with np.errstate(divide='ignore', over='ignore', invalid='ignore', under='ignore'):
            value = sigma+1/transformed if transformed != 0 and np.isfinite(transformed) else complex(np.nan)
        if not np.isfinite(value):
            discarded_transform += 1
            continue
        vector = vectors[:,i]
        if exact_zero_a:
            # LU already established regularity of this special pencil. The
            # 32-eps budget allows modest LU/Arnoldi rounding, not an absolute
            # eigenvalue tolerance: sigma*mu is unitless and should be -1.
            with np.errstate(over='ignore', invalid='ignore', under='ignore'):
                consistency = abs(1+sigma*transformed)
            if not np.isfinite(consistency) or consistency > 32*np.finfo(float).eps:
                discarded_residual += 1
                continue
            value = 0j
        try:
            residual = _residual(residual_a,residual_b,residual_a_norm,residual_b_norm,value,vector)
        except _ResidualRangeError:
            discarded_numeric_range += 1
            residual = np.inf
        if residual > residual_tol:
            discarded_residual += 1
            continue
        values.append(value)
        valid_vectors.append(vector)
        residuals.append(residual)
    values = np.asarray(values,dtype=np.complex128)
    vectors = np.column_stack(valid_vectors) if valid_vectors else np.empty((n,0),dtype=np.complex128)
    residuals = np.asarray(residuals,dtype=float)
    order = np.argsort(abs(values-sigma),kind='stable')
    returned = len(values)
    converged = arpack_converged and returned == k and not (discarded_transform or discarded_residual)
    status = 'converged' if converged else ('partial' if returned else 'failed')
    metadata = dict(requested=k, returned=returned, converged=converged, status=status,
                    sigma=sigma,k=k,tolerance=tol,residual_tolerance=residual_tol,
                    seed=seed,ncv=ncv,maxiter=maxiter,arpack_converged=arpack_converged,
                    candidate_count=len(mu),discarded_transform=discarded_transform,
                    discarded_residual=discarded_residual,
                    discarded_numeric_range=discarded_numeric_range,lu_seconds=lu_seconds,
                    iteration_seconds=iteration_seconds,lu_nnz=int(lu.L.nnz+lu.U.nnz),
                    linear_solves=linear_solves,residual_normalization='frobenius',
                    A_frobenius=a_norm,B_frobenius=b_norm)
    result = EigenResult(values[order],vectors[:,order],residuals[order],metadata)
    if not converged and (not allow_partial or not returned):
        raise EigenSolveError(
            f'{status}: {returned}/{k} valid pairs; ARPACK converged={arpack_converged}; '
            f'discarded transform={discarded_transform}, residual={discarded_residual}; '
            f'numeric range={discarded_numeric_range}',result) from failure
    return result


def solve_finite(pencil, **kwargs):
    """Solve a FinitePencil; retrieve independent A and B snapshots only once.

    Retain the pencil together with these free-face eigenvectors for field
    reconstruction via pencil.grid.unpack/interpolate. Optical eigenvalues,
    delta and alpha use um^-1. This function does not track modes or certify
    domain/grid/cutoff convergence.
    """
    if not isinstance(pencil, FinitePencil):
        raise TypeError('pencil must be a FinitePencil')
    return solve_generalized(pencil.A,pencil.B,**kwargs)
