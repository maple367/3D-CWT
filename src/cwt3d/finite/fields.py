"""Finite complex envelopes and signed power budgets, not electromagnetic watts.

Normalize free faces using P=hx*hy*||Bv||²=1. With this amplitude convention
centers/faces have units um^-1, P is dimensionless, and power-budget terms
are rates in um^-1. Intensity is the incoherent sum of four envelope moduli
squared, NOT the microscopic interference-resolved electromagnetic intensity.
The phase reference is optional and never changes lambda=delta+i*alpha.
"""
from dataclasses import dataclass
from numbers import Integral, Complex
from types import MappingProxyType

import numpy as np

from .assembly import FinitePencil
from .grid import WaveFields, _complex_array, _snapshot
from .eigensolver import (EigenResult, _start_vector, _residual, _residual_pencil,
                          _frobenius)


@dataclass(frozen=True, slots=True, init=False, eq=False)
class FiniteMode:
    """Immutable normalized mode; every public ndarray has its own bytes header.

    ``residual`` is recomputed from this pencil (Frobenius normwise), not
    blindly copied from EigenResult. Reconstruction accepts non-eigenvectors
    for diagnostics: a large residual/balance error is reported, not hidden.
    ``boundary_error`` is max incoming face amplitude (exact zero by layout).
    ``reconstruction_error`` is max |interpolated faces - Bv|.
    """
    grid: object
    eigenvalue: complex
    faces: WaveFields
    centers: WaveFields
    budget: object
    residual: float
    boundary_error: float
    reconstruction_error: float
    normalization: str
    units: object
    _vector: bytes
    _center: bytes
    _intensities: bytes

    @property
    def free_vector(self):
        return np.ndarray((self.grid.n_dof,),dtype=complex,buffer=self._vector)

    @property
    def center_vector(self):
        return np.ndarray((self.grid.n_dof,),dtype=complex,buffer=self._center)

    @property
    def component_intensities(self):
        return np.ndarray((4,self.grid.Ny,self.grid.Nx),dtype=float,buffer=self._intensities)

    @property
    def intensity(self):
        return _snapshot(np.sum(self.component_intensities,axis=0))

    @property
    def delta(self):
        return self.eigenvalue.real

    @property
    def alpha(self):
        return self.eigenvalue.imag


def reconstruct_field(pencil, vector, eigenvalue, *, coupling=None, phase_fix=True):
    """Reconstruct one free-face vector with unit cell-center integrated power.

    Input scaling is removed BEFORE squaring; no zero/nonfinite vector is
    accepted. Optional CouplingResult must exactly match the pencil's C.
    Generic complex C gives only signed coupling power (including gain), not
    an invented absorption/radiation decomposition. Crad is zero-order vertical
    radiation; only if all auxiliary orders are evanescent is it all radiative
    coupling in this lossless optical model. No eigenvalue sign is changed.
    """
    if not isinstance(pencil,FinitePencil):
        raise TypeError('pencil must be FinitePencil')
    if isinstance(eigenvalue,(bool,np.bool_)) or not isinstance(eigenvalue,Complex) or not np.isfinite(eigenvalue):
        raise ValueError('eigenvalue must be a finite complex scalar')
    if not isinstance(phase_fix,(bool,np.bool_)):
        raise ValueError('phase_fix must be boolean')
    if coupling is not None:
        from ..coupling import CouplingResult
        if not isinstance(coupling,CouplingResult):
            raise TypeError('coupling must be CouplingResult')
        if coupling.units != 'um^-1' or not np.array_equal(coupling.C,pencil.C):
            raise ValueError('coupling must match pencil.C and units exactly')
        parts = []
        for name in ('C1D','Crad','C2D'):
            part = _complex_array(getattr(coupling,name),'coupling component '+name)
            if part.shape != (4,4):
                raise ValueError('coupling components must have shape (4,4)')
            parts.append(part)
        part_scale = max(float(np.max(abs(part))) for part in parts)
        if part_scale:
            # Compare at the component scale, without a dimensionful floor;
            # dividing first also avoids overflow in a cancelling component sum.
            summed = sum(part/part_scale for part in parts)
            if np.max(abs(summed-pencil.C/part_scale)) > 64*np.finfo(float).eps:
                raise ValueError('coupling components do not sum to pencil.C')
        elif np.any(pencil.C):
            raise ValueError('coupling components do not sum to pencil.C')
    g = pencil.grid
    vector = _complex_array(vector,'vector')
    if vector.shape != (g.n_dof,):
        raise ValueError(f'vector must have shape ({g.n_dof},)')
    v = _start_vector(vector,g.n_dof,None)
    a,b = pencil.A,pencil.B
    # Fail closed when physical units cannot be represented, rather than
    # silently returning zero/inf diagnostics. Common large/small input
    # amplitude has already been removed safely by _start_vector.
    try:
        with np.errstate(over='raise',invalid='raise',divide='raise'):
            area = g.hx_um*g.hy_um
            if not np.isfinite(area) or area <= 0:
                raise ValueError('cell area outside numeric range')
            u = b@v
            norm = np.linalg.norm(u)
            denominator = np.sqrt(area)*norm
            if not np.isfinite(denominator) or denominator <= 0:
                raise ValueError('center power is zero or outside numeric range')
            v /= denominator
            if phase_fix:
                anchor = v[np.argmax(abs(v))]
                v *= np.conj(anchor/abs(anchor))
            u = b@v
            faces = g.unpack(v)
            centers = g.interpolate(faces)
            arrays = tuple(getattr(centers,n) for n in ('Rx','Sx','Ry','Sy'))
            interpolated = np.array(arrays).ravel()
            reconstruction_error = float(np.max(abs(interpolated-u)))
            intensities = abs(u.reshape(4,g.Ny,g.Nx))**2
            power = float(area*np.sum(intensities))
            edge = float(g.hy_um*(np.sum(abs(faces.Rx[:,-1])**2)+np.sum(abs(faces.Sx[:,0])**2))
                         +g.hx_um*(np.sum(abs(faces.Ry[-1,:])**2)+np.sum(abs(faces.Sy[0,:])**2)))
            uc = u.reshape(4,-1)
            def contribution(c):
                return float(2*area*np.vdot(uc,c@uc).imag)
            pc = contribution(pencil.C)
            threshold = float(2*complex(eigenvalue).imag*power)
            difference = threshold-pc-edge
            # Sum magnitudes of physical channels, no unit-sized floor.
            scale = abs(threshold)+abs(pc)+edge
            relative = abs(difference)/scale if scale else (0. if difference == 0 else np.inf)
            budget = dict(center_power=power,edge=edge,coupling=pc,threshold=threshold,
                          balance_signed_error=difference,balance_absolute_error=abs(difference),
                          balance_relative_error=relative,
                          radiation_scope='unspecified: generic coupling matrix')
            if coupling is not None:
                for name in ('C1D','Crad','C2D'):
                    budget[name] = contribution(getattr(coupling,name))
                budget['component_sum_error'] = pc-sum(budget[n] for n in ('C1D','Crad','C2D'))
                budget['all_aux_evanescent'] = bool(coupling.all_aux_evanescent)
                budget['radiation_scope'] = ('all radiative coupling (auxiliary orders evanescent)'
                    if coupling.all_aux_evanescent else 'zero-order only; auxiliary orders may radiate')
            if not all(np.isfinite(x) for x in (power,edge,pc,threshold,relative,reconstruction_error)):
                raise ValueError('field/power diagnostics outside numeric range')
    except FloatingPointError as exc:
        raise ValueError('field/power diagnostics outside numeric range') from exc
    aa,bb = _residual_pencil(a,b)
    residual = _residual(aa,bb,_frobenius(aa),_frobenius(bb),complex(eigenvalue),v)
    incoming = (faces.Rx[:,0],faces.Sx[:,-1],faces.Ry[0,:],faces.Sy[-1,:])
    result = object.__new__(FiniteMode)
    entries = dict(grid=g,eigenvalue=complex(eigenvalue),faces=faces,centers=centers,
                   budget=MappingProxyType(budget),residual=float(residual),
                   boundary_error=max(float(np.max(abs(x))) for x in incoming),
                   reconstruction_error=reconstruction_error,
                   normalization='hx*hy*sum_cells,sum_components |Bv|^2 = 1',
                   units=MappingProxyType(dict(length='um',envelope='um^-1',intensity='um^-2',
                                               center_power='1',power_budget='um^-1',eigenvalue='um^-1')),
                   _vector=v.tobytes(),_center=u.tobytes(),_intensities=intensities.tobytes())
    for name,value in entries.items():
        object.__setattr__(result,name,value)
    return result


def _result_arrays(pencil,result):
    if not isinstance(pencil,FinitePencil) or not isinstance(result,EigenResult):
        raise TypeError('expected FinitePencil and EigenResult')
    values,vectors = result.eigenvalues,result.eigenvectors
    if (values.ndim != 1 or vectors.shape != (pencil.grid.n_dof,len(values))
            or result.residuals.shape != values.shape):
        raise ValueError('EigenResult arrays do not match pencil and returned count')
    return values,vectors


def reconstruct_mode(pencil,result,index,**kwargs):
    """Analyze one actually returned column; never wrap negative indices."""
    values,vectors = _result_arrays(pencil,result)
    if isinstance(index,(bool,np.bool_)) or not isinstance(index,Integral):
        raise TypeError('index must be an integer, not bool')
    if not 0 <= index < len(values):
        raise IndexError('index outside returned modes')
    return reconstruct_field(pencil,vectors[:,index],values[index],**kwargs)


def reconstruct_modes(pencil,result,**kwargs):
    """Tuple of returned modes only, including honest partial EigenResult sets."""
    values,vectors = _result_arrays(pencil,result)
    return tuple(reconstruct_field(pencil,vectors[:,i],value,**kwargs)
                 for i,value in enumerate(values))
