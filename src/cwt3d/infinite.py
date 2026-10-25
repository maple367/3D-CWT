"""Infinite-periodic four-mode eigenproblem, not mode tracking or Q fitting."""
from dataclasses import dataclass
import numpy as np
from .coupling import CouplingResult, _Snapshots, _pack


@dataclass(frozen=True)
class InfiniteModes(_Snapshots):
    """Unsorted np.linalg.eig pairs: lambda=delta+i*alpha, all in um^-1.

    delta=Re(lambda) is propagation detuning; alpha=Im(lambda) is required
    AMPLITUDE gain/vertical loss in the coupled-wave convention. Power loss
    is 2*alpha, not abs(Im(lambda)). No frequency/group-index or Q conversion.
    Columns of eigenvectors follow eigenvalues; order is NOT stable tracking.
    residuals are ||Cv-lambda*v|| / ((||C||2+|lambda|)||v||2), zero if both
    denominator and numerator vanish.
    """
    _arrays: tuple
    units: str = 'um^-1'


def solve_infinite(coupling):
    if not isinstance(coupling,CouplingResult):
        raise TypeError('Expected CouplingResult from build_coupling')
    C = coupling.C
    values,vectors = np.linalg.eig(C)
    errors = np.linalg.norm(C@vectors-vectors*values[None,:],axis=0)
    denominator = (np.linalg.norm(C,2)+abs(values))*np.linalg.norm(vectors,axis=0)
    residuals = np.divide(errors,denominator,out=np.zeros(4),where=denominator!=0)
    return InfiniteModes(_pack(eigenvalues=values,eigenvectors=vectors,
                               delta=values.real,alpha=values.imag,
                               power_loss=2*values.imag,residuals=residuals))
