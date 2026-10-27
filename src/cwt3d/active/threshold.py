"""Fixed-carrier optical spectra and mode-specific uniform threshold gains."""
from ..finite.eigensolver import solve_generalized
from .cavity import ActiveCavity, _scalar


def solve_active_modes(cavity, N, material, *, photon_density_um3=0., **options):
    """Solve the frozen sparse pencil using the existing residual-gated solver.

    This fixed gain/index analysis is not a nonlinear lasing-state solve.
    Solver shifts and eigenvalues retain the passive um^-1 convention.
    """
    if not isinstance(cavity, ActiveCavity):
        raise TypeError('cavity must be an ActiveCavity')
    A, B = cavity.frozen_pencil(N, material, photon_density_um3=photon_density_um3)
    return solve_generalized(A, B, **options)


def uniform_threshold_gain(alpha, *, confinement, internal_loss_um_inverse=0.):
    """Return (2 alpha + alpha_i)/Gamma in um^-1 for one passive mode.

    A mode-specific uniform unsaturated gain identity, not a guarantee of
    the global first threshold when competing modes or index shifts change.
    """
    alpha = _scalar(alpha, 'alpha')
    gamma = _scalar(confinement, 'confinement', positive=True, upper=1)
    loss = _scalar(internal_loss_um_inverse, 'internal_loss_um_inverse')
    result = (2*alpha+loss)/gamma
    import math
    if not math.isfinite(result):
        raise ValueError('threshold gain outside finite numeric range')
    return result
