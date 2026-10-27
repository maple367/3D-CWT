"""Opt-in active-cavity analysis on the shared passive optical core.

Units: um, ns, densities um^-3, material power gain/loss um^-1,
current density A/um^2. The deterministic first module requires an explicit
initial optical seed; thermal transport and stochastic linewidth are not
included. Passive APIs are unchanged.
"""
from importlib import import_module

_EXPORTS = {
    'CarrierMaterial': 'materials',
    'ActiveCavity': 'cavity',
    'ActiveState': 'results',
    'ActiveResult': 'results',
    'simulate': 'dynamics',
    'solve_active_modes': 'threshold',
    'uniform_threshold_gain': 'threshold',
    'diffusion_operator': 'carriers',
    'carrier_rhs': 'carriers',
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    return getattr(import_module('.' + _EXPORTS[name], __name__), name)
