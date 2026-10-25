"""Homogeneous, reflection-free Green operator and delta-source oracles."""

import numpy as np
import pytest

from ._legacy import LEGACY_BASELINE, legacy_class


@pytest.fixture
def green_case():
    assert LEGACY_BASELINE == "520f4d8-unfixed"
    model = legacy_class("Model")()
    model.k0, model.beta0 = 1.0, 3.0
    # Preserve Python class mangling: beta_z calls self.__eps_profile_z.
    model._Model__eps_profile_z = lambda z: 1.0 + 0j
    model.kappa_p = lambda order: (1.0, 0.0, 0.0, 1.0)
    return model, (2, 0)


def decaying_green(z, source, gamma):
    """(d_z^2-gamma^2)G=-delta, decaying at both infinities."""
    return np.exp(-gamma * np.abs(z - source)) / (2 * gamma)


def relative_ode_residual(kernel, z, source, gamma, h=1e-4):
    value = kernel(z, source)
    second = (kernel(z + h, source) - 2 * value + kernel(z - h, source)) / h**2
    return abs(second - gamma**2 * value) / abs(gamma**2 * value)


@pytest.mark.parametrize("distance", [-0.7, -0.2, 0.2, 0.7])
def test_legacy_green_violates_evanescent_operator(green_case, distance):
    model, order = green_case
    source = 0.31
    gamma = model.beta_z_func_higher_order(source, order)
    assert gamma**2 == pytest.approx(35.0)
    old = lambda z, zp: model.Green_func_higher_order(z, zp, order)
    # Both sides solve the oscillatory operator instead: normalized defect ~2.
    assert relative_ode_residual(old, source + distance, source, gamma) == pytest.approx(2.0, abs=2e-6)
    reference = lambda z, zp: decaying_green(z, zp, np.sqrt(35.0))
    assert relative_ode_residual(reference, source + distance, source, gamma) < 2e-6
    assert old(source + distance, source) != pytest.approx(reference(source + distance, source))


def test_reference_green_decay_continuity_and_source_jump():
    gamma, source = np.sqrt(35.0), 0.31
    kernel = lambda z: decaying_green(z, source, gamma)
    assert kernel(source) == pytest.approx(1 / (2 * gamma))
    assert kernel(source + 0.4) == pytest.approx(kernel(source - 0.4))
    assert 0 < kernel(source + 0.8) < kernel(source + 0.4)
    h = 1e-6
    assert kernel(source - h) == pytest.approx(kernel(source + h))
    right = (kernel(source + h) - kernel(source)) / h
    left = (kernel(source) - kernel(source - h)) / h
    assert right - left == pytest.approx(-1.0, abs=4e-6)
    # Integral of the operator across the source tends to -1 as well.
    integral_g = (1 - np.exp(-gamma * h)) / gamma**2
    assert right - left - gamma**2 * integral_g == pytest.approx(-1, abs=4e-6)
