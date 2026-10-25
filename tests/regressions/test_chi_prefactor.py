"""Characterize the missing legacy chi prefactor; not a repaired solver."""

from types import SimpleNamespace

import numpy as np
import pytest

from ._legacy import LEGACY_BASELINE, legacy_class


@pytest.fixture
def chi_case():
    assert LEGACY_BASELINE == "520f4d8-unfixed"
    solver = legacy_class("CWT_solver")()
    solver._cut_off = 1
    # Only the four diagonal harmonics survive m**2+n**2 > 1.
    # Two dimensionless layer coefficients, with a deliberately absent layer.
    solver.xi_calculator_collect = [
        {(m, n): 2.0 for m in (-1, 1) for n in (-1, 1)},
        None,
        {(m, n): 6.0 for m in (-1, 1) for n in (-1, 1)},
    ]
    solver.model = SimpleNamespace(_xi_weight=[0.25, 0.0, 0.75], k0=2.0, beta0=3.0)
    solver.get_varsigma = lambda order, direction: 0.5 if direction == "x" else -0.25j
    return solver


def chi_oracle(dimensionless_sum, k0, beta0):
    """Equation (3.35): chi has inverse-length units, unlike the sum."""
    return -k0**2 / (2 * beta0) * dimensionless_sum


@pytest.mark.parametrize("direction,total", [("x", 10.0), ("y", -5j)])
def test_legacy_chi_is_bare_sum_not_equation_335(chi_case, direction, total):
    actual = chi_case._chi_func((0, 0, 1, 0), direction)
    assert actual == pytest.approx(total)  # 4 * (0.25*2 + 0.75*6) * varsigma
    correct = chi_oracle(total, chi_case.model.k0, chi_case.model.beta0)
    assert actual != pytest.approx(correct)
    assert correct == pytest.approx(-2 * total / 3)


def test_legacy_chi_lacks_inverse_length_scaling(chi_case):
    before = chi_case._chi_func((0, 0, 1, 0), "x")
    correct = chi_oracle(10.0, 2.0, 3.0)
    # A 1/length quantity grows by s when both wavenumbers grow by s;
    # dimensionless xi, mu/nu-derived varsigma and overlap weights stay fixed.
    scale = 7.0
    chi_case.model.k0 *= scale
    chi_case.model.beta0 *= scale
    after = chi_case._chi_func((0, 0, 1, 0), "x")
    assert after == pytest.approx(before)
    assert after != pytest.approx(scale * before)
    assert chi_oracle(10.0, chi_case.model.k0, chi_case.model.beta0) == pytest.approx(scale * correct)
