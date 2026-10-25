"""Execute the historical DFT return path; keep Fourier targets independent."""

import traceback

import numpy as np
import pytest

from ._legacy import LEGACY_BASELINE, legacy_class


@pytest.mark.parametrize("index", [(0, 0), (1, 2), (-1, 1)])
def test_legacy_dft_first_normal_index_call_raises_at_return(index):
    assert LEGACY_BASELINE == "520f4d8-unfixed"
    calculator = legacy_class("xi_calculator_DFT")()
    calculator.resolution = 5
    calculator.xi_array = np.arange(25).reshape(5, 5).astype(complex) + 0.25j
    calculator.array = {}
    assert not hasattr(calculator, "_xi")
    with pytest.raises(AttributeError, match="_xi") as caught:
        calculator._cal(index)
    # The normal assignment succeeded; failure is not an index/build/import error.
    assert calculator.array[index] == calculator.xi_array[index]
    frame = traceback.extract_tb(caught.value.__traceback__)[-1]
    assert frame.filename.replace("\\", "/").endswith("calculator/_calc.py")
    assert frame.name == "_cal"
    assert frame.line.strip() == "return self._xi"
    assert not hasattr(calculator, "_xi")


def direct_coefficient(field, x, y, period, order):
    """Unit-cell quadrature with positive phase; axes are explicitly (x,y)."""
    m, n = order
    weights = np.exp(2j * np.pi * (m * x + n * y) / period)
    # Uniform periodic endpoint-excluded quadrature: dx*dy / cell_area.
    dx = dy = period / field.shape[0]
    return np.sum(field * weights) * dx * dy / period**2


@pytest.mark.parametrize("size", [16, 31])
def test_direct_fourier_oracle_normalization_sign_axes_and_translation(size):
    period = 2.3
    axis = (np.arange(size) + 0.5) * period / size
    x, y = np.meshgrid(axis, axis, indexing="ij")
    m, n = 2, -3
    amplitude, background = 0.4 + 0.7j, 2.5
    shift_x, shift_y = 0.17, -0.09
    field = background + amplitude * np.exp(
        -2j * np.pi * (m * (x - shift_x) + n * (y - shift_y)) / period)
    expected = amplitude * np.exp(2j * np.pi * (m * shift_x + n * shift_y) / period)
    assert direct_coefficient(field, x, y, period, (m, n)) == pytest.approx(expected)
    assert direct_coefficient(field, x, y, period, (0, 0)) == pytest.approx(background)
    assert abs(direct_coefficient(field, x, y, period, (-m, -n))) < 1e-13
    assert abs(direct_coefficient(field, x, y, period, (n, m))) < 1e-13
    reconstructed = background + expected * np.exp(-2j * np.pi * (m * x + n * y) / period)
    np.testing.assert_allclose(reconstructed, field, atol=1e-13)
