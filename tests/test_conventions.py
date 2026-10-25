"""Independent analytic checks for the conventions, not a Fourier solver."""

import numpy as np
import pytest

from cwt3d import conventions as c


def test_time_sign():
    assert c.TIME_EXPONENT_SIGN == 1
    assert c.time_phase(1.0, np.pi / 2) == pytest.approx(1j)


def test_basic_waves_and_polarizations():
    assert c.BASIC_WAVE_ORDER == ("Rx", "Sx", "Ry", "Sy")
    assert [(w.name, w.m, w.n, w.electric_component) for w in c.BASIC_WAVES] == [
        ("Rx", 1, 0, "Ey"), ("Sx", -1, 0, "Ey"),
        ("Ry", 0, 1, "Ex"), ("Sy", 0, -1, "Ex"),
    ]
    with pytest.raises((AttributeError, TypeError)):
        c.BASIC_WAVES[0].m = 2


def test_spatial_sign_and_broadcasting():
    assert c.SPATIAL_EXPONENT_SIGN == -1
    assert c.FOURIER_COEFFICIENT_EXPONENT_SIGN == 1
    phase = c.spatial_phase(1, 0, beta0_per_um=1.0,
                            x_um=np.array([0, np.pi / 2]), y_um=0.0)
    np.testing.assert_allclose(phase, [1, -1j], atol=1e-15)


def test_positive_fourier_coefficient_and_negative_reconstruction():
    # Translated, complex plane wave distinguishes signs and x/y ordering.
    period_um = 2.3
    beta = 2 * np.pi / period_um
    axis = np.arange(32) * period_um / 32
    x, y = np.meshgrid(axis, axis, indexing="ij")
    m, n = 2, -3
    amplitude = 0.4 + 0.7j
    field = amplitude * np.exp(-1j * beta * (m * (x - 0.17) + n * (y + 0.09)))
    coefficient = np.mean(field * c.fourier_coefficient_phase(
        m, n, beta0_per_um=beta, x_um=x, y_um=y))
    expected = amplitude * np.exp(1j * beta * (m * 0.17 - n * 0.09))
    assert coefficient == pytest.approx(expected)
    wrong_mode = np.mean(field * c.fourier_coefficient_phase(
        -m, -n, beta0_per_um=beta, x_um=x, y_um=y))
    assert abs(wrong_mode) < 1e-14
    reconstructed = coefficient * c.spatial_phase(
        m, n, beta0_per_um=beta, x_um=x, y_um=y)
    np.testing.assert_allclose(reconstructed, field, atol=1e-14)


@pytest.mark.parametrize("value", [2.5, np.array([0.0, 2.5, -1.0])])
def test_length_and_inverse_length_units(value):
    assert c.INTERNAL_LENGTH_UNIT == "um"
    assert c.INTERNAL_INVERSE_LENGTH_UNIT == "um^-1"
    length_cm = c.convert_length(value, from_unit="um", to_unit="cm")
    inverse_cm = c.convert_inverse_length(value, from_unit="um^-1", to_unit="cm^-1")
    np.testing.assert_allclose(length_cm, np.asarray(value) * 1e-4)
    np.testing.assert_allclose(inverse_cm, np.asarray(value) * 1e4)
    np.testing.assert_allclose(c.convert_length(length_cm, from_unit="cm", to_unit="um"), value)
    np.testing.assert_allclose(c.convert_inverse_length(
        inverse_cm, from_unit="cm^-1", to_unit="um^-1"), value)
    assert isinstance(length_cm, float if np.isscalar(value) else np.ndarray)
    assert isinstance(inverse_cm, float if np.isscalar(value) else np.ndarray)
    # Dimensionless phase/attenuation is invariant under consistent conversion.
    np.testing.assert_allclose(length_cm * inverse_cm, np.asarray(value) ** 2)


@pytest.mark.parametrize("converter,unit", [
    (c.convert_length, "um"), (c.convert_inverse_length, "um^-1")])
def test_conversion_requires_explicit_dimension_correct_units(converter, unit):
    with pytest.raises(TypeError):
        converter(1.0)
    for bad in ("nm", "UM", "cm-1", "", None):
        with pytest.raises(ValueError):
            converter(1.0, from_unit=bad, to_unit=unit)
        with pytest.raises(ValueError):
            converter(1.0, from_unit=unit, to_unit=bad)
    other_dimension = "um^-1" if unit == "um" else "um"
    with pytest.raises(ValueError):
        converter(1.0, from_unit=other_dimension, to_unit=unit)


@pytest.mark.parametrize("value", [True, "1", 1 + 2j, [1, float("nan")], float("inf")])
@pytest.mark.parametrize("converter,unit", [
    (c.convert_length, "um"), (c.convert_inverse_length, "um^-1")])
def test_invalid_numeric_conversion_rejected(value, converter, unit):
    with pytest.raises((TypeError, ValueError)):
        converter(value, from_unit=unit, to_unit=unit)


@pytest.mark.parametrize("value", [1.2, np.array([0.0, 1.2, -0.5])])
def test_amplitude_and_power_loss(value):
    # E(s)=E(0) exp(-alpha*s); P(s)/P(0)=exp(-2*alpha*s).
    power_loss = c.amplitude_loss_to_power_loss(value)
    np.testing.assert_allclose(power_loss, 2 * np.asarray(value))
    np.testing.assert_allclose(np.abs(np.exp(-np.asarray(value) * 0.7)) ** 2,
                               np.exp(-power_loss * 0.7))
    assert isinstance(power_loss, float if np.isscalar(value) else np.ndarray)
