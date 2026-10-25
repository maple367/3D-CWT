import numpy as np
import pytest

from cwt3d.geometry import CircleHole, RightTriangleHole, SampledPermittivity, sample_permittivity


def test_circle_broadcast_material_and_average():
    hole = CircleHole(0.2, period_um=(1, 1.4), center_um=(0.4, 0.7), eps_background=4+2j, eps_hole=1-1j)
    assert hole.fill_fraction == pytest.approx(np.pi * .2**2 / 1.4)
    assert hole.mean_epsilon == pytest.approx(4+2j + (-3-3j)*hole.fill_fraction)
    values = hole(np.array([[.4, .9]]), np.array([[.7], [1.2]]))
    np.testing.assert_array_equal(values, [[1-1j, 4+2j], [4+2j, 4+2j]])


def test_triangle_unequal_legs_centroid_and_area():
    t = RightTriangleHole((.3, .6), center_um=(.5, .5), eps_background=4, eps_hole=1)
    assert t.fill_fraction == pytest.approx(.09)
    np.testing.assert_allclose(t.vertices_um.mean(axis=0), [.5, .5])
    assert t(.5, .5) == 1
    assert t(.9, .9) == 4
    assert t.mean_epsilon == pytest.approx(3.73)


def test_cell_center_sampling_and_snapshot():
    g = sample_permittivity(lambda x, y: x+2j*y, (2, 3), (5, 8))
    assert g.values.shape == (5, 8)
    assert g.values[0, 0] == pytest.approx(.125+.6j)
    a = np.ones((3, 4)); s = SampledPermittivity(a)
    a[:] = 9
    assert np.all(s.values == 1)
    assert not s.values.flags.writeable


@pytest.mark.parametrize('kwargs', [dict(radius_um=0), dict(radius_um=-1), dict(radius_um=.6), dict(radius_um=.1, center_um=(0, .5)), dict(radius_um=.1, period_um=(1, 0)), dict(radius_um=np.nan), dict(radius_um=.1, eps_hole=np.inf), dict(radius_um=.1, center_um=(.5+1j,.5))])
def test_bad_circle(kwargs):
    with pytest.raises((ValueError, TypeError)):
        CircleHole(**kwargs)


@pytest.mark.parametrize('legs', [(0,.2),(-.1,.2),(.2,2),(np.nan,.2),(.2,),(.2+1j,.3)])
def test_bad_triangle(legs):
    with pytest.raises((ValueError, TypeError)):
        RightTriangleHole(legs)


@pytest.mark.parametrize('values', [[], [1,2], [[np.nan]], [['text']]])
def test_bad_grid(values):
    with pytest.raises((ValueError, TypeError)):
        SampledPermittivity(values)


@pytest.mark.parametrize('shape', [(0,5),(5,2.5),(True,5)])
def test_bad_sampling_shape(shape):
    with pytest.raises((ValueError, TypeError)):
        sample_permittivity(lambda x,y: 1, (1,1), shape)
