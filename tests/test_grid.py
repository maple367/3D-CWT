"""Task 8: independent staggered-layout, elimination and snapshot oracles."""
from dataclasses import FrozenInstanceError
import subprocess
import sys

import numpy as np
import pytest

from cwt3d.conventions import BASIC_WAVE_ORDER
from cwt3d.finite import StaggeredGrid, WaveFields


NAMES = ('Rx', 'Sx', 'Ry', 'Sy')


def test_01_rectangular_manual_indices_and_complete_cover():
    g = StaggeredGrid(2, 3, 8., 9.)
    assert BASIC_WAVE_ORDER == NAMES
    assert (g.n_dof, g.hx_um, g.hy_um) == (24, 4., 3.)
    expected = {
        'Rx': [[-1, 0, 1], [-1, 2, 3], [-1, 4, 5]],
        'Sx': [[6, 7, -1], [8, 9, -1], [10, 11, -1]],
        'Ry': [[-1, -1], [12, 13], [14, 15], [16, 17]],
        'Sy': [[18, 19], [20, 21], [22, 23], [-1, -1]],
    }
    ids = []
    for name, table in expected.items():
        assert g.component_shape(name) == np.shape(table)
        for row, line in enumerate(table):
            for col, wanted in enumerate(line):
                actual = g.free_index(name, row, col)
                assert actual == wanted
                if actual >= 0:
                    ids.append(actual)
    assert sorted(ids) == list(range(24))


def test_02_coordinates_row_y_column_x_and_distinct_lengths():
    g = StaggeredGrid(2, 3, 8., 9.)
    for name in NAMES:
        x, y = g.coordinates(name)
        xx = [0., 4., 8.] if name.endswith('x') else [2., 6.]
        yy = [1.5, 4.5, 7.5] if name.endswith('x') else [0., 3., 6., 9.]
        np.testing.assert_array_equal(x, np.tile(xx, (len(yy), 1)))
        np.testing.assert_array_equal(y, np.tile(yy, (len(xx), 1)).T)
    x, y = g.coordinates()
    np.testing.assert_array_equal(x, [[2, 6]]*3)
    np.testing.assert_array_equal(y, [[1.5]*2, [4.5]*2, [7.5]*2])


def test_03_complex_roundtrip_exact_boundaries_and_outgoing():
    g = StaggeredGrid(2, 3, 8, 9)
    v = np.arange(1, 25) + 1j*np.arange(30, 54)
    original = v.copy()
    f = g.unpack(v)
    np.testing.assert_array_equal(f.Rx, [[0, 1+30j, 2+31j], [0, 3+32j, 4+33j], [0, 5+34j, 6+35j]])
    assert np.all(f.Rx[:, 0] == 0) and np.all(f.Sx[:, -1] == 0)
    assert np.all(f.Ry[0, :] == 0) and np.all(f.Sy[-1, :] == 0)
    assert all(np.all(a != 0) for a in (f.Rx[:, -1], f.Sx[:, 0], f.Ry[-1, :], f.Sy[0, :]))
    np.testing.assert_array_equal(g.pack(f), original)
    again = g.unpack(g.pack(f))
    for name in NAMES:
        np.testing.assert_array_equal(getattr(again, name), getattr(f, name))
    v[:] = 0
    np.testing.assert_array_equal(g.pack(f), original)


def test_04_one_cell_intersections_are_not_all_dirichlet():
    g = StaggeredGrid(1, 1, 2, 7)
    f = g.unpack([1+2j, 3+4j, 5+6j, 7+8j])
    for name, expected in zip(NAMES, ([[0, 1+2j]], [[3+4j, 0]], [[0], [5+6j]], [[7+8j], [0]])):
        np.testing.assert_array_equal(getattr(f, name), expected)
    assert g.n_dof == 4
    np.testing.assert_array_equal(g.pack(f), [1+2j, 3+4j, 5+6j, 7+8j])


def test_05_center_interpolation_linear_bc_and_constants():
    g = StaggeredGrid(2, 3, 8, 9)
    x, _ = g.coordinates('Rx')
    _, y = g.coordinates('Ry')
    f = WaveFields((1+2j)*x, (2-1j)*(8-x), (3+1j)*y, (4-2j)*(9-y))
    g.pack(f)  # manufactured fields satisfy exactly their own incoming edge
    center = g.interpolate(f)
    xc, yc = g.coordinates()
    for name, expected in zip(NAMES, ((1+2j)*xc, (2-1j)*(8-xc), (3+1j)*yc, (4-2j)*(9-yc))):
        np.testing.assert_array_equal(getattr(center, name), expected)
    constants = WaveFields(*(np.full(g.component_shape(n), 2+3j) for n in NAMES))
    # Interpolation is geometric and does not impose BCs on arbitrary fields.
    for n in NAMES:
        np.testing.assert_array_equal(getattr(g.interpolate(constants), n), np.full((3, 2), 2+3j))


@pytest.mark.filterwarnings('ignore:Setting the .* on a NumPy array has been deprecated:DeprecationWarning')
@pytest.mark.parametrize('metadata', ['shape', 'dtype', 'strides'])
def test_06_public_metadata_and_bytes_snapshot_isolation(metadata):
    g = StaggeredGrid(2, 3, 8, 9)
    v = np.arange(24)+1j
    f = g.unpack(v)
    packed = g.pack(f)
    expected = packed.copy()
    retained_sibling = g.unpack(expected)
    center_before = g.interpolate(f)
    center_expected = tuple(getattr(center_before, n).copy() for n in NAMES)
    coords_expected = tuple(a.copy() for a in g.coordinates('Rx'))
    for public in (f.Rx, f.Sx, f.Ry, f.Sy, packed, *g.coordinates('Rx'), center_before.Rx):
        assert not public.flags.writeable
        with pytest.raises(ValueError):
            public.setflags(write=True)
        current = public
        while isinstance(current, np.ndarray):
            parent = current.base
            if metadata == 'shape':
                current.shape = (current.size,)
            elif metadata == 'dtype':
                current.dtype = np.uint8
            else:
                current.strides = (0,)*current.ndim
            current = parent
    np.testing.assert_array_equal(g.pack(f), expected)
    np.testing.assert_array_equal(g.pack(retained_sibling), expected)
    np.testing.assert_array_equal(g.pack(g.unpack(expected)), expected)
    for actual, wanted in zip(g.coordinates('Rx'), coords_expected):
        np.testing.assert_array_equal(actual, wanted)
    for n, wanted in zip(NAMES, center_expected):
        np.testing.assert_array_equal(getattr(center_before, n), wanted)
        np.testing.assert_array_equal(getattr(g.interpolate(f), n), wanted)
    with pytest.raises((FrozenInstanceError, AttributeError, TypeError)):
        g.Nx = 99
    with pytest.raises((FrozenInstanceError, AttributeError, TypeError)):
        f.Rx = np.zeros((3, 3))
    with pytest.raises((FrozenInstanceError, AttributeError, TypeError)):
        f.grid = StaggeredGrid(1, 1, 1, 1)
    raw = [getattr(f, n).copy() for n in NAMES]
    snapshot = WaveFields(*raw)
    raw[0][:] = 77
    np.testing.assert_array_equal(g.pack(snapshot), expected)


@pytest.mark.parametrize('value', [True, np.bool_(False), 2., 0, -1, np.nan, np.inf, 1+0j, '2'])
@pytest.mark.parametrize('axis', ['Nx', 'Ny'])
def test_07_invalid_cell_counts(value, axis):
    args = dict(Nx=2, Ny=3, Lx_um=8, Ly_um=9)
    args[axis] = value
    with pytest.raises((ValueError, TypeError)):
        StaggeredGrid(**args)


@pytest.mark.parametrize('value', [True, np.bool_(True), 0, -1, np.nan, np.inf, 1+0j, '2', [2]])
@pytest.mark.parametrize('axis', ['Lx_um', 'Ly_um'])
def test_08_invalid_lengths(value, axis):
    args = dict(Nx=2, Ny=3, Lx_um=8, Ly_um=9)
    args[axis] = value
    with pytest.raises((ValueError, TypeError)):
        StaggeredGrid(**args)


def test_09_underflow_spacing_and_numpy_scalars():
    with pytest.raises(ValueError):
        StaggeredGrid(2, 3, np.nextafter(0., 1.), 9)
    g = StaggeredGrid(np.int64(2), np.int32(3), np.float64(8), np.float32(9))
    assert g.n_dof == 24


@pytest.mark.parametrize('name,row,col', [('Rx', -1, 0), ('Sx', 0, -1), ('Ry', 4, 0), ('Sy', 0, 2), ('Rx', 0., 0), ('Rx', True, 0), ('bad', 0, 0)])
def test_10_invalid_index_not_negative_alias(name, row, col):
    with pytest.raises((ValueError, IndexError, TypeError)):
        StaggeredGrid(2, 3, 8, 9).free_index(name, row, col)


@pytest.mark.parametrize('vec', [np.zeros((24, 1)), np.zeros(23), np.full(24, np.nan), np.full(24, np.inf), ['bad']*24])
def test_11_invalid_vectors(vec):
    with pytest.raises((ValueError, TypeError)):
        StaggeredGrid(2, 3, 8, 9).unpack(vec)


@pytest.mark.parametrize('name', NAMES)
def test_12_reject_each_nonzero_incoming_boundary(name):
    g = StaggeredGrid(2, 3, 8, 9)
    arrays = {n: getattr(g.unpack(np.ones(24)), n).copy() for n in NAMES}
    pos = {'Rx': (0, 0), 'Sx': (0, -1), 'Ry': (0, 0), 'Sy': (-1, 0)}[name]
    arrays[name][pos] = 1e-300j
    with pytest.raises(ValueError, match='incoming'):
        g.pack(WaveFields(**arrays))


@pytest.mark.parametrize('bad', [np.zeros((2, 2)), np.full((3, 3), np.nan), np.full((3, 3), np.inf)])
def test_13_invalid_full_fields(bad):
    g = StaggeredGrid(2, 3, 8, 9)
    with pytest.raises((ValueError, TypeError)):
        f = WaveFields(bad, np.zeros((3, 3)), np.zeros((4, 2)), np.zeros((4, 2)))
        g.pack(f)
    with pytest.raises((ValueError, TypeError)):
        f = WaveFields(bad, np.zeros((3, 3)), np.zeros((4, 2)), np.zeros((4, 2)))
        g.interpolate(f)


def test_14_fresh_import_and_each_operation_without_legacy():
    code = '''
import sys
import numpy as np
from cwt3d.finite import StaggeredGrid
for nx, ny in [(1,1), (2,3), (3,2)]:
    g = StaggeredGrid(nx, ny, 8, 9)
    f = g.unpack(np.arange(g.n_dof)+1j)
    g.pack(f); g.interpolate(f); g.coordinates('Rx'); g.free_index('Sy', 0, 0)
assert not any(n.split('.')[0] in {'model','calculator','utils','mph','numba'} for n in sys.modules)
'''
    subprocess.run([sys.executable, '-c', code], check=True)
