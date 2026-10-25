"""Small-grid checks for the chapter 4 circular-hole example."""
import csv
import importlib.util
import inspect
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'examples' / 'liang_70um.py'


@pytest.fixture(scope='module')
def example():
    spec = importlib.util.spec_from_file_location('liang_70um', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def case(example):
    return example.run_case(N=4, D=1, q=8, F=16, k=3)


def test_structure_and_bragg(example):
    a, eps, thickness, layers = example.build_structure()
    hole = layers[6]
    assert a == .295
    assert hole.fill_fraction == pytest.approx(.16)
    assert hole.radius_um == pytest.approx(a * np.sqrt(.16 / np.pi))
    assert hole.center_um == (a / 2, a / 2)
    np.testing.assert_allclose(eps, [3.307**2, 3.524**2, 3.553**2,
        3.524**2, 3.279**2, 3.524**2, hole.mean_epsilon.real, 3.524**2, 3.307**2])
    np.testing.assert_allclose(thickness, [.030, .064, .040, .040, .022, .118, .020])
    reference = example.solve_reference_mode(a, eps, thickness)
    assert abs(reference.mode.beta - 2 * np.pi / a) / reference.mode.beta < 1e-10
    assert reference.bracket[0] < reference.k0 < reference.bracket[1]


def test_defaults(example):
    args = example.parser().parse_args([])
    assert (args.L_um, args.N, args.D, args.q, args.F, args.k) == (70, 64, 16, 48, 64, 1)
    assert args.backend == 'prepared'
    assert args.output == Path('results/liang-70um')
    assert args.sigma is None
    assert inspect.signature(example.run_case).parameters['L_um'].default == 70


def test_api_residual_and_order(case):
    assert case.pencil.grid.Lx_um == 70
    values = case.eigenpairs.eigenvalues
    assert np.all(np.diff(values.imag) >= 0)
    np.testing.assert_array_equal(values, [m.eigenvalue for m in case.modes])
    a, b = case.pencil.A, case.pencil.B
    for i, mode in enumerate(case.modes):
        vector = case.eigenpairs.eigenvectors[:, i]
        assert np.linalg.norm(a @ vector - values[i] * (b @ vector)) / np.linalg.norm(vector) < 1e-8
        assert mode.residual < 1e-9
        assert mode.budget['balance_relative_error'] < 1e-7
        assert mode.budget['center_power'] == pytest.approx(1)
    bandedge = min(np.linalg.eigvals(case.coupling.C).real)
    assert case.summary['shift_um_inverse'] == pytest.approx([bandedge - .6 / 70, .25 / 70])
    assert 'targeted subset' in case.summary['spectrum_scope']
    json.dumps(case.summary, allow_nan=False)


def test_export_consistency(example, case, tmp_path):
    out = tmp_path / 'export'
    example.export_case(case, out)
    summary = json.loads((out / 'summary.json').read_text())
    with (out / 'modes.csv').open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    with np.load(out / 'fields.npz', allow_pickle=False) as fields:
        assert fields['centers'].shape == (3, 4, 4, 4)
        assert fields['faces_Rx'].shape == (3, 4, 5)
        assert fields['faces_Ry'].shape == (3, 5, 4)
        assert fields['eigenvectors'].shape == (64, 3)
        np.testing.assert_array_equal(fields['eigenvalues'], case.eigenpairs.eigenvalues)
        np.testing.assert_allclose(fields['C'], sum(fields[n] for n in ('C1D', 'Crad', 'C2D')))
        for i, row in enumerate(rows):
            value = fields['eigenvalues'][i]
            assert float(row['alphaL']) == pytest.approx(value.imag * 70)
            assert float(row['deltaL']) == pytest.approx(value.real * 70)
            assert summary['modes'][i]['alpha_um_inverse'] == value.imag
            np.testing.assert_array_equal(fields['centers'][i, 0], case.modes[i].centers.Rx)
            assert fields['faces_Rx'][i, :, 0].max() == 0
            u = fields['centers'][i].reshape(4, -1)
            area = (70 / 4)**2
            assert area * np.vdot(u, u).real == pytest.approx(1)
            edge = (70 / 4) * sum(np.sum(abs(face)**2) for face in (
                fields['faces_Rx'][i, :, -1], fields['faces_Sx'][i, :, 0],
                fields['faces_Ry'][i, -1, :], fields['faces_Sy'][i, 0, :]))
            radiation = 2 * area * np.vdot(u, fields['Crad'] @ u).imag
            assert edge == pytest.approx(float(row['edge']))
            assert radiation == pytest.approx(float(row['Crad']), abs=1e-13)
            assert 2 * value.imag == pytest.approx(edge + float(row['coupling']), abs=1e-9)
        assert all(np.isfinite(fields[name]).all() for name in fields.files)
    with pytest.raises(ValueError, match='nonempty'):
        example.export_case(case, out)


def run_cli(tmp_path, *args):
    return subprocess.run([sys.executable, '-I', str(SCRIPT), *args],
        cwd=tmp_path, capture_output=True, text=True, timeout=90)


def test_explicit_shift(example):
    sigma = -.08 + .012j
    result = example.run_case(N=3, D=1, q=8, F=16, k=2, sigma=sigma)
    assert result.eigenpairs.sigma == sigma
    assert result.summary['shift_um_inverse'] == [sigma.real, sigma.imag]
    assert result.summary['shift_rule'] == 'user supplied'
    assert np.all(np.diff(result.eigenpairs.eigenvalues.imag) >= 0)


def test_signed_alpha_preserved(example, case, monkeypatch):
    original = case.eigenpairs
    values = original.eigenvalues.copy()
    values[0] = complex(values[0].real, -values[0].imag)
    fake = example.EigenResult(values, original.eigenvectors, original.residuals,
                               dict(original.metadata))
    monkeypatch.setattr(example, 'solve_finite', lambda *a, **kw: fake)
    result = example.run_case(N=4, D=1, q=8, F=16, k=3)
    assert result.summary['modes'][0]['alphaL'] < 0
    assert result.modes[0].alpha == values[0].imag


def test_strict_json_export(example, case, tmp_path):
    summary = dict(case.summary, invalid=float('nan'))
    invalid = example.CaseResult(case.reference, case.coupling, case.pencil,
                                  case.eigenpairs, case.modes, summary)
    out = tmp_path / 'nonfinite'
    with pytest.raises(ValueError):
        example.export_case(invalid, out)
    assert not out.exists()


def test_cli_isolated(tmp_path):
    out = tmp_path / 'cli'
    process = run_cli(tmp_path, '--N', '3', '--D', '1', '--q', '8', '--F', '16',
                      '--k', '2', '--output', str(out))
    assert process.returncode == 0, process.stderr
    assert json.loads((out / 'summary.json').read_text())['solver']['returned'] == 2


@pytest.mark.parametrize('args', [('--N', '0'), ('--D', '0'), ('--q', '0'),
    ('--F', '0'), ('--k', '999999'), ('--L-um', 'nan'), ('--sigma', 'nan')])
def test_cli_bad_input(tmp_path, args):
    out = tmp_path / 'invalid'
    process = run_cli(tmp_path, '--output', str(out), *args)
    assert process.returncode != 0
    assert not out.exists()


def test_nonempty_rejected_before_compute(example, tmp_path, monkeypatch):
    (tmp_path / 'keep.txt').write_text('unchanged')
    def forbidden(*args, **kwargs):
        pytest.fail('expensive calculation started')
    monkeypatch.setattr(example, 'build_structure', forbidden)
    with pytest.raises(ValueError, match='nonempty'):
        example.run_case(output=tmp_path)
    assert (tmp_path / 'keep.txt').read_text() == 'unchanged'
    process = run_cli(tmp_path, '--output', str(tmp_path))
    assert process.returncode != 0
    assert 'nonempty' in process.stderr
