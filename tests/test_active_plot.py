"""Plot/export regressions, including an actual unseeded active CLI run."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('active_plot', ROOT/'scripts/plot_active_trajectory.py')
plot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plot)


@pytest.mark.parametrize('shape', [(4,), (4, 2, 2)], ids=['photons', 'carriers'])
@pytest.mark.parametrize('reference,candidate,expected', [
    (0., 0., (0., 0.)),
    (0., 2., (2., None)),
    (4., 6., (2., .5)),
])
def test_difference_metrics(shape, reference, candidate, expected):
    assert plot._difference_metrics(np.full(shape, candidate), np.full(shape, reference)) == expected


@pytest.fixture(scope='module')
def zero_export(tmp_path_factory):
    output = tmp_path_factory.mktemp('zero-active')/'demo'
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    subprocess.run([sys.executable, str(ROOT/'examples/active_cavity.py'),
                    '--N', '2', '--D', '1', '--q', '4', '--F', '8',
                    '--duration-ns', '.002', '--samples', '4', '--initial-photons', '0',
                    '--output', str(output)], check=True, env=env,
                   capture_output=True, text=True, timeout=60)
    with np.load(output/'trajectory.npz', allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    np.testing.assert_array_equal(arrays['photon_number'], 0.)
    np.testing.assert_array_equal(arrays['field_vectors'], 0.)
    return output, arrays


def test_actual_zero_field_plot_cli(zero_export, tmp_path):
    directory, _ = zero_export
    figure, metrics = tmp_path/'plot.png', tmp_path/'metrics.json'
    result = subprocess.run([sys.executable, str(ROOT/'scripts/plot_active_trajectory.py'),
        '--reference', str(directory), '--compare', str(directory),
        '--output', str(figure), '--metrics', str(metrics)],
        capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert figure.read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
    rows = json.loads(metrics.read_text())['rows']
    assert len(rows) == 2
    for row in rows:
        assert row['max_photon_difference_over_reference_peak'] == 0.
        assert row['max_photon_absolute_difference'] == 0.
        assert row['max_carrier_absolute_difference'] == 0.
    assert json.loads(result.stdout)['rows'] == rows


def test_zero_carrier_reference(zero_export, tmp_path):
    directory, arrays = zero_export
    case = tmp_path/'zero-carriers'
    case.mkdir()
    (case/'summary.json').write_bytes((directory/'summary.json').read_bytes())
    arrays = dict(arrays, carrier_density_um3=np.zeros_like(arrays['carrier_density_um3']))
    np.savez_compressed(case/'trajectory.npz', **arrays)
    metrics = tmp_path/'metrics.json'
    plot.main(['--reference', str(case), '--output', str(tmp_path/'plot.png'), '--metrics', str(metrics)])
    row = json.loads(metrics.read_text())['rows'][0]
    assert row['max_carrier_difference_over_reference_peak'] == 0.
    assert row['max_carrier_absolute_difference'] == 0.


@pytest.mark.parametrize('invalid', ['nonfinite-metric', 'fingerprint'])
def test_invalid_export_leaves_no_half_output(zero_export, tmp_path, invalid):
    directory, arrays = zero_export
    case = tmp_path/'invalid'
    case.mkdir()
    summary = json.loads((directory/'summary.json').read_text())
    if invalid == 'nonfinite-metric':
        summary['final_window_relative_photon_variation'] = float('nan')
    else:
        summary['environment']['active_algorithm_digest'] = 'different'
    (case/'summary.json').write_text(json.dumps(summary))
    np.savez_compressed(case/'trajectory.npz', **arrays)
    figure, metrics = tmp_path/'plot.png', tmp_path/'metrics.json'
    with pytest.raises(ValueError):
        plot.main(['--reference', str(directory), '--compare', str(case),
                   '--output', str(figure), '--metrics', str(metrics)])
    assert not figure.exists()
    assert not metrics.exists()
