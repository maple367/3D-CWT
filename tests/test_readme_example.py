"""Check README links, example code blocks, and the optional plotting script."""
import importlib.util
from pathlib import Path
import re

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_plot():
    spec = importlib.util.spec_from_file_location(
        'plot_liang_70um', ROOT / 'examples/plot_liang_70um.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_readme_local_links_exist():
    text = (ROOT / 'README.md').read_text(encoding='utf-8')
    for link in re.findall(r'\]\(([^)]+)\)', text):
        if not link.startswith(('https://', 'http://', '#')):
            assert (ROOT / link.split('#')[0]).is_file(), link


def test_documented_python_blocks_compile():
    for file in ('README.md', 'docs/api.md', 'docs/usage.md',
                 'docs/examples/liang-70um.md'):
        text = (ROOT / file).read_text(encoding='utf-8')
        for block in re.findall(r'```python\n(.*?)```', text, flags=re.S):
            compile(block, file, 'exec')


def test_plot_exported_numeric_envelope(tmp_path):
    pytest.importorskip('matplotlib')
    module = load_plot()
    # Deliberately rectangular cells catch swapped x/y axes.
    nx, ny = 5, 4
    centers = np.ones((1, 4, ny, nx), dtype=complex)
    centers[0, 0] *= np.arange(nx) + 1
    source = tmp_path / 'result'
    source.mkdir()
    np.savez_compressed(source / 'fields.npz', centers=centers,
        eigenvalues=np.array([-.09+.003j]),
        x_centers_um=(np.arange(nx)+.5)*70/nx,
        y_centers_um=(np.arange(ny)+.5)*70/ny,
        x_faces_um=np.arange(nx+1)*70/nx,
        y_faces_um=np.arange(ny+1)*70/ny)
    target = tmp_path / 'figure.png'
    assert module.plot_result(source, target) == target
    assert target.read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
    with pytest.raises(ValueError, match='mode'):
        module.plot_result(source, target, mode=-1)
