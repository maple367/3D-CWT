"""Public active-module release documentation and archive gates."""
import importlib.util
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def test_readmes_expose_active_analysis_without_stale_exclusion():
    main=(ROOT/'README.md').read_text(encoding='utf-8')
    pypi=(ROOT/'README.pypi.md').read_text(encoding='utf-8')
    for text in (main,pypi):
        assert 'cwt3d.active' in text
        assert 'Carrier transport and above-threshold' not in text
        assert 'deterministic' in text
        assert 'illustrative' in text
    assert 'examples/active_cavity.py' in main
    assert 'scripts/plot_active_trajectory.py' in main
    assert 'docs/active-cavity.md' in main
    assert '## Active-cavity analysis' in main


def test_active_guide_is_available_in_release_not_only_development():
    text=(ROOT/'docs/active-cavity.md').read_text(encoding='utf-8')
    assert '(development module)' not in text
    assert 'not the published' not in text
    assert '1.4.0' in text
    assert 'coupling.k0' in text
    assert 'not calibrated' in text or 'not a calibrated' in text


def test_release_gate_requires_active_example_docs_and_plot_regression():
    spec=importlib.util.spec_from_file_location('active_release_gate',ROOT/'scripts/check_release.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert {'examples/active_cavity.py','scripts/plot_active_trajectory.py',
            'docs/active-cavity.md','tests/test_active_plot.py',
            'docs/validation/active-cavity/README.md'}<=module.REQUIRED_SDIST
