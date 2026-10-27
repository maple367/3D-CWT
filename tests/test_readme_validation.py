"""Public literature report, citation placement and archive coverage."""
from pathlib import Path
import csv
import re

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'docs/validation/liang-reproduction'


def test_citations_precede_examples_and_legacy_readme_is_removed():
    text = (ROOT / 'README.md').read_text(encoding='utf-8')
    assert text.index('## Citing 3D-CWT') < text.index('## Example:')
    assert '10.3788/COL202624.043602' in text
    assert '10.1088/1361-6641/adb7fd' in text
    for old in ('## Previous versions', '# Release Notes', '# Requirements',
                'Latest - Version 2.0.0', 'original code and its version notes'):
        assert old not in text


def test_public_report_links_and_privacy():
    paths = [ROOT / 'README.md', *REPORT.rglob('*.md')]
    assert (REPORT / 'README.md').is_file()
    assert (REPORT / 'liang-reproduction-report.pdf').read_bytes().startswith(b'%PDF')
    for path in paths:
        text = path.read_text(encoding='utf-8')
        for link in re.findall(r'\]\(([^)]+)\)', text):
            if not link.startswith(('https://', 'http://', '#')):
                assert (path.parent / link.split('#')[0]).is_file(), (path, link)
        assert not re.search(r'(?i)(\b[A-Z]:[/\\]|/Users/|/home/|cache/scratch|deleg_[a-f0-9])', text)
    assert '*.pdf' in (ROOT / 'MANIFEST.in').read_text()


def test_readme_thresholds_match_archived_measurements():
    with (REPORT / 'finite/modes.csv').open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 60
    selected = [r for r in rows if int(r['N']) == 96 and float(r['L_um']) == 70]
    assert len(selected) == 8
    text = (ROOT / 'README.md').read_text(encoding='utf-8')
    for row in selected:
        assert f"{float(row['alphaL']):.6f}" in text
    assert '6.98%' in text
    assert 'not' in text and 'convergence' in text


def test_readme_documents_default_on_reciprocity_and_fallback():
    text = (ROOT / 'README.md').read_text(encoding='utf-8')
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib
    version = tomllib.loads((ROOT / 'pyproject.toml').read_text(encoding='utf-8'))['project']['version']
    for value in (version, '--no-reciprocity', 'reciprocity=False',
                  'enabled by default', 'geometric symmetry',
                  'docs/validation/reciprocal-green-overlap/README.md'):
        assert value in text
