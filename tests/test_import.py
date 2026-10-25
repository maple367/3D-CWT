"""Import isolation checks run in fresh interpreters, not pytest's module cache."""

from pathlib import Path
import subprocess
import sys
import textwrap


def test_core_import_without_optional_or_legacy_modules():
    source = Path(__file__).resolve().parents[1] / "src"
    code = textwrap.dedent(f"""
        import importlib.abc
        import sys
        sys.path.insert(0, {str(source)!r})

        class BlockOptional(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in {{'mph', 'numba', 'model'}}:
                    raise AssertionError('Forbidden import: ' + fullname)

        sys.meta_path.insert(0, BlockOptional())
        import cwt3d
        from cwt3d import conventions
        assert conventions.BASIC_WAVE_ORDER == ('Rx', 'Sx', 'Ry', 'Sy')
        assert not any(name.split('.')[0] in {{'mph', 'numba', 'model'}}
                       for name in sys.modules)
    """)
    result = subprocess.run([sys.executable, "-I", "-c", code],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
