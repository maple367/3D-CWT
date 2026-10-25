"""Synthetic archive validation fixtures, not evidence of a real package build."""
from __future__ import annotations

import importlib.util
import io
from pathlib import Path
import subprocess
import sys
import tarfile
import zipfile

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_release.py"


@pytest.fixture
def checker():
    spec = importlib.util.spec_from_file_location("release_checker", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def archives(tmp_path, checker):
    project = {"name": "cwt3d", "version": "0.1.0", "requires-python": ">=3.10"}
    root = tmp_path / "project"
    (root / "src/cwt3d/finite").mkdir(parents=True)
    source = b"# synthetic fixture\n"
    for name in ("__init__.py", "finite/eigensolver.py"):
        (root / "src/cwt3d" / name).write_bytes(source)
    sdist = {name: b"fixture\n" for name in checker.REQUIRED_SDIST}
    sdist.update({"src/cwt3d/__init__.py": source, "src/cwt3d/finite/eigensolver.py": source})
    metadata = ("Metadata-Version: 2.4\nName: cwt3d\nVersion: 0.1.0\n"
                "Requires-Python: >=3.10\nLicense-Expression: MIT\n"
                'Requires-Dist: numba>=0.61; extra == "numba"\n\n').encode()
    wheel = {"cwt3d/__init__.py": source, "cwt3d/finite/eigensolver.py": source,
             "cwt3d-0.1.0.dist-info/METADATA": metadata,
             "cwt3d-0.1.0.dist-info/WHEEL": b"Wheel-Version: 1.0\n",
             "cwt3d-0.1.0.dist-info/RECORD": b"",
             "cwt3d-0.1.0.dist-info/licenses/LICENSE": b"MIT fixture\n"}
    dist = tmp_path / "dist"
    dist.mkdir()
    def build(wheel_changes=None, sdist_changes=None):
        w, s = dict(wheel), dict(sdist)
        for target, changes in ((w, wheel_changes), (s, sdist_changes)):
            for name, value in (changes or {}).items():
                if value is None:
                    target.pop(name, None)
                else:
                    target[name] = value
        with zipfile.ZipFile(dist / "cwt3d-0.1.0-py3-none-any.whl", "w") as archive:
            for name, data in w.items():
                archive.writestr(name, data)
        with tarfile.open(dist / "cwt3d-0.1.0.tar.gz", "w:gz") as archive:
            for name, data in s.items():
                info = tarfile.TarInfo("cwt3d-0.1.0/" + name)
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
        return checker.check_artifacts(dist, project, root)
    return build, dist, metadata, project, root


def test_complete_synthetic_archives(archives):
    result = archives[0]()
    assert len(result) == 2
    assert all(len(digest) == 64 for _, digest in result)


@pytest.mark.parametrize("tag", ["0.1.0", "v0.1.1", " v0.1.0", "refs/tags/v0.1.0", "v0.1.0\n"])
def test_tag_literal(checker, tag):
    with pytest.raises(checker.ReleaseError, match="tag"):
        checker.check_tag(tag, "0.1.0")
    checker.check_tag("v0.1.0", "0.1.0")


@pytest.mark.parametrize("name", ["../escape.py", "/escape.py", "C:/escape.py", "cwt3d/../escape.py", "cwt3d\\escape.py", "cwt3d//escape.py", "legacy.py", "model/_Model.py", "cwt3d/.env", "cwt3d/.venv/x", "cwt3d/credentials.json", "cwt3d/x.mph", "cwt3d/x.pth", "cwt3d/x.pkl", "cwt3d/x.egg-info/PKG-INFO"])
def test_wheel_bad_files(checker, archives, name):
    with pytest.raises(checker.ReleaseError):
        archives[0]({name: b"bad"})


@pytest.mark.parametrize("name", ["model/_Model.py", "calculator/_calc.py", "examples/liang_70um.py", "docs/api.md"])
def test_sdist_required(checker, archives, name):
    with pytest.raises(checker.ReleaseError, match="missing"):
        archives[0](sdist_changes={name: None})


@pytest.mark.parametrize("old,new", [(b"Name: cwt3d", b"Name: other"), (b"Version: 0.1.0", b"Version: 9.9"), (b">=3.10", b">=3.11")])
def test_metadata_mismatch(checker, archives, old, new):
    with pytest.raises(checker.ReleaseError):
        archives[0]({"cwt3d-0.1.0.dist-info/METADATA": archives[2].replace(old, new)})


@pytest.mark.parametrize("requirement", ["numba>=0.61", "mph>=1", "COMSOL", 'numba; python_version > "3.9"', 'numba; extra == "numba" or python_version > "3.9"'])
def test_runtime_dependencies(checker, archives, requirement):
    data = archives[2].replace(b"\n\n", ("\nRequires-Dist: " + requirement + "\n\n").encode())
    with pytest.raises(checker.ReleaseError, match="dependency"):
        archives[0]({"cwt3d-0.1.0.dist-info/METADATA": data})


def test_core_sources(checker, archives):
    with pytest.raises(checker.ReleaseError, match="source"):
        archives[0]({"cwt3d/__init__.py": b"# changed\n"})
    with pytest.raises(checker.ReleaseError, match="source"):
        archives[0]({"cwt3d/extra.py": b"# extra\n"})
    archives[0]({"cwt3d/__init__.py": b"# synthetic fixture\r\n"})


def test_exact_pair(checker, archives):
    (archives[1] / "unexpected.zip").write_bytes(b"bad")
    with pytest.raises(checker.ReleaseError):
        archives[0]()
    (archives[1] / "unexpected.zip").unlink()
    (archives[1] / "SHA256SUMS").write_text("fixture", encoding="utf-8")
    archives[0]()


def test_tar_links_and_traversal(checker, archives):
    build, dist, _, project, root = archives
    build()
    for name, kind in [("cwt3d-0.1.0/../outside", tarfile.REGTYPE), ("cwt3d-0.1.0/link", tarfile.SYMTYPE)]:
        with tarfile.open(dist / "cwt3d-0.1.0.tar.gz", "w:gz") as archive:
            info = tarfile.TarInfo(name)
            info.type = kind
            info.linkname = "outside"
            archive.addfile(info)
        with pytest.raises(checker.ReleaseError):
            checker.check_artifacts(dist, project, root)


def test_ancestry_read_only(checker, tmp_path, monkeypatch):
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 1)
    monkeypatch.setattr(checker.subprocess, "run", run)
    with pytest.raises(checker.ReleaseError, match="origin/latest"):
        checker.check_main_ancestor(tmp_path)
    assert calls == [["git", "merge-base", "--is-ancestor", "HEAD", "refs/remotes/origin/latest"]]


def test_real_git_ancestry(checker, tmp_path):
    def git(*args):
        return subprocess.run(["git", *args], cwd=tmp_path, check=True,
                              capture_output=True, text=True).stdout.strip()
    git("init", "--quiet")
    git("-c", "user.name=Release Test", "-c", "user.email=release@example.invalid",
        "commit", "--allow-empty", "-m", "fixture")
    git("update-ref", "refs/remotes/origin/latest", "HEAD")
    before = git("status", "--porcelain"), git("rev-parse", "HEAD")
    checker.check_main_ancestor(tmp_path)
    assert before == (git("status", "--porcelain"), git("rev-parse", "HEAD"))
    git("-c", "user.name=Release Test", "-c", "user.email=release@example.invalid",
        "commit", "--allow-empty", "-m", "not on latest")
    with pytest.raises(checker.ReleaseError):
        checker.check_main_ancestor(tmp_path)


@pytest.mark.parametrize("name", ["cwt3d-0.1.0.dist-info/licenses/LICENSE", "cwt3d-0.1.0.dist-info/METADATA", "cwt3d-0.1.0.dist-info/WHEEL", "cwt3d-0.1.0.dist-info/RECORD", "cwt3d/__init__.py"])
def test_wheel_missing_required(checker, archives, name):
    with pytest.raises(checker.ReleaseError):
        archives[0]({name: None})


def test_zip_symlink_and_duplicate(checker, archives):
    build, dist, _, project, root = archives
    path = dist / "cwt3d-0.1.0-py3-none-any.whl"
    build()
    with zipfile.ZipFile(path, "a") as archive:
        link = zipfile.ZipInfo("cwt3d/link.py")
        link.create_system = 3
        link.external_attr = 0o120777 << 16
        archive.writestr(link, "outside")
    with pytest.raises(checker.ReleaseError, match="links"):
        checker.check_artifacts(dist, project, root)
    build()
    with zipfile.ZipFile(path, "a") as archive:
        with pytest.warns(UserWarning, match="Duplicate"):
            archive.writestr("cwt3d/__init__.py", "duplicate")
    with pytest.raises(checker.ReleaseError, match="duplicate"):
        checker.check_artifacts(dist, project, root)


@pytest.mark.parametrize("name", [".env", "data/credentials.json", "data/model.mph", "data/weights.pth", "data/x.pickle"])
def test_sdist_sensitive_files(checker, archives, name):
    with pytest.raises(checker.ReleaseError):
        archives[0](sdist_changes={name: b"bad"})


def test_cli_bad_tag():
    result = subprocess.run([sys.executable, str(SCRIPT), "--tag", "v999"], capture_output=True, text=True)
    assert result.returncode != 0
    assert "tag" in result.stderr
