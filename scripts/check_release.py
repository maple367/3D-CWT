#!/usr/bin/env python3
"""Read-only release gates. Build with ``python -m build`` before --artifacts.

Python 3.10 needs tomli; otherwise only the standard library is used. Archives
are inspected in memory, never extracted. Synthetic tests do not replace a
real sdist-to-wheel build and clean-environment installation smoke test.
"""
from __future__ import annotations

import argparse
from email.parser import BytesParser
from email.policy import default
import hashlib
from pathlib import Path
import re
import stat
import subprocess
import sys
import tarfile
import zipfile

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


REQUIRED_SDIST = frozenset({
    "pyproject.toml", "README.pypi.md", "README.md", "LICENSE",
    "src/cwt3d/finite/eigensolver.py", "examples/liang_70um.py",
    "examples/plot_liang_70um.py", "tests/test_end_to_end.py",
    "tests/regressions/_legacy.py", "model/_Model.py", "calculator/_calc.py",
    "docs/images/liang-70um.png", "docs/api.md",
    "examples/active_cavity.py", "scripts/plot_active_trajectory.py",
    "tests/test_active_integration.py", "tests/test_active_plot.py",
    "docs/active-cavity.md", "docs/validation/active-cavity/README.md",
    "docs/validation/active-cavity/refinement.png", "docs/validation/active-cavity/refinement.json",
    "docs/validation/active-cavity/long-n4.png", "docs/validation/active-cavity/long-n4.json",
})
MAX_FILE_SIZE = 32 * 1024 * 1024
MAX_TOTAL_SIZE = 128 * 1024 * 1024


class ReleaseError(ValueError):
    """A release gate rejected the candidate."""


def check_tag(tag: str, version: str) -> None:
    if tag != "v" + version:
        raise ReleaseError(f"tag must equal v{version} literally")


def check_main_ancestor(root: Path) -> None:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", "HEAD", "refs/remotes/origin/latest"],
        cwd=root, capture_output=True, check=False,
    )
    if result.returncode != 0:
        raise ReleaseError("HEAD is not a verified ancestor of origin/latest; fetch origin/latest first")


def _safe_path(name: str, directory: bool = False) -> str:
    if directory and name.endswith("/"):
        name = name[:-1]
    parts = name.split("/")
    if (not name or "\\" in name or ":" in name or
            any(part in {"", ".", ".."} for part in parts) or
            any(ord(char) < 32 or ord(char) == 127 for char in name)):
        raise ReleaseError("unsafe archive path")
    return name


def _sensitive(name: str, *, wheel: bool) -> None:
    for part in name.lower().split("/"):
        if (part in {".git", ".venv", "venv", "__pycache__", ".pytest_cache"} or
                part.startswith(".env") or "credential" in part or
                part.endswith((".mph", ".pth", ".pkl", ".pickle", ".pyc", ".pyo", ".pem", ".key")) or
                (wheel and part.endswith(".egg-info"))):
            raise ReleaseError("forbidden archive content")


def _read_archive(path: Path, *, wheel: bool, roots: set[str]) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    seen: set[str] = set()
    total = 0
    def record(name, size, directory):
        nonlocal total
        clean = _safe_path(name, directory)
        _sensitive(clean, wheel=wheel)
        if clean.split("/")[0] not in roots:
            raise ReleaseError("unexpected archive root")
        if wheel and not directory and clean.startswith("cwt3d/"):
            if not (clean.endswith(".py") or clean == "cwt3d/py.typed"):
                raise ReleaseError("wheel contains unexpected non-source package data")
        if clean.casefold() in seen:
            raise ReleaseError("duplicate archive path")
        seen.add(clean.casefold())
        total += size
        if size < 0 or size > MAX_FILE_SIZE or total > MAX_TOTAL_SIZE:
            raise ReleaseError("archive size limit exceeded")
        return clean
    if wheel:
        with zipfile.ZipFile(path) as archive:
            for item in archive.infolist():
                mode = item.external_attr >> 16
                if stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR):
                    raise ReleaseError("archive links or special entries are forbidden")
                name = record(item.filename, item.file_size, item.is_dir())
                if not item.is_dir():
                    files[name] = archive.read(item)
    else:
        with tarfile.open(path, "r:gz") as archive:
            for item in archive:
                if not (item.isfile() or item.isdir()):
                    raise ReleaseError("archive links or special entries are forbidden")
                name = record(item.name, item.size, item.isdir())
                if item.isfile():
                    stream = archive.extractfile(item)
                    if stream is None:
                        raise ReleaseError("unreadable archive entry")
                    files[name] = stream.read()
    return files


def _source_hash(data: bytes) -> str:
    return hashlib.sha256(data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")).hexdigest()


def _metadata(files: dict[str, bytes], project: dict, info: str) -> None:
    for required in ("METADATA", "WHEEL", "RECORD"):
        if info + "/" + required not in files:
            raise ReleaseError(f"wheel missing {required}")
    message = BytesParser(policy=default).parsebytes(files[info + "/METADATA"])
    if message.defects:
        raise ReleaseError("invalid wheel metadata")
    for key, expected in (("Name", project["name"]), ("Version", project["version"]),
                          ("Requires-Python", project["requires-python"])):
        if message.get_all(key) != [expected]:
            raise ReleaseError(f"wheel metadata {key} mismatch")
    if not any(name.startswith(info + "/") and Path(name).name.upper() == "LICENSE"
               and data.strip() for name, data in files.items()):
        raise ReleaseError("wheel missing license")
    for dependency in message.get_all("Requires-Dist", []):
        match = re.match(r"^\s*([A-Za-z0-9_.-]+)", dependency)
        if match is None:
            raise ReleaseError("invalid dependency")
        name = re.sub(r"[-_.]+", "-", match.group(1)).lower()
        if name in {"numba", "mph", "comsol"}:
            # Fail closed: allow only an explicit extra-only marker.
            marker = dependency.partition(";")[2]
            extra_only = re.fullmatch(
                r"\s*extra\s*==\s*(['\"])[A-Za-z0-9_.-]+\1\s*", marker
            )
            if not extra_only:
                raise ReleaseError(f"forbidden mandatory dependency: {name}")


def check_artifacts(directory: Path, project: dict, root: Path) -> list[tuple[str, str]]:
    if not directory.is_dir():
        raise ReleaseError("artifact directory does not exist")
    entries = [p for p in directory.iterdir() if p.name != "SHA256SUMS"]
    if any(p.is_symlink() or not p.is_file() for p in entries):
        raise ReleaseError("unexpected artifact directory entry")
    wheels = [p for p in entries if p.name.endswith(".whl")]
    sdists = [p for p in entries if p.name.endswith(".tar.gz")]
    if len(entries) != 2 or len(wheels) != 1 or len(sdists) != 1:
        raise ReleaseError("expected exactly one wheel and one sdist")
    name = re.sub(r"[-_.]+", "_", project["name"])
    version = project["version"]
    prefix = f"{name}-{version}"
    if sdists[0].name != prefix + ".tar.gz" or not wheels[0].name.startswith(prefix + "-"):
        raise ReleaseError("artifact filename name/version mismatch")
    wheel = _read_archive(wheels[0], wheel=True, roots={"cwt3d", prefix + ".dist-info"})
    packed = _read_archive(sdists[0], wheel=False, roots={prefix})
    if any(not key.startswith(prefix + "/") for key in packed):
        raise ReleaseError("sdist root name/version mismatch")
    sdist = {key[len(prefix) + 1:]: data for key, data in packed.items()}
    missing = REQUIRED_SDIST - sdist.keys()
    if missing:
        raise ReleaseError("sdist missing: " + ", ".join(sorted(missing)))
    info = prefix + ".dist-info"
    if any(key.split("/")[0] not in {"cwt3d", info} for key in wheel):
        raise ReleaseError("wheel must contain only cwt3d and its dist-info")
    _metadata(wheel, project, info)
    wheel_sources = {key: value for key, value in wheel.items() if key.startswith("cwt3d/") and key.endswith(".py")}
    sdist_sources = {key[4:]: value for key, value in sdist.items() if key.startswith("src/cwt3d/") and key.endswith(".py")}
    local_sources = {"cwt3d/" + p.relative_to(root / "src/cwt3d").as_posix(): p.read_bytes()
                     for p in (root / "src/cwt3d").rglob("*.py")}
    if not {"cwt3d/__init__.py", "cwt3d/finite/eigensolver.py"} <= wheel_sources.keys():
        raise ReleaseError("wheel missing core source modules")
    if wheel_sources.keys() != sdist_sources.keys() or wheel_sources.keys() != local_sources.keys():
        raise ReleaseError("core source file sets differ between checkout, sdist and wheel")
    for key, data in wheel_sources.items():
        if len({_source_hash(data), _source_hash(sdist_sources[key]), _source_hash(local_sources[key])}) != 1:
            raise ReleaseError(f"core source differs: {key}")
    return [(p.name, hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(entries)]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path)
    parser.add_argument("--tag")
    parser.add_argument("--require-main-ancestor", action="store_true")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    try:
        with (root / "pyproject.toml").open("rb") as stream:
            project = tomllib.load(stream)["project"]
        if args.tag is not None:
            check_tag(args.tag, project["version"])
        if args.require_main_ancestor:
            check_main_ancestor(root)
        hashes = check_artifacts(args.artifacts, project, root) if args.artifacts is not None else []
    except (ReleaseError, OSError, ValueError, KeyError, tarfile.TarError, zipfile.BadZipFile, RuntimeError) as exc:
        # OSError text can include absolute local paths; do not disclose it.
        detail = str(exc) if isinstance(exc, ReleaseError) else type(exc).__name__
        print(f"release check failed: {detail}", file=sys.stderr)
        return 1
    print(f"release check passed: {project['name']} {project['version']}")
    for filename, digest in hashes:
        print(f"{digest}  {filename}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
