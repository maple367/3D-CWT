# Publishing cwt3d to PyPI

The workflow is [`.github/workflows/publish.yml`](../.github/workflows/publish.yml).
It runs tests, builds a source distribution, builds the wheel from that source
archive, checks metadata and archive contents, and tests the installed wheel
from the extracted source tree. It does not change the numerical core.

## One-time PyPI authorization

This uses PyPI Trusted Publishing, not a stored API token. Sign in to your own
PyPI account and open <https://pypi.org/manage/account/publishing/>. Under
**Add a new pending publisher → GitHub**, enter:

| Field | Value |
| --- | --- |
| PyPI project name | `cwt3d` |
| Owner | `maple367` |
| Repository name | `3D-CWT` |
| Workflow name | `publish.yml` |
| Environment name | `pypi` |

Use the workflow filename, not its display title or full path. All five fields
must match. A pending publisher creates the project on its first successful
upload; it does not reserve the name beforehand. If the project already exists,
its owner must add the same publisher in that project's publishing settings.

GitHub's `pypi` environment should allow **tags matching `v*` only**. Do not add
PyPI credentials to repository secrets, workflow files, or chat messages.
See [PyPI's pending-publisher documentation](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/).

## Verify without uploading

Open **Actions → Publish to PyPI → Run workflow**, select `latest`, and run it.
The manual event runs tests and creates downloadable distribution artifacts;
the PyPI upload job is always skipped. Artifacts expire after 14 days.
This is a build check, not proof that PyPI authorization has been configured.

## Publish a version

1. Set `[project].version` in `pyproject.toml` to the intended, unused PyPI
   version and commit it to `latest`. The first stable package version is `1.0.0`;
   historical application release numbers are separate.
2. Create a GitHub Release with the exact tag `v` followed by that version
   (for example, `v1.0.0`), targeting the tested commit on `latest`.
3. Publish it as a normal release. Drafts do not run the workflow; GitHub
   releases marked **pre-release** run checks but do not upload to PyPI.
4. Wait for the **Upload to PyPI** job to finish and check the files and version
   at <https://pypi.org/project/cwt3d/>.

A tag/version mismatch, a tagged commit outside `latest`'s ancestry, a failed
test, or an invalid distribution stops publication. Every new upload needs a
new version: PyPI files cannot be overwritten, and the workflow deliberately
does not silently skip existing files. If an upload fails after one file was
accepted, inspect the project files before retrying; use a new version when
necessary rather than deleting or replacing a published release.

The build/test jobs have read-only repository permissions. Only the isolated
upload job requests `id-token: write`; PyPI exchanges that GitHub identity for
a short-lived credential and records distribution attestations. Action
references are pinned to full commit hashes.

## Local checks

From a clean checkout:

```bash
python -m pip install ".[dev,plot]" build twine
python -m pytest -q
python -m build
python scripts/check_release.py --artifacts dist --tag v1.0.0
python -m twine check --strict dist/*
```

Replace the example tag when changing versions. `--require-main-ancestor` adds
the same ancestry gate used by the release workflow; it needs a fetched
`origin/latest` reference. The source archive includes examples, documentation
and regression fixtures; the wheel contains only the library and its package
metadata/license. Build outputs, local environments and experimental results
are not release inputs.

PyPI distribution is a software release, not a claim of fully converged PCSEL
predictions. The model's assumptions and numerical limits remain those stated
in the API and example documentation.
