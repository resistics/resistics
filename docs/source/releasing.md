(release-verification)=
# Release verification

This page defines the verified local checks for a Resistics release candidate.
It does not authorise publishing, tagging, branch promotion, or repository
configuration changes.

## Local candidate checklist

Start from the paired checkout described in {ref}`contributing` and inspect the
worktree so that generated or unrelated files are not accidentally included.
Confirm that dependency metadata and the complete locked environment agree:

```console
uv lock --check
uv sync --locked --all-groups
```

Run the maintained quality, test, coverage, and documentation gates:

```console
uv run --locked --no-sync ruff format --check resistics tests scripts
uv run --locked --no-sync ruff check resistics tests scripts
uv run --locked --no-sync pydoclint --config=pyproject.toml resistics
uv run --locked --no-sync pyrefly check
uv run --locked --no-sync pytest
uv run --locked --no-sync python scripts/check_coverage.py
uv run --locked --no-sync python scripts/check_documentation.py
uv run --locked --no-sync python scripts/check_documentation.py links
```

Then verify dependency floors, known vulnerabilities, legacy packaging
boundaries, and distribution construction:

```console
uv run --locked --no-sync python scripts/check_minimum_dependencies.py
uv run --locked --no-sync python scripts/audit_dependencies.py
uv run --locked --no-sync python scripts/check_no_legacy_packaging.py
uv build --no-sources
```

The minimum-dependency check builds both local packages and smoke-tests their
installed wheels on Python 3.12. The dependency audit is networked and covers
every supported Python minor. The strict documentation gate emits and supports
HTML only. Review the wheel and source archive contents before treating the
candidate as ready, including `resistics/py.typed`, and install the artifacts
with the RegressionInC candidate in an isolated environment rather than
importing either source checkout.

Record the exact commits, interpreter versions, commands, results, artifact
paths, and any accepted advisory with its owner and review date. A local pass
demonstrates candidate readiness; it is not proof that a registry-only
installation or hosted deployment works.

## Owner follow-ups: not active or verified

The following work remains outside the current local hardening gate and must
not be described as already operational:

- Release RegressionInC through its own reviewed boundary before expecting a
  standalone registry installation of Resistics.
- Protect release tags and the publishing environment, then configure PyPI
  Trusted Publishing with OpenID Connect. Remove token secrets only after the
  protected path has been proven.
- Rebuild hosted CI with full-commit-SHA action pins and minimum permissions.
  Cover Linux on Python 3.12-3.14, the minimum and maximum supported versions
  on Windows and macOS, and run the quality and coverage gates once.
- Configure hosted documentation to check out RegressionInC beside Resistics,
  install the locked uv documentation group, run the same strict gate, and
  request HTML output only.

These follow-ups are owned by the repository owner because they change remote
repositories, credentials, protected environments, or publishing state. They
are deliberately not release commands for contributors to run locally.
