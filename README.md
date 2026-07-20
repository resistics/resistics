## Welcome

[![PyPI Latest Release](https://img.shields.io/pypi/v/resistics.svg)](https://pypi.org/project/resistics/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/resistics)](https://pypi.org/project/resistics/)
[![Documentation Status](https://readthedocs.org/projects/resistics/badge/?version=latest)](https://resistics.readthedocs.io/en/latest/?badge=latest)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/resistics/resistics.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/resistics/resistics/context:python)
[![codecov](https://codecov.io/gh/resistics/resistics/branch/master/graph/badge.svg?token=CXLJC9J7AW)](https://codecov.io/gh/resistics/resistics)
[![Code style: Ruff](https://img.shields.io/badge/code%20style-Ruff-D7FF64.svg)](https://docs.astral.sh/ruff/)

Resistics is a native Python 3.11-3.14 package for the processing of
magnetotelluric (MT) data. It incorporates robust processing methods and adopts
a modular approach to processing which allows for customisation and future
improvements to be quickly adopted.

## Latest news

Resistics is moving to version 1.0.0, which will be a breaking change versus
the current stable version of 0.0.6. The `mth5` branch contains the in-development
1.0 line.

- Documentation for 1.0.0: https://resistics.readthedocs.io/
- Documentation for 0.0.6: https://resistics.io/

When version 1.0.0 reaches a stable release the documentation will move to the
main resistics.io site.

## Audience

Resistics is intended for people who use magnetotelluric methods to estimate the
subsurface resistivity. This may be for furthering geological understanding, for
geothermal prospecting or for other purposes.

The package may have utility for the wider electromagnetic geophysics community.

## Getting started

To install the stable 0.0.6 version:

```console
python -m pip install resistics
```

For development of the 1.0 line, use the paired local setup below.

## Developing the mth5 branch

The `mth5` hardening branch intentionally uses an editable checkout of
RegressionInC. Keep both repositories beside each other with these directory
names:

```text
<development-directory>/
├── regressioninc/
└── resistics/
```

From the `resistics` directory, create the complete locked environment with:

```console
uv sync --locked --all-groups
```

Install the repository's pre-commit hook once, then run the same maintained
file-hygiene and locked Python quality checks on demand with:

```console
uv run --locked --no-sync pre-commit install
uv run --locked --no-sync pre-commit run --all-files
```

The complete local production gate is:

```console
uv run --locked --no-sync ruff format --check resistics tests scripts
uv run --locked --no-sync ruff check resistics tests scripts
uv run --locked --no-sync pydoclint resistics
uv run --locked --no-sync pyrefly check
uv run --locked --no-sync pytest
uv run --locked --no-sync pytest --cov=resistics --cov-branch --cov-report=term --cov-report=html --cov-report=xml
uv run --locked --no-sync sphinx-build -D sphinx_gallery_conf.plot_gallery=0 -b html docs/source .artifacts/hardening/documentation/html
uv build --no-sources
uv audit --locked
uv run --locked --no-sync python scripts/check_no_legacy_packaging.py
```

The gallery-disabled documentation command retains the measured legacy warning
backlog until the Phase 7 documentation migration. Dependency-audit policy is
hardened in Phase 6. Pyrefly is the sole type checker. All production modules
are error- and warning-clean under the pinned checker, and the distribution
advertises its inline public annotations through PEP 561. The committed empty
baseline makes any new error-level finding fail the local gate; updating it is
a deliberate maintenance action, not part of the normal local gate.

The Python pre-commit hooks use `uv run --locked --no-sync` internally. Run
the sync command above after pulling a lockfile or dependency change; hooks
will fail rather than silently alter that environment.

The sibling path is declared in `pyproject.toml` as `../regressioninc`. If uv
reports that this path does not exist, clone or move the RegressionInC
repository into the layout above before syncing. This development branch does
not currently claim standalone installation from a package registry.

## Support and feature requests

Feel free to submit issues, feature requests or ideas for improvements in the
Github issues section.
