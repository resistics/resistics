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

From the `resistics` directory, create the complete locked environment and run
the test suite with:

```console
uv sync --locked --all-groups
uv run --locked --no-sync pytest
```

Build the current narrative and API documentation without re-running the
known-stale executable gallery examples with:

```console
uv run --locked --no-sync sphinx-build -D sphinx_gallery_conf.plot_gallery=0 -b html docs/source .artifacts/hardening/documentation/html
```

The sibling path is declared in `pyproject.toml` as `../regressioninc`. If uv
reports that this path does not exist, clone or move the RegressionInC
repository into the layout above before syncing. This development branch does
not currently claim standalone installation from a package registry.

## Support and feature requests

Feel free to submit issues, feature requests or ideas for improvements in the
Github issues section.
