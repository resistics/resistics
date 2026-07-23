# Contributing to Resistics

The maintained contributor workflow, quality gates, documentation standards,
coverage policy, and focused TUI performance checks are documented in
[`docs/source/contributing.md`](docs/source/contributing.md).

Resistics 1.0 development uses an editable sibling checkout of RegressionInC:

```text
<development-directory>/
├── regressioninc/
└── resistics/
```

From the `resistics` directory, prepare the complete locked environment and run
the local file checks with:

```console
uv sync --locked --all-groups
uv run --locked --no-sync pre-commit install
uv run --locked --no-sync pre-commit run --all-files
```

Use the [release verification guide](docs/source/releasing.md) when evaluating
a release candidate. Publishing, protected tags, Trusted Publishing, and
hosted documentation deployment remain owner-managed follow-ups.
