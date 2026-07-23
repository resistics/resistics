(contributing)=
# Contributing

Resistics uses one locked uv environment for development, tests, documentation,
and packaging. The maintained workflow is local and reproducible; hosted
automation is not currently part of the verified contract.

## Prepare the checkout

Use Python 3.12, 3.13, or 3.14 and install
[uv](https://docs.astral.sh/uv/). Resistics 1.0 uses an editable sibling
checkout of RegressionInC, so keep the repositories beside each other:

```text
<development-directory>/
├── regressioninc/
└── resistics/
```

From the `resistics` directory, create or update the complete environment:

```console
uv sync --locked --all-groups
uv run --locked --no-sync pre-commit install
```

The `--locked` flag prevents a contributor command from silently changing the
resolved environment. Run the sync again after pulling a dependency or lockfile
change.

## Work on a change

Add focused tests for the behaviour being changed, then run the relevant test
module while iterating. Before handing off a change, run the complete source
suite and coverage gate:

```console
uv run --locked --no-sync pytest -q tests/test_project.py
uv run --locked --no-sync pytest
uv run --locked --no-sync python scripts/check_coverage.py
```

Coverage is branch-aware and must remain at or above `75.95%`. This precise
floor protects the measured baseline without claiming a rounded 76 percent.
Do not lower the floor, omit difficult modules, or add a coverage pragma to
make a change pass. Exercise new branches directly. A pragma is acceptable
only for genuinely unreachable defensive or instrumentation code, and its
reason must be visible beside the exclusion.

The coverage command combines the normal source suite with the same Sphinx
doctest builder that owns the 808 executable docstring prompts. It excludes
only tests marked `performance`, because coverage instrumentation invalidates
their elapsed-time measurements; run those focused tests separately below.

Run formatting, lint, docstring, and type checks with:

```console
uv run --locked --no-sync ruff format --check resistics tests scripts
uv run --locked --no-sync ruff check resistics tests scripts
uv run --locked --no-sync pydoclint --config=pyproject.toml resistics
uv run --locked --no-sync pyrefly check
uv run --locked --no-sync pre-commit run --all-files
```

Ruff is the sole formatter and general linter. Pydoclint checks that MyST
field lists agree with signatures and annotations. Pyrefly is the sole type
checker, and `pyrefly-baseline.json` must remain an empty JSON object: do not
regenerate it to accept new errors.

## Write documentation

All maintained documentation is Markdown with MyST extensions. API reference
pages use the repository's single MyST-aware standard-autodoc adapter; do not
add an alternative API generator or authored reStructuredText.

The {ref}`docstring authoring guide <docstring-authoring>` defines required
coverage, the content checklist, private-helper criteria, and executable
example and plot conventions. In particular, rich examples and plots belong
with the function or class they document when that is where users will find
them most useful.

Run the permanent local documentation gate after changing documentation or a
public API:

```console
uv run --locked --no-sync python scripts/check_documentation.py
```

It creates a fresh, nitpicky, warning-fatal HTML site, executes the six
tutorials, verifies every protected Matplotlib artifact, and runs the fenced
docstring examples with Sphinx's doctest builder. HTML is the only supported
published output. External links require network access and have their own
explicit check:

```console
uv run --locked --no-sync python scripts/check_documentation.py links
```

## Check TUI performance

Footer binding checks must remain cached and free of project or filesystem I/O.
Binding refreshes must remain owned by state transitions. Run their focused
regressions after changing TUI actions, selection state, refresh behaviour, or
project-service boundaries:

```console
uv run --locked --no-sync pytest -q tests/test_tui.py -k "cached_action_checks_are_fast or binding_refreshes_follow_owned_state_transitions"
```

For changes that can affect startup or imports, record a fresh-process sample
in the ignored hardening artifact directory:

```console
uv run --locked --no-sync python scripts/hardening_report.py --import-samples 5 --output .artifacts/hardening/performance/tui-import.json
```

Compare like-for-like samples on the same machine. These focused checks support
the responsiveness contract; they do not replace the complete source suite.

(justified-suppressions)=
## Justify suppressions

Fix the underlying code or improve its annotation or documentation first. When
a tool is wrong or a framework contract cannot be expressed directly, a
suppression must:

1. apply to the smallest declaration or expression possible;
2. name only the necessary rule or diagnostic;
3. include a nearby explanation of the invariant or framework constraint; and
4. retain a focused test when the suppressed contract affects behaviour.

Do not add broad file, package, or global exclusions merely to make a gate
green. A shared configuration exception needs an equally narrow file pattern
and an explanatory comment. Public API pydoclint findings and Pyrefly errors
must not be moved into baselines. Review existing suppressions when touching
their code and remove them when the tool or implementation no longer needs
them.

## Before hand-off

Keep unrelated worktree changes intact. Run `git diff --check`, review the
rendered documentation when it changed, and record intentional changes to
coverage, performance, public contracts, or tool suppressions. The
{ref}`release verification guide <release-verification>` contains the broader
candidate checklist.
