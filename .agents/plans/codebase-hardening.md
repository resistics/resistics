# Resistics Codebase Hardening Plan

Status: proposed for review  
Created: 2026-07-18  
Scope: resistics and, where explicitly identified, the sibling `regressioninc`
package

## Purpose

Harden resistics without attempting one large rewrite. The work should improve
terminal UI responsiveness, simplify maintenance, remove Poetry completely,
modernise development and release tooling, make static analysis useful again,
and reduce architectural weak spots while preserving existing behaviour.

This is an incremental delivery plan. Every numbered checkpoint is intended to
be a safe opportunity to commit and push a coherent change to GitHub. Later
checkpoints should not be bundled into an earlier commit merely because they
are in the same phase.

## Desired Outcomes

- The test suite has a documented green baseline and protects important
  processing, serialization, plotting, and TUI contracts.
- Poetry and all Poetry-generated or Poetry-specific references are gone from
  resistics and `regressioninc`.
- uv is the primary Python environment, dependency, build, and publishing tool.
- Ruff replaces Black and Flake8; pydoclint replaces darglint.
- A fast, useful type checker replaces mypy, with existing findings baselined
  and new regressions rejected.
- The TUI remains responsive while projects and metadata are loaded.
- Expensive filesystem, HDF5, YAML, JSON, validation, and plotting work is not
  performed from frequently called UI state checks.
- Large modules are split at stable boundaries without changing public APIs in
  the same commit.
- CI tests supported Python versions and operating systems, and publishing uses
  reproducible builds and PyPI Trusted Publishing.
- Dependency, security, type, coverage, and performance checks are useful
  signals rather than permanently failing or ignored jobs.

## Constraints and Guardrails

- MTH5 remains the only public input format.
- New and migrated models use Pydantic v2 APIs.
- Core resistics remains standalone and must not depend on app, GUI, web, IPC,
  or app-specific state.
- Numerical complex-domain regression remains in `regressioninc`; resistics
  owns MT-specific preparation and result packaging.
- Preserve the current command-line and TUI entry points unless a deprecation
  is explicitly agreed.
- Preserve user work already present in the worktree. In particular, the
  current edits to `.github/workflows/commit_flow.yml` and
  `.github/workflows/publish_flow.yml` must be incorporated, not overwritten.
- Do not combine behaviour changes, broad formatting, file moves, and dependency
  updates in one commit.
- Do not publish a package, create a release, push a tag, or alter repository
  settings merely because the corresponding workflow has been prepared.
- Changes in the sibling `regressioninc` repository require a separately
  reviewed branch and commits in that repository.

## Audit Baseline

The following figures were observed on branch `mth5` at commit `9873a47` on
2026-07-18. Re-measure them before starting implementation because the branch
may have moved.

- Production Python: approximately 21,011 lines across 37 files.
- Tests: approximately 5,941 lines.
- Since the first substantial TUI commit, production code grew by about 5,756
  net lines and tests by about 2,882 net lines.
- `resistics/tui.py`: 2,673 lines, 13 classes, and 189 functions or methods.
- `resistics/plot.py`: 1,200 lines; `plot_job` and `plot_flow` contain substantial
  parallel rendering logic.
- Other large modules include `time.py`, `gather.py`, `window.py`,
  `testing.py`, and `project.py`.
- Tests: 372 collected, 371 passed, and one transfer-function doctest failed
  because expected channel case no longer matches actual canonical output.
- Branch coverage: approximately 76% overall. Lower-covered important modules
  include `gather`, `project`, `calibrate`, and the TUI.
- Flake8 reported roughly 40 findings, including an undefined
  `GatherCriteria` reference and several high-complexity functions.
- mypy reported 210 errors across 17 files and was not operating as a useful
  gate.
- Importing `resistics.tui` took approximately 2.18 seconds in the audit
  environment and eagerly imported plotting and scientific dependencies.
- The local `.venv` occupied approximately 897 MB. This is environment weight,
  not source-code bloat, and should not drive architectural rewrites.

Likely TUI performance causes identified during the audit:

- Project and MTH5 loading occurs synchronously on the UI thread in important
  paths.
- Project mounting scans directories, walks HDF5 contents, parses configuration
  files, and validates jobs before the screen becomes fully usable.
- `check_action` and binding refresh paths can perform filesystem access,
  parsing, validation, solution loading, and MTH5 queries.
- Cursor movement and tab activation trigger redundant binding refreshes.
- Plotting/scientific libraries are imported eagerly by the TUI module.
- Unconditional `tqdm` output in regression work can compete with Textual for
  terminal rendering.

## Commit and Review Strategy

Each checkpoint below should normally become one commit. A checkpoint may be
split further whenever its diff becomes difficult to review.

Before each commit:

1. Inspect `git diff --check` and the complete staged diff.
2. Run the smallest focused test set that exercises the change.
3. Run the configured formatter and linter for touched files.
4. Record intentional baseline changes, such as coverage or type findings.
5. Avoid staging unrelated user changes.

Before each push or pull request:

1. Run the full locally available quality suite.
2. Confirm generated locks and build metadata are consistent.
3. Confirm public API changes are documented or explicitly absent.
4. Add a short before/after measurement for performance-related changes.

Soft review limits:

- Prefer one concern and fewer than roughly 400 changed logical lines per
  commit.
- Mechanical formatting or file moves may exceed that limit, but must be
  isolated so GitHub can recognise renames and reviewers can separate movement
  from behaviour.
- Stop at every phase review gate. Do not start the next phase merely to make a
  pull request appear more substantial.

Suggested branch strategy:

- Use a short-lived branch per phase or tightly related group of checkpoints.
- Push checkpoint commits as backups even when a pull request is still a draft.
- Squashing is optional, but retain separate commits when they materially help
  review, `git bisect`, or rollback.

## Phase 0: Restore and Measure the Baseline

Goal: establish a trustworthy starting point before changing tools or
architecture.

### Checkpoint 0.1: Make the existing suite green

- Fix the transfer-function doctest expectation or implementation according to
  the canonical lowercase channel naming contract.
- Fix the undefined `GatherCriteria` name and other genuine correctness errors
  already identified by the current lint configuration.
- Avoid broad cleanup in this commit.

Verification:

- Full pytest suite passes.
- Existing Black and Flake8 checks pass or the remaining known findings are
  documented for the Ruff migration.

Suggested commit: `fix: restore green test and lint baseline`

### Checkpoint 0.2: Add durable quality and performance measurements

- Set coverage failure at the confirmed baseline, initially 76%, so coverage
  cannot silently regress.
- Add focused tests proving frequently called TUI action-state checks do not
  perform project, filesystem, MTH5, YAML, or JSON I/O.
- Add a repeatable cold-import timing command and representative TUI response
  benchmark or profiling fixture. Keep timing assertions tolerant enough for CI
  variance; use them primarily for before/after evidence.
- Capture current module sizes and complexity in documentation or a lightweight
  reporting command rather than creating hard line-count gates.

Suggested commit: `test: establish hardening baselines`

### Phase 0 review gate

- Tests are green.
- Baseline measurements can be repeated by another contributor.
- No production refactor has been mixed into measurement work.

## Phase 1: Remove Poetry and Modernise Packaging

Goal: remove Poetry from the complete dependency and release path. The sibling
package must be handled first because resistics currently resolves it through a
local uv source and its build metadata still requires `poetry-core`.

### Checkpoint 1.1 (`regressioninc`): Convert package metadata

- Replace `[tool.poetry]` metadata with PEP 621 `[project]` metadata.
- Replace `poetry-core` with Hatchling or another agreed PEP 517 backend; use
  Hatchling by default for consistency with resistics.
- Align the supported Python range with the versions actually supported by both
  packages and their dependencies.
- Represent development and documentation dependencies using uv groups.
- Generate and verify `uv.lock` without relying on Poetry.

Verification:

- A clean `uv sync` works.
- `uv build --no-sources` produces both wheel and source distribution.
- Tests pass against the built wheel.

Suggested commit in `regressioninc`:
`build: migrate package metadata from Poetry to uv`

### Checkpoint 1.2 (`regressioninc`): Replace automation and references

- Replace Poetry commands in CI, publishing, Read the Docs, documentation, and
  contributor instructions.
- Add a repository check that rejects `poetry`, `poetry-core`, and `pypoetry`
  references, excluding deliberate migration history if any must remain.
- Prepare Trusted Publishing, but treat repository/PyPI configuration and an
  actual release as explicit manual follow-up actions.

Suggested commit in `regressioninc`:
`ci: remove remaining Poetry workflows`

### Checkpoint 1.3 (`regressioninc`): Prepare a resolvable release boundary

- Decide and document the first compatible release version, provisionally an
  alpha such as `0.1.0a1` if the public interface is not stable.
- Verify wheel and source distribution in isolated environments.
- Publish only after explicit approval and successful dry-run validation.

Suggested commit in `regressioninc`:
`docs: document regressioninc release contract`

Release/tag/publish is a manual GitHub checkpoint, not part of this commit.

### Checkpoint 1.4 (`resistics`): Remove the local package workaround

- Replace the editable sibling source override with a normal bounded
  `regressioninc` dependency once an installable release exists.
- Remove the local `[tool.uv.sources]` entry.
- Confirm a clean checkout can resolve and test without the sibling directory.

Suggested commit: `build: use released regressioninc package`

### Checkpoint 1.5: Replace commit CI

- Preserve the user's intended Python 3.11 and 3.14 coverage from the current
  workflow edits.
- Use current, SHA-pinned GitHub actions and `astral-sh/setup-uv`.
- Test supported Python versions on Linux, Windows, and macOS using a deliberate
  matrix that controls cost.
- Run quality and coverage once on a representative Python version rather than
  redundantly in every matrix cell.
- Verify the lock file is current.

Suggested commit: `ci: replace Poetry test workflow with uv`

### Checkpoint 1.6: Replace publishing workflow

- Trigger from protected version tags or GitHub releases according to the
  agreed release policy.
- Run `uv build --no-sources`.
- Inspect and test both the wheel and source distribution in isolated
  environments.
- Use PyPI Trusted Publishing with GitHub OIDC and a protected environment.
- Remove token-based publishing secrets after the trusted publisher is proven.
- Pin third-party actions by full commit SHA.

Suggested commit: `ci: publish verified distributions with trusted publishing`

Enabling the PyPI trusted publisher and making a real release are separate
manual GitHub/PyPI checkpoints.

### Checkpoint 1.7: Remove final Poetry references

- Convert `.readthedocs.yaml` to native uv installation using the docs group.
- Update contributor documentation, comments, badges, notebooks, and captured
  notebook output that still references Poetry paths.
- Delete obsolete Poetry lock/configuration files if any remain.
- Add a CI or repository script check for case-insensitive occurrences of
  `poetry`, `poetry-core`, or `pypoetry`.

Suggested commit: `chore: remove final Poetry references`

### Phase 1 review gate

- A clean checkout builds, tests, and builds documentation using uv only.
- Both packages build without `poetry-core`.
- The repository search finds no unexplained Poetry references.
- Publishing has been dry-run and inspected; no real release is required to
  approve the code changes.

## Phase 2: Consolidate Formatting, Linting, and Documentation Checks

Goal: replace the overlapping legacy toolchain with fast, explicit checks.

### Checkpoint 2.1: Introduce Ruff linting

- Add Ruff with explicit rule families rather than accepting changing defaults.
- Initial candidates: `E`, `F`, `W`, `I`, `UP`, `B`, `A`, `S`, `PT`, `C4`,
  `SIM`, `PERF`, `RUF`, and `C90`.
- Start complexity thresholds near the current code, then ratchet them down.
- Configure justified per-file exceptions for tests, docs, and scientific
  conventions.
- Fix correctness and import findings; baseline or defer large complexity
  refactors to their owning phases.

Suggested commit: `build: replace Flake8 linting with Ruff`

### Checkpoint 2.2: Switch formatting to Ruff

- Configure `ruff format` to preserve the existing Black-compatible style.
- Apply formatting in an isolated mechanical commit.
- Remove Black dependencies, configuration, badges, and hooks only after the
  formatting diff is reviewed.

Suggested commit: `style: replace Black formatting with Ruff`

### Checkpoint 2.3: Replace darglint with pydoclint

- Configure pydoclint for the project's NumPy-style docstrings.
- Establish a measured baseline rather than weakening checks globally until
  they pass.
- Enforce documentation checks first on new or touched public APIs.
- Fix high-value public API documentation separately from mechanical tooling
  configuration if the diff is substantial.

Suggested commits:

- `build: replace darglint with pydoclint`
- `docs: fix public API docstring contracts` (only if needed)

### Checkpoint 2.4: Simplify pre-commit and CI quality jobs

- Replace obsolete hook revisions and the deprecated Prettier mirror.
- Keep local hooks fast; leave expensive full-suite checks to CI.
- Ensure local and CI commands invoke the same pinned tools through uv.
- Remove `.flake8` and redundant configuration only after Ruff covers the
  intended rules.

Suggested commit: `ci: simplify automated quality checks`

### Phase 2 review gate

- Ruff format and lint are the only Python style/lint tools.
- pydoclint is the only docstring contract checker.
- Black, Flake8, and darglint are absent from dependencies, hooks, CI, docs, and
  badges.
- The full test suite remains green after the mechanical formatting commit.

## Phase 3: Replace mypy with a Useful Type-Checking Gate

Goal: select one fast checker based on evidence, prevent new type regressions,
and reduce existing issues incrementally. Do not maintain multiple mandatory
type checkers.

### Candidate order

1. **Pyrefly** is the preferred candidate because it is fast, has a language
   server, supports baselines, and includes explicit Pydantic v2 support.
2. **Basedpyright** is the conservative fallback because it has a mature
   Pyright foundation, strong diagnostics, convenient uv/PyPI installation,
   `.venv` discovery, and baseline support.
3. **ty** should be evaluated because it is extremely fast and aligns with uv
   and Ruff, but it remains pre-1.0 and its specialised third-party-library
   behaviour is still evolving.
4. Plain Pyright remains a valid reference point but is less convenient for
   this uv-first, editor-independent project than Basedpyright.

### Checkpoint 3.1: Run and record a bounded checker evaluation

- Run Pyrefly, Basedpyright, and ty against representative modules:
  `flow`, `job`, `project`, `gather`, `regression`, and `tui`.
- Compare:
  - cold and warm runtime;
  - incremental/editor responsiveness;
  - number of genuine defects found;
  - false-positive and suppression burden;
  - Pydantic v2 model, validator, serialization, and generated-constructor
    behaviour;
  - NumPy, SciPy, ObsPy, MTH5, Plotly, and Textual type information;
  - diagnostic clarity and navigation;
  - baseline and CI ergonomics.
- Record the decision and rationale in this plan or a short architecture
  decision record.
- Do not commit three permanent checker dependencies merely to perform the
  evaluation; use temporary isolated invocations.

Suggested commit: `docs: record type checker selection`

### Checkpoint 3.2: Install the selected checker and remove mypy

- Default to Pyrefly if the evaluation confirms its expected Pydantic and
  diagnostic advantages; otherwise select Basedpyright.
- Add one pinned development dependency, one configuration, one local command,
  and one CI job.
- Generate and commit a baseline for existing valid findings.
- Configure CI to reject new findings without forcing an immediate repository-
  wide annotation rewrite.
- Remove mypy dependencies, configuration, caches, hooks, and documentation.
- Do not translate every old mypy exception automatically; retain only
  suppressions demonstrated to be necessary with the selected checker.

Suggested commit: `build: replace mypy with <selected-checker>`

### Checkpoint 3.3: Harden core contracts module by module

Reduce the committed baseline in small commits, in this order unless evaluation
shows a stronger dependency sequence:

1. `flow`, `job`, `project`, and `mask`.
2. TUI state, DTO, and service boundaries.
3. `gather` and `regression` adapters.
4. `time`, `decimate`, `window`, and `spectra`.
5. Remaining public modules.

Each module group should be a separate commit. Prefer correcting contracts,
narrowing types, and adding focused tests over casts or broad ignores.

Suggested commit pattern: `refactor(<module>): harden type contracts`

### Phase 3 review gate

- mypy is completely removed.
- Exactly one type checker is mandatory in CI.
- Existing debt is represented by a shrinking baseline, not scattered blanket
  ignores.
- New or edited public APIs are checked.
- Checker runtime is short enough that contributors will run it locally.
- Reassess ty after it reaches an agreed maturity point or gains capabilities
  materially relevant to resistics; do not run a second permanent CI gate in
  anticipation.

## Phase 4: Make the TUI Responsive

Goal: remove blocking and repeated work from the Textual event loop before
splitting the large module.

### Checkpoint 4.1: Make action-state checks pure and cheap

- Introduce explicit cached screen/application state used by `check_action`.
- Remove filesystem, MTH5, YAML, JSON, solution, flow, parameter, and job reads
  from `check_action` and equivalent binding predicates.
- Update state only when the underlying selection or data changes.
- Add tests that fail if action-state evaluation performs I/O.

Suggested commit: `perf(tui): make action checks side-effect free`

### Checkpoint 4.2: Eliminate redundant refreshes

- Map every `refresh_bindings()` call to the state transition that requires it.
- Remove duplicate refreshes on tab activation, cursor movement, and nested
  control updates.
- Batch table/tree updates where Textual permits it.
- Measure handler time before and after.

Suggested commit: `perf(tui): reduce redundant binding refreshes`

### Checkpoint 4.3: Add a cached project explorer index

- Introduce a UI-neutral `ProjectExplorerIndex` or equivalent service result.
- Build it from project path and relevant file identity such as modification
  time and size.
- Centralise directory scans, HDF5 summaries, flow/parameter/criteria parsing,
  and job listings.
- Define explicit invalidation after create, edit, delete, processing, and
  external refresh operations.
- Test cache hits, invalidation, stale files, malformed metadata, and closed
  MTH5 handles.

Suggested commit: `perf(project): cache explorer metadata`

### Checkpoint 4.4: Move blocking loads to workers

- Show the project screen and loading state immediately.
- Run synchronous MTH5 and filesystem APIs in Textual thread workers.
- Return immutable results or messages to the UI thread; mutate widgets only
  through Textual's thread-safe mechanisms.
- Load inactive tab data lazily.
- Support cancellation or stale-result rejection if the user changes project
  or selection while a worker is running.

Suggested commit: `perf(tui): load project metadata in workers`

### Checkpoint 4.5: Defer heavy imports

- Move Plotly, Matplotlib, SciPy, ObsPy, and other feature-specific imports out
  of the TUI startup path where practical.
- Import plot and processing services when the corresponding action begins.
- Avoid hiding real missing-dependency errors; report them at the feature
  boundary with actionable messages.
- Re-measure cold import and first-screen time.

Suggested commit: `perf(tui): defer feature-specific imports`

### Checkpoint 4.6: Replace terminal progress output

- Remove unconditional `tqdm` rendering from code used under Textual.
- Use structured progress callbacks/events that standalone callers can adapt to
  `tqdm` and the TUI can render natively.
- Test callback ordering, completion, cancellation, and failure.

Suggested commit: `refactor: expose structured processing progress`

### Phase 4 review gate

- `check_action` and binding predicates perform zero I/O.
- Representative handlers complete in under 50 ms when no worker result is
  awaited.
- Project opening displays responsive loading state rather than freezing.
- Cold `resistics.tui` import time is reduced by at least 60% from the confirmed
  baseline, or remaining unavoidable imports are documented.
- No worker directly mutates Textual widgets from a background thread.

## Phase 5: Split Large Modules at Stable Boundaries

Goal: reduce cognitive load after behaviour is protected and performance work
has exposed natural interfaces.

### Checkpoint 5.1: Convert the TUI module to a package skeleton

- Preserve `resistics.tui:main` and existing import paths.
- First create a package facade and move code without behavioural edits.
- Let Git detect moves; do not mix formatting or logic changes.

Proposed structure:

```text
resistics/tui/
├── __init__.py
├── app.py
├── state.py
├── services.py
└── screens/
    ├── launcher.py
    ├── project.py
    └── dialogs.py
```

Suggested commit: `refactor(tui): introduce package boundaries`

### Checkpoint 5.2: Extract dialogs and launcher screens

- Move low-coupling dialogs first.
- Preserve widget ids, messages, bindings, and tests.
- Keep each move separate if GitHub no longer presents a readable rename.

Suggested commit: `refactor(tui): extract launcher and dialogs`

### Checkpoint 5.3: Extract project screen state and services

- Separate presentation from project indexing, validation, plotting targets,
  deletion previews, and execution orchestration.
- Keep service interfaces usable without constructing a Textual application.
- Add unit tests for services and retain pilot tests for screen wiring.

Suggested commits:

- `refactor(tui): extract project explorer services`
- `refactor(tui): extract project screen`

### Checkpoint 5.4: Unify flow and job plot rendering

- Extract shared node, edge, layout, title, metadata, hover, data-type label,
  and layering behaviour.
- Keep flow/job-specific model adaptation separate from rendering.
- Protect title spacing, text-over-edge layering, edge-label font size,
  parameter summary wording, and clean hover text with focused tests.
- Preserve public plotting functions as thin adapters.

Suggested commit: `refactor(plot): share flow graph rendering`

### Checkpoint 5.5: Split gather responsibilities

- Separate criteria and validation, MTH5/project discovery, planning, and data
  assembly.
- Break high-complexity methods through named domain operations rather than
  moving the same complexity into private helpers mechanically.
- Narrow broad exception handling and preserve context in user-facing errors.

Suggested commit pattern: `refactor(gather): extract <responsibility>`

### Checkpoint 5.6: Finish MTH5 boundary cleanup

- Remove incomplete compatibility stubs such as `Measurement = None` and
  `Site = None` once callers have migrated.
- Make file ownership and close behaviour explicit.
- Remove dead legacy paths only after search, tests, docs, and release notes
  confirm they are no longer public or required for migration.

Suggested commit: `refactor(project): complete MTH5-only boundary`

### Checkpoint 5.7: Remove or relocate dead and test-only code

- Review the empty `resq.py`, empty notebook, commented-out time-processing
  blocks, and unused compatibility helpers.
- Reduce `resistics/testing.py` carefully: retain intentional public doctest
  helpers and move assertion-heavy factories into tests where appropriate.
- Delete only after call-site and documentation searches.

Suggested commit: `chore: remove verified dead code`

### Phase 5 review gate

- Public imports and entry points remain compatible or have explicit,
  documented deprecations.
- New TUI modules are preferably under 800 lines and individual screen modules
  under roughly 500 lines; exceptions require a clear cohesion argument.
- Plot layout and hover behaviour remain covered by tests.
- File moves are reviewable independently from behaviour changes.

## Phase 6: Dependencies, Security, and Compatibility

Goal: reduce dependency weight and improve supply-chain and platform confidence
using evidence rather than speculative migrations.

### Checkpoint 6.1: Audit direct dependencies

- Map every direct dependency to production imports, optional features, docs,
  tests, or development tooling.
- Remove dependencies that are only transitive or unused.
- Move plotting, notebook, documentation, and testing dependencies to suitable
  optional extras or uv groups where public behaviour permits.
- Review `prettyprinter`, direct Matplotlib requirements, and other narrow-use
  packages specifically.

Suggested commit: `build: remove unused direct dependencies`

### Checkpoint 6.2: Test the lowest supported dependency set

- Add a scheduled or manual CI job that resolves the lowest supported direct
  versions where practical.
- Keep normal lock-based CI reproducible.
- Document intentional upper bounds and compatibility constraints.

Suggested commit: `ci: test supported dependency bounds`

### Checkpoint 6.3: Add dependency and workflow monitoring

- Configure Dependabot or an agreed equivalent for uv and GitHub Actions.
- Add `uv audit` with a documented policy for unavailable fixes and accepted
  risks.
- Keep all third-party actions pinned by full SHA and update them through
  reviewed automation.
- Set minimal workflow permissions and protected publishing environments.

Suggested commit: `ci: add dependency and workflow security checks`

### Checkpoint 6.4: Decide uv versus Pixi from CI evidence

- Keep uv as the default while wheels and supported builds work across the
  Python/OS matrix.
- Record failures caused by native scientific dependencies separately from
  application or workflow failures.
- Trial Pixi only if ObsPy, MTH5, HDF5, or another compiled dependency produces
  repeated unsupported-platform or solver problems that uv/PyPI cannot
  reasonably address.
- If a trial is needed, run it on a branch and compare lock reproducibility,
  build/publish flow, contributor setup, CI time, and maintenance burden.
- Do not maintain uv and Pixi as equal permanent paths unless they solve
  demonstrably different supported use cases.

This checkpoint may result in a decision record with no package-manager change.

Suggested commit: `docs: record environment manager decision`

### Phase 6 review gate

- Every direct dependency has a documented production or development purpose.
- Supported Python/OS combinations install from a clean checkout.
- Dependency vulnerabilities and action updates have visible ownership.
- Any Pixi adoption is justified by reproduced failures, not concern alone.

## Phase 7: Documentation and Final Hardening

Goal: make the strengthened workflows discoverable and remove migration debris.

### Checkpoint 7.1: Update contributor and release documentation

- Document uv setup, tests, Ruff, pydoclint, the selected type checker,
  documentation builds, and release verification.
- Document how to run focused TUI performance checks.
- Add a release checklist covering tag protection, Trusted Publishing, wheel
  and source-distribution smoke tests, and rollback/yank decisions.

Suggested commit: `docs: update contributor and release workflows`

### Checkpoint 7.2: Reconcile active architecture plans

- Review `.agents/plans/modernization.md` against the implementation.
- Mark completed or superseded current-state claims rather than leaving stale
  instructions for future agents.
- Ensure naming, project rules, examples, and TUI guidelines match the final
  code.

Suggested commit: `docs: reconcile architecture guidance`

### Checkpoint 7.3: Re-run the complete audit

- Re-measure tests, coverage, complexity, module sizes, import time, TUI handler
  responsiveness, dependency weight, type baseline, and build/install time.
- Compare results with the Phase 0 baseline.
- Document remaining risks and create narrowly scoped follow-up issues rather
  than expanding this programme indefinitely.

Suggested commit: `docs: record hardening outcomes`

### Final review gate

- All desired outcomes are met or explicitly deferred with an owner and reason.
- The repository can be cloned, synced, tested, documented, built, and type
  checked without Poetry.
- Release artifacts are reproducible and independently smoke-tested.
- TUI responsiveness improvements are supported by before/after evidence.
- Remaining large or complex modules have focused follow-up issues rather than
  vague cleanup tasks.

## Recommended Initial Pull Request Sequence

The first work should remain deliberately small:

1. Phase 0.1: restore the green baseline.
2. Phase 0.2: commit measurements and regression tests.
3. `regressioninc` Phase 1.1: PEP 621 and uv packaging in its own repository.
4. `regressioninc` Phase 1.2: automation cleanup in its own repository.
5. Resistics Phase 1.4: consume a released `regressioninc`.
6. Resistics Phase 1.5: uv test CI.
7. Resistics Phase 1.6: publishing workflow.
8. Resistics Phase 1.7: final Poetry removal.

After that foundation is merged, proceed through Ruff/pydoclint, checker
selection, TUI performance, and only then structural module splits.

## Commands to Standardise During Implementation

Exact commands may change as tool selection is completed, but the final
developer workflow should provide simple equivalents of:

```console
uv sync --locked --all-groups
uv run pytest
uv run pytest --cov=resistics --cov-branch
uv run ruff format --check .
uv run ruff check .
uv run pydoclint resistics
uv run <selected-type-checker>
uv build --no-sources
uv audit
```

During the Poetry-removal transition, use the existing `.venv` for tests when a
clean uv resolution is blocked by `regressioninc`; do not conceal that blocker
with an undocumented local install.

## References to Re-verify When Implementing

Tool and action versions change. Re-check current official guidance at the time
of each implementation phase:

- uv GitHub Actions: <https://docs.astral.sh/uv/guides/integration/github/>
- uv packaging and publishing: <https://docs.astral.sh/uv/guides/package/>
- Read the Docs uv integration:
  <https://docs.readthedocs.com/platform/stable/build-customization.html#install-dependencies-with-uv>
- Ruff formatter: <https://docs.astral.sh/ruff/formatter/>
- pydoclint configuration: <https://jsh9.github.io/pydoclint/config_options.html>
- Pyrefly Pydantic support: <https://pyrefly.org/en/docs/pydantic/>
- Pyrefly configuration and baselines:
  <https://pyrefly.org/en/docs/configuration/>
- Basedpyright baseline:
  <https://docs.basedpyright.com/latest/benefits-over-pyright/baseline/>
- ty documentation: <https://docs.astral.sh/ty/>
- Textual workers: <https://textual.textualize.io/guide/workers/>
- GitHub Actions security:
  <https://docs.github.com/en/actions/reference/security/secure-use>
- Dependabot supported ecosystems:
  <https://docs.github.com/en/code-security/reference/supply-chain-security/supported-ecosystems-and-repositories>
- Pixi Conda/PyPI environments:
  <https://pixi.prefix.dev/latest/concepts/conda_pypi/>

## Out of Scope Without Separate Approval

- Publishing either package or creating a GitHub release.
- Changing PyPI Trusted Publisher, GitHub environment, branch-protection, or
  repository-secret settings.
- Editing or committing the sibling `regressioninc` repository as part of a
  resistics commit.
- Rewriting numerical algorithms solely to reduce line count.
- Adopting Pixi without a reproduced dependency problem and a reviewed decision.
- Breaking public APIs merely to satisfy a type checker or complexity metric.
