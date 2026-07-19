# Resistics Codebase Hardening Plan

Status: ready for implementation
Created: 2026-07-18
Last reviewed: 2026-07-19
Scope: resistics and, where explicitly identified, the sibling `regressioninc`
package

## Purpose

Harden resistics without attempting one large rewrite. The work should improve
terminal UI responsiveness, simplify maintenance, remove Poetry completely,
modernise development and release tooling, make static analysis useful again,
and reduce architectural weak spots while preserving existing behaviour.

This is an incremental delivery plan. Numbered checkpoints are verification
milestones and natural opportunities to back up coherent work; they are not a
requirement to split this broad programme into small pull requests or commits.

## Desired Outcomes

- The test suite has a documented green baseline and protects important
  processing, serialization, plotting, and TUI contracts.
- Poetry and all Poetry-generated or Poetry-specific references are gone from
  resistics and `regressioninc`.
- uv is the primary Python environment, dependency, build, and publishing tool
  unless the evidence gate in Phase 6 justifies moving environment management
  to Pixi.
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
- Public APIs and non-trivial internal operations have useful docstrings, with
  executable examples and plots kept beside the code when they materially help
  users understand behaviour.
- The `mth5` branch reaches production-quality code, tests, tooling, and
  documentation before its owner decides how to promote it.

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
- Changes in the sibling `regressioninc` repository require a separate worktree
  and commit history in that repository.

## Docstring Strategy

Docstrings are part of the implementation contract, not optional commentary.
They should explain behaviour that cannot be learned reliably from the name and
type signature alone. The goal is useful documentation rather than inflated
coverage through placeholder prose.

### Required coverage

- Require docstrings for every public module, class, function, method,
  property, command entry point, and public callback. Treat names exported via
  `__all__` or documented as supported API as public even if their spelling is
  unusual.
- Require docstrings for private classes and functions when they implement a
  domain operation, numerical algorithm, state transition, worker boundary,
  cache/invalidation rule, file-format boundary, or other non-obvious contract.
- Do not require separate docstrings for trivial private delegators, inherited
  methods whose contract is unchanged, or conventional dunder methods. Add one
  when behaviour, invariants, side effects, or failure modes differ from the
  inherited or conventional contract.
- Document constructor arguments and class-wide invariants on the class unless
  `__init__` has an independently useful contract. Avoid duplicating the same
  text on both objects.
- Tests are exempt from blanket public-API docstring rules, but reusable
  fixtures and non-obvious test helpers should explain the behaviour they
  provide. Test names remain the primary description of individual cases.
- Pydantic model fields must have meaningful `Field(description=...)` text or
  supported attribute documentation when their name and annotation do not
  fully express units, constraints, defaults, or semantics.

### Content requirements

A useful docstring should contain the applicable parts of this list:

- A concise summary of what the object does in domain terms.
- Important preconditions, invariants, units, coordinate conventions, array
  shapes/order, accepted representations, and the meaning of sentinel values
  such as `None`.
- Mutation, file ownership, I/O, caching, concurrency, and other side effects
  that affect callers.
- Parameter, return, yield, and raised-exception contracts. Types belong in
  Python annotations and must not be duplicated in the final MyST docstring.
- Warnings about lossy operations, expensive work, numerical limitations, or
  lifecycle requirements when relevant.
- Cross-references to the related process, model, or workflow rather than
  repeating its full explanation.

Do not accept docstrings that merely restate the function name, repeat type
annotations, claim that an object "does something", or preserve stale examples
that no longer execute.

### Examples and plots

- Include at least one example for a non-trivial public API when correct use
  requires object assembly, configuration, a particular call sequence, or
  interpretation of a domain-specific result. Simple accessors and obvious
  value objects do not need ceremonial examples.
- Keep examples in the docstring of the function or class they explain. A
  tutorial may build a longer workflow from them, but it does not replace the
  co-located example.
- Use fenced `{doctest}` examples for small executable contracts. They must be
  deterministic, fast, network-free, and based on synthetic data or compact
  repository fixtures.
- Keep a plot in its docstring whenever the visual result is part of
  understanding that API. Convert existing `.. plot::` blocks to fenced
  `{plot}` blocks in place; do not move them to a gallery merely to simplify
  documentation generation.
- Plot examples must use deterministic data, avoid external downloads, close
  figures where necessary, and remain small enough for routine documentation
  builds. Fix or explicitly remove an obsolete example rather than silently
  skipping execution.
- Prefer focused examples that demonstrate one contract. Longer end-to-end
  workflows belong in MyST-NB tutorials and should cross-reference the relevant
  API objects.

### Authoring and enforcement

- Until the MyST foundation in Phase 7.1 is active, new docstrings follow the
  repository's existing NumPy convention so current documentation remains
  valid. After that checkpoint, all new and materially edited docstrings use
  MyST with Sphinx-style field lists.
- Ruff `D` rules enforce the presence and basic form of public docstrings;
  pydoclint checks signature, parameter, return, yield, and exception
  contracts. The Sphinx doctest and HTML builders execute examples and plots
  and validate cross-references.
- Configure pydoclint to check short docstrings rather than allowing a one-line
  placeholder to bypass contract checks. Use a baseline only for identified
  existing debt, never for newly added or materially changed public APIs.
- Pre-commit checks changed Python files for docstring regressions, while CI
  checks the complete production package and documentation build.
- Any suppression of a public docstring rule must state why generated,
  inherited, protocol-mandated, or callback-specific behaviour makes the rule
  inappropriate. Broad module-wide exemptions are temporary migration debt and
  must have an owner or removal checkpoint.

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

Checkpoint commits should keep a coherent technical purpose, but the work may
span several commits where that is the most practical way to reach a verified
milestone.

Before each commit:

1. Inspect `git diff --check` and the complete staged diff.
2. Run the smallest focused test set that exercises the change.
3. Run the configured formatter and linter for touched files.
4. Confirm changed public APIs and non-trivial internal operations satisfy the
   docstring strategy, including an example or plot where the contract calls
   for one.
5. Record intentional baseline changes, such as coverage or type findings.
6. Avoid staging unrelated user changes.

Before each push:

1. Run the full locally available quality suite.
2. Confirm generated locks and build metadata are consistent.
3. Confirm public API changes are documented or explicitly absent.
4. Add a short before/after measurement for performance-related changes.

Working-branch strategy:

- Perform this hardening programme on `mth5`; use it as the long-lived branch
  for the completed work and its verification history.
- Back up coherent checkpoints to GitHub as useful, but no pull request or
  primary-branch promotion is part of this plan.
- The owner will decide separately when and how the production-ready `mth5`
  work moves to the primary branch.

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

- Set coverage failure at the confirmed baseline, initially 75.95% (76% when
  rounded to a whole percentage), so coverage cannot silently regress.
- Add focused instrumentation and tests identifying which frequently called
  TUI action-state checks perform project, filesystem, MTH5, YAML, or JSON I/O.
  Protect action paths already backed by in-memory state at this checkpoint,
  record the remaining I/O branches as Phase 4.1 debt, and convert the
  inventory into a strict zero-I/O regression gate during that refactor.
- Add a repeatable cold-import timing command and representative TUI response
  benchmark or profiling fixture. Keep timing assertions tolerant enough for CI
  variance; use them primarily for before/after evidence.
- Capture current module sizes and complexity in documentation or a lightweight
  reporting command rather than creating hard line-count gates.
- Inventory missing public docstrings, non-trivial undocumented private
  operations, existing executable examples, and existing plot directives. Use
  this as migration input rather than a permanent lower quality threshold.
- Standardise generated local reports under the already ignored `.artifacts/`
  directory, including coverage, profiling, documentation, and build-inspection
  output.

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

### Checkpoint 1.5: Harden resistics package metadata and artifacts

- Reconcile the package version with its development-status classifier; the
  current alpha version and Beta classifier must not disagree.
- Replace the current `<=3.14` Python constraint with the semantically correct
  upper bound, provisionally `>=3.11,<3.15`, if the compatibility matrix
  confirms support for every Python 3.11–3.14 minor release.
- Check project URLs, license metadata, package data, and wheel/sdist contents.
- Build without local uv sources, build a wheel from the sdist, and smoke-test
  imports and the `resistics` entry point from the resulting wheel in an
  isolated environment.
- Defer adding a PEP 561 `py.typed` marker until Phase 3 confirms the supported
  public API meets the agreed type-checking standard.

Suggested commit: `build: harden package metadata and artifacts`

### Checkpoint 1.6: Replace commit CI

- Preserve the user's intended Python 3.11 and 3.14 coverage from the current
  workflow edits.
- Use current, SHA-pinned GitHub actions and `astral-sh/setup-uv`.
- Test Python 3.11–3.14 on Linux, and test the minimum and maximum supported
  versions on Windows and macOS. Expand that matrix if failures reveal
  version-specific platform behaviour.
- Run quality and coverage once on a representative Python version rather than
  redundantly in every matrix cell.
- Verify the lock file is current.

Suggested commit: `ci: replace Poetry test workflow with uv`

### Checkpoint 1.7: Replace publishing workflow

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

### Checkpoint 1.8: Remove final Poetry references

- Convert `.readthedocs.yaml` to native uv installation using the docs group.
- Update contributor documentation, comments, badges, notebooks, and captured
  notebook output that still references Poetry paths.
- Delete obsolete Poetry lock/configuration files if any remain.
- Add a CI or repository script check for case-insensitive occurrences of
  `poetry`, `poetry-core`, or `pypoetry` in active code, configuration,
  automation, and user/contributor documentation. Explicitly exclude this
  migration plan while it remains the historical implementation record.

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
  `SIM`, `PERF`, `RUF`, `C90`, and `D` for the docstring-presence/style checks
  currently supplied by Flake8 plugins.
- Map every existing Flake8 plugin to a Ruff rule or an explicit, documented
  omission so the migration does not silently reduce coverage.
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

- Use pydoclint's native CLI and baseline support rather than running it as a
  Flake8 plugin.
- Initially configure its NumPy style for the existing docstrings, including
  consistency between their legacy type entries and Python annotations. This
  is a temporary compatibility mode until the MyST migration in Phase 7.
- Establish a measured baseline rather than weakening checks globally until
  they pass.
- Enforce documentation checks first on new or touched public APIs.
- Fix high-value public API documentation separately from mechanical tooling
  configuration if the diff is substantial.

Suggested commits:

- `build: replace darglint with pydoclint`
- `docs: fix public API docstring contracts` (only if needed)

### Checkpoint 2.4: Enforce the docstring contract

- Enable Ruff `D100`–`D104` for production code. Enable `D105`–`D107` only where
  they match the strategy above, since conventional dunders and constructors
  documented at class level should not acquire duplicate prose.
- Define narrow exceptions for generated, inherited, or framework-mandated
  objects.
- Bring the supported public API to complete docstring presence rather than
  permanently ignoring whole legacy modules.
- Configure pydoclint with `skip-checking-short-docstrings = false` and a
  measured legacy baseline so newly added one-line placeholders cannot evade
  parameter, return, yield, or exception checks.
- Add or improve examples for the most frequently used non-trivial public APIs,
  prioritising project creation, flow/parameter/job construction, processing
  operations, result access, and plotting.
- Record existing docstring plots and examples as protected documentation
  assets before the MyST syntax migration. Add focused tests where an example
  exposes behaviour not otherwise protected.
- Document the strategy above in contributor guidance or a dedicated authoring
  page, including when a private helper needs a docstring and when an example or
  plot is expected.

Suggested commit: `docs: establish production docstring requirements`

### Checkpoint 2.5: Simplify pre-commit and CI quality jobs

- Replace obsolete hook revisions and remove the deprecated Prettier mirror.
  Do not retain a JavaScript formatter unless a maintained non-Python asset has
  an explicit formatting need.
- Use uv-backed local hooks so pre-commit, contributors, and CI run the same
  locked tool versions rather than isolated hook environments.
- Invoke uv-backed hooks with `uv run --locked --no-sync` after the documented
  environment sync, preventing hooks from silently changing the environment.
- Run Ruff format and Ruff check for staged Python files, retain YAML,
  end-of-file, and trailing-whitespace hygiene checks, and run pydoclint for
  changed production Python files.
- Revisit a type-checking pre-push hook in Phase 3 only if the selected checker's
  measured runtime is short enough for routine local use; CI remains the
  mandatory full-project type gate.
- Keep the full test suite, coverage, documentation build, package build,
  security scan, and slow type checking in CI.
- Verify hook installation and execution from a clean
  `uv sync --locked --all-groups` environment and document the setup command.
- Remove `.flake8` and redundant configuration only after Ruff covers the
  intended rules.

Suggested commit: `build: modernise uv-backed pre-commit checks`

### Phase 2 review gate

- Ruff format and lint are the only Python style/lint tools.
- pydoclint is the only docstring contract checker.
- Pre-commit uses maintained file-hygiene hooks and uv-backed Python quality
  hooks.
- Every supported public object has a substantive docstring, and new or changed
  public APIs cannot bypass contract checking with a placeholder summary.
- Existing examples and plots have an inventory and regression path for the
  MyST migration.
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
   and Ruff, but it remains pre-1.0, its specialised third-party-library
   behaviour is still evolving, and it does not currently offer the same
   project-baseline workflow as the leading candidates.
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
- Require either a committed project baseline that rejects new findings or a
  clean initial result. Do not adopt a checker by generating repository-wide
  inline ignore comments or globally disabling meaningful rule families.
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

Reduce the committed baseline in these logical module groups unless evaluation
shows a stronger dependency sequence:

1. `flow`, `job`, `project`, and `mask`.
2. TUI state, DTO, and service boundaries.
3. `gather` and `regression` adapters.
4. `time`, `decimate`, `window`, and `spectra`.
5. Remaining public modules.

Prefer correcting contracts, narrowing types, and adding focused tests over
casts or broad ignores. Once the supported public API meets the agreed standard,
add and verify a PEP 561 `py.typed` marker in both wheel and sdist.

Suggested commit pattern: `refactor(<module>): harden type contracts`

### Phase 3 review gate

- mypy is completely removed.
- Exactly one type checker is mandatory in CI.
- Existing debt is represented by a shrinking baseline, not scattered blanket
  ignores.
- New or edited public APIs are checked.
- Checker runtime is short enough that contributors will run it locally.
- If the distribution advertises inline types through `py.typed`, the marker is
  present in artifacts and public annotations satisfy the documented support
  level.
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
- Keep file movement mechanically distinct from behavioural edits where mixing
  them would make regressions difficult to diagnose.

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
- File moves do not obscure behavioural changes or their regression coverage.

## Phase 6: Dependencies, Security, and Compatibility

Goal: reduce dependency weight and improve supply-chain and platform confidence
using evidence rather than speculative migrations.

### Checkpoint 6.1: Audit direct dependencies

- Map every direct dependency to production imports, optional features, docs,
  tests, or development tooling.
- Remove dependencies that are only transitive or unused.
- Rebase legacy lower bounds on versions that actually support Python 3.11 and
  the current package APIs; several present bounds predate the supported Python
  range and are not credible compatibility promises.
- Add upper bounds only for reproduced incompatibilities, with a comment or
  issue explaining each cap.
- Move plotting, notebook, documentation, and testing dependencies to suitable
  optional extras or uv groups where public behaviour permits.
- Review `prettyprinter`, direct Matplotlib requirements, and other narrow-use
  packages specifically.

Suggested commit: `build: remove unused direct dependencies`

### Checkpoint 6.2: Test the lowest supported dependency set

- Add a scheduled or manual CI job that resolves the lowest supported direct
  versions where practical and exercises the built distribution, not only the
  source checkout.
- Keep normal lock-based CI reproducible.
- Document intentional upper bounds and compatibility constraints.
- Fail clearly when a declared lower bound cannot install on the minimum
  supported Python version; fix the metadata instead of accepting an
  untestable promise.

Suggested commit: `ci: test supported dependency bounds`

### Checkpoint 6.3: Add dependency and workflow monitoring

- Configure Dependabot or an agreed equivalent for uv and GitHub Actions.
- Add `uv audit` for the locked project, including relevant Python-version
  variants, with a documented policy for unavailable fixes and accepted risks.
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
- If a trial is needed, keep it as an isolated experiment on `mth5` and compare
  lock reproducibility, build/publish flow, contributor setup, CI time, and
  maintenance burden before changing the primary environment workflow.
- Do not maintain uv and Pixi as equal permanent paths unless they solve
  demonstrably different supported use cases.

This checkpoint may result in a decision record with no package-manager change.

Suggested commit: `docs: record environment manager decision`

### Phase 6 review gate

- Every direct dependency has a documented production or development purpose.
- Supported Python/OS combinations install from a clean checkout.
- Dependency vulnerabilities and action updates have visible ownership.
- Any Pixi adoption is justified by reproduced failures, not concern alone.

## Phase 7: MyST Documentation Modernisation

Goal: retain Sphinx and Furo while making MyST the sole authored documentation
language. Rich documentation must remain co-located with the functions and
classes it explains. This includes plots, executable examples, warnings, notes,
maths, and cross-references currently embedded in docstrings.

Documentation output is HTML only. PDF, EPUB, and other legacy formats are not
supported by this phase.

### Target documentation architecture

- Use MyST for every maintained documentation page and Python docstring.
- Use MyST-NB text notebooks for executable tutorials.
- Prefer `sphinx-autodoc2` for static API discovery and MyST docstring rendering
  if the Phase 7.1 prototype proves complete for resistics' public API. MyST is
  the required authoring language; the API generator remains evidence-gated.
- Use MyST field lists for parameter, return, yield, and exception contracts.
- Keep types in Python annotations instead of duplicating them in docstrings.
- Apply the repository-wide docstring strategy to public APIs and non-trivial
  internal operations; syntax conversion must not reduce useful explanation.
- Keep `matplotlib.sphinxext.plot_directive`; invoke it through fenced `{plot}`
  blocks in the same docstrings that currently contain `.. plot::`.
- Execute ordinary examples through fenced `{doctest}` blocks in their original
  docstrings.
- Render Plotly tutorial output through notebook MIME output without Chrome or
  Kaleido.
- Organise the site into tutorials, how-to guides, explanation, and API
  reference.

The final MyST configuration must enable field lists and restrict implicit
cross-reference resolution to the standard and Python domains. API discovery
must cover all supported public modules without importing the full scientific
dependency graph, or isolate unavoidable imports when static discovery cannot
render an accurate public API.

### Checkpoint 7.1: Establish the MyST foundation

- Add MyST-NB and `sphinx-autodoc2` to the uv documentation group for the
  prototype.
- Test static discovery against representative Pydantic models, inherited
  members, aliases, overloads, signatures, `__all__`, cross-references, and
  source links. Compare that output with the current API documentation.
- Select one API generator after the prototype. Prefer autodoc2 when it reaches
  parity; otherwise retain standard autodoc with an isolated import strategy
  and prove that it renders MyST docstrings correctly.
- During migration only, use per-object parser rules so converted MyST
  docstrings and unconverted reStructuredText docstrings can coexist.
- Keep the temporary compatibility rules explicit and remove them in
  Checkpoint 7.5.
- Add a minimal MyST page and representative API object proving that internal
  references, intersphinx references, field lists, `{doctest}`, and `{plot}`
  work from docstrings before beginning the bulk conversion.

Suggested commit: `docs: establish MyST documentation toolchain`

### Checkpoint 7.2: Convert site structure and narrative pages

- Convert the landing page, getting started, lower-level usage, custom process,
  literature, navigation, and API entry pages from `.rst` to MyST `.md`.
- Use explicit, stable labels for cross-references that should survive heading
  renames.
- Replace reStructuredText toctrees, roles, directives, includes, and labels
  with their MyST equivalents.
- Rewrite the landing content to describe the current MTH5, flow, parameter,
  job, TUI, and standalone-library contracts rather than the historic 0.0.6 to
  1.0 transition.
- Do not mix API or behavioural rewrites into the mechanical syntax conversion.

Suggested commit: `docs: migrate narrative documentation to MyST`

### Checkpoint 7.3: Convert docstrings without moving their content

- Preserve every useful example and plot beside its existing function or
  class. Co-location is an acceptance requirement, not an implementation
  detail.
- Review each migrated public docstring for domain meaning, units, shapes,
  invariants, side effects, failure modes, and cross-references. Do not treat a
  syntactically valid conversion as complete when the original documentation
  was only a placeholder.
- Convert directives in place:
  - `.. plot::` to fenced `{plot}`;
  - `.. warning::`, `.. note::`, and `.. math::` to their fenced MyST forms;
  - Sphinx roles such as `:class:` and `:func:` to MyST roles such as
    `{py:class}` and `{py:func}`;
  - executable prompt examples to fenced `{doctest}` blocks.
- Convert NumPy parameter sections to MyST-compatible field lists using
  `:param name:`, `:return:`, `:yield:`, and `:raises Exception:`.
- Do not add `:type:` or `:rtype:` fields when the type is already present in
  the Python signature.
- During migration, run pydoclint in NumPy mode over unconverted modules and in
  Sphinx mode over converted modules using explicit module lists. Do not force
  mixed docstring styles through one parser configuration.
- For converted docstrings, make annotations the sole type source by selecting
  Sphinx style and disabling docstring argument, return, and yield type checks
  while retaining signature type checks. Use baselines only for unrelated
  existing contract findings.
- Record the intended pydoclint settings explicitly:
  `arg-type-hints-in-docstring = false`,
  `arg-type-hints-in-signature = true`, `check-return-types = false`, and
  `check-yield-types = false`.
- Migrate the following logical module groups in an order that keeps temporary
  parser compatibility manageable:
  1. `common`, `sampling`, and `transfunc`;
  2. `time` metadata and data containers;
  3. `time` readers and processors;
  4. `decimate` and `window`;
  5. `spectra`, `calibrate`, and `regression`;
  6. `flow`, `job`, `project`, `mask`, and `gather`;
  7. plotting and TUI-facing public APIs.
- Temporary parser routing may target individual modules or classes so a large
  module can migrate incrementally where that reduces risk.
- Add missing co-located `{doctest}` or `{plot}` examples for non-trivial public
  APIs identified by the Phase 0 inventory, while keeping examples focused and
  deterministic.

Suggested commit pattern: `docs(<module>): migrate docstrings to MyST`

### Checkpoint 7.4: Replace the obsolete example gallery

- Replace Sphinx-Gallery scripts with curated MyST-NB text notebooks.
- Provide at minimum:
  - first MTH5 project;
  - project discovery and navigation;
  - flow and parameter creation;
  - job validation and execution;
  - calibration and remote reference;
  - result and flow/job plotting.
- Execute small deterministic tutorials on every documentation build.
- Use synthetic or repository fixtures; normal documentation builds must not
  download data or require network access.
- Delete rather than translate examples for removed ASCII, bz2, binary,
  `letsgo`, legacy `Configuration`, or `resp` workflows.
- Render Plotly figures using notebook MIME output and verify their interactive
  HTML representation.

Suggested commit: `docs: replace legacy gallery with MyST tutorials`

### Checkpoint 7.5: Remove the transitional documentation stack

- Make MyST the catch-all docstring parser for the selected API generator.
- Keep exactly one API-generation path. If the prototype selected autodoc2,
  remove standard autodoc; if it exposed material API gaps, retain standard
  autodoc and remove autodoc2 instead.
- Remove Sphinx-Gallery, Napoleon, autodoc-pydantic, Kaleido, the unselected API
  generator, and the temporary mixed-parser and dual-pydoclint configuration.
- Remove obsolete gallery-only dependencies such as `seedir` and `emoji` if no
  maintained documentation uses them.
- Remove `docs/requirements.txt`; the uv documentation group becomes the only
  dependency declaration.
- Delete or replace every remaining maintained `.rst` source.
- Add repository checks that reject authored `.rst` files and legacy
  reStructuredText directive syntax in documentation and docstrings.
- Preserve Pydantic field documentation through source annotations, field
  descriptions, and attribute docstrings rendered by the selected API
  generator.

Suggested commit: `docs: remove reStructuredText documentation path`

### Checkpoint 7.6: Enforce documentation in CI and Read the Docs

- Build HTML with nitpicky cross-reference checking, warnings as errors, and
  keep-going enabled.
- Run the Sphinx doctest builder for fenced `{doctest}` examples.
- Ensure fenced `{plot}` examples execute as part of the HTML build and produce
  their expected figures.
- Remove pytest's raw module-docstring scanning only after the Sphinx doctest
  builder covers the converted examples.
- Run external link checking on a schedule so transient external failures do
  not block every push.
- Convert Read the Docs from Poetry to native uv installation using the docs
  group and a supported Python version.
- Request HTML output only.

Suggested commit: `ci: enforce MyST documentation builds`

### Checkpoint 7.7: Update contributor and release documentation

- Document uv setup, tests, Ruff, pydoclint, the selected type checker, MyST
  authoring, fenced docstring directives, documentation builds, and release
  verification.
- Include the required-coverage rules, content checklist, private-helper
  criteria, example/plot expectations, and justified-suppression process from
  the docstring strategy.
- State explicitly that rich examples and plots belong with their documented
  objects when that is where users will find them most useful.
- Document how to run focused TUI performance checks.
- Add a release checklist covering tag protection, Trusted Publishing, wheel
  and source-distribution smoke tests, documentation deployment, and
  rollback/yank decisions.

Suggested commit: `docs: finalise documentation guidance`

### Phase 7 review gate

- `sphinx-build -nW --keep-going -b html` succeeds.
- `sphinx-build -W --keep-going -b doctest` succeeds.
- Every `{plot}` example executes and produces its expected figure.
- Representative internal, Python-object, and intersphinx references resolve.
- Plotly tutorials render interactively without Chrome or Kaleido.
- pydoclint passes using the field-list configuration and agreed baseline.
- The pydoclint baseline contains no finding for a supported public API.
- Exactly one API generator remains, and representative generated API pages
  match the public interfaces and signatures in the source.
- No authored `.rst` files remain under maintained documentation or examples.
- No legacy `.. plot::`, `.. warning::`, `.. note::`, or `.. math::` syntax
  remains.
- Existing valuable docstring examples remain with their original functions or
  classes.
- Every supported public object has a substantive MyST docstring; non-trivial
  public workflows and processing APIs have co-located examples where required
  by the strategy.
- Every retained `{doctest}` and `{plot}` block is executed in the appropriate
  documentation build rather than being present but unchecked.
- API documentation includes current flow, job, project, mask, plotting, and
  TUI contracts.
- A clean uv environment builds the documentation locally and on Read the Docs.
- Only HTML documentation output is produced or supported.

## Phase 8: Final Reconciliation and Audit

Goal: reconcile working guidance with the completed implementation and record
the outcome of the hardening programme.

### Checkpoint 8.1: Reconcile active architecture plans

- Review `.agents/plans/modernization.md` against the implementation.
- Mark completed or superseded current-state claims rather than leaving stale
  instructions for future agents.
- Ensure naming, project rules, examples, documentation rules, and TUI
  guidelines match the final code.

Suggested commit: `docs: reconcile architecture guidance`

### Checkpoint 8.2: Re-run the complete audit

- Re-measure tests, coverage, complexity, module sizes, import time, TUI handler
  responsiveness, dependency weight, type baseline, public docstring coverage,
  executable example/plot counts, documentation build time, and package
  build/install time.
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
- Documentation is MyST-only, warnings-as-errors clean, and built through uv.
- The docstring strategy is enforced in pre-commit and CI, with complete public
  coverage and no unexplained broad suppressions.
- Remaining large or complex modules have focused follow-up issues rather than
  vague cleanup tasks.
- `mth5` satisfies every non-deferred production-readiness gate; branch
  promotion remains entirely outside this plan.

## Recommended Initial Implementation Sequence

Establish the packaging and measurement foundation in this order:

1. Phase 0.1: restore the green baseline.
2. Phase 0.2: commit measurements and regression tests.
3. `regressioninc` Phase 1.1: PEP 621 and uv packaging in its own repository.
4. `regressioninc` Phase 1.2: automation cleanup in its own repository.
5. `regressioninc` Phase 1.3: establish its installable release boundary.
6. Resistics Phase 1.4: consume a released `regressioninc`.
7. Resistics Phase 1.5: harden package metadata and artifacts.
8. Resistics Phase 1.6: uv test CI.
9. Resistics Phase 1.7: publishing workflow.
10. Resistics Phase 1.8: final Poetry removal.

After that foundation is complete, proceed through Ruff/pydoclint, checker
selection, TUI performance, structural module splits, documentation
modernisation, and the final audit.

## Commands to Standardise During Implementation

Exact commands may change as tool selection is completed, but the final
developer workflow should provide simple equivalents of:

```console
uv sync --locked --all-groups
uv run pre-commit run --all-files
uv run pytest
uv run pytest --cov=resistics --cov-branch
uv run ruff format --check .
uv run ruff check .
uv run pydoclint resistics
uv run <selected-type-checker>
uv run sphinx-build -nW --keep-going -b html docs/source .artifacts/docs/html
uv run sphinx-build -W --keep-going -b doctest docs/source .artifacts/docs/doctest
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
- uv dependency auditing: <https://docs.astral.sh/uv/reference/cli/#uv-audit>
- Read the Docs uv integration:
  <https://docs.readthedocs.com/platform/stable/build-customization.html#install-dependencies-with-uv>
- Ruff formatter: <https://docs.astral.sh/ruff/formatter/>
- pydoclint configuration: <https://jsh9.github.io/pydoclint/config_options.html>
- MyST roles, directives, and cross-references:
  <https://myst-parser.readthedocs.io/en/latest/syntax/roles-and-directives.html>
- MyST-NB authoring and execution: <https://myst-nb.readthedocs.io/en/latest/>
- sphinx-autodoc2 MyST docstrings:
  <https://sphinx-autodoc2.readthedocs.io/en/latest/docstrings.html>
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

- Opening a pull request, promoting `mth5`, renaming the primary branch, or
  changing the GitHub default branch.
- Publishing either package or creating a GitHub release.
- Changing PyPI Trusted Publisher, GitHub environment, branch-protection, or
  repository-secret settings.
- Editing or committing the sibling `regressioninc` repository as part of a
  resistics commit.
- Rewriting numerical algorithms solely to reduce line count.
- Adopting Pixi without a reproduced dependency problem and a reviewed decision.
- Breaking public APIs merely to satisfy a type checker or complexity metric.
