# Resistics Code-Hardening Implementation Record

Status: in progress; Checkpoint 2.4 verified with public docstring enforcement
Created: 2026-07-19
Last updated: 2026-07-19
Working branch: `mth5`
Starting HEAD: `c345ae6`
Governing plan: [codebase-hardening.md](codebase-hardening.md)

## Purpose

This file is the durable execution record for the code-hardening programme. It
is designed for work that spans many sessions and may be resumed by a different
agent with no access to earlier conversation context.

The governing plan defines scope, technical intent, checkpoint requirements,
and acceptance criteria. This implementation record defines execution order,
current state, evidence, decisions, blockers, and the exact next action. If the
two files disagree, stop implementation, reconcile them explicitly, and record
the decision here before continuing.

Do not copy detailed design requirements into this file unless they are needed
to preserve an implementation decision. Link to the governing checkpoint so
the two documents do not drift independently.

## Current State

- Programme state: `in_progress`
- Active phase: Phase 2 - Ruff, pydoclint, docstrings, and pre-commit
- Active checkpoint: `2.5` (`resistics`) - Simplify pre-commit and local quality commands
- Checkpoint state: `not_started`
- Last completed checkpoint: `2.2` at `070c414`
- Last verified checkpoint: `2.4` in `S013`
- Last session: `S013`
- Last verified commit: resistics `070c414`; regressioninc `9eb11a4` is the
  base of uncommitted Checkpoints 1.1 and 1.2 work
- Current blocker: none
- Next exact action: begin Checkpoint 2.5 by inventorying the remaining hygiene
  and Prettier hooks, then replace the isolated environments with locked,
  uv-backed local quality commands.

Current worktree caveat:

- Resistics `070c414` records the accumulated work through Checkpoint 2.2. The
  current dirty Resistics files contain the scoped, verified Checkpoints 2.3
  and 2.4 migrations plus their S012-S013 verification records.
- The owner's Python 3.11/3.14 CI intent remains a requirement of deferred
  Checkpoint 1.6 after the obsolete hosted workflows were removed.
- `../regressioninc/regressioninc/base.py` contains a pre-existing user change
  for Pydantic 2 and must not be reverted or absorbed silently into packaging
  work.
- `../regressioninc/pyproject.toml`, deletion of
  `../regressioninc/poetry.lock`, and `../regressioninc/uv.lock` are the scoped
  Checkpoint 1.1 changes.
- Regressioninc's CI and publishing workflows, Read the Docs configuration,
  README, `.gitignore`, documentation configuration, legacy-reference checker,
  and Matplotlib compatibility edits are the scoped Checkpoint 1.2 changes.
- Ruff format is the sole Resistics Python formatter. Pydoclint 0.9.1 now owns
  NumPy-style docstring contract checks through an explicit committed baseline;
  short docstrings are checked. Ruff enforces public production docstring
  presence, with narrow Pydantic-validator and Textual-callback exceptions.
- These files must not be reverted or overwritten during hardening work.

## Authority and Boundaries

- Perform resistics implementation on `mth5`.
- Do not open a pull request, promote `mth5`, rename the primary branch, publish
  either package, create a release, push a tag, or change repository settings.
- Hosted CI, Read the Docs deployment, registry-only installation, and
  publishing automation are recorded follow-ups and do not gate the local
  code/tooling/documentation hardening programme.
- Work in the sibling `regressioninc` repository requires a separate worktree,
  explicit scope, and its own commit history. Track its checkpoint status here,
  but never mix its files into a resistics commit.
- Preserve unrelated dirty-worktree changes.
- Do not create commits or push changes unless the user explicitly requests it.
  Record when a checkpoint is verified and ready to commit so the user has a
  safe commit opportunity.
- A checkpoint may span any number of sessions. Session boundaries do not relax
  its acceptance criteria.
- Keep only one checkpoint `in_progress` unless an explicit dependency requires
  a second, clearly recorded workstream.

## Status Vocabulary

Use only these states in the checkpoint tracker:

- `not_started`: no implementation work has begun.
- `in_progress`: work has begun but the governing acceptance criteria are not
  yet satisfied.
- `blocked`: progress requires a user decision, external release, unavailable
  dependency, or another checkpoint. Record the exact unblock condition.
- `verified`: implementation and checkpoint-level verification are complete;
  the work may still be uncommitted.
- `complete`: verified work is safely recorded at the stated commit or other
  durable boundary.
- `deferred`: deliberately excluded with a reason, owner, and consequence.

Never mark a checkpoint `verified` because most work is done. Never mark it
`complete` without recording its durable boundary and verification evidence.

## Multi-Session Protocol

### At the start of every implementation session

1. Read the Current State, checkpoint tracker, decision log, blocker log, and
   latest session handoff in this file.
2. Read the active checkpoint and its phase gate in the governing plan.
3. Run `git branch --show-current`, `git status --short`, and
   `git log -1 --oneline` before editing.
4. Confirm that intentional dirty files still match the latest handoff.
5. Re-run any previous verification whose result may have become stale because
   the worktree, lock file, environment, or dependency state changed.
6. Change the active checkpoint to `in_progress` only when implementation work
   actually begins.
7. State the session goal and the smallest evidence needed to finish it.

### While working

- Keep the governing checkpoint in scope. Record newly discovered work as a
  follow-up or plan amendment rather than silently expanding the checkpoint.
- Add focused regression tests before or with behavioural fixes.
- Preserve docstrings, examples, and plots. Apply the governing docstring
  strategy to every new or materially changed public API and non-trivial
  internal operation.
- Record material decisions when they are made, including rejected alternatives
  that a later session might otherwise reconsider.
- Record before/after measurements for performance, dependency, coverage,
  typing, documentation, and packaging work.
- If verification exposes unrelated pre-existing failures, record them
  separately and do not claim they were caused by the active checkpoint.

### Before ending every session

1. Run `git diff --check` and inspect the complete relevant diff.
2. Run the focused verification for the work performed.
3. Run broader checks required by the checkpoint when it is being marked
   `verified`.
4. Update Current State and the checkpoint tracker.
5. Update the measurement, decision, and blocker logs when applicable.
6. Append a session handoff using the template at the end of this file.
7. State the exact next command or file to inspect, not merely a broad phase.
8. Leave the worktree in a state another session can understand. Do not hide
   failing or incomplete work behind a `complete` status.

## Dependency and Execution Order

The numbered phases remain the default execution order. The following
dependencies are hard constraints:

```text
Phase 0 baseline
    |
    +--> regressioninc 1.1 -> 1.2
    |                         |
    |                         v
    +--------------------> resistics 1.4 -> 1.5 -> 1.8
                                              |
                                              v
Phase 2 quality/docstring tools -> Phase 3 type checker
              |                         |
              |                         +--> typed public API / py.typed decision
              v
Phase 4 TUI performance -> Phase 5 stable module boundaries
              |                         |
              +-------------------------+
                                        v
Phase 6 dependency/platform evidence -> Phase 7 final MyST documentation
                                        |
                                        v
                                  Phase 8 final audit

Deferred owner follow-up after local hardening:
regressioninc 1.3 -> hosted CI 1.6 -> publishing 1.7
```

Additional sequencing rules:

- Phase 1.4 retains the editable sibling regressioninc boundary and must refresh
  the resistics lock whenever that package's metadata changes.
- Checkpoints 1.3, 1.6, and 1.7 are owned deferrals. They do not block Phases
  2-8, but the final audit must not claim standalone registry installation,
  hosted automation, or publication readiness.
- Ruff and pydoclint must be installed before their pre-commit hooks become
  mandatory.
- The type-checker evaluation must precede removal of mypy.
- TUI responsiveness work precedes splitting `resistics/tui.py`, so module
  movement follows behaviour and performance protection.
- The Phase 7.1 parser prototype must pass before bulk docstring conversion.
- The documentation migration follows stable public/module boundaries where
  possible, avoiding avoidable rewrites of freshly converted documentation.
- Phase 8 cannot begin while a local code/tooling/documentation gate is
  incomplete or any deferral lacks an owner and documented consequence.

## Checkpoint Tracker

The completion evidence column should contain a commit, session id, report path,
or short command/result reference. Detailed output belongs in the session log or
`.artifacts/`, not in this table.

| ID | Repository | State | Completion evidence or unblock condition |
| --- | --- | --- | --- |
| 0.1 | resistics | `complete` | `ebede10`; S001 verification |
| 0.2 | resistics | `complete` | `b350893`-`d9911e3`; S002 evidence |
| Gate 0 | resistics | `complete` | `d9911e3`; 373 tests; repeatable report |
| 1.1 | regressioninc | `verified` | S003; clean sync/build/wheel tests |
| 1.2 | regressioninc | `verified` | S004; locked CI/docs/build and guard checks |
| 1.3 | regressioninc | `deferred` | Owner; after resistics and regressioninc review |
| 1.4 | resistics | `verified` | S006; clean paired sync and 373 tests |
| 1.5 | resistics | `verified` | S007-S008; tsdownsample; Python 3.11-3.14 |
| 1.6 | resistics | `deferred` | Owner; automate stable local commands later |
| 1.7 | resistics | `deferred` | Owner; after regressioninc release decision |
| 1.8 | resistics | `verified` | S009; active-reference guard and uv local gate |
| Gate 1 | both | `verified` | S003-S009; local uv sync/test/build/docs gate |
| 2.1 | resistics | `verified` | S010; Ruff 0.15.22 clean; 376 tests |
| 2.2 | resistics | `verified` | S011; Ruff format clean; 376 tests |
| 2.3 | resistics | `verified` | S012; pydoclint 0.9.1 baseline; 376 tests |
| 2.4 | resistics | `verified` | S013; public docs clean; 384 tests |
| 2.5 | resistics | `not_started` | Depends on 2.1-2.4 |
| Gate 2 | resistics | `not_started` | Depends on 2.1-2.5 |
| 3.1 | resistics | `not_started` | Evidence-based checker evaluation |
| 3.2 | resistics | `not_started` | Depends on 3.1 |
| 3.3 | resistics | `not_started` | Depends on 3.2 |
| Gate 3 | resistics | `not_started` | Depends on 3.1-3.3 |
| 4.1 | resistics | `not_started` | Pure action-state checks |
| 4.2 | resistics | `not_started` | Binding refresh measurements |
| 4.3 | resistics | `not_started` | Explorer index and invalidation |
| 4.4 | resistics | `not_started` | Worker/cancellation boundaries |
| 4.5 | resistics | `not_started` | Cold-import measurements |
| 4.6 | both | `not_started` | Structured progress boundary |
| Gate 4 | resistics | `not_started` | Depends on 4.1-4.6 |
| 5.1 | resistics | `not_started` | TUI package facade |
| 5.2 | resistics | `not_started` | Depends on 5.1 |
| 5.3 | resistics | `not_started` | Depends on 4.3-4.4 and 5.1 |
| 5.4 | resistics | `not_started` | Shared plot rendering |
| 5.5 | resistics | `not_started` | Gather boundaries |
| 5.6 | resistics | `not_started` | MTH5-only cleanup |
| 5.7 | resistics | `not_started` | Verified dead-code cleanup |
| Gate 5 | resistics | `not_started` | Depends on 5.1-5.7 |
| 6.1 | resistics | `not_started` | Direct dependency map |
| 6.2 | resistics | `not_started` | Credible lower-bound matrix |
| 6.3 | resistics | `not_started` | Audit and workflow monitoring |
| 6.4 | resistics | `not_started` | uv/Pixi evidence decision |
| Gate 6 | resistics | `not_started` | Depends on 6.1-6.4 |
| 7.1 | resistics | `not_started` | MyST/API-generator prototype |
| 7.2 | resistics | `not_started` | Depends on 7.1 |
| 7.3 | resistics | `not_started` | Depends on 2.3-2.4 and 7.1 |
| 7.4 | resistics | `not_started` | Depends on 7.1 |
| 7.5 | resistics | `not_started` | Depends on 7.2-7.4 |
| 7.6 | resistics | `not_started` | Depends on 7.5 |
| 7.7 | resistics | `not_started` | Depends on 7.6 |
| Gate 7 | resistics | `not_started` | Depends on 7.1-7.7 |
| 8.1 | resistics | `not_started` | Reconcile active plans |
| 8.2 | resistics | `not_started` | Repeat complete audit |
| Final gate | resistics | `not_started` | All gates complete or owned deferral |

## Phase Execution Briefs

These briefs describe how to divide work across sessions. The governing plan
contains the full requirements.

### Phase 0 - Baseline

- Session work: reproduce failures, fix correctness issues, establish test,
  coverage, docstring, module-size, import-time, and TUI measurements.
- Durable outputs: commands in project guidance and machine-readable or text
  reports under `.artifacts/hardening/baseline/` where they are not committed.
- Exit evidence: green existing suite, repeatable measurements, and Gate 0
  acceptance recorded.

### Phase 1 - Local packaging and uv

- Session work: keep regressioninc as a documented editable sibling, refresh
  the paired lock, harden resistics metadata and paired artifacts, and remove
  Poetry from active local development and documentation paths.
- Do not spend programme time on hosted CI, registry publication, or standalone
  dependency resolution. Preserve those as owned follow-ups with their
  production consequence stated explicitly.
- Exit evidence: clean paired uv-only sync/build/test/docs paths and inspected
  local wheel/sdist artifacts from both packages.

### Phase 2 - Ruff, pydoclint, docstrings, and pre-commit

- Session work: map legacy rules, introduce Ruff lint then formatting, install
  pydoclint in temporary NumPy mode, enforce the docstring strategy, and switch
  pre-commit to locked project tools.
- Preserve a formatting-only boundary even if it shares a session with other
  work, so behavioural regressions remain diagnosable.
- Exit evidence: legacy tools absent, public docstrings complete, baselines
  recorded, and pre-commit/full local commands invoking locked project tools.

### Phase 3 - Type checking

- Session work: benchmark candidates on the same modules, record the decision,
  install exactly one checker, remove mypy, and reduce its baseline by domain
  group.
- Do not add `py.typed` until public annotation quality is supportable and the
  built artifacts have been checked for the marker.
- Exit evidence: one fast mandatory checker, no new findings, shrinking debt,
  and a recorded public typing-support level.

### Phase 4 - TUI responsiveness

- Session work: make action checks pure, reduce refreshes, index project data,
  move blocking work to cancellable workers, defer imports, and replace raw
  terminal progress.
- Every optimisation session records the same before/after workload and keeps
  correctness tests separate from timing evidence.
- Exit evidence: zero I/O in action predicates, responsive loading, safe worker
  boundaries, and measured import/handler improvements.

### Phase 5 - Stable module boundaries

- Session work: package the TUI, extract screens/state/services, unify plotting,
  split gather responsibilities, finish MTH5 cleanup, and remove verified dead
  code.
- Preserve public imports and entry points while moving code. Do not combine a
  move with unrelated behaviour or formatting work that hides regressions.
- Exit evidence: stable public API, focused modules, shared plot contracts, and
  tests covering moved behaviour.

### Phase 6 - Dependencies and platforms

- Session work: map direct dependencies, correct unsupported lower bounds, test
  built artifacts across dependency/Python/OS boundaries, add auditing, and make
  the evidence-based uv/Pixi decision.
- A Pixi experiment is not an adoption. Record the reproduced uv limitation and
  comparative result before changing the primary environment workflow.
- Exit evidence: credible dependency metadata, clean supported installs,
  visible vulnerability ownership, and a recorded environment-manager decision.

### Phase 7 - MyST documentation

- Session work: prove the parser/API-generator path, migrate narrative pages,
  convert docstrings without moving their examples or plots, replace the legacy
  gallery, remove the transitional stack, and enforce documentation builds.
- Track converted and unconverted modules explicitly while dual parser and
  pydoclint configurations exist.
- Exit evidence: MyST-only authored documentation, one API generator, complete
  public docstrings, executable doctests/plots, and warning-clean HTML builds.

### Phase 8 - Reconciliation and final audit

- Session work: reconcile active plans, repeat every baseline measurement, and
  record remaining owned risks without expanding the programme indefinitely.
- Exit evidence: every governing outcome satisfied or explicitly deferred, and
  `mth5` demonstrably production-ready. Promotion remains outside scope.

## Verification Evidence Standard

Record exact commands, exit status, and meaningful totals. A statement such as
"tests pass" is insufficient without the command and result.

Minimum evidence by change type:

| Change type | Required evidence |
| --- | --- |
| Behaviour/correctness | Focused regression test and applicable full suite |
| Formatting/lint | Ruff format check, Ruff lint, and pre-commit |
| Docstring | Ruff `D`, pydoclint, doctest, and HTML build as applicable |
| Type contract | Selected checker plus runtime regression tests |
| TUI | Pilot/unit tests, no-I/O assertions, and before/after measurement |
| Packaging | Locked sync, build, artifact inspection, isolated smoke test |
| Dependencies | Resolution/install matrix and built-artifact tests |
| Workflow | Syntax/config validation and safe dry-run where possible |
| Documentation | Warning-clean HTML, doctest, plot execution, references |

Full-suite commands and tool names may change during migration. Record the
actual command used in the session log rather than rewriting historical entries.

## Measurement Ledger

All starting values below come from the 2026-07-18 audit in the governing plan
and must be re-measured in Phase 0 before they are treated as verified.

| Metric | Audit value | Verified baseline | Latest value | Evidence |
| --- | --- | --- | --- | --- |
| Tests | 372 collected; 371 passed; 1 failed | 373 passed | 373 passed | S002 |
| Branch coverage | approximately 76% | 75.96% | 75.96% | S002; coverage XML |
| Production Python | approximately 21,011 lines | 21,015 | 21,015 | S002 report |
| Tests | approximately 5,941 lines | 6,051 | 6,051 | S002 report |
| `resistics/tui.py` | 2,673 lines | 2,673 | 2,673 | S002 report |
| `resistics/plot.py` | 1,200 lines | 1,200 | 1,200 | S002 report |
| Flake8 | approximately 40 findings | 43 findings | 43 findings | S001 |
| Legacy complexity | not recorded | 17 CCR001; 5 C901; 4 ECE001 | same | S002 |
| Black format | not recorded | 26 files differ | 26 files differ | S001 |
| mypy | 210 errors across 17 files | 209 across 17 files | same | S002 |
| TUI cold import | approximately 2.18 seconds | 2.2659 s median | same | S002 report |
| Cached TUI action checks | not measured | 1,200 in 0.002322 s; zero instrumented I/O | same | S002 XML |
| Public docstring coverage | not measured | 80.2%; 566/706 | same | S002 report |
| Executable docstring examples | not measured | 778 prompts | same | S002 report |
| Executable docstring plots | not measured | 16 directives | same | S002 report |
| Local `.venv` size | approximately 897 MB | 910 MB | 910 MB | S002 |

For timing measurements, record the machine/runtime context and multiple runs.
Do not compare one cold run with one warm run or turn machine-specific timings
into brittle automated assertions.

## Decision Log

Do not rewrite historical decisions. Append a superseding decision and link the
old id when evidence changes the direction.

- `D001` (2026-07-19): Work on `mth5`; it is the hardening integration branch.
- `D002` (2026-07-19): Production promotion is out of scope and remains the
  owner's responsibility.
- `D003` (2026-07-19): Use uv unless Phase 6 evidence justifies Pixi; do not
  maintain two equivalent environment paths.
- `D004` (2026-07-19): Replace Black, Flake8, darglint, and mypy with Ruff,
  pydoclint, and one evidence-selected type checker.
- `D005` (2026-07-19): MyST is the final authored documentation language; do
  not permanently maintain MyST and reStructuredText sources.
- `D006` (2026-07-19): Keep useful examples and plots in docstrings because
  co-location is a user requirement.
- `D007` (2026-07-19): Evidence-gate the API generator. MyST is mandatory, but
  autodoc2 is not if API parity fails.
- `D008` (2026-07-19): Require substantive public docstrings; presence alone is
  not production documentation.
- `D009` (2026-07-19): Add `py.typed` only after public annotations are
  supportable; do not advertise PEP 561 support prematurely.
- `D010` (2026-07-19): Phase 0 protects the TUI action predicates already
  backed by in-memory state and records the existing plot/deletion I/O paths as
  explicit debt. Phase 4.1 will refactor those paths and make zero I/O a strict
  gate; Phase 0 must not claim a property the current production code lacks.
- `D011` (2026-07-19): Set the initial coverage gate to 75.95% with two-decimal
  reporting. The measured 75.96% appears as 76% at whole-number precision, but
  a literal 76% threshold rejects the confirmed green baseline.
- `D012` (2026-07-19): Set regressioninc's package contract to Python
  `>=3.11,<3.15` and Pydantic `>=2.0`, matching resistics' intended Python
  matrix and the existing Pydantic 2 migration. Cross-platform matrix evidence
  remains part of Checkpoint 1.6 and Phase 6.
- `D013` (2026-07-19): Preserve regressioninc's declared runtime dependency set
  during the build-backend migration. Direct-dependency rationalisation and
  credible lower bounds belong to Phase 6, not the Poetry-removal checkpoint.
- `D014` (2026-07-19): Regressioninc automation uses a locked uv environment,
  tests the minimum and maximum supported Python versions on Linux and Windows,
  and runs legacy Flake8/darglint only once until Phase 2 replaces them.
- `D015` (2026-07-19): Regressioninc publishing remains a manually dispatched,
  two-job workflow. The publish job has only OIDC permission and the `pypi`
  environment; configuring that environment and the matching PyPI Trusted
  Publisher remains a manual owner action.
- `D016` (2026-07-19): Keep the narrow Matplotlib and Sphinx Gallery
  compatibility fixes discovered by the clean docs environment in Checkpoint
  1.2. They are required for the replacement Read the Docs path to execute and
  do not pre-empt the broader Phase 7 documentation migration.
- `D017` (2026-07-19): Supersede the release-first dependency sequence. Keep
  regressioninc as the editable sibling uv source throughout resistics
  hardening; refresh and verify the paired lock rather than publishing an alpha
  solely to remove the local source.
- `D018` (2026-07-19): Hosted CI, Read the Docs deployment, standalone registry
  installation, and publishing automation are owner follow-ups. They do not
  gate Ruff/pydoclint, type checking, TUI performance, module boundaries,
  dependency auditing, MyST documentation, or the local final audit.
- `D019` (2026-07-19): Checkpoint 1.4 verifies the current narrative/API docs
  with Sphinx Gallery execution disabled. The full build's 18 obsolete example
  failures are measured Phase 7 debt, not a reason to mix gallery migration
  into the local dependency-boundary checkpoint.
- `D020` (2026-07-19, superseded by D022): Declare and verify Python
  `>=3.11,<3.15`. Python 3.11 initially resolved NumPy `<2` because the
  compatible lttbc extension line was not NumPy-2 compatible.
- `D021` (2026-07-19): Restrict the source distribution to the package source,
  tests, build metadata, README, changelog, and license. Repository plans,
  workflows, generated documentation, notebooks, and 68 MB of example datasets
  remain available from the repository but are not package-source contents.
- `D022` (2026-07-19): Replace lttbc with `tsdownsample>=0.1.5.1,<0.2` and
  remove Resistics' provisional Python 3.11 NumPy upper bound. Use standard LTTB
  for finite arrays and the documented NaN-aware MinMaxLTTB implementation when
  a floating-point series contains gaps. Apply returned indices to the original
  arrays to preserve dtype and large-index precision.
- `D023` (2026-07-19): Delete the obsolete Resistics commit, publishing, and
  Read the Docs deployment configurations instead of partially translating
  them. The editable sibling dependency has no hosted checkout design yet, so
  an incomplete uv rewrite would imply automation that cannot work. Preserve
  the owner's Python 3.11/3.14 matrix intent for Checkpoint 1.6 and rebuild
  hosted documentation and publication only in their owned deferrals.
- `D024` (2026-07-19): Replace Resistics' Flake8 stack with Ruff 0.15.22 and an
  explicit stable rule selection: `A`, `B`, `C4`, `C90`, `E`, `F`, `I`,
  `PERF`, `PT`, `RUF`, `S`, `SIM`, `UP`, and `W`. Core Flake8, builtins,
  Bandit, pytest-style, and McCabe coverage map to those families. Existing
  flake8-docstrings checks were disabled by the old selection, so Ruff `D`
  activation remains owned by Checkpoint 2.4 rather than being mixed into the
  lint migration. Ruff has no equivalent for the cognitive- or
  expression-complexity plugins; preserve their measured 17- and 4-finding
  baselines as explicit Phase 4/5 refactoring debt. Four current McCabe
  violations use function-level suppressions so new complexity remains checked.
- `D025` (2026-07-19): Make Ruff 0.15.22 the sole Resistics Python formatter
  with an explicit Black-compatible 88-column, double-quote, space-indent, and
  magic-trailing-comma contract. Keep docstring code formatting disabled so
  embedded examples, plots, and directive syntax remain author-controlled.
  Ruff continues to honour the existing `fmt: off/on` data-layout boundary.
  Remove Black from Resistics, while treating the remaining Black name in
  `uv.lock` as sibling regressioninc development metadata rather than a
  resolved Resistics tool.
- `D026` (2026-07-19): Replace darglint with pydoclint 0.9.1's native CLI in
  NumPy mode. Keep argument, return, yield, exception, annotation, and class
  attribute consistency checks enabled, and commit the measured 233-finding
  compatibility baseline instead of disabling those checks. Baseline
  regeneration is manual so hooks cannot silently accept new debt; definition
  lines are the canonical location for native `noqa` exceptions. Retain
  `skip-checking-short-docstrings = true` only through Checkpoint 2.3 because
  Checkpoint 2.4 explicitly measures and enables the stricter setting.
- `D027` (2026-07-19): Enforce Ruff D100-D104 throughout production source and
  require NumPy-style contracts even for short docstrings. Exclude Pydantic
  field/model validators by their fully qualified decorators and exclude only
  D102 for `resistics/tui.py`, whose public-looking methods are Textual
  callbacks and actions rather than supported library APIs. Do not enable
  D105-D107: constructors belong to their class contract, and conventional
  dunders/framework entry points should not acquire duplicate prose. Preserve
  API-local examples, plots, and Sphinx directives in place; protect their
  current inventory with AST-based tests and normal doctest collection.

## Blocker Log

Add one row whenever a checkpoint becomes `blocked`. Close it by recording the
unblock evidence; do not delete it.

| ID | Opened | Checkpoint | Blocker | Unblock condition | State |
| --- | --- | --- | --- | --- | --- |
| B000 | - | - | No blockers recorded | - | `closed` |

## Artifact Layout

Use the ignored `.artifacts/` directory for generated evidence that should not
be committed:

```text
.artifacts/
`-- hardening/
    |-- baseline/
    |-- coverage/
    |-- dependencies/
    |-- documentation/
    |-- packaging/
    |-- performance/
    `-- typing/
```

Commit durable configuration, baselines required by tools, decision records,
and small human-readable summaries when they are part of the product or
developer workflow. Do not commit large generated HTML, profiles, wheels, or
temporary environments.

## Session Handoff Template

Append new sessions below the existing log. Do not edit an old handoff except to
correct a factual error; note the correction explicitly.

```markdown
### SNNN - YYYY-MM-DD - <checkpoint and concise goal>

- Checkpoint state at start:
- Starting HEAD and worktree:
- Session objective:
- Work completed:
- Files changed:
- Decisions added or superseded:
- Verification commands and results:
- Measurements/artifacts:
- Known failures or incomplete work:
- Checkpoint state at end:
- Commit readiness or commit id:
- Exact next action:
```

## Session Log

### S000 - 2026-07-19 - Initialize the implementation record

- Checkpoint state at start: implementation tracking did not yet exist.
- Starting HEAD and worktree: `c345ae6` on `mth5`; the governing plan and two
  GitHub workflow files were already modified.
- Session objective: create a multi-session execution and handoff structure for
  the governing code-hardening plan.
- Work completed: created this tracker, dependency map, phase briefs,
  verification standard, measurement ledger, decision log, blocker log, and
  handoff template.
- Files changed: `.agents/plans/code-hardening-implementation.md`.
- Decisions added or superseded: recorded D001-D009 from the governing plan and
  user direction; no new implementation decision was made.
- Verification commands and results: `git diff --check` passed after file
  creation.
- Measurements/artifacts: none; the audit values remain unverified.
- Known failures or incomplete work: Phase 0 has not started.
- Checkpoint state at end: `0.1` remains `not_started`.
- Commit readiness or commit id: planning file is ready; no commit requested.
- Exact next action: begin S001 by refreshing the current baseline and
  reproducing Phase 0.1 failures without modifying production code first.

### S001 - 2026-07-19 - Restore the green test baseline

- Checkpoint state at start: `0.1` was `not_started`.
- Starting HEAD and worktree: `c345ae6` on `mth5`; the governing plan, this
  implementation record, and two GitHub workflow files were already dirty.
- Session objective: reproduce the current test and legacy quality baseline,
  then fix only confirmed Checkpoint 0.1 correctness failures.
- Work completed: updated the transfer-function doctest to the canonical
  lowercase channel output and made the lazily imported `GatherCriteria` type
  visible to static analysis through a `TYPE_CHECKING` import.
- Files changed: `resistics/transfunc.py`, `resistics/templates.py`, and this
  implementation record.
- Decisions added or superseded: none. Formatting, complexity, unused-import,
  test-only security, and expression-complexity findings remain recorded debt
  for their governing Ruff/refactor checkpoints.
- Verification commands and results:
  - Initial `.venv/bin/pytest -q`: 371 passed, one known transfer-function
    doctest failure.
  - Focused doctest and project tests: 9 passed.
  - Final `.venv/bin/pytest -q`: 372 passed in 20.39 seconds.
  - `.venv/bin/flake8 resistics tests --statistics`: 43 remaining findings;
    the `GatherCriteria` F821 finding is gone. Counts are 17 CCR001, 5 C901,
    4 E302, 4 ECE001, 3 F401, 3 S311, 2 E201, and one each of E126, E131, E226,
    F841, and S108.
  - Black 26.5.1 checked files individually after whole-tree invocations
    stalled: 26 files would be reformatted. The touched `templates.py` passes;
    `transfunc.py` already contained legacy format differences.
  - `git diff --check`: passed.
- Measurements/artifacts: test, Flake8, and Black values were added to the
  measurement ledger. No generated artifacts were retained.
- Known failures or incomplete work: legacy Black and Flake8 checks are not
  green; Checkpoint 0.1 permits these known findings to be carried into the
  Ruff migration. The default whole-tree Black invocation stalled twice and
  was replaced by bounded per-file checks for this baseline.
- Checkpoint state at end: `0.1` is `verified` but uncommitted; `0.2` is queued.
- Commit readiness or commit id: the two scoped production changes and tracker
  update are ready for a checkpoint commit; no commit was requested.
- Exact next action: start S002 by measuring Checkpoint 0.2 coverage, code size,
  public docstrings, executable examples/plots, TUI import time, and action-state
  I/O behaviour before adding durable gates.

### S002 - 2026-07-19 - Establish durable quality and performance baselines

- Checkpoint state at start: `0.2` was `in_progress`; the user had recorded
  Checkpoint 0.1 at `ebede10` after S001.
- Starting HEAD and worktree: `ebede10` on `mth5`; this implementation record
  and the two GitHub workflow files were dirty. The workflow changes remain
  user-owned and untouched.
- Session objective: measure the complete Phase 0 baseline, add repeatable
  reports and coverage enforcement, and protect frequently evaluated cached
  TUI action predicates without mixing in a production refactor.
- Work completed: added a 75.95% branch-coverage floor with two-decimal
  reporting; moved generated coverage output below `.artifacts/hardening/`;
  added a standard-library baseline reporter for code size, public/private
  docstring inventory, embedded examples/plots, environment context, and
  fresh-process TUI imports; and added an instrumented benchmark covering 1,200
  cached Footer/action checks.
- Files changed: `.agents/plans/codebase-hardening.md`,
  `.agents/plans/code-hardening-implementation.md`, `pyproject.toml`,
  `scripts/hardening_report.py`, and `tests/test_tui.py`.
- Decisions added or superseded: D010 records the phased TUI no-I/O gate and
  D011 records the exact coverage threshold. The governing Checkpoint 0.2 text
  was reconciled with the measured code instead of claiming all existing
  action predicates were already I/O-free.
- Verification commands and results:
  - `.venv/bin/pytest -q tests/test_tui.py -k cached_action_checks`: one passed;
    1,200 checks completed in 0.002322 seconds with zero calls through the
    instrumented filesystem, project, YAML, JSON, MTH5-listing, or
    job-validation seams.
  - `.venv/bin/pytest -q --cov=resistics --cov-branch --cov-report=term
    --cov-report=html --cov-report=xml`: 373 passed in 34.31 seconds; measured
    branch coverage 75.96% and the 75.95% floor passed.
  - Final `.venv/bin/pytest -q`: 373 passed in 20.73 seconds after the
    filesystem instrumentation was added to the focused TUI benchmark.
  - `.venv/bin/black --check scripts/hardening_report.py` and
    `.venv/bin/flake8 scripts/hardening_report.py --statistics`: passed with no
    findings.
  - `.venv/bin/flake8 resistics tests --statistics`: reproduced 43 legacy
    findings, including 17 CCR001, 5 C901, and 4 ECE001 complexity findings.
  - `.venv/bin/mypy resistics`: reproduced 209 errors across 17 of 21 source
    files with mypy 2.1.0.
  - `git diff --check`: passed at the final handoff.
- Measurements/artifacts: `.artifacts/hardening/baseline/codebase.json` records
  21,015 production lines, 6,051 test lines, 80.2% documented public objects
  (566/706), 140 missing public docstrings, 19 substantial undocumented private
  operations, 778 doctest prompts, 16 plot directives, and five TUI import
  samples on CPython 3.13.5/WSL2. The final cold-import range was
  2.2158-2.3125 seconds with a 2.2659-second median. Coverage HTML/XML and the
  TUI timing property are stored under `.artifacts/hardening/coverage/` and
  `.artifacts/hardening/performance/`. The local `.venv` is 910 MB.
- Known failures or incomplete work: plot action predicates still call project
  summary/data lookup, MTH5 run listing, flow YAML loading, and job validation;
  the data-deletion predicate requests a filesystem-backed deletion preview.
  D010 assigns these production changes to Phase 4.1, where the existing
  benchmark becomes a strict all-action zero-I/O gate. Legacy lint and typing
  debt remains assigned to Phases 2 and 3.
- Checkpoint state at end: `0.2` and Gate 0 are `verified`; no production code
  was changed.
- Commit readiness or commit id: the scoped Checkpoint 0.2 changes are ready
  for a user-approved commit; no commit was requested or created.
- Exact next action: run `git status --short`, review the five scoped files
  (including the new reporter), and record Checkpoint 0.2 durably before
  locating the sibling `regressioninc` repository for Checkpoint 1.1.

### S003 - 2026-07-19 - Convert regressioninc metadata to PEP 621 and uv

- Checkpoint state at start: `1.1` was `not_started`; Phase 0 and Checkpoint
  0.2 were verified but remained uncommitted in resistics.
- Starting HEAD and worktree: regressioninc `9eb11a4` on `main` with a
  pre-existing user change in `regressioninc/base.py`; resistics remained at
  `ebede10` on `mth5` with the documented Checkpoint 0.2 and workflow changes.
- Session objective: replace regressioninc's Poetry package metadata and build
  backend, establish a uv lock, and prove clean source and wheel workflows
  without modifying or absorbing the user's source change.
- Work completed: converted `[tool.poetry]` metadata to PEP 621 `[project]`;
  moved development, documentation, and test requirements into uv dependency
  groups; selected Hatchling; aligned the package with Python `>=3.11,<3.15`
  and Pydantic 2; generated `uv.lock`; and removed the obsolete `poetry.lock`.
- Files changed by this checkpoint: `../regressioninc/pyproject.toml`,
  `../regressioninc/uv.lock`, and deletion of `../regressioninc/poetry.lock`.
  The dirty `../regressioninc/regressioninc/base.py` was inspected but not
  edited.
- Decisions added or superseded: D012 records the Python/Pydantic contract;
  D013 defers dependency rationalisation to its evidence-based phase.
- Verification commands and results:
  - Baseline `../resistics/.venv/bin/pytest -q -p no:cacheprovider`: 28 passed
    in 2.20 seconds before packaging changes.
  - `uv lock` and final `uv lock --check`: resolved 105 packages for the full
    Python 3.11-3.14 contract; no Poetry package or backend occurs in the lock.
  - Clean `UV_PROJECT_ENVIRONMENT=<temporary> uv sync --frozen`: installed 98
    default-group packages and built the editable project successfully.
  - Pytest from the clean uv environment: 28 passed in 3.17 seconds.
  - `uv build --no-sources`: built
    `regressioninc-0.1.0a0.tar.gz` and
    `regressioninc-0.1.0a0-py3-none-any.whl`; Hatchling built the wheel from the
    source distribution.
  - The rebuilt wheel was installed into a second temporary environment. Its
    import resolved from `site-packages`, reported version `0.1.0a0`, and its
    unit tests plus installed-package doctests passed: 28 in 3.31 seconds.
  - Final `git diff --check`: passed in regressioninc.
- Measurements/artifacts: `uv.lock` contains 2,356 lines. The wheel is 10,715
  bytes and contains only the package, dist-info metadata, and MIT license. The
  150,616-byte sdist contains `uv.lock` and no `poetry.lock`. Core metadata is
  version 2.4 with the expected Python range, MIT license expression, and seven
  preserved runtime requirements.
- Known failures or incomplete work: Poetry commands remain in regressioninc's
  GitHub workflow and Read the Docs configuration, while legacy Poetry-era
  tools remain in pre-commit and dependency groups. These are deliberately
  assigned to Checkpoints 1.2 and 2. The built worktree also contains the
  pre-existing uncommitted Pydantic source change; it must be reviewed and
  committed independently from the three packaging changes. Resistics' own
  `uv lock --check` now reports stale because its editable sibling source has
  new metadata; Checkpoint 1.4 owns that expected intermediate coupling, and
  no resistics lock update was made in S003.
- Checkpoint state at end: `1.1` is `verified`; Gate 1 remains open.
- During-session repository update: the owner durably recorded resistics
  Checkpoint 0.2 in four commits, `b350893`, `19fdba2`, `5f55c6b`, and
  `d9911e3`. Current State and the tracker now mark Checkpoint 0.2 and Gate 0
  complete at `d9911e3`; this does not alter the historical S003 starting HEAD.
- Commit readiness or commit id: the three scoped packaging changes are ready
  for a regressioninc commit; no commit was requested or created.
- Exact next action: run `git -C ../regressioninc status --short`, review only
  `pyproject.toml`, `uv.lock`, and the `poetry.lock` deletion, then durably
  record Checkpoint 1.1 before starting Checkpoint 1.2 automation cleanup.

### S004 - 2026-07-19 - Replace regressioninc automation and references

- Checkpoint state at start: `1.2` was `in_progress`; Checkpoint 1.1 remained
  verified but uncommitted in regressioninc.
- Starting HEAD and worktree: regressioninc `9eb11a4` on `main` with the
  documented uncommitted Checkpoint 1.1 files and the owner's independent
  Pydantic 2 edit in `regressioninc/base.py`; resistics was at `9052136` on
  `mth5` with two pre-existing workflow edits.
- Session objective: remove remaining Poetry-era automation and contributor
  references, establish a repeatable repository guard, and prepare but not run
  a Trusted Publishing workflow.
- Work completed: replaced regressioninc's CI with locked uv minimum/maximum
  Python jobs; added a separated manual build/OIDC publish workflow; switched
  Read the Docs to its native uv installation method; documented the uv
  developer and publishing paths; added a tracked/unignored-file repository
  guard; and removed the obsolete `.gitignore` commentary. Clean documentation
  verification also exposed and fixed current Matplotlib colormap calls, the
  Sphinx Gallery temporary-file matcher, an absent static directory, and an
  unpickleable gallery sort configuration.
- Files changed by this checkpoint: `../regressioninc/.github/workflows/commit_flow.yml`,
  `../regressioninc/.github/workflows/publish_flow.yml`,
  `../regressioninc/.readthedocs.yaml`, `../regressioninc/.gitignore`,
  `../regressioninc/README.md`, `../regressioninc/docs/source/conf.py`,
  `../regressioninc/regressioninc/testing/complex.py`,
  `../regressioninc/regressioninc/testing/real.py`, and
  `../regressioninc/scripts/check_no_legacy_packaging.py`. The owner's dirty
  `regressioninc/base.py` was preserved without modification.
- Decisions added or superseded: D014 records the regressioninc CI shape; D015
  records the manual, least-privilege Trusted Publishing boundary; D016 records
  the compatibility fixes needed to make the modern docs environment run.
- Verification commands and results:
  - `python scripts/check_no_legacy_packaging.py`: passed against the real
    working tree. A deliberate untracked probe produced the expected exit 1
    and exact file/line finding.
  - PyYAML `yaml.compose` parsed both GitHub workflows and
    `.readthedocs.yaml`; final `git diff --check` passed.
  - `uv lock --check`: resolved the existing 105-package lock without changes.
  - Clean `uv sync --locked --no-default-groups --group dev --group tests
    --group docs`: installed 98 packages and built regressioninc successfully.
  - Flake8 and darglint passed against `regressioninc` in the clean environment.
  - A headless `plot_2d` smoke test exercised the updated real-data colormap
    path and produced a figure successfully.
  - `pytest --cov=regressioninc --cov-branch`: 28 passed on Python 3.13.5;
    measured branch coverage was 43%.
  - A second clean docs-only locked environment installed 57 packages.
    `sphinx-build -W --keep-going -b html` then executed all seven gallery
    examples and completed with zero warnings after fetching the configured
    intersphinx inventories.
  - `uv build --no-sources` produced the wheel and source distribution. An
    extracted-artifact scan found no forbidden packaging reference outside the
    deliberately self-describing repository checker.
- Measurements/artifacts: temporary environments, HTML, and distributions
  were kept under `/tmp`; no generated artifacts were added to either worktree.
- Known failures or incomplete work: GitHub Actions and publishing were not
  executed. Before any manual publish, the owner must configure the GitHub
  `pypi` environment and a matching PyPI Trusted Publisher for
  `publish_flow.yml`. Checkpoints 1.1 and 1.2 remain uncommitted together with
  the separate owner-authored `base.py` edit.
- Checkpoint state at end: `1.2` is `verified`; Gate 1 remains open.
- Commit readiness or commit id: the scoped Checkpoint 1.2 changes are ready
  for a regressioninc commit; no commit was requested or created.
- Exact next action: begin Checkpoint 1.3 by inspecting regressioninc's version
  exports and PEP 621 metadata, selecting and documenting the compatible alpha
  release boundary, then repeating isolated artifact verification without
  publishing or changing repository settings.

### S005 - 2026-07-19 - Retain the local regressioninc boundary

- Checkpoint state at start: `1.3` had been set `in_progress`, but no release
  boundary edit had landed.
- Starting HEAD and worktree: resistics `9052136` on `mth5` with the tracker and
  two pre-existing workflow files dirty; regressioninc `9eb11a4` on `main` with
  the documented Checkpoints 1.1/1.2 work and owner-authored `base.py` edit.
- Session objective: assess and record the owner's decision to prioritise local
  production-quality code and modern tooling while retaining the sibling
  regressioninc installation and deferring hosted/release work.
- Work completed: revised the governing plan and this tracker so the editable
  sibling source is intentional; replaced the release-first dependency chain
  with a local uv boundary; changed package, quality, typing, dependency, and
  documentation gates to use repeatable local commands; and moved hosted CI,
  Read the Docs deployment, registry-only installation, and publishing into
  explicit owner follow-ups.
- Files changed: `.agents/plans/codebase-hardening.md` and
  `.agents/plans/code-hardening-implementation.md`.
- Decisions added or superseded: D017 supersedes the release-first sequence;
  D018 makes hosted and publishing work non-gating.
- Verification commands and results:
  - Regressioninc status/diff review confirmed the aborted 1.3 edit changed no
    product file and its version remains `0.1.0a0`.
  - `uv lock --check` in resistics resolved 194 packages and reported that the
    lock needs updating, confirming the first revised Checkpoint 1.4 action.
  - `git diff --check` passed after the plan reconciliation.
- Measurements/artifacts: none.
- Known failures or incomplete work: resistics locked sync is not current until
  Checkpoint 1.4 deliberately refreshes the local regressioninc metadata.
  Standalone installation, hosted CI/Read the Docs, and publishing readiness
  must not be claimed during the local hardening programme.
- Checkpoint state at end: `1.3` is `deferred`; revised `1.4` is queued.
- Commit readiness or commit id: the two planning-file updates are ready; no
  commit was requested or created.
- Exact next action: begin revised Checkpoint 1.4 by inspecting the local
  regressioninc entries in `uv.lock`, run a deliberate `uv lock`, and verify a
  locked paired-repository sync before changing production code.

### S006 - 2026-07-19 - Formalise the local sibling boundary

- Checkpoint state at start: revised `1.4` was `not_started`; regressioninc
  release work and hosted automation were already deferred by S005.
- Starting HEAD and worktree: resistics `1e5b208` on `mth5` with both plan files
  and two owner workflow edits dirty; regressioninc remained at `9eb11a4` on
  `main` with the documented Checkpoints 1.1/1.2 and Pydantic 2 changes.
- Session objective: make the sibling regressioninc source an explicit,
  reproducible local boundary and prove it from a clean paired directory layout.
- Work completed: refreshed `uv.lock` against regressioninc's PEP 621 metadata;
  documented the required sibling directory names, locked setup/test commands,
  missing-path recovery, and non-registry scope in `README.md`; and clarified
  that Phase 7 owns execution of the obsolete Sphinx Gallery examples.
- Files changed by this checkpoint: `README.md`, `uv.lock`,
  `.agents/plans/codebase-hardening.md`, and this implementation record. No
  production Python file was changed.
- Decisions added or superseded: D019 keeps legacy gallery execution out of the
  dependency-boundary checkpoint while preserving the failure evidence for
  Phase 7.
- Verification commands and results:
  - `uv lock` and final `uv lock --check`: resolved 194 packages. The 27-line
    lock diff updates regressioninc's Pydantic requirement to `>=2.0` and
    records its uv dependency groups; no unrelated package version changed.
  - `uv sync --locked --all-groups` succeeded in the working tree and in a
    clean temporary paired layout, where it built both editable projects and
    installed 184 packages.
  - The documented missing-sibling case failed immediately with
    `Distribution not found at: .../regressioninc`, matching the README recovery
    guidance.
  - The full suite passed in both environments: 373 tests in 22.52 seconds and
    373 tests in 23.04 seconds respectively.
  - Installed metadata resolved regressioninc and resistics from their expected
    sibling paths; the `resistics` console entry point loaded
    `resistics.tui:main` as a callable. Direct `resistics.project` and
    `resistics.gather` imports also passed.
  - The unmodified full Sphinx build reproduced 18 legacy gallery failures:
    removed modules, removed pandas arguments, and Chrome-dependent Plotly
    rendering. With gallery execution disabled, the narrative/API HTML build
    succeeded with 98 existing warnings. This is the measured Phase 7 baseline,
    not a regressioninc-boundary failure.
  - Final `git diff --check` passed.
- Measurements/artifacts: local HTML evidence is under
  `.artifacts/hardening/documentation/checkpoint-1.4-html-no-gallery/`; temporary
  paired and missing-sibling environments are under `/tmp` only.
- Known failures or incomplete work: executable legacy gallery documentation
  and warning cleanup remain assigned to Phase 7. Standalone registry
  installation, hosted CI/Read the Docs, and publishing remain the D018 owner
  deferrals.
- Checkpoint state at end: `1.4` is `verified`; local Gate 1 remains open.
- Commit readiness or commit id: the scoped README/lock/plan changes are ready;
  no commit was requested or created.
- Exact next action: begin Checkpoint 1.5 by comparing resistics' alpha version,
  classifier, Python range, URLs, license, and package contents before changing
  metadata or building paired local artifacts.

### S007 - 2026-07-19 - Harden metadata and paired artifacts

- Checkpoint state at start: `1.5` was `in_progress` after Checkpoint 1.4 had
  verified the paired editable-repository boundary.
- Starting HEAD and worktree: resistics `1e5b208` on `mth5` with the two plan
  files, Checkpoint 1.4 README/lock work, and two owner workflow edits dirty;
  regressioninc remained at `9eb11a4` with its documented hardening changes.
- Session objective: align Resistics metadata with the actual alpha release,
  prove its supported Python range, reduce accidental distribution contents,
  and verify wheel/sdist installation with a locally built regressioninc wheel.
- Work completed: normalized version `1.0.0a3`; changed the Beta classifier to
  Alpha; declared Python 3.11-3.14 classifiers and `>=3.11,<3.15`; made the
  MIT license file explicit; corrected the documentation URL; required a
  PEP-639-capable Hatchling; added a minimal sdist allow-list; updated stale
  README Python/install text; and made the Windower example deterministic
  across NumPy 1 and 2 while retaining it in the class docstring.
- Compatibility correction: the first clean Python 3.11 run exposed an lttbc
  wheel compiled against NumPy 1 being installed with NumPy 2. The package now
  declares NumPy `<2` on Python 3.11 and modern NumPy on Python 3.12+, allowing
  the resolver to select the compatible lttbc line without a uv-only override.
- Files changed by this checkpoint: `pyproject.toml`, `README.md`,
  `resistics/window.py`, `uv.lock`, the governing plan carried from earlier
  sessions, and this implementation record. Owner workflow changes were not
  modified.
- Decisions added or superseded: D020 records the evidence-based NumPy marker;
  D021 defines the source-distribution boundary. The planned `py.typed` marker
  remains deferred until Phase 3.
- Verification commands and results:
  - Final locked suites passed against the exact working-tree source on CPython
    3.11.15, 3.12.13, 3.13.5, and 3.14.6: 373 tests on every minor. The final
    recorded runs took 33.64, 32.55, 21.77, and 30.03 seconds respectively;
    three matrix runs were intentionally concurrent and are not performance
    baselines.
  - `uv lock --check` resolved 195 packages and passed. The split lock selects
    NumPy 1.26.4/lttbc 0.2.4 below Python 3.12 and NumPy 2.5.0/lttbc 0.3.0 on
    Python 3.12+.
  - `uv build` produced `resistics-1.0.0a3.tar.gz` and the universal wheel. The
    wheel contains only the 21 production modules and dist-info, including
    `LICENCE.txt`; metadata and `resistics = resistics.tui:main` are correct.
  - The allow-listed sdist contains package source, 16 test files, README,
    changelog, license, and build metadata. It is 185,303 bytes, down from the
    39,903,321-byte default artifact, and excludes internal plans, workflows,
    notebooks, generated docs, and example datasets.
  - Isolated Python 3.14 wheel and Python 3.12 sdist installs each resolved the
    locally built `regressioninc-0.1.0a0` wheel, imported both packages from
    site-packages, reported the expected versions/Python range, and exercised
    the console launcher usage path with its expected exit status.
  - The Read the Docs 1.0 URL, GitHub repository, and issue tracker resolved;
    wheel and sdist metadata carry the corrected values and explicit MIT
    license expression/file.
  - Final `git diff --check` passed.
- Measurements/artifacts: inspected distributions are under
  `.artifacts/hardening/package/checkpoint-1.5/`; matrix and isolated install
  environments are under `/tmp/resistics-checkpoint-1-5-*` only.
- Known failures or incomplete work: registry-only regressioninc resolution,
  hosted automation, publishing, and cross-platform coverage remain D018 owner
  deferrals. Phase 3 owns the typed-package decision, and Phase 6 will perform
  the broader lower-bound/dependency review.
- Checkpoint state at end: `1.5` is `verified`; Checkpoint 1.8 is next and local
  Gate 1 remains open.
- Commit readiness or commit id: the scoped metadata, compatibility, artifact,
  README/lock, and tracking changes are ready; no commit was requested or
  created.
- Exact next action: begin Checkpoint 1.8 by inventorying active Poetry
  references case-insensitively, preserving historical plan evidence while
  removing obsolete local, documentation, workflow, notebook, and config paths.

### S008 - 2026-07-19 - Replace lttbc with tsdownsample

- Checkpoint state at start: `1.5` had been verified in S007 but was reopened
  after the owner approved replacing the unmaintained lttbc dependency rather
  than retaining its Python 3.11 NumPy workaround.
- Starting HEAD and worktree: resistics remained at `1e5b208` on `mth5` with
  the accumulated verified hardening work and two pre-existing owner workflow
  edits dirty; regressioninc remained at `9eb11a4` with its documented changes.
- Session objective: replace lttbc with an actively released LTTB backend,
  preserve plotting behaviour, remove the NumPy upper bound, and repeat all
  Checkpoint 1.5 compatibility and artifact gates.
- Work completed: replaced `lttbc>=0.2.1` with
  `tsdownsample>=0.1.5.1,<0.2`; changed the wrapper to select original-array
  indices without float32 conversion; made strided inputs contiguous only for
  the native call; selected NaN-aware MinMaxLTTB for floating-point gaps; and
  updated plotting docstrings to use the implementation-neutral LTTB term.
- Tests added: exact existing point selection remains unchanged; large int64
  indices and source dtypes are preserved; non-contiguous views are accepted;
  and a NaN marker survives downsampling so Plotly does not bridge a data gap.
- Files changed by this amendment: `pyproject.toml`, `uv.lock`,
  `resistics/plot.py`, `resistics/time.py`, `resistics/spectra.py`,
  `resistics/decimate.py`, `tests/test_plot.py`, and this implementation record.
- Decisions added or superseded: D022 supersedes D020's temporary NumPy marker
  while retaining the verified Python `>=3.11,<3.15` support declaration.
- Verification commands and results:
  - `uv lock` resolved 194 packages, removed lttbc 0.2.4 and 0.3.0, and added
    tsdownsample 0.1.5.1. Resistics once again declares unqualified
    `numpy>=1.20.2`; no lttbc package remains in the lock or artifacts.
  - Six focused LTTB test cases passed, including the three new precision,
    strided-view, and NaN-gap contracts.
  - Clean paired locked environments passed 376 tests on CPython 3.11.15,
    3.12.13, and 3.14.6 in 35.00, 32.34, and 31.80 seconds. The working CPython
    3.13.5 environment passed 376 tests in 21.42 seconds.
  - The temporary Python 3.11 environment was then upgraded from NumPy 1.26.4
    to NumPy 2.4.6 outside the lock and passed all 376 tests in 22.98 seconds,
    directly verifying that the former binary incompatibility is gone.
  - In an isolated comparison using the existing Resistics wrapper contract,
    standard tsdownsample LTTB selected exactly the same indices as lttbc for
    the existing example and synthetic 100,000- and 1,000,000-point signals.
    It was approximately 3.5-4 times faster while preserving original dtypes.
  - Fresh Python 3.11 wheel and Python 3.14 sdist installs resolved the local
    regressioninc wheel, installed tsdownsample with NumPy 2, omitted lttbc,
    retained a NaN gap through the public wrapper, imported from site-packages,
    and exercised the console launcher usage path successfully.
  - Final wheel metadata contains `tsdownsample<0.2,>=0.1.5.1`, contains no
    lttbc dependency or Python-specific NumPy restriction, and retains the
    inspected S007 license, URL, entry-point, and package-content boundary.
  - Final `uv lock --check` and `git diff --check` passed.
- Measurements/artifacts: rebuilt distributions remain under
  `.artifacts/hardening/package/checkpoint-1.5/`; clean matrix, comparison, and
  artifact environments are under `/tmp/resistics-*` only.
- Known failures or incomplete work: tsdownsample uses native Rust wheels, so
  Phase 6 should retain wheel-availability review for any future niche target.
  Its current release provides wheels for the declared Python minors on the
  mainstream Linux, macOS, and Windows targets.
- Checkpoint state at end: amended `1.5` is `verified`; Checkpoint 1.8 is again
  next and local Gate 1 remains open.
- Commit readiness or commit id: the dependency replacement and all prior
  verified Checkpoint 1.4/1.5 changes are ready; no commit was requested or
  created.
- Exact next action: begin Checkpoint 1.8 by inventorying active Poetry
  references case-insensitively, preserving historical plan evidence while
  removing obsolete local, documentation, workflow, notebook, and config paths.

### S009 - 2026-07-19 - Remove Poetry from active Resistics paths

- Checkpoint state at start: `1.8` was `not_started`; amended Checkpoint 1.5
  was committed and verified, while hosted CI, Read the Docs deployment, and
  publishing remained explicit owner deferrals.
- Starting HEAD and worktree: Resistics `61e1396` on `mth5` with the owner's
  Python 3.11/3.14 edits still present in both obsolete workflow files;
  regressioninc remained at `9eb11a4` with its documented hardening changes.
- Session objective: remove every active Poetry dependency path, make local
  documentation usage explicitly uv-only, and prevent legacy packaging
  references from returning without prematurely designing hosted automation.
- Work completed: deleted the obsolete commit and publishing workflows and the
  Poetry-backed Read the Docs configuration; documented the locked uv Sphinx
  command with legacy gallery execution disabled; removed a captured pypoetry
  environment path from a notebook warning; and added a case-insensitive
  repository checker covering tracked and unignored files.
- Guard boundary: the checker excludes only its own source and the two
  hardening plans, which must name retired tools as historical evidence. It
  scans active code, configuration, automation, documentation, notebooks, lock
  files, and untracked contributor work.
- Files changed by this checkpoint: deletion of
  `.github/workflows/commit_flow.yml`,
  `.github/workflows/publish_flow.yml`, and `.readthedocs.yaml`; updates to
  `README.md` and `notebooks/check_standalone_single.ipynb`; addition of
  `scripts/check_no_legacy_packaging.py`; and this implementation record.
- Decisions added or superseded: D023 records why obsolete hosted files are
  removed rather than replaced before the sibling-checkout design exists. It
  preserves the owner's Python matrix intent for the deferred CI checkpoint.
- Verification commands and results:
  - `uv sync --locked --all-groups` resolved 194 packages and checked all 184
    packages in the complete local environment.
  - The exact documented `uv run --locked --no-sync sphinx-build` command built
    HTML successfully with gallery execution disabled. It reported the 93
    currently visible legacy warnings, retained as measured Phase 7 debt.
  - The full source suite passed: 376 tests in 21.67 seconds.
  - `uv build --no-sources` rebuilt the 1.0.0a3 wheel and source distribution
    under `.artifacts/hardening/package/gate-1/`; the sdist retained its narrow
    43-member package/test/metadata boundary.
  - Final `uv lock --check` resolved 194 packages and passed.
  - Both repositories' legacy-packaging checkers reported no active references.
    Resistics' final `git diff --check` also passed.
- Measurements/artifacts: local HTML is under
  `.artifacts/hardening/documentation/checkpoint-1.8-html-uv/`; the rebuilt
  distributions are under `.artifacts/hardening/package/gate-1/`.
- Known failures or incomplete work: no hosted CI, Read the Docs deployment, or
  publishing workflow is active. D018/D023 assign their replacement to the
  repository owner after stable local commands and a sibling-checkout design
  exist. The documentation warnings and obsolete gallery remain Phase 7 debt.
- Checkpoint state at end: `1.8` and local Gate 1 are `verified`; Phase 2 can
  start at Checkpoint 2.1.
- Commit readiness or commit id: the scoped Checkpoint 1.8 removals, guard, and
  documentation changes are ready; no commit was requested or created.
- Exact next action: inventory Black and Flake8 configuration, dependencies,
  suppressions, scripts, and pre-commit hooks for Checkpoint 2.1, then map every
  active rule to Ruff before changing enforcement.

### S010 - 2026-07-19 - Replace Flake8 linting with Ruff

- Checkpoint state at start: `2.1` was `not_started`; Checkpoint 1.8 and the
  local Phase 1 gate had been committed at Resistics `a45a864` on `mth5`.
- Starting HEAD and worktree: Resistics was clean at `a45a864`; regressioninc
  remained at `9eb11a4` with its separately documented uncommitted hardening
  changes.
- Session objective: replace the overlapping Resistics Flake8/plugin stack with
  an explicit Ruff lint contract, preserve or document every legacy rule
  boundary, and establish a green fast baseline without starting the formatting
  or docstring-contract migrations.
- Baseline inventory: the legacy command reported 44 findings in 4.18 seconds:
  10 core/style/security findings, 5 McCabe findings, 17 cognitive-complexity
  findings, 4 expression-complexity findings, and 8 associated test/scientific
  findings. Black formatting and darglint were deliberately left installed.
- Work completed: added and locked Ruff 0.15.22; selected the stable families in
  D024 with Python 3.11 and 88-column settings; configured narrow scientific,
  test, Pydantic, and Textual exceptions; retained CSV pytest parameter names;
  replaced the Flake8 pre-commit hook with the matching official Ruff hook;
  removed all Resistics Flake8 packages and `.flake8`; and applied Ruff's safe
  import, annotation, and syntax modernisations.
- Residual review: unsafe fixes were never applied wholesale. Equivalent
  transformations were selected by rule and backed by the full suite, including
  explicit `zip(strict=False)`, exception chaining, warning stack levels,
  boolean dtype validation, Python 3.11 aliases, and strengthened exception
  assertions. Four pre-existing McCabe failures have function-level `C901`
  suppressions tied to their owning refactoring phases.
- Rule mapping and omissions: core `E/F/W`, builtins `A`, Bandit `S`,
  pytest-style `PT`, and McCabe `C90` are active. The formerly installed
  pydocstyle integration had no enabled `D` rules, so docstring presence/style
  remains Checkpoint 2.4 work. Ruff has no cognitive- or expression-complexity
  equivalent; their measured debt is recorded in D024 rather than silently
  presented as covered.
- Files changed by this checkpoint: `.flake8` deletion; Ruff dependency,
  configuration, lock, and pre-commit changes; mechanically modernised imports,
  annotations, and selected lint findings across production source, tests, and
  repository scripts; and this implementation record.
- Verification commands and results:
  - `ruff check resistics tests scripts`: all checks passed in 0.030 seconds,
    approximately 139 times faster than the 4.18-second legacy baseline.
  - The pinned `pre-commit run ruff-check --all-files` hook passed. Its notices
    about the still-legacy hygiene, Prettier, Black, and darglint hooks remain
    owned by Checkpoints 2.2-2.5.
  - Black confirmed every changed Python file still matches the current
    formatter boundary.
  - The full suite passed: 376 tests in 23.43 seconds.
  - `uv lock --check` resolved 179 packages and passed; the environment removed
    the 17 resolved Flake8/plugin support packages and added Ruff 0.15.22.
  - The legacy-packaging guard and final `git diff --check` passed.
- Known failures or incomplete work: regressioninc's metadata still records its
  own legacy development lint group inside the paired lock, although those
  packages are not resolved into Resistics' environment. Its quality-tool
  migration remains separate repository work. Black, darglint, and the old
  pre-commit hygiene/Prettier hooks remain intentionally active until their
  sequenced Phase 2 checkpoints.
- Checkpoint state at end: `2.1` is `verified`; Checkpoint 2.2 is next.
- Commit readiness or commit id: the Ruff configuration, mechanical fixes,
  Flake8 removal, and verification record are ready; no commit was requested or
  created.
- Exact next action: compare Ruff formatting against Black on this exact tree,
  review the formatting-only diff, then remove Black and its badge/hook/config
  after the Ruff-formatted suite passes.

### S011 - 2026-07-19 - Switch formatting from Black to Ruff

- Checkpoint state at start: `2.2` was `not_started`; Checkpoint 2.1 had been
  committed at Resistics `f2d7c0d` on `mth5`, and the worktree was clean.
- Starting HEAD and worktree: Resistics was clean at `f2d7c0d`; regressioninc
  remained at `9eb11a4` with its separately documented uncommitted hardening
  changes.
- Session objective: establish Ruff as the sole Resistics formatter while
  keeping the format-only diff isolated, reviewed, and behaviour-neutral.
- Comparison and review: the committed tree passed Black. Ruff identified six
  files for formatting and left 33 unchanged. The reviewed delta removed extra
  class-body blank lines, normalized three f-string expressions, joined one
  adjacent mask-error string, and selected equivalent wrapping for chained
  Textual calls in production and tests. It changed no control flow or values.
- Work completed: formatted the six files; added the explicit D025 Ruff format
  contract; removed Black from the Resistics development group, configuration,
  and pre-commit; added the pinned Ruff formatting hook after Ruff lint fixes;
  and replaced the README Black badge with Ruff.
- Files changed by this checkpoint: `resistics/calibrate.py`,
  `resistics/mask.py`, `resistics/time.py`, `resistics/transfunc.py`,
  `resistics/tui.py`, `tests/test_tui.py`, `pyproject.toml`, `uv.lock`,
  `.pre-commit-config.yaml`, `README.md`, and this implementation record.
- Decisions added or superseded: D025 records the explicit formatter contract
  and protects embedded docstring examples/directives from automatic rewriting.
- Verification commands and results:
  - `ruff format --check resistics tests scripts`: all 39 files formatted.
  - `ruff check resistics tests scripts`: all checks passed.
  - Both pinned `ruff-check` and `ruff-format` pre-commit hooks passed against
    all files.
  - The full suite passed: 376 tests in 22.54 seconds.
  - `uv lock` resolved 176 packages and removed Black 26.5.1 plus its
    transitive-only Click and pytokens packages; `uv lock --check` passed.
  - Black's executable is absent from the synced environment. Resistics source,
    configuration, README, and hooks contain no active Black reference.
  - The legacy-packaging guard and final `git diff --check` passed.
- Known failures or incomplete work: `uv.lock` still records the Black name in
  sibling regressioninc's development metadata, but Black is not a resolved
  Resistics package. Darglint and the remaining old pre-commit hygiene/Prettier
  hooks stay active until Checkpoints 2.3 and 2.5.
- Checkpoint state at end: `2.2` is `verified`; Checkpoint 2.3 is next.
- Commit readiness or commit id: the isolated formatter diff, Black removal,
  Ruff hook/configuration, and verification record are ready; no commit was
  requested or created.
- Exact next action: measure darglint's findings and runtime, install pydoclint
  in NumPy mode, and establish the reviewed Checkpoint 2.3 baseline without
  weakening new or touched public API checks.

### S012 - 2026-07-19 - Replace darglint with pydoclint

- Checkpoint state at start: `2.3` was `not_started`; Checkpoint 2.2 had been
  committed at Resistics `070c414` on `mth5`, and the worktree was clean.
- Starting HEAD and worktree: Resistics was clean at `070c414`; regressioninc
  remained at `9eb11a4` with its separately documented uncommitted hardening
  changes.
- Session objective: replace unmaintained darglint with a fast, maintained
  native docstring contract checker while preserving NumPy docstrings and
  preventing inherited documentation debt from weakening checks on new work.
- Baseline inventory: darglint 1.8.1 reported 19 findings in 15.264 seconds.
  Pydoclint 0.9.1 reported 233 findings in 0.411 seconds: 79 DOC105, 4 DOC107,
  3 DOC201, 44 DOC203, 7 DOC301, 1 DOC501, 3 DOC502, 8 DOC503, 32 DOC601,
  32 DOC603, and 20 DOC606 findings. Most newly visible debt is legacy type
  spelling/annotation drift and incomplete class-attribute documentation.
- Work completed: replaced the development dependency and pinned pre-commit
  hook; removed `.darglint`; added the explicit D026 NumPy/type-consistency
  policy to `pyproject.toml`; and generated the committed
  `pydoclint-baseline.txt`. Automatic baseline regeneration is disabled.
- Baseline validation: the package and a single baselined file both passed. A
  temporary unbaselined public API with mismatched arguments and no return
  contract failed with DOC103, DOC201, and DOC203, proving that new findings
  remain blocking. The temporary file was deleted after the check.
- Files changed by this checkpoint: `.darglint` deletion,
  `.pre-commit-config.yaml`, `pyproject.toml`, `uv.lock`,
  `pydoclint-baseline.txt`, and this implementation record.
- Decisions added or superseded: D026 records the native baseline policy,
  manual debt acceptance, type-consistency coverage, and the intentionally
  temporary short-docstring setting owned by Checkpoint 2.4.
- Verification commands and results:
  - `pydoclint resistics`: no unbaselined violations; runtime remains about
    0.4 seconds, roughly 37 times faster than the darglint baseline.
  - The pinned `pre-commit run pydoclint --all-files` hook passed.
  - `ruff format --check resistics tests scripts`: all 39 files formatted;
    `ruff check resistics tests scripts`: all checks passed.
  - The full suite passed: 376 tests in 22.53 seconds.
  - `uv lock --check` resolved 178 packages and passed. Darglint was removed;
    pydoclint 0.9.1 and its Click/docstring-parser-fork dependencies were added.
  - The legacy-packaging guard and final `git diff --check` passed.
- Known failures or incomplete work: the baseline intentionally records 233
  existing findings for later documentation work. Short docstrings are still
  exempt until Checkpoint 2.4. `uv.lock` may retain the darglint name only in
  sibling regressioninc's development metadata; it is not a resolved
  Resistics package. The old hygiene and Prettier hooks remain Checkpoint 2.5
  work.
- Checkpoint state at end: `2.3` is `verified`; Checkpoint 2.4 is next.
- Commit readiness or commit id: the pydoclint dependency, configuration,
  committed baseline, hook replacement, darglint removal, and verification
  record are ready; no commit was requested or created.
- Exact next action: inventory Ruff `D100`-`D107` findings and rerun pydoclint
  with `skip-checking-short-docstrings = false`, then define the measured public
  docstring presence and contract baseline for Checkpoint 2.4.

### S013 - 2026-07-19 - Enforce the production docstring contract

- Checkpoint state at start: `2.4` was `not_started`; Checkpoint 2.3 was
  verified but uncommitted on top of Resistics `070c414` on `mth5`.
- Starting HEAD and worktree: Resistics remained at `070c414` with only the
  scoped Checkpoint 2.3 migration dirty; regressioninc remained at `9eb11a4`
  with its separately documented uncommitted hardening changes.
- Session objective: make public production docstrings mandatory, prevent short
  placeholders from evading contract checks, preserve API-local documentation
  assets, and give contributors an explicit authoring policy.
- Baseline inventory: Ruff reported 190 D100-D107 findings: 1 D100, 15 D101,
  119 D102, 5 D103, 19 D105, and 31 D107. The required D100-D104 subset was
  140 findings. Decorator-specific Pydantic exclusions and the TUI D102
  framework boundary reduced this to 42 supported-API findings, all fixed.
  Enabling short-docstring checks exposed 1,472 findings beyond the earlier
  233-finding baseline. The final strict baseline contains 1,823 inherited
  findings after the high-value documentation improvements.
- Work completed: enabled Ruff D100-D104 for production code; documented every
  actionable public module, class, function, and method; corrected three class
  docstring placements; enabled strict pydoclint short-docstring checks; and
  regenerated the manual baseline. D105-D107 remain deliberately outside the
  contract under D027.
- Authoring and asset work: added the Sphinx docstring authoring page covering
  public/private requirements, NumPy sections, annotations, examples,
  constructors, directives, suppressions, and baseline policy. Added or
  improved examples for project, flow, parameter, and job construction. The
  production inventory now has at least 86 example-bearing docstrings and 14
  API-local plot directives. Two AST-based tests protect those assets, while
  normal doctest collection executes examples.
- Enforcement validation: a temporary undocumented public API failed Ruff with
  D100/D103. A temporary one-line documented callable failed pydoclint with
  DOC101, DOC103, DOC201, and DOC203. Both temporary files were removed.
- Files changed by this checkpoint: `pyproject.toml`, documentation additions
  and index, focused docstring additions across twelve production modules,
  `pydoclint-baseline.txt`, `tests/test_docstring_contract.py`, and this
  implementation record.
- Decisions added or superseded: D027 defines the enforced presence boundary,
  the narrow framework exceptions, constructor/dunder policy, and protection
  for examples, plots, and embedded directives.
- Verification commands and results:
  - `ruff format --check resistics tests scripts`: all 40 files formatted;
    `ruff check resistics tests scripts`: all checks passed, including public
    production docstring presence.
  - `pydoclint resistics` and the pinned pydoclint pre-commit hook passed with
    short-docstring checking enabled and automatic baseline updates disabled.
  - The final full suite passed: 384 tests in 20.82 seconds. This includes the
    two asset tests, four new construction-example doctests, and two existing
    examples recovered by correcting their class-docstring placement.
  - The gallery-disabled Sphinx HTML build succeeded at
    `.artifacts/hardening/documentation/checkpoint-2.4-html`. Its 99 warnings
    are the deferred legacy gallery, external inventory, missing legacy module,
    and dependency/API-import debt already owned by Phase 7.
  - `uv lock --check` resolved 178 packages and passed. The
    legacy-packaging guard and final `git diff --check` passed.
- Known failures or incomplete work: the 1,823 pydoclint findings are explicit
  inherited debt, not a greenfield standard. D105-D107 are intentionally not
  enabled. Textual callback documentation remains governed by framework need
  rather than public-library presence. The Sphinx warning backlog remains
  Phase 7 work.
- Checkpoint state at end: `2.4` is `verified`; Checkpoint 2.5 is next.
- Commit readiness or commit id: Checkpoints 2.3 and 2.4 are both verified and
  ready together; no commit was requested or created.
- Exact next action: inventory the remaining pre-commit hygiene and Prettier
  hooks, then replace isolated hook environments with locked uv-backed local
  commands for Checkpoint 2.5.
