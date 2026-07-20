# Resistics Code-Hardening Implementation Record

Status: in progress; Checkpoint 4.3 verified with cached explorer indexing
Created: 2026-07-19
Last updated: 2026-07-20
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
- Active phase: Phase 4 - Make the TUI responsive
- Active checkpoint: `4.4` (`resistics`) - Move blocking loads to workers
- Checkpoint state: `not_started`
- Last completed checkpoint: `4.3` in `S025`
- Last verified checkpoint: `4.3` in `S025`
- Last session: `S025`
- Last verified commit: resistics `148a877` plus the verified uncommitted
  Checkpoint 4.3 worktree; regressioninc `9eb11a4` is the base of uncommitted
  Checkpoints 1.1 and 1.2 work
- Current blocker: none
- Next exact action: inventory synchronous project-opening and explorer loads,
  define immutable worker results and generation-based stale-result rejection,
  then add a responsiveness test proving the screen mounts before discovery
  completes.

Current worktree caveat:

- Resistics `148a877` records the tracked hardening work through Checkpoint 4.2,
  including owned binding refreshes and handler timing coverage.
  The empty `pyrefly-baseline.json` remains untracked and is a required Phase 3
  artifact. The verified Checkpoint 4.3 explorer, TUI, and test edits are
  uncommitted.
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
- Pre-commit uses maintained v6 file-hygiene hooks plus locked uv-backed local
  Ruff and pydoclint hooks. The local hook is installed in this checkout.
- Pyrefly 1.1.1 is the sole mandatory type checker. Its locked project command
  checks all 21 production modules without error- or warning-level findings;
  the committed error baseline is empty, and new findings fail locally and in
  pre-commit. Two narrow demonstrated suppressions remain. The complete inline
  typing contract is advertised by `py.typed`, verified in wheel and sdist.
  mypy and its project cache/configuration are absent.
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
| 2.5 | resistics | `verified` | S014; clean sync/hooks; 384 tests |
| Gate 2 | resistics | `verified` | S010-S014; sole Ruff/pydoclint uv gate |
| 3.1 | resistics | `verified` | S015; Pyrefly 1.1.1 selected |
| 3.2 | resistics | `verified` | S016; locked Pyrefly gate, 175-entry baseline |
| 3.3 | resistics | `verified` | S017-S021; baseline 175 -> 0; PEP 561 artifacts verified |
| Gate 3 | resistics | `verified` | S022; Pyrefly retained, 0 diagnostics, PEP 561 artifacts verified |
| 4.1 | resistics | `verified` | S023; 5,200 checks; zero instrumented I/O |
| 4.2 | resistics | `verified` | S024; owned refreshes; handlers below 50 ms |
| 4.3 | resistics | `verified` | S025; identity cache and owned invalidation |
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
| Tests | 372 collected; 371 passed; 1 failed | 373 passed | 396 passed | S025 |
| Branch coverage | approximately 76% | 75.96% | 76.19% | S014; coverage XML |
| Production Python | approximately 21,011 lines | 21,015 | 21,015 | S002 report |
| Tests | approximately 5,941 lines | 6,051 | 6,051 | S002 report |
| `resistics/tui.py` | 2,673 lines | 2,673 | 2,673 | S002 report |
| `resistics/plot.py` | 1,200 lines | 1,200 | 1,200 | S002 report |
| Flake8 | approximately 40 findings | 43 findings | 43 findings | S001 |
| Legacy complexity | not recorded | 17 CCR001; 5 C901; 4 ECE001 | same | S002 |
| Black format | not recorded | 26 files differ | 26 files differ | S001 |
| mypy | 210 errors across 17 files | 209 across 17 files | removed | S016 |
| Pyrefly | not installed | 175 errors across 17 files | 0 new errors | S016 |
| TUI cold import | approximately 2.18 seconds | 2.2659 s median | same | S002 report |
| Cached TUI action checks | not measured | 5,200 in 0.002322 s; zero instrumented I/O | same | S023 XML |
| TUI binding refresh ownership | not measured | calls 200/3/3/1 | calls 100/1/1/0; 13.007 ms maximum | S024 XML |
| Explorer resource parsing | repeated by table, job, and selection | one parse per file identity | zero reads on cache hits | S025 tests |
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
- `D028` (2026-07-19): Keep only `pre-commit-hooks` v6.0.0 as an external hook
  repository for YAML, end-of-file, and trailing-whitespace hygiene. Run Ruff
  and pydoclint as `repo: local`, `language: system` hooks whose entries use
  `uv run --locked --no-sync`; this makes the lock the single Python-tool
  version authority and prevents commit-time environment mutation. Remove the
  deprecated Prettier mirror because no maintained non-Python asset requires a
  JavaScript formatter. Document the complete local command surface, but leave
  the mandatory type command explicitly empty until Phase 3 selects one useful
  mypy replacement. Include `uv audit --locked` in the local gate; immediately
  apply available focused transitive security upgrades rather than accepting a
  failing audit without policy.
- `D029` (2026-07-20): Select Pyrefly 1.1.1 for Checkpoint 3.2. On the common
  8,031-line evaluation slice it combined a bounded 71-error baseline, the
  clearest Pydantic v2 modelling, useful NumPy/SciPy inference, sub-second CLI
  checks, a 0.01-second watched recheck, and an explicit non-mutating baseline
  update workflow. Basedpyright 1.39.9 found useful additional Pandas and
  Textual contracts but its recommended defaults produced 1,304 baseline
  entries, rejected valid lax Pydantic constructors, and took about 3.7 seconds
  warm. ty 0.0.61 was fastest and accepted Pydantic coercion, but lost
  serialization and SciPy results to `Unknown` and exposed no project-baseline
  command. Retain neither rejected candidate as a project dependency. Reassess
  ty at the Phase 3 review gate or when it gains a baseline workflow and the
  missing third-party/Pydantic result precision.
- `D030` (2026-07-20): Pin Pyrefly exactly at 1.1.1 in the development group
  and make `uv run --locked --no-sync pyrefly check` the sole type command.
  Configure its default preset for the Python 3.11 minimum and all 21
  `resistics` production modules. Commit error-level debt in the root
  `pyrefly-baseline.json`; baseline regeneration requires an explicit
  `--update-baseline` command and is never part of the normal gate. Run the
  project command as an always-run, filename-independent local pre-commit hook
  so configuration and cross-module findings cannot escape a file-scoped
  invocation. Keep the 15 lower-severity warnings visible to editors but below
  the mandatory CLI threshold until Checkpoint 3.3 triages them. Remove mypy,
  its configuration, cache ignores, generated project cache, documentation,
  and lock-only dependencies instead of translating its old exceptions.

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

### S014 - 2026-07-19 - Simplify pre-commit and the local quality gate

- Checkpoint state at start: `2.5` was `not_started`; the owner had committed
  the tracked Checkpoint 2.3/2.4 changes as Resistics `312710b`, while the new
  pydoclint baseline and docstring authoring page/index remained uncommitted.
- Starting HEAD and worktree: Resistics `312710b` on `mth5` with only those
  three documented Checkpoint 2.4 paths dirty; regressioninc remained at
  `9eb11a4` with its separately documented uncommitted hardening changes.
- Session objective: remove obsolete and duplicated pre-commit environments,
  make the lock authoritative for Python quality tools, and document and prove
  a repeatable local production gate.
- Hook inventory and work completed: replaced `pre-commit-hooks` v4.4.0 with
  current v6.0.0; removed the deprecated Prettier mirror; replaced the isolated
  Ruff and pydoclint repositories with three local `language: system` hooks
  using `uv run --locked --no-sync`; and installed the repository pre-commit
  hook. The configuration now has one maintained external hygiene repository
  and no isolated Python quality-tool environments.
- Hygiene migration: the upgraded end-of-file hook removed one extra terminal
  blank line from `.gitignore`. Every hook passed across all tracked product
  files and the untracked Checkpoint 2.4 documentation/baseline files. The
  managed workspace exposes `.agents` as read-only, so the mutating
  end-of-file hook cannot open those files during `--all-files`; all non-mutating
  hooks passed them, and the product-tree command passed without an exclusion
  in repository configuration.
- Clean-environment proof: a fresh temporary Python 3.14.6 environment synced
  all groups from the 178-package lock, installed 168 packages including the
  editable sibling regressioninc, validated the configuration, and passed all
  six product hooks. Re-syncing that environment after the security refresh
  installed the current lock and the same hook set passed again without sync.
- Local-gate guidance: expanded the README with exact locked setup,
  installation, pre-commit, Ruff, pydoclint, test, coverage, Sphinx, build,
  dependency-audit, and legacy-packaging commands. The type slot is explicitly
  unassigned until Phase 3 selects one checker; mypy is not presented as a
  production signal during that transition.
- Security audit: the initial OSV-backed `uv audit --locked` found eight Pillow
  12.2.0 advisories and one setuptools 82.0.1 advisory, all with fixes. A
  focused lock refresh selected Pillow 12.3.0 and setuptools 83.0.0. The final
  audit found no known vulnerabilities or adverse statuses in 177 audited
  packages.
- Files changed by this checkpoint: `.pre-commit-config.yaml`, `.gitignore`,
  `README.md`, `uv.lock`, and this implementation record. The untracked
  pydoclint baseline and docstring authoring files remain required Checkpoint
  2.4 work and must be included at the next durable boundary.
- Decisions added or superseded: D028 defines the locked local-hook topology,
  removal of Prettier, temporary absence of a mandatory type command, and
  treatment of immediately fixable audit findings.
- Verification commands and results:
  - The installed `.git/hooks/pre-commit` points to the synced project
    environment and `.pre-commit-config.yaml`; configuration validation passed.
  - All six hooks passed the complete product tree from both the main synced
    environment and the fresh Python 3.14 environment.
  - Ruff format, Ruff lint, and pydoclint passed through those locked hooks.
  - The full suite passed: 384 tests in 21.42 seconds.
  - Branch coverage passed its configured threshold at 76.19%; HTML and XML
    reports were generated under `.artifacts/hardening/coverage/`.
  - The gallery-disabled Sphinx HTML build succeeded with the known 99-warning
    Phase 7 backlog at
    `.artifacts/hardening/documentation/checkpoint-2.5-html`.
  - `uv build --no-sources` produced the 189,714-byte source distribution and
    161,242-byte wheel. `uv lock --check`, the final zero-finding OSV audit,
    the legacy-packaging guard, and `git diff --check` passed.
- Known failures or incomplete work: a full `pre-commit run --all-files` cannot
  run the mutating end-of-file hook against the sandbox-owned read-only
  `.agents` paths in this managed session; this is not encoded as a repository
  exclusion. Type checking remains intentionally unassigned until Phase 3. The
  documentation warning backlog remains Phase 7 work.
- Checkpoint state at end: `2.5` and the Phase 2 review gate are `verified`;
  Checkpoint 3.1 is next.
- Commit readiness or commit id: the Checkpoint 2.4 remainder and Checkpoint
  2.5 hook/guidance/security changes are verified and ready; no commit was
  requested or created.
- Exact next action: install and benchmark Pyrefly, basedpyright, and ty one at
  a time against the same representative modules, then record the Checkpoint
  3.1 selection evidence without making any candidate mandatory prematurely.

### S015 - 2026-07-20 - Evaluate and select a modern type checker

- Checkpoint state at start: `3.1` was `not_started`; Resistics was at
  `312710b` with the documented verified Checkpoint 2.4 remainder and
  Checkpoint 2.5 worktree still present.
- Starting branch, HEAD, and worktree: `mth5` at `312710b`; the only dirty
  paths were the S014-documented implementation record, hook, guidance,
  hygiene, documentation, baseline, and lock paths. No candidate dependency or
  configuration was already installed in the project.
- Session objective: compare Pyrefly, basedpyright, and ty on the same bounded
  production slice, select one checker using diagnostic and adoption evidence,
  and leave installation and the repository-wide baseline to Checkpoint 3.2.
- Evaluation environment and method: Linux 6.18.33.1 WSL2 x86_64, CPython
  3.13.5 project environment, Python 3.11 target, and uv 0.11.25. Temporary
  isolated uv invocations installed Pyrefly 1.1.1, basedpyright 1.39.9 (Pyright
  1.1.411), and ty 0.0.61 one at a time. Every checker analysed the same
  `flow`, `job`, `project`, `gather`, `regression`, and `tui` files: 8,031
  physical source lines. The existing `.venv` supplied third-party packages.
- Runtime results from fresh checker processes after candidate download:
  Pyrefly took 0.501 seconds on the first run and 0.477, 0.383, and 0.382
  seconds subsequently; basedpyright took 3.924 seconds initially and 3.899,
  3.694, and 3.663 seconds subsequently; ty took 0.279 seconds initially and
  0.282, 0.282, and 0.259 seconds subsequently. These are local comparative
  timings, not automated performance thresholds.
- Incremental/editor evidence: all three candidates provide a language server
  and a watched check. Touching `flow.py` while each watcher was live produced
  a prompt recheck; Pyrefly reported 0.01 seconds internally. basedpyright and
  ty did not expose an internal watched-recheck duration in their terminal
  output, so no invented numeric comparison is recorded for them.
- Finding volume: Pyrefly reported 71 errors plus five lower-severity warnings
  (30 gather, 22 TUI, eight project, eight regression, and three flow). ty
  reported 69 diagnostics with nearly the same core set. basedpyright reported
  137 error-level findings; its recommended default set generated 1,304 total
  baseline entries, dominated by unknown/Any propagation, unused results,
  unannotated class attributes, and strictness warnings.
- Defect usefulness: manual triage confirmed at least ten distinct actionable
  contract families rather than ten duplicate line reports: nullable process
  output/name invariants; the obsolete `Site = None` gather boundary; nullable
  gather channel lists; possibly absent gathered metadata; incomplete MTH5
  inspection-mixin protocols; validator-established but statically optional
  transfer-function dimensions; the undeclared `Regressor.fit`/`coef`
  protocol; incompatible override parameter contracts; Textual app/callback
  optionality; and un-narrowed path/plot payloads. The dominant false-positive
  or suppression burden was repeated dynamic MTH5 group access, Textual's
  generic `App` boundary, broad Pandas indexing unions, and Pydantic invariants
  established only by validators.
- Pydantic and library probe: a runtime-verified Pydantic v2 model accepted the
  lax generated constructor `Reading(samples="3")`. Pyrefly and ty accepted
  it, while basedpyright rejected it. Pyrefly retained `dict[str, Any]` for
  `model_dump`, NumPy shape and complex dtype information, and useful partial
  SciPy array types. basedpyright also retained useful NumPy/SciPy types but
  warned about missing stubs for SciPy, ObsPy, MTH5, and Plotly. ty reduced the
  serialized Pydantic result and SciPy results to `Unknown` and lost the NumPy
  dtype. All three identified ObsPy, MTH5, Plotly, and Textual nominal types;
  Textual remained `App[Unknown]` in all three.
- Diagnostic and navigation review: all candidates emitted source ranges,
  stable rule names, and LSP-capable navigation. ty had the richest default
  source-frame presentation, basedpyright supplied detailed subtype chains but
  often very long output, and Pyrefly's full-text mode combined source frames
  with concise narrowing suggestions while its minimal/JSON/GitHub/JUnit modes
  support automation.
- Baseline evidence: Pyrefly generated a 71-entry JSON baseline only when
  explicitly passed `--update-baseline`; a subsequent check returned zero
  errors, and a deliberately added assignment regression was still rejected.
  basedpyright generated a 1,304-entry baseline and subsequently returned zero,
  but its documented default automatically rewrites removed baseline entries.
  ty 0.0.61 exposed suppression generation but no project-baseline command, so
  its non-clean result cannot satisfy the governing new-finding gate without a
  custom wrapper or broad rule suppression.
- Selection: D029 selects Pyrefly 1.1.1. It provides the best combined
  Pydantic v2 accuracy, useful scientific-library inference, bounded baseline,
  non-mutating gate ergonomics, diagnostic clarity, and sub-second runtime.
  basedpyright remains the evidence-backed fallback; ty remains a future
  reassessment candidate rather than a second installed checker.
- Repository changes: only this implementation record changed during S015.
  Temporary candidate environments, probe files, candidate baselines, and the
  basedpyright evaluation configuration were outside the tracked project; no
  checker was added to `pyproject.toml` or `uv.lock`.
- Verification: every candidate completed against all six modules with
  resolved project dependencies; Pydantic probe runtime succeeded; Pyrefly's
  baseline rejected a new finding; watcher rechecks completed; `git
  diff --check` passed; and the final worktree contained no unexpected path.
  Product tests were not rerun because the checkpoint changed no production,
  test, dependency, or mandatory-tooling file.
- Checkpoint state at end: `3.1` is `verified`; Checkpoint 3.2 is next.
- Commit readiness or commit id: the S015 selection record is verified and can
  share the next owner-selected documentation boundary; no commit was requested
  or created.
- Exact next action: add one pinned Pyrefly development dependency and its
  project configuration, generate and prove a repository-wide baseline, then
  remove mypy and make the locked Pyrefly command mandatory in Checkpoint 3.2.

### S016 - 2026-07-20 - Install Pyrefly and retire mypy

- Checkpoint state at start: `3.2` was `not_started`; Checkpoint 3.1 had
  selected Pyrefly 1.1.1, but no checker dependency, configuration, baseline,
  command, or hook was yet installed.
- Starting branch, HEAD, and worktree: `mth5` at `312710b`; the documented
  verified Checkpoint 2.4 remainder, Checkpoint 2.5 changes, and S015 selection
  record were the only dirty paths. They were preserved throughout.
- Session objective: install exactly one selected checker, establish a
  non-mutating new-finding gate over production source, remove mypy completely,
  and prove the locked command from direct, hook, dependency, and test paths.
- Checker installation and configuration: pinned `pyrefly==1.1.1` in the
  development dependency group. Added one `[tool.pyrefly]` configuration using
  the default preset, Python 3.11 target, `resistics` project include, concise
  CLI output, and the root `pyrefly-baseline.json`. Configuration discovery
  resolved the project interpreter, editable sibling, and all 21 production
  modules without hard-coded site-package paths.
- Baseline: generated 175 error-level entries across 17 production modules;
  the largest families are 51 bad argument types, 37 missing attributes, 18
  unsupported operations, 13 override parameter-name mismatches, and ten bad
  assignments. Fifteen lower-severity warnings remain below the CLI threshold
  and visible in editor diagnostics. The baseline contains only relative
  `resistics/` paths and no temporary probe entry.
- Regression-gate proof: the normal configured command reported zero new
  errors in about 0.70 seconds. A temporary new production module assigning a
  string to an annotated integer failed with one unsuppressed `bad-assignment`;
  removing that module restored zero. Baseline regeneration requires the
  explicit `--baseline pyrefly-baseline.json --update-baseline` command, so the
  normal checker and hook cannot accept new debt silently.
- Hook and guidance: added an always-run local Pyrefly hook using
  `uv run --locked --no-sync pyrefly check`, with filenames disabled so it
  checks the configured project once rather than producing file-scoped results.
  Added the same command to the README production gate and documented the
  deliberate baseline-update policy.
- Mypy removal: deleted the test-group mypy dependency, `mypy.ini`, all project
  mypy cache ignores and guidance, and the generated `.mypy_cache`. The lock
  removed mypy 2.1.0 plus its four lock-only support packages `ast-serialize`,
  `librt`, `mypy-extensions`, and `pathspec`. Active project files and the lock
  contain no mypy reference; similarly named modules remaining inside `.venv`
  belong to unrelated installed dependencies and are not a Resistics checker.
- Lock and environment: the refreshed lock resolves 174 packages and the synced
  CPython 3.13.5 environment installed Pyrefly while uninstalling mypy and its
  support packages. The OSV-backed audit found no known vulnerabilities or
  adverse statuses across 173 audited packages.
- Files changed by this checkpoint: `.gitignore`, `.pre-commit-config.yaml`,
  `README.md`, `pyproject.toml`, `uv.lock`, deletion of `mypy.ini`, new
  `pyrefly-baseline.json`, and this implementation record. Some tracked paths
  already contain the separately documented S014 changes; those were not
  reverted or attributed to S016.
- Decisions added or superseded: D030 defines the pinned version, production
  scope, minimum Python target, explicit baseline policy, mandatory hook shape,
  warning threshold, and complete mypy retirement.
- `D031` (2026-07-20): For the first Checkpoint 3.3 contract group, make the
  MTH5 inspection mixin's required operations explicit and model mask artifact
  names as fixed class identities rather than optional user parameters. Retain
  only two demonstrated, rule-specific Pyrefly suppressions in this group: the
  intentional Pydantic instance-field-to-`ClassVar` override protected by a
  runtime model-field test, and NumPy's incorrect `savez_compressed(**arrays)`
  stub interpretation. Do not replace either contract with a broad ignore or
  a cast. Keep third-party missing-stub and intentional Pandas scalar
  normalisation diagnostics warning-only for later Checkpoint 3.3 triage.
- `D032` (2026-07-20): At Textual boundaries, represent plot requests as a
  discriminated payload union, express the minimal tree-node service as a
  protocol, narrow screen ownership to `ResisticsTui` with a runtime-checked
  helper, and centralise nullable focus traversal over generic widgets. These
  contracts replace all TUI baseline entries without casts, checker ignores,
  or coupling application code to Textual's private tree-node module. Reject
  malformed plot payloads before data access and preserve the existing
  launcher/navigation behavior through real Textual integration tests.
- `D033` (2026-07-20): Remove the obsolete directory-based gather API instead
  of formalising compatibility protocols for code that cannot operate on the
  MTH5-only `Project`. Delete `get_site_evals_metadata`,
  `get_site_level_wins`, `get_site_wins`, `Selection`, `Selector`,
  `ProjectGather`, and the `Measurement = None`/`Site = None` placeholders.
  Preserve `GatherCriteria`, `GatherSelection`, `EvaluationFrequencyGather`,
  `Gather`, `QuickGather`, and the shared combined-data containers. Model the
  spectra regression preparer as a sibling with its own flow inputs rather
  than a subtype of the gathered-data preparer, and express solver plugins by
  the minimal fit/coefficient protocol they must satisfy. Do not add a legacy
  import shim or checker suppression for removed APIs.
- `D034` (2026-07-20): Pull the time-ingestion portion of the planned MTH5-only
  cleanup forward into Checkpoint 3.3. Remove the NumPy/ASCII directory readers
  and writers, their tests, examples, and standalone notebooks rather than
  hardening obsolete persistence contracts. Make MTH5 `RunTS.dataset` the sole
  ingestion boundary and store `TimeData` internally as an xarray array with
  explicit `channel` and `time` dimensions, while retaining a mutable NumPy
  view for the established SciPy numerical processors. Preserve serialisable
  survey, station, run, and channel MTH5 metadata. Keep compressed persistence
  for derived decimated, windowed, spectra, and mask artifacts behind one
  shared helper and one demonstrated NumPy-stub suppression. Declare xarray as
  a direct dependency because production code now imports it directly.
- `D035` (2026-07-20): Advertise complete inline PEP 561 typing for Resistics
  after all 21 production modules reached zero error- and warning-level
  diagnostics under pinned Pyrefly 1.1.1 and the error baseline became empty.
  Add the maintained PyYAML and tqdm stub packages to the locked development
  environment, document the support level, and include an empty `py.typed`
  marker. Keep the two demonstrated rule-specific suppressions from D031;
  neither weakens a public contract. Verify the marker by inspecting both the
  wheel and source distribution before treating Checkpoint 3.3 as complete.
- `D036` (2026-07-20): Retain Pyrefly 1.1.1 as the sole mandatory checker at
  the Phase 3 review gate. A refreshed ephemeral reassessment confirmed that
  ty remains at 0.0.61: it checked the production directory in about 0.17
  seconds but reported 29 diagnostics, primarily from lost Pandas `NaT` and
  numeric union narrowing, and still exposes inline-ignore generation rather
  than a project-baseline workflow. Pyrefly checks the same 21 production
  modules with zero warning-level diagnostics in 0.52-0.63 seconds and retains
  the explicit empty-baseline gate. Do not add ty as a dependency or second
  permanent checker; reassess it only after a material capability release.
- `D037` (2026-07-20): Make `ProjectExplorerScreen.check_action` a pure
  in-memory predicate. Cache project and selected-data plot targets, validated
  flow paths, project-data deletion eligibility, and existing job validation
  summaries while their owning catalogue or selection transitions already do
  the required reads. Refresh those values on mount, explicit resource
  refreshes, data selection, deletion, and terminal job events. Action
  invocation may perform its owned load, but Footer eligibility evaluation may
  not read the filesystem, MTH5, YAML, JSON, solutions, flows, parameters, or
  jobs.
- `D038` (2026-07-20): Refresh Footer bindings at the outer state transition
  after all related model and widget mutations have completed. Catalogue
  population helpers do not publish intermediate binding state. Full project,
  terminal-job, YAML-resource, create, delete, and restore rebuilds suspend
  Textual repaints through `App.batch_update`; their owner publishes one final
  binding refresh. Tab activation publishes once, data metadata rendering
  publishes none, and table-cursor refreshes are limited to the focused table
  on its active resource tab.
- `D039` (2026-07-20): Put explorer discovery behind the UI-neutral
  `ProjectExplorerIndex`. Cache only handle-free project DTOs, run summaries,
  parsed Pydantic resources, validation errors, job summaries, and resolved job
  validations; never retain an HDF5 object. Identify YAML cache entries by
  project-local path, nanosecond modification time, and byte size. A cache hit
  performs no project or filesystem read. Resource changes invalidate their
  namespace and dependent jobs; data deletion and processing invalidate
  project state; explicit external refresh invalidates all sections. Unchanged
  file identities may reuse parsed models across invalidation.
- Verification commands and results:
  - `uv lock --check` passed with 174 resolved packages, and locked all-group
    sync completed successfully.
  - `pyrefly dump-config` found the correct project interpreter, import roots,
    editable sibling, and 21 covered production files.
  - Direct locked Pyrefly checks and the isolated-cache pre-commit Pyrefly hook
    passed with zero new errors; the deliberate regression probe failed as
    required.
  - Ruff formatting checked 40 files, Ruff lint passed, and pydoclint reported
    no violations.
  - The full suite passed: 384 tests in 21.80 seconds.
  - The OSV audit, baseline schema/path assertions, active mypy-reference scan,
    project-cache absence check, and `git diff --check` passed.
- Environment-specific note: pre-commit's default cache is read-only in the
  managed workspace. Re-running it with `PRE_COMMIT_HOME` set to a temporary
  writable cache initialized the maintained hook repository and passed the new
  Pyrefly hook; no repository workaround or exclusion was added.
- Checkpoint state at end: `3.2` is `verified`; Checkpoint 3.3 is next. The
  Phase 3 review gate remains open until baseline reduction is complete.
- Commit readiness or commit id: the Pyrefly dependency, configuration,
  baseline, hook, guidance, mypy removal, lock refresh, and S016 record are
  verified and ready for an owner-selected commit; no commit was requested or
  created.
- Exact next action: inventory the baseline entries for `flow`, `job`,
  `project`, and `mask`, then add focused regression tests before correcting
  the first Checkpoint 3.3 contract group.

### S017 - 2026-07-20 - Harden flow, project, and mask contracts

- Checkpoint state at start: `3.3` was `in_progress`; Checkpoint 3.2 had
  established the mandatory Pyrefly gate with 175 baseline errors across 17
  production modules.
- Starting branch, HEAD, and worktree: `mth5` at `312710b`; the documented
  verified Checkpoint 2.4 remainder and Checkpoints 2.5/3.2 worktree were
  preserved. No unrelated dirty path was modified or reverted.
- Session objective: complete the first Checkpoint 3.3 module group by
  inventorying `flow`, `job`, `project`, and `mask`, correcting their core
  contracts, adding focused runtime protection, and shrinking both applicable
  baselines without weakening either mandatory gate.
- Inventory: the group contained 17 Pyrefly baseline errors: three in `flow`,
  zero in `job`, eight in `project`, and six in `mask`. They represented a
  nullable process output invariant, nullable mask identity, an undeclared
  mask calculation method, an incomplete MTH5 inspection-mixin contract,
  writer override shape, two intentional fixed-name Pydantic overrides, and a
  NumPy keyword-stub mismatch.
- Contract corrections: `process_descriptor` now narrows the concrete
  process output invariant before constructing its non-optional DTO. The MTH5
  inspection mixin declares the sampling-frequency, survey, station, run, and
  table-filter operations used by its shared behavior, and `MTH5File` now
  matches that typed filter signature. `WindowMaskWriter` preserves the base
  writer signature and rejects non-mask `ResisticsData` explicitly.
  `WindowMaskProcess` declares a fixed class-level artifact name and its
  calculation signature, removing nullable-name handling from flow validation
  and mask persistence while preserving the existing Pydantic API.
- Suppression evidence: the only two source suppressions are rule-specific.
  One covers Pyrefly's rejection of the intentional Pydantic field-to-class
  identity override; tests prove mask names cannot become model fields or user
  parameters. The other covers NumPy's stub treating string-keyed compressed
  arrays as the unrelated `allow_pickle` parameter; the existing round-trip
  test exercises the real supported call.
- Baselines: explicit Pyrefly regeneration reduced the error baseline from 175
  to 158 entries and left no entry for any module in the first group. Updating
  full NumPy-style docs for the edited public contracts and explicitly
  regenerating the line-sensitive pydoclint baseline removed ten stale
  findings, from 1,823 to 1,813. Normal gate commands remain non-mutating.
- Warning triage: Pyrefly reports 16 below-threshold diagnostics. In this group,
  missing PyYAML stubs and deliberate integer normalisation at Pandas/MTH5 DTO
  boundaries remain warnings rather than being hidden or converted into type
  debt; dependency-level warning policy remains open for the final module
  group.
- Tests: extended the fixed mask-name regression to cover the abstract process
  model and added a writer-contract test proving other `ResisticsData` values
  fail before filesystem mutation. The focused `flow`, `job`, `project`, and
  `mask` suite passed 47 tests; the full suite passed 385 tests in 21.23
  seconds.
- Files changed by this contract group: `resistics/flow.py`,
  `resistics/project.py`, `resistics/mask.py`, `tests/test_mask.py`,
  `pyrefly-baseline.json`, `pydoclint-baseline.txt`, and this implementation
  record. Previously documented dirty paths were preserved.
- Decisions added or superseded: D031 records the mixin and fixed mask identity
  contracts, the evidence for both narrow suppressions, and the deferred
  warning-level dependency/scalar-normalisation review.
- Verification commands and results:
  - `uv run --locked --no-sync ruff format --check resistics tests scripts`
    checked 40 files; Ruff lint passed on the same supported paths.
  - `uv run --locked --no-sync pydoclint resistics` reported no new
    violations; `uv run --locked --no-sync pyrefly check` reported zero new
    errors with two demonstrated suppressions.
  - The focused suite passed 47 tests in 2.81 seconds and the full suite passed
    385 tests in 21.23 seconds.
  - The regenerated Pyrefly baseline contains 158 errors and no `flow`, `job`,
    `project`, or `mask` path. `git diff --check` passed.
- Checkpoint state at end: the first logical group of Checkpoint 3.3 is
  `verified`; Checkpoint 3.3 remains `in_progress` until the remaining four
  groups and the PEP 561 artifact check are complete.
- Commit readiness or commit id: the first contract group and both reduced
  baselines are verified and ready for an owner-selected commit; no commit was
  requested or created.
- Exact next action: inventory the Pyrefly findings at TUI state, DTO, and
  service boundaries, add focused tests for the affected behavior, and correct
  the second Checkpoint 3.3 module group.

### S018 - 2026-07-20 - Harden TUI state, DTO, and service boundaries

- Checkpoint state at start: `3.3` was `in_progress`; S017 had verified the
  first logical module group and reduced the Pyrefly baseline from 175 to 158.
- Starting branch, HEAD, and worktree: `mth5` at `312710b`; all documented
  Checkpoint 2.4, 2.5, 3.2, and S017 worktree changes were preserved. No
  unrelated path was modified or reverted.
- Session objective: complete the second Checkpoint 3.3 group by inventorying
  TUI state, DTO, and service-boundary findings, protecting affected runtime
  behavior, and removing the complete `resistics/tui.py` error baseline.
- Inventory: the TUI contained 22 baseline errors: 11 bad argument types,
  eight missing attributes, one overload mismatch, one unsupported operation,
  and one non-iterable payload. The concrete families were nine nullable-focus
  list operations, six custom application-method calls through Textual's
  generic `App`, one string-typed directory path, two untyped tree-node-map
  operations, three broad plot payload operations, and an imprecise spectra
  reader result.
- Boundary corrections: introduced a discriminated `PlotTarget` union for
  flow, job, project, time, spectra, and transfer-function requests. Plot
  construction now validates external payload shapes before reading project
  data and narrows the metadata-or-data spectra result before plotting.
  Introduced a minimal structural tree-node protocol instead of importing a
  private Textual implementation type, and typed the ancestor-node map with
  its real `str | None` keys. Directory navigation now normalises Textual's
  path value explicitly to `Path`.
- State and application contracts: centralised relative focus traversal over
  `Sequence[Widget]`, including safe handling for no focus, unrelated focus,
  and an empty enabled-control set while preserving each dialog's previous
  fallback policy. A runtime-checked application helper now proves project
  screens are owned by `ResisticsTui` before calling launcher-specific methods;
  generic Textual methods remain accessed through the normal `App` contract.
- Regression coverage: added three malformed plot-payload cases covering time,
  spectra, and transfer-function requests. Existing real Textual tests continue
  to exercise dialog focus, launcher navigation, project creation/open/close,
  typed plot construction, data-tree population, YAML state, and worker
  progress. The focused TUI suite increased from 25 to 28 passing tests.
- Baselines: explicit Pyrefly regeneration removed all 22 TUI entries and
  reduced the repository error baseline from 158 to 136, with no new source
  suppression. Explicit pydoclint regeneration after fully documenting the
  edited contracts removed ten stale findings, from 1,813 to 1,803. The 16
  below-threshold Pyrefly warnings are unchanged and none is in `tui.py`.
- Files changed by this contract group: `resistics/tui.py`,
  `tests/test_tui.py`, `pyrefly-baseline.json`, `pydoclint-baseline.txt`, and
  this implementation record. Previously documented dirty paths were
  preserved.
- Decisions added or superseded: D032 records the discriminated plot DTO,
  tree-node protocol, runtime application narrowing, and shared focus-state
  contract. No cast, private Textual import, or checker suppression was added.
- Verification commands and results:
  - `uv run --locked --no-sync ruff format --check resistics tests scripts`
    checked 40 files, and Ruff lint passed on the same supported paths.
  - `uv run --locked --no-sync pydoclint resistics` reported no new
    violations; `uv run --locked --no-sync pyrefly check` reported zero new
    errors with only the two S017-demonstrated suppressions repository-wide.
  - The focused TUI suite passed 28 tests in 22.77 seconds. The full suite
    passed 388 tests in 22.09 seconds.
  - The regenerated Pyrefly baseline contains 136 errors and no TUI path; the
    pydoclint baseline contains 1,803 entries; `git diff --check` passed.
- Checkpoint state at end: the first two logical groups of Checkpoint 3.3 are
  `verified`; Checkpoint 3.3 remains `in_progress` until the final three groups
  and PEP 561 artifact verification are complete.
- Commit readiness or commit id: both verified contract groups and their
  reduced baselines are ready for an owner-selected commit; no commit was
  requested or created.
- Exact next action: inventory the `gather` and `regression` Pyrefly findings,
  add focused tests for their adapter boundaries, and correct the third
  Checkpoint 3.3 logical group.

### S019 - 2026-07-20 - Remove legacy gathering and harden regression adapters

- Checkpoint state at start: `3.3` was `in_progress`; S017-S018 had verified
  the first two logical groups and reduced the Pyrefly baseline from 175 to
  136.
- Starting branch, HEAD, and worktree: `mth5` at `312710b`; all previously
  documented hardening changes were preserved. The owner explicitly preferred
  removing legacy APIs and building for the MTH5-only future over retaining
  compatibility structures.
- Session objective: complete the third Checkpoint 3.3 group by resolving 30
  `gather` and eight `regression` errors, deleting obsolete adapters when they
  no longer represented a supported project model, and protecting the
  replacement contracts with focused tests.
- Inventory: the 38 errors comprised 19 missing attributes, 11 bad argument
  types, three bad indices, three override parameter-name mismatches, one bad
  return, and one overload mismatch. The dominant source was the
  `Measurement = None`/`Site = None` compatibility boundary and directory
  gather code expecting the pre-MTH5 project hierarchy. Remaining families
  were validator-established channels and dimensions, an invalid preparer
  inheritance relationship, an underspecified regressioninc base type, and a
  flow executor parameter-name mismatch.
- Legacy removal: deleted `get_site_evals_metadata`, `get_site_level_wins`,
  `get_site_wins`, `Selection`, `Selector`, and `ProjectGather`, plus the two
  `None` aliases in `project.py`. Retained the supported MTH5
  `GatherCriteria`/`GatherSelection`/`EvaluationFrequencyGather`/`Gather`
  pipeline, standalone `QuickGather`, and the shared `SiteCombinedMetadata`,
  `SiteCombinedData`, `GatheredData`, and evaluation-locator containers. No
  deprecated import shim or placeholder protocol was left behind.
- Supported gather contracts: both MTH5 and quick gathering now resolve the
  validated cross-channel invariant once and pass a concrete `list[str]`
  through extraction and metadata construction. Updated the module narrative
  and `QuickGather` cross-reference to describe only supported paths. Replaced
  the removed legacy example with a current, executable `GatherCriteria`
  remote-reference example so the API-local example inventory remains at or
  above its protected threshold.
- Regression contracts: `RegressionPreparerSpectra` is now a sibling
  `ResisticsProcess` with explicit spectra flow inputs, rather than an invalid
  subtype of `RegressionPreparerGathered`. Shared observation/predictor
  transformations have one implementation. Transfer-function dimensions are
  narrowed before NumPy allocation. The solver boundary is a minimal protocol
  requiring `fit` and optional coefficients; a missing fitted coefficient now
  raises a clear `ValueError`. `SolutionWriter.execute` now matches the parent
  context parameter contract.
- Regression coverage: added an explicit removed-API test for all six gather
  symbols and both project placeholders, a spectra-preparer flow-contract test,
  and a solver-adapter failure test for regressors that produce no
  coefficients. Existing numerical tests continue to cover quick/MTH5 gather,
  gathered and spectra preparation, random transfer-function shapes, and OLS
  solutions.
- Baselines: explicit Pyrefly regeneration removed all 38 group entries and
  reduced the repository baseline from 136 to 98, with no new source
  suppression. Explicit pydoclint regeneration removed 24 stale findings,
  from 1,803 to 1,779. The 16 lower-severity warnings are unchanged; the
  untyped `tqdm` import remains visible for final dependency-warning triage.
- Files changed by this group: `resistics/gather.py`,
  `resistics/project.py`, `resistics/regression.py`, `tests/test_gather.py`,
  `tests/test_regression.py`, `pyrefly-baseline.json`,
  `pydoclint-baseline.txt`, and this implementation record. Previously
  documented dirty paths were preserved.
- Decisions added or superseded: D033 records the owner-authorised legacy API
  deletion and the supported gather/regression adapter contracts.
- Verification commands and results:
  - The focused gather/regression suite passed 36 tests in 3.54 seconds; the
    documentation inventory and gather doctests passed three tests in 1.72
    seconds.
  - `uv run --locked --no-sync ruff format --check resistics tests scripts`
    checked 40 files, and Ruff lint passed on the same supported paths.
  - `uv run --locked --no-sync pydoclint resistics` reported no new
    violations; `uv run --locked --no-sync pyrefly check` reported zero new
    errors with only the two S017-demonstrated suppressions repository-wide.
  - The full suite passed 391 tests in 21.55 seconds. The regenerated Pyrefly
    baseline contains 98 errors and no `gather` or `regression` path; the
    pydoclint baseline contains 1,779 entries; `git diff --check` passed.
- Checkpoint state at end: the first three logical groups of Checkpoint 3.3 are
  `verified`; Checkpoint 3.3 remains `in_progress` until the final two groups
  and PEP 561 artifact verification are complete.
- Commit readiness or commit id: the three verified contract groups, legacy
  deletion, and reduced baselines are ready for an owner-selected commit; no
  commit was requested or created.
- Exact next action: inventory the `time`, `decimate`, `window`, and `spectra`
  findings, add focused numerical and persistence-contract tests, and correct
  the fourth Checkpoint 3.3 logical group.

### S020 - 2026-07-20 - Adopt MTH5-only labelled time data and harden the numerical pipeline

- Checkpoint state at start: `3.3` was `in_progress`; S017-S019 had verified
  the first three logical groups and reduced the Pyrefly baseline from 175 to
  98.
- Starting branch, HEAD, and worktree: `mth5` at `312710b`; all previously
  documented hardening changes were preserved. During inventory the owner
  clarified that NumPy/ASCII time readers no longer require support and asked
  for `TimeData` to align more closely with MTH5 data.
- Session objective: complete the fourth Checkpoint 3.3 group, remove obsolete
  time ingestion, make the MTH5 boundary explicit and labelled, harden shared
  metadata and persistence invariants, and protect the entire numerical path
  with focused tests.
- Inventory: the group started with 59 baseline findings: 32 in `time`, 11 in
  `spectra`, ten in `decimate`, and six in `window`. They comprised 19 bad
  argument types, nine override parameter-name mismatches, eight unsupported
  operations, six mutable-attribute overrides, four overload mismatches, three
  invalid annotations, three bad assignments, three missing arguments, two
  non-iterable values, one bad return, and one bad specialization. Shared
  causes were validator-established channel counts and process names still
  annotated as nullable, derived decimation lists annotated as optional,
  inconsistent writer overrides, repeated NumPy archive calls, and legacy time
  readers that no longer represented the MTH5-only product.
- MTH5-only time boundary: removed `TimeReader`, `TimeReaderJSON`,
  `TimeReaderAscii`, `TimeReaderNumpy`, `TimeWriterAscii`, and
  `TimeWriterNumpy`. Deleted their three read examples, the obsolete project
  configuration example, and two standalone notebooks that imported the
  removed readers. No compatibility shim remains, and a repository-wide search
  finds no reference to the removed symbols.
- Labelled `TimeData`: production code now declares xarray directly and stores
  samples as a `channel`/`time` labelled `DataArray`. `data` remains the mutable
  NumPy view used by existing SciPy processors; `dataset` and `to_xarray`
  expose MTH5-compatible labelled forms. Construction validates unique channel
  names, channel presence, dimensions, and metadata shape. The MTH5 adapter
  consumes a real `RunTS.dataset`, keeps its time coordinate, and preserves
  serialisable survey, station, run, data-logger, location, channel, sensor,
  and source-attribute metadata.
- Numerical and persistence contracts: channel counts and process names are
  concrete after validation; constrained integers use `Annotated`/`Field`;
  derived decimation fields are concrete lists; spectral grouping and
  single-channel colour bars are total; Fourier detrending uses SciPy's literal
  contract; process errors always identify their process. Decimated, windowed,
  spectra, and mask archives now use one compressed-array helper. All derived
  writers match the base writer signature, narrow their runtime data type, and
  produce an explicit `TypeError` for the wrong artifact.
- Regression coverage: replaced the directory-reader tests with a real MTH5
  `RunTS` ingestion test proving labels, source identifiers, station location,
  and round-tripped channel variables. Added an end-to-end test from labelled
  time data through decimation, windowing, Fourier transformation, and all
  three derived persistence round trips, plus wrong-artifact tests for every
  derived writer. Focused time/decimation/numerical tests passed 62 tests.
- Baselines: explicit Pyrefly regeneration removed all group findings plus
  shared contract findings, reducing the repository error baseline from 98 to
  32 with no additional suppression; the two demonstrated S017 suppressions
  remain repository-wide. The remaining findings are 13 in `testing`, 12 in
  `transfunc`, three in `calibrate`, and one each in `templates`, `sampling`,
  `plot`, and `common`. Converting the two touched metadata models to proper
  class-level attribute documentation reduced the pydoclint baseline from
  1,779 to 1,759.
- Files changed by this group: `pyproject.toml`, `uv.lock`, `resistics/common.py`,
  `resistics/time.py`, `resistics/decimate.py`, `resistics/window.py`,
  `resistics/spectra.py`, the shared mask writer call, the gather criteria name
  narrowing, `tests/test_time.py`, new `tests/test_numerical_pipeline.py`, both
  baselines, deletion of four legacy examples and two standalone notebooks,
  and this implementation record. Previously documented dirty paths were
  preserved.
- Decisions added or superseded: D034 records the owner-authorised MTH5-only
  time-I/O deletion, xarray-labelled internal representation, direct dependency,
  metadata-preservation boundary, and shared derived-persistence helper.
- Verification commands and results:
  - Focused time/decimation/numerical tests passed 62 tests in 2.95 seconds;
    the real MTH5 `RunTS` smoke test passed independently.
  - `uv run --locked --no-sync ruff check resistics tests scripts` passed and
    Ruff format checked 41 files. Pydoclint reported no new violations;
    Pyrefly reported zero new errors with two demonstrated suppressions and 14
    below-threshold warnings.
  - The full suite passed 381 tests in 20.75 seconds. The reduced total reflects
    removal of obsolete directory-reader tests, partially offset by five new
    MTH5/numerical contract tests.
  - `PRE_COMMIT_HOME=/tmp/resistics-pre-commit uv run --locked --no-sync
    pre-commit run --all-files` passed all seven hooks after the pinned hook
    environment was fetched outside the network-restricted sandbox. `git diff
    --check` passed.
- Checkpoint state at end: the first four logical groups of Checkpoint 3.3 are
  `verified`; Checkpoint 3.3 remains `in_progress` for the final 32 findings,
  public typing-support decision, and built-artifact marker verification.
- Commit readiness or commit id: the four verified contract groups, MTH5-only
  time deletion, labelled data boundary, tests, and reduced baselines are ready
  for an owner-selected commit; no commit was requested or created.
- Exact next action: inventory and correct the final 32 findings across
  `testing`, `transfunc`, `calibrate`, `templates`, `sampling`, `plot`, and
  `common`, then verify the public typing level and built-artifact marker
  decision to complete Checkpoint 3.3.

### S021 - 2026-07-20 - Complete core contract hardening and advertise inline typing

- Checkpoint state at start: `3.3` was `in_progress`; S017-S020 had verified
  the first four groups and reduced the Pyrefly baseline from 175 to 32.
- Starting branch, HEAD, and worktree: `mth5` at `312710b`; all documented
  hardening and owner changes were preserved.
- Session objective: resolve the final 32 findings in the remaining public
  modules, decide the supported public typing level, verify PEP 561 artifacts,
  and complete Checkpoint 3.3.
- Contract changes: transfer-function fields are concrete after validation;
  serialized dimensions must match channel lists; registered transfer-function
  dictionaries restore models and unknown names fail instead of escaping as
  untyped dictionaries. Gather and regression now rely on these invariants.
  Calibration interpolation declares its actual complex-array result, process
  execution validates its dynamic run adapter, timeline and high-resolution
  datetime boundaries return concrete types, template dictionaries are
  explicitly widened, and testing helpers no longer change annotated types.
- Coverage: added five transfer-function tests for derived invariants,
  registered dispatch, unknown types, inconsistent dimensions, and constrained
  variations, plus a complex calibration interpolation test. Affected focused
  tests passed 44 tests after the final invariant cleanup.
- Typing support: removed all 32 remaining baseline entries and explicitly
  regenerated `pyrefly-baseline.json` to an empty error list. Removed ten
  redundant conversion warnings and added locked `types-PyYAML` and
  `types-tqdm` development stubs, leaving zero diagnostics at warning severity.
  The two demonstrated D031 suppressions remain the only suppressions. D035
  records complete inline PEP 561 support; README documents the level and an
  empty `resistics/py.typed` marker is included.
- Artifact evidence: `uv build --no-sources` produced the 1.0.0a3 wheel and
  sdist under `/tmp/resistics-checkpoint-3-3-dist-20260720`. Direct ZIP and tar
  inspection found `resistics/py.typed` in both; wheel metadata reports
  `Requires-Python: <3.15,>=3.11`.
- Documentation baseline: documented all strengthened public attributes and
  exceptions instead of baselining them. Explicit regeneration reduced the
  pydoclint baseline from 1,759 to 1,723 and the normal command reports no new
  violations.
- Verification commands and results:
  - The full suite passed 387 tests in 25.31 seconds. The post-cleanup focused
    calibration, transfer-function, gather, and regression suite passed 44.
  - Ruff format checked 42 files and Ruff lint passed. Pyrefly reported zero
    diagnostics at warning severity; `git diff --check` passed.
  - All seven pre-commit hooks passed using the existing isolated hook cache;
    the file-hygiene pass required the established out-of-sandbox invocation
    because tracked `.agents` guidance is read-only inside the sandbox.
- Checkpoint state at end: all five groups of Checkpoint 3.3 are `verified`;
  the Pyrefly baseline is empty and both PEP 561 artifacts are verified. The
  Phase 3 review gate is next.
- Commit readiness or commit id: the complete Checkpoint 3.3 hardening stack is
  ready for an owner-selected commit; no commit was requested or created.
- Exact next action: run the Phase 3 review gate, including the planned ty
  reassessment, and verify the sole-checker, empty-baseline, performance,
  suppression, public typing, and artifact criteria.

### S022 - 2026-07-20 - Verify the Phase 3 type-checking review gate

- Gate state at start: Checkpoints 3.1-3.3 were `verified`; the Phase 3 review
  gate was `not_started` pending the planned ty reassessment and consolidated
  checker, performance, suppression, and artifact evidence.
- Starting branch, HEAD, and worktree: `mth5` at owner commit `419c495`. That
  commit contains the tracked hardening stack through S021. The pydoclint and
  Pyrefly baselines and `resistics/py.typed` remain untracked; they were
  preserved and are explicitly required in the next owner-selected commit.
- Tooling audit: project configuration, pre-commit, the lock, and the synced
  development tree contain exactly one checker, Pyrefly 1.1.1. `mypy.ini` is
  deleted and active tooling contains no mypy, BasedPyright, Pyright, or ty
  dependency/configuration. The Pyrefly error baseline is an empty list. The
  only source suppressions remain the two rule-specific D031 boundaries.
- Performance: three sequential locked Pyrefly checks covered all 21
  production modules and their 899 dependencies with zero warning-level
  diagnostics. Wall times were 0.63, 0.52, and 0.56 seconds; reported analysis
  times were 0.64, 0.57, and 0.58 seconds with about 503-505 MiB physical
  memory. This remains short enough for the always-run local hook.
- ty reassessment: refreshed PyPI metadata and ran current ty 0.0.61
  ephemerally against `resistics`, Python 3.11, and the project environment.
  It completed in about 0.17 seconds but reported 29 diagnostics, largely
  Pandas `NaT` and numeric union-narrowing losses, plus the two already
  demonstrated NumPy/Pydantic boundary cases. Its CLI still has no
  project-baseline workflow. D036 retains Pyrefly and leaves ty uninstalled.
- Public typing and artifact evidence: a fresh isolated `uv build --no-sources`
  produced the 1.0.0a3 wheel and sdist under
  `/tmp/resistics-phase3-gate-20260720`. ZIP and tar inspection found
  `resistics/py.typed` in both. Wheel metadata reports Python `>=3.11,<3.15`.
- Verification commands and results:
  - `uv lock --check` resolved the locked 177-package environment without
    changes. Ruff format checked 42 files; Ruff lint and pydoclint passed;
    Pyrefly reported zero warning-level diagnostics; `git diff --check` passed.
  - The full suite passed 387 tests in 24.01 seconds.
  - All seven pre-commit hooks passed using the existing isolated hook cache.
- Gate state at end: the Phase 3 review gate is `verified`. Mypy is absent,
  Pyrefly is the sole fast mandatory checker, the error baseline is empty,
  edited public APIs are covered, suppressions are narrow and demonstrated,
  and complete inline typing is documented and present in both artifacts.
- Commit readiness or commit id: no commit was requested or created. The plan
  update plus the three required untracked baseline/marker artifacts are ready
  for the next owner-selected commit.
- Exact next action: begin Checkpoint 4.1 by inventorying every `check_action`
  and equivalent binding predicate, then add a zero-I/O benchmark that fails on
  filesystem, MTH5, YAML, JSON, solution, flow, parameter, or job reads.

### S023 - 2026-07-20 - Make TUI action-state checks pure and cheap

- Checkpoint state at start: `4.1` was `not_started`; the Phase 3 review gate
  was verified in S022.
- Starting branch, HEAD, and worktree: `mth5` at owner commit `35096de`. The
  owner had committed the PEP 561 marker, implementation-record update, and
  pydoclint baseline. The required empty `pyrefly-baseline.json` remained
  untracked and was preserved.
- Session objective: inventory all Footer eligibility predicates, establish a
  failing zero-I/O test, cache state at its owning transitions, and verify that
  repeated `check_action` calls are pure and cheap.
- Inventory and failing evidence: `ProjectExplorerScreen.check_action` was the
  only action predicate. Project plotting called `file_summary`; data plotting
  traversed project/MTH5 and solution state; flow plotting parsed YAML; job
  plotting validated job resources; and project-data deletion previewed the
  filesystem. Extending the existing benchmark to `plot` failed with 100
  repeated `project.file_summary` calls before implementation.
- Work completed: added `_ProjectActionState` for project/data plot targets,
  validated flow paths, and derived-data deletion eligibility. Catalogue
  population, selection transitions, explicit refreshes, deletion, and
  terminal job events now update that cache while already owning the relevant
  reads. Job eligibility reuses existing cached summaries and validation.
  Split the former complex predicate into small in-memory helpers; action
  methods retain responsibility for loads performed only after invocation.
- Test coverage: expanded the action-state test across all 13 actions and four
  meaningful Project, MTH5 Data, Flow, and Job states. It instruments project
  summary/catalogue/deletion methods, job validation, YAML loading, and common
  `Path` read/query methods, and fails on any I/O during 5,200 eligibility
  checks.
- Files changed: `resistics/tui.py`, `tests/test_tui.py`, and this implementation
  record. The pre-existing untracked Pyrefly baseline was not changed.
- Decisions added or superseded: D037 records the pure predicate boundary and
  cache invalidation ownership.
- Verification commands and results:
  - The action benchmark completed 5,200 checks in 0.002322 seconds with zero
    instrumented I/O; the result is recorded in
    `.artifacts/hardening/performance/tui-actions.xml`.
  - The complete TUI suite passed 28 tests in 20.66 seconds; the full suite
    passed 387 tests in 22.63 seconds.
  - Ruff format checked 42 maintained Python files and Ruff lint passed.
    Pydoclint reported no unbaselined violations; Pyrefly reported zero
    warning-level diagnostics with the two established suppressions.
  - All seven configured pre-commit hooks passed, and `git diff --check`
    passed.
- Known failures or incomplete work: none within Checkpoint 4.1.
- Checkpoint state at end: `4.1` is `verified`; Checkpoint 4.2 is next.
- Commit readiness or commit id: the cached-state implementation, zero-I/O
  regression test, and S023 record are verified and ready for an
  owner-selected commit; no commit was requested or created.
- Exact next action: inventory every `refresh_bindings()` call, map it to its
  owning state transition, record handler-time baselines, and remove duplicate
  refreshes on tab activation, cursor movement, and nested control updates.

### S024 - 2026-07-20 - Eliminate redundant TUI binding refreshes

- Checkpoint state at start: `4.2` was `not_started`; Checkpoint 4.1 was
  verified in S023 and committed by the owner.
- Starting branch, HEAD, and worktree: `mth5` at owner commit `521b783`. Only
  the required empty `pyrefly-baseline.json` was untracked; it was preserved.
- Session objective: map each binding refresh to its owning state transition,
  measure representative handlers, remove duplicate publications, and batch
  related table and tree repaint work without changing Footer behavior.
- Inventory and failing evidence: 17 explicit refresh sites mixed final state
  transitions with intermediate catalogue population and content-only work.
  Tab activation refreshed twice; full project refresh and terminal job
  completion each refreshed three times; data metadata rendering refreshed
  once despite changing no action eligibility. The failing regression probe
  recorded refresh counts `200/3/3/1` for 100 tab handlers, one project refresh,
  one terminal progress event, and one metadata render.
- Work completed: introduced one batched full-view population boundary; removed
  intermediate refreshes from Data and Job population; made metadata rendering
  content-only; limited table-highlight refreshes to the focused table on its
  active tab; and moved edit/save/discard, job creation, deletion, restoration,
  processing, and refresh publications to the end of their owning transition.
  Full-view, resource, terminal-job, create, delete, and restore rebuilds use
  Textual's repaint batching boundary.
- Test coverage: added a regression benchmark that instruments actual binding
  publications for tab, full-refresh, terminal-progress, and metadata handlers,
  asserts counts `100/1/1/0`, and enforces the Phase 4 representative-handler
  target of less than 50 ms.
- Files changed: `resistics/tui.py`, `tests/test_tui.py`, and this implementation
  record. The pre-existing untracked Pyrefly baseline was not changed.
- Decisions added or superseded: D038 records outer-transition refresh
  ownership and the repaint batching boundary.
- Measurements and artifacts:
  - Before: 100 tab handlers took 0.000712 seconds; project refresh took
    0.013549 seconds; terminal progress took 0.003499 seconds; metadata
    rendering took 0.000306 seconds; refresh counts were `200/3/3/1`.
  - After: 100 tab handlers took 0.000294 seconds; project refresh took
    0.013007 seconds; terminal progress took 0.003279 seconds; metadata
    rendering took 0.000304 seconds; refresh counts were `100/1/1/0`.
  - The ignored before/after JUnit evidence is under
    `.artifacts/hardening/performance/tui-refresh-before.xml` and
    `.artifacts/hardening/performance/tui-refresh-after.xml`.
- Verification commands and results:
  - The complete TUI suite passed 29 tests in 19.25 seconds; the full suite
    passed 388 tests in 23.38 seconds.
  - Ruff format checked 42 maintained Python files and Ruff lint passed.
    Pydoclint reported no unbaselined violations; Pyrefly reported zero
    warning-level diagnostics with the two established suppressions.
  - All seven configured pre-commit hooks passed, and `git diff --check`
    passed.
- Known failures or incomplete work: none within Checkpoint 4.2.
- Checkpoint state at end: `4.2` is `verified`; Checkpoint 4.3 is next.
- Commit readiness or commit id: the refresh-ownership implementation,
  repaint batching, regression benchmark, and S024 record are verified and
  ready for an owner-selected commit; no commit was requested or created.
- Exact next action: inventory repeated explorer directory, MTH5, flow,
  parameter, criteria, and job scans; define file-identity cache keys and
  invalidation owners; then add failing cache-hit and stale-file tests.

### S025 - 2026-07-20 - Add the cached project explorer index

- Checkpoint state at start: `4.3` was `not_started`; Checkpoint 4.2 was
  verified in S024 and committed by the owner.
- Starting branch, HEAD, and worktree: `mth5` at owner commit `148a877`. Only
  the required empty `pyrefly-baseline.json` was untracked; it was preserved.
- Session objective: centralise explorer discovery and parsing behind a
  UI-neutral cache, define file identities and invalidation ownership, and
  protect cache hits, stale files, malformed resources, and MTH5 lifecycle.
- Inventory and failing evidence: overview and Data independently queried
  project state; every resource table rescanned and reparsed YAML; job listing
  reparsed every referenced flow, parameter set, and criteria file per job;
  selection validated the same job again; and job-form options repeated the
  scans. The initial seven-test explorer contract failed at collection because
  no index service existed.
- Work completed: added `ProjectExplorerIndex` with handle-free project state,
  lazy run summaries, non-fatal discovery issues, parsed resource records, and
  cached job summaries/validations. YAML identities combine path,
  nanosecond-resolution modification time, and byte size. Section cache hits
  perform no reads; invalidated unchanged identities reuse their parsed model.
  Split `ProjectJobs.validate_loaded` from file resolution so indexed jobs use
  already-parsed resources without changing standalone `validate` behavior.
- TUI integration: overview, Data, resource tables, job options, and selection
  now consume indexed state. Resource edits, copies, deletes, creation, and
  default restoration invalidate their namespace and dependent jobs. Derived
  data deletion invalidates project state; terminal processing invalidates
  project and jobs; explicit refresh invalidates every section. Job tables are
  rebuilt when a referenced flow, parameter set, or criteria definition
  changes.
- Test coverage: seven UI-neutral tests cover cache identity, section-isolated
  invalidation, stale and deleted files, malformed YAML error caching, parsed
  resource reuse across job validation, dependent-job invalidation, and cached
  DTO access after the project MTH5 handle closes. A TUI integration test maps
  external refresh, flow edit, job create/delete, derived-data deletion, and
  processing completion to their exact invalidation calls.
- Files changed: new `resistics/explorer.py` and `tests/test_explorer.py`, plus
  `resistics/job.py`, `resistics/tui.py`, `tests/test_tui.py`, the pydoclint
  baseline, and this implementation record. The pre-existing untracked Pyrefly
  baseline was not changed.
- Decisions added or superseded: D039 records handle-free cache contents, file
  identity, dependency invalidation, and explicit external-refresh semantics.
- Verification commands and results:
  - The focused explorer, job, and TUI suites passed 52 tests; the full suite
    passed 396 tests in 23.42 seconds.
  - Ruff format checked 44 maintained Python files and Ruff lint passed.
    Pydoclint reported no unbaselined violations; explicit regeneration removed
    nine stale line-sensitive entries, leaving 1,714. Pyrefly reported zero
    warning-level diagnostics with the two established suppressions.
  - All seven configured pre-commit hooks passed, and `git diff --check`
    passed.
- Known failures or incomplete work: none within Checkpoint 4.3.
- Checkpoint state at end: `4.3` is `verified`; Checkpoint 4.4 is next.
- Commit readiness or commit id: the explorer index, loaded-resource job
  validation boundary, invalidation integration, tests, baseline update, and
  S025 record are verified and ready for an owner-selected commit; no commit
  was requested or created.
- Exact next action: inventory synchronous project-opening and explorer loads,
  define immutable worker results and generation-based stale-result rejection,
  and add a responsiveness test proving the screen mounts before discovery
  completes.
