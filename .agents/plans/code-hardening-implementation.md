# Resistics Code-Hardening Implementation Record

Status: initialized; implementation not started
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
- Active phase: Phase 0 - Restore and Measure the Baseline
- Active checkpoint: `0.2` - Add durable quality and performance measurements
- Checkpoint state: `not_started`
- Last completed checkpoint: none
- Last verified checkpoint: `0.1` in `S001`; changes are uncommitted
- Last session: `S001`
- Last verified commit: none; `c345ae6` is only the starting repository HEAD
- Current blocker: none
- Next exact action: begin Checkpoint 0.2 by defining repeatable commands for
  coverage, source/test size, public docstrings, examples/plots, cold TUI import,
  and representative TUI action-state I/O before adding regression gates.

Current worktree caveat:

- `.agents/plans/codebase-hardening.md` contains intentional planning edits.
- `.github/workflows/commit_flow.yml` and
  `.github/workflows/publish_flow.yml` contain pre-existing user changes.
- `resistics/templates.py` and `resistics/transfunc.py` contain the verified
  Checkpoint 0.1 fixes from S001.
- These files must not be reverted or overwritten while establishing the
  baseline.

## Authority and Boundaries

- Perform resistics implementation on `mth5`.
- Do not open a pull request, promote `mth5`, rename the primary branch, publish
  either package, create a release, push a tag, or change repository settings.
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
    +--> regressioninc 1.1 -> 1.2 -> 1.3
    |                              |
    |                              v
    +--------------------------> resistics 1.4 -> 1.5 -> 1.6 -> 1.7 -> 1.8
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
```

Additional sequencing rules:

- Phase 1.4 cannot complete until an installable `regressioninc` boundary
  exists. Publishing that boundary remains a manual user action.
- Ruff and pydoclint must be installed before their pre-commit hooks become
  mandatory.
- The type-checker evaluation must precede removal of mypy.
- TUI responsiveness work precedes splitting `resistics/tui.py`, so module
  movement follows behaviour and performance protection.
- The Phase 7.1 parser prototype must pass before bulk docstring conversion.
- The documentation migration follows stable public/module boundaries where
  possible, avoiding avoidable rewrites of freshly converted documentation.
- Phase 8 cannot begin while a phase gate is incomplete or deferred without an
  owner and documented production consequence.

## Checkpoint Tracker

The completion evidence column should contain a commit, session id, report path,
or short command/result reference. Detailed output belongs in the session log or
`.artifacts/`, not in this table.

| ID | Repository | State | Completion evidence or unblock condition |
| --- | --- | --- | --- |
| 0.1 | resistics | `verified` | S001: 372 tests pass; F821 removed |
| 0.2 | resistics | `not_started` | Depends on 0.1 |
| Gate 0 | resistics | `not_started` | Depends on 0.1-0.2 |
| 1.1 | regressioninc | `not_started` | Separate repository/worktree |
| 1.2 | regressioninc | `not_started` | Depends on 1.1 |
| 1.3 | regressioninc | `not_started` | Depends on 1.2; release is manual |
| 1.4 | resistics | `not_started` | Needs installable regressioninc |
| 1.5 | resistics | `not_started` | Depends on 1.4 |
| 1.6 | resistics | `not_started` | Depends on 1.5 |
| 1.7 | resistics | `not_started` | Workflow only; no publication |
| 1.8 | resistics | `not_started` | Depends on active uv paths |
| Gate 1 | both | `not_started` | Depends on 1.1-1.8 |
| 2.1 | resistics | `not_started` | Ruff lint migration |
| 2.2 | resistics | `not_started` | Depends on 2.1 |
| 2.3 | resistics | `not_started` | pydoclint in NumPy mode initially |
| 2.4 | resistics | `not_started` | Public docstring contract |
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

### Phase 1 - Packaging, uv, and automation

- Session work: handle `regressioninc` packaging in its repository, establish
  its installable boundary, remove the local source override, harden resistics
  artifacts, and replace test/publish workflows.
- Keep release configuration separate from an actual release. Record any manual
  user action as a blocker with the precise artifact/version required.
- Exit evidence: clean uv-only sync/build/test/docs paths and inspected wheel and
  sdist artifacts from both packages.

### Phase 2 - Ruff, pydoclint, docstrings, and pre-commit

- Session work: map legacy rules, introduce Ruff lint then formatting, install
  pydoclint in temporary NumPy mode, enforce the docstring strategy, and switch
  pre-commit to locked project tools.
- Preserve a formatting-only boundary even if it shares a session with other
  work, so behavioural regressions remain diagnosable.
- Exit evidence: legacy tools absent, public docstrings complete, baselines
  recorded, pre-commit and CI invoking the same project versions.

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
| Tests | 372 collected; 371 passed; 1 failed | 372 passed | 372 passed | S001 |
| Branch coverage | approximately 76% | pending | pending | Phase 0 |
| Production Python | approximately 21,011 lines | pending | pending | Phase 0 |
| Tests | approximately 5,941 lines | pending | pending | Phase 0 |
| `resistics/tui.py` | 2,673 lines | pending | pending | Phase 0 |
| `resistics/plot.py` | 1,200 lines | pending | pending | Phase 0 |
| Flake8 | approximately 40 findings | 43 findings | 43 findings | S001 |
| Black format | not recorded | 26 files differ | 26 files differ | S001 |
| mypy | 210 errors across 17 files | pending | pending | Phase 0 |
| TUI cold import | approximately 2.18 seconds | pending | pending | Phase 0 |
| Public docstring coverage | not measured | pending | pending | Phase 0 |
| Executable docstring examples | not measured | pending | pending | Phase 0 |
| Executable docstring plots | not measured | pending | pending | Phase 0 |
| Local `.venv` size | approximately 897 MB | pending | pending | Phase 0 |

For timing measurements, record the machine/runtime context and multiple runs.
Do not compare one cold run with one warm run or turn machine-specific timings
into brittle CI assertions.

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
