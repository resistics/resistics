# Resistics Code-Hardening Implementation Record

Status: in progress; Checkpoint 7.2 verified; Checkpoint 7.3 ready
Created: 2026-07-19
Last updated: 2026-07-22
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
- Active phase: Phase 7 - MyST documentation modernisation
- Active checkpoint: `7.3` (`resistics`) - Convert docstrings without moving their content
- Checkpoint state: `not_started`
- Last completed checkpoint: `7.2` in `S043`
- Last verified checkpoint: `7.2` in `S043`
- Last session: `S043`
- Last verified commit: resistics `70a84f2`, recording verified hardening
  through Checkpoint 6.3;
  regressioninc `9eb11a4` is the base of uncommitted Checkpoints 1.1 and 1.2
  work
- Current blocker: none
- Next exact action: migrate the `common`, `sampling`, and `transfunc`
  docstrings to MyST in place and configure explicit NumPy- and Sphinx-style
  pydoclint invocations for the unconverted and converted module sets.

Current worktree caveat:

- Resistics `70a84f2` records verified hardening through Checkpoint 6.3. The
  verified Checkpoint 6.4 uv/Pixi decision, environment-manager guidance, and
  execution record are uncommitted. Checkpoint 7.1's selected standard-autodoc
  MyST bridge, bounded representative fixture, dependencies, lock, page, and
  execution record are stacked on that work and are also uncommitted.
  Checkpoint 7.2's maintained MyST pages, current landing content, expanded API
  navigation, Sphinx configuration, and execution record are stacked on those
  changes. The required empty
  `pyrefly-baseline.json` remains intentionally untracked.
- The owner's minimum/maximum Python CI intent remains a requirement of
  deferred Checkpoint 1.6 after the obsolete hosted workflows were removed;
  D053 makes that current matrix Python 3.12/3.14.
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
  checks all 35 production modules without error-level findings;
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
| 4.3 | resistics | `verified` | S025-S026; identity cache, Pydantic DTOs, owned invalidation |
| 4.4 | resistics | `verified` | S027; 403 tests; responsive/lazy/stale-result coverage |
| 4.5 | resistics | `verified` | S028; 0.1748 s import; 0.235886 s first screen; 405 tests |
| 4.6 | both | `verified` | S029; structured lifecycle events; no raw terminal rendering |
| Gate 4 | resistics | `verified` | S023-S029; responsive, zero-I/O, 92.29% import reduction |
| 5.1 | resistics | `verified` | S030; facade/entry point preserved; 413 tests; artifacts verified |
| 5.2 | resistics | `verified` | S031; stable screen owners; 414 tests; artifacts verified |
| 5.3 | resistics | `verified` | S032-S033; UI-neutral service and session diagnostics; 421 tests |
| 5.4 | resistics | `verified` | S034; thin adapters; shared renderer; 421 tests |
| 5.5 | resistics | `verified` | S035; responsibility split; 425 tests; artifacts verified |
| 5.6 | resistics | `verified` | S036; explicit handle ownership; 423 tests; artifacts verified |
| 5.7 | resistics | `verified` | S037; 422 tests; root Ruff and artifacts verified |
| Gate 5 | resistics | `verified` | S030-S037; public removals documented; graph behavior protected |
| 6.1 | resistics | `verified` | S038; 128-package lock; ownership map; artifacts verified |
| 6.2 | resistics | `verified` | S039; Python 3.12 paired wheels; 17 floors; 89 doctests |
| 6.3 | resistics | `verified` | S040; Python 3.12-3.14 OSV audit; zero accepted risks |
| 6.4 | resistics | `verified` | S041; uv retained; 3x3 binary resolution matrix |
| Gate 6 | resistics | `verified` | S038-S041; owned dependencies; paired wheels; OSV clean |
| 7.1 | resistics | `verified` | S042; standard autodoc selected; 5 doctests; plot/reference proof |
| 7.2 | resistics | `verified` | S043; 28 MyST pages; 62-page HTML build |
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

#### Checkpoint 6.1 dependency ownership map

| Runtime dependency | Resistics owner |
| --- | --- |
| NumPy | Array-backed processing and data models throughout production |
| SciPy | Filtering, resampling, decimation, and spectral operations |
| Pandas | Project/run summaries, jobs, masks, gathers, and tabular metadata |
| tsdownsample | Public Plotly time-series downsampling with NaN preservation |
| attotime | High-precision sampling and timestamp conversion |
| Plotly | Public plotting APIs, flow/job figures, and TUI plot presentation |
| fast-sugiyama | Ranked flow/job graph layout |
| Loguru | Package diagnostics, processing logs, and TUI log capture |
| Pydantic | Public process, data, configuration, job, and TUI DTO contracts |
| PyYAML | Configuration, process, flow, and job serialization |
| Textual with syntax support | Installed ``resistics`` TUI entry point and YAML editors |
| regressioninc | Numerical regression implementations used by Resistics adapters |
| MTH5 | Sole public project/time-series input boundary |
| Xarray | Labelled ``TimeData`` channel/sample storage |

| Dependency group | Owner |
| --- | --- |
| shared: Matplotlib | Pytest-executed API plots and Sphinx plot directives |
| dev: pre-commit, Ruff, pydoclint, Pyrefly, types-PyYAML | Local repository quality gates |
| docs: Sphinx, Furo, Sphinx-Gallery, OpenGraph, autodoc-pydantic, copybutton, Kaleido | Transitional documentation build and Plotly static rendering |
| tests: pytest, pytest-cov | Executable examples, tests, and branch coverage |

The separate notebook group has no owner after Checkpoint 5.7 removed every
tracked notebook. Plotly and Textual remain runtime requirements because they
back supported public behavior; Matplotlib remains outside runtime metadata.
The transitional ``docs/requirements.txt`` file remains until Checkpoint 7.5,
but its now-unused IPython, nbformat, and seedir entries were removed here.

#### Checkpoint 6.3 hosted security follow-ups

| Follow-up | Owner and timing | Required boundary | Consequence while deferred |
| --- | --- | --- | --- |
| Dependabot | Repository owner; Checkpoint 1.6 | Monitor the uv/Python lock and GitHub Actions dependencies | Vulnerabilities are found by the documented local OSV gate, not continuously hosted |
| Action pinning | Repository owner; Checkpoint 1.6 | Pin every third-party action to a reviewed full commit SHA | No future workflow may be treated as supply-chain hardened before this review |
| Workflow permissions | Repository owner; Checkpoints 1.6-1.7 | Default to read-only and grant only the minimum job-specific permissions | Hosted automation and publication remain outside the production-readiness claim |
| Protected publishing | Repository owner; Checkpoint 1.7 | Use a protected publishing environment, required approval, and a matching trusted publisher | Releases remain a manual owner action and this branch is not publication-ready |

#### Checkpoint 6.4 environment-manager evidence

| Evidence | Classification | Outcome |
| --- | --- | --- |
| The former lttbc wheel loaded against an incompatible NumPy ABI | Reproduced native-wheel incompatibility | Replaced by tsdownsample in S008; all Python-minor installs then passed |
| fast-sugiyama 0.5.3 has no wheel usable on manylinux 2.28 | Current platform boundary | Its wheels require manylinux 2.34; older glibc is not a verified wheel-only target |
| Isolated builds could not fetch Hatchling inside the managed sandbox | Network/cache restriction | Every network-enabled retry built successfully; not a solver or package compatibility defect |
| regressioninc's old statsmodels floor failed a Python 3.11 source build | Stale sibling lower-bound metadata | Owned by regressioninc's separate dependency review; current Resistics floors install on Python 3.12 and 3.14 |
| Coverage-instrumented TUI timing and legacy gallery failures | Application/test/documentation behavior | Unrelated to dependency solving or native package availability |

uv 0.11.25 resolved the complete runtime and default group set using binary
distributions for Python 3.12, 3.13, and 3.14 on x86-64 manylinux 2.34, macOS,
and Windows: all nine cells passed, as did locked dry-run syncs for the same
matrix. The lock records corresponding supported-minor wheels for the critical
NumPy, SciPy, Pandas, h5py, tsdownsample, PyProj, Pydantic Core, scikit-learn,
and statsmodels native stack. S039 and S041 executed paired installed wheels
and 89 doctests at the Python 3.12 and 3.14 endpoints. No repeated uv/PyPI
limitation remains on the declared local contract, so Checkpoint 6.4's
threshold for a Pixi trial was not met.

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
| Tests | 372 collected; 371 passed; 1 failed | 373 passed | 422 passed | S037 |
| Branch coverage | approximately 76% | 75.96% | 80.59%; threshold met, two coverage-instrumented TUI failures | S038 coverage XML |
| Production Python | approximately 21,011 lines | 21,015 | 24,071 | S037 report |
| Tests | approximately 5,941 lines | 6,051 | 7,757 | S037 report |
| `resistics/tui.py` | 2,673 lines | 2,673 | `app.py` 436; project/logging modules 2,923 | S033 report |
| `resistics/plot.py` | 1,200 lines | 1,200 | plot 700; flow graph 617 | S034 report |
| `resistics/gather.py` | 1,235 lines | 1,235 | facade 183; criteria 247; data 582; plan 312; project 645 | S035 report |
| `resistics/project.py` | 1,224 lines | 1,224 | project 1,209; private MTH5 boundary 291 | S036 report |
| `resistics/testing.py` | 1,296 lines | 1,296 | 816; suite-only factories 195 under tests | S037 report |
| Flake8 | approximately 40 findings | 43 findings | 43 findings | S001 |
| Legacy complexity | not recorded | 17 CCR001; 5 C901; 4 ECE001 | same | S002 |
| Black format | not recorded | 26 files differ | 26 files differ | S001 |
| mypy | 210 errors across 17 files | 209 across 17 files | removed | S016 |
| Pyrefly | not installed | 175 errors across 17 files | 0 new errors | S016 |
| TUI cold import | approximately 2.18 seconds | 2.2659 s median | 0.2715 s median | S033 report |
| TUI first screen | not measured | 2.076730 s median | 0.235886 s median | S028 probe |
| Cached TUI action checks | not measured | 5,200 in 0.002322 s; zero instrumented I/O | same | S023 XML |
| TUI binding refresh ownership | not measured | calls 200/3/3/1 | calls 100/1/1/0; 13.007 ms maximum | S024 XML |
| Explorer resource parsing | repeated by table, job, and selection | one parse per file identity | zero reads on cache hits | S025 tests |
| TUI project/explorer loading | synchronous before first project screen | loading surface before blocked open/summary release | inactive tabs lazy; stale results rejected | S027 tests |
| Terminal progress rendering | two unconditional `tqdm` loops | same | zero; serializable callbacks through TUI | S029 tests |
| TUI module boundary | one module | one 2,673-line module | app 436; project 499; project mixins 63-756; logging 322; services 642; state 156 | S033 artifacts |
| Public docstring coverage | not measured | 80.2%; 566/706 | 86.8%; 638/735 | S033 report |
| Executable docstring examples | not measured | 778 prompts | 803 prompts | S032 report |
| Executable docstring plots | not measured | 16 directives | same | S002 report |
| Locked packages | not recorded | 174 | 128 | S038 lock |
| Local `.venv` size | approximately 897 MB | 910 MB | 744 MB | S038 |

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
- `D054` (2026-07-22): Make one repository script the dependency-security
  authority. Derive Python 3.12-3.14 from package classifiers and run uv's
  locked OSV audit against the complete default environment for each minor;
  this checks marker-specific runtime, development, documentation, shared, and
  test resolutions without requiring each interpreter to be installed. Keep
  the networked check in the production gate rather than pre-commit. Begin with
  no accepted risks. An advisory without a compatible fix may be suppressed
  only through a complete, owned record with a review no more than 90 days away
  and uv
  ``--ignore-until-fixed``; available fixes must be updated and locked. Defer
  Dependabot, full-SHA action pinning, least-privilege workflow permissions,
  and protected trusted publishing to their existing owner checkpoints without
  implying that hosted monitoring currently exists.
- `D055` (2026-07-22): Retain uv 0.11.25 as the sole environment and dependency
  manager and do not run a Pixi trial. Paired installed wheels pass at both
  supported Python endpoints, and the complete dependency set resolves with
  binary distributions across the tested Python 3.12-3.14 by x86-64 Linux,
  macOS, and Windows matrix. The only current wheel boundary is
  fast-sugiyama's manylinux 2.34 floor; an older-glibc target would need a
  separately verified source build or a new platform decision. Treat sandbox
  DNS/cache failures, stale sibling minimum metadata, and application/test/docs
  failures separately from package-manager limitations. Trial Pixi only after
  a required target repeatedly fails because uv/PyPI cannot supply or
  reasonably build a native dependency; do not add a second lock for concern
  alone.
- `D056` (2026-07-22): Retain standard Sphinx autodoc as the Phase 7 API
  generator. The autodoc2 0.5.0 prototype did not reach the current API's
  fidelity: it omitted public facade re-exports, collapsed overloads to the
  implementation signature, and rendered Pydantic constructors as generic
  ``**data`` input. Standard autodoc preserved those contracts and can render
  Python 3.12 ``type`` aliases through an explicit data directive. Route
  converted docstrings through the local ``myst_autodoc`` adapter so one
  dynamic generator supplies discovery and source links while selected content
  is parsed as MyST. Keep the regex-based MyST/RST boundary explicit until
  Checkpoint 7.5. The adapter also supplies the missing MyST state-machine hook
  required by Matplotlib's fenced ``{plot}`` directive; remove that hook when
  upstream MyST implements ``insert_input``. Keep unavoidable imports confined
  to the locked documentation environment. Retain the prototype-only autodoc2
  dependency until Checkpoint 7.5 removes the unselected transitional stack.
- `D057` (2026-07-22): Make every maintained site-structure, narrative, and
  API module-entry page a MyST source now, including newly visible entry pages
  for the current explorer, flow, job, mask, templates, and TUI modules. Delete
  the obsolete generated-module-list source instead of translating it. Keep
  standard autodoc directives inside explicit fenced ``{eval-rst}`` blocks
  until the selected generator's transitional parser stack is removed in
  Checkpoint 7.5. Retain one fenced RST citation definition for the legacy
  Sphinx Gallery pages only until Checkpoint 7.4 deletes that gallery; all
  maintained prose, navigation, labels, and cross-references remain native
  MyST.

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
- `D040` (2026-07-20): Use frozen Pydantic v2 models for every public explorer
  result, including file identities, issues, indexed resources, indexed jobs,
  and project state. These objects cross the service/UI boundary and must offer
  validation, serialization, and JSON schema. Reserve frozen dataclasses for
  private implementation records; the parsed-resource cache uses one private
  dataclass key containing only its namespace and filesystem identity. Inherit
  the public results directly from Pydantic `BaseModel`, because the existing
  `ResisticsModel.summary()` method conflicts with the established `summary`
  fields on explorer job and project results.
- `D041` (2026-07-20): Present project-opening and explorer loading surfaces
  immediately, then run their synchronous project, MTH5, filesystem, YAML, and
  job discovery through Textual-managed async workers backed by short-lived
  dedicated thread executors. Do not use the event loop's default executor:
  Textual 8.2.8 under Python 3.13 waits for those threads during test-loop
  shutdown, which can freeze cancellation of an uninterruptible synchronous
  call. Workers return private frozen dataclass messages containing public
  frozen Pydantic explorer DTOs; only UI-thread worker lifecycle handlers
  mutate widgets. Load only the active explorer tab. Treat cancellation as
  generation rejection because Python cannot terminate an active synchronous
  thread safely; close superseded project handles deterministically. Protect
  the shared explorer cache with section epochs so a late invalidated thread
  cannot repopulate current state. Defer MTH5 closure after explorer unmount
  until every already-running discovery exits, and reject discovery that
  starts after closure was requested.
- `D042` (2026-07-21): Keep the `resistics.tui` module import limited to the
  standard library, Loguru, and Textual. Import project, MTH5, explorer, job,
  flow, spectra, regression, transfer-function, and plotting services only
  when their screen or action begins; retain their annotations behind
  `TYPE_CHECKING`. Feature boundaries convert a missing required dependency
  into an actionable reinstall message while preserving ordinary runtime
  errors. Tests patch dependencies in their owning modules rather than relying
  on eager aliases in `resistics.tui`. Protect the boundary with a fresh-
  process module-inventory test instead of a brittle timing threshold; keep
  repeatable multi-sample timings as checkpoint evidence.
- `D043` (2026-07-21): Replace raw regression-loop rendering with the frozen
  Pydantic `ProcessingProgressEvent` contract and `ProcessingProgressState`
  lifecycle. Processes opt into progress and cancellation through keyword
  arguments; the base flow adapter forwards executor-owned callbacks only when
  those arguments are declared. Flow execution enriches events with stage,
  node, and qualified process identity; `JobProgressEvent` carries the exact
  nested event to standalone, app, and TUI consumers. A private dataclass owns
  mutable frequency counters. Remove `tqdm`, `types-tqdm`, and their orphaned
  `types-requests` dependency rather than retaining a renderer in core. The
  sibling regressioninc inventory found no terminal renderer and requires no
  change. Register pydoclint's external `DOC` code family with Ruff so six
  definition-line suppressions can document pydoclint's inability to resolve
  callback type aliases without weakening either checker.
- `D044` (2026-07-21): Convert `resistics.tui` from one module into an explicit
  package facade while keeping all implementation behavior in
  `resistics.tui.app` for Checkpoint 5.1. Re-export only the TUI's own public
  constants, aliases, screens, application class, and launchers; imported
  Textual and standard-library names are not part of the supported facade.
  Private-helper tests import the owning `app` module. Create documented empty
  `state`, `services`, and `screens` boundaries now, but defer class movement
  to Checkpoints 5.2-5.3 so the package conversion remains mechanical. Preserve
  `resistics.tui:main`, update only the path-sensitive Ruff ignore and
  pydoclint baseline, and verify both built distributions before extraction.
- `D045` (2026-07-21): Give the eight modal dialogs and their immutable
  deletion request to `screens.dialogs`, and give the header plus three
  pre-project screens to `screens.launcher`. Preserve their source bodies,
  widget ids, messages, bindings, callback wiring, facade identities, and the
  transitional `resistics.tui.app` aliases. Move only the three helpers shared
  across these screens into `services`; use a function-local application import
  in the existing runtime type check so screen-module imports cannot create an
  app/screens cycle. Extend the existing Textual-only Ruff exceptions to the
  screen modules and regenerate the line/path-sensitive pydoclint baseline;
  do not combine the move with DTO or behavior changes. Keep the 693-line
  dialogs module as one cohesive modal-screen boundary for now: splitting its
  repeated dialog CSS and focus behavior during a mechanical checkpoint would
  obscure the movement, while it remains below the Phase 5 module ceiling.
- `D046` (2026-07-21): Make `ProjectExplorerService` the UI-neutral owner of
  project indexing, validation, YAML mutations, deletion previews, plot-target
  resolution, job execution, and project-close coordination. The service may
  be constructed and tested without a Textual application; existing frozen
  Pydantic explorer DTOs cross its public boundary. Move the public
  `ProjectDataDeletionRequest` to `state.py` and convert it to a frozen
  Pydantic v2 model, while retaining dataclasses for private worker and mutable
  action records. This deliberately narrows D042's lightweight-import rule:
  the public TUI facade may now import Pydantic for its public DTO contract,
  but project, MTH5, plotting, processing, and other feature-heavy modules
  remain deferred. Keep the concrete `ProjectExplorerScreen` in
  `screens.project`; private data, job, and resource presentation mixins own
  cohesive UI behaviour only. Preserve public facade and transitional app
  aliases, but remove the unsupported internal screen aliases for the index
  and job repository rather than carrying legacy internals forward. Every new
  production module remains below 800 lines and the concrete screen remains
  below 500 lines.
- `D047` (2026-07-21): Add a separate session Logs tab without weakening the
  structured, per-job Activity surface. Represent each UI/app-facing diagnostic
  as a frozen Pydantic `DiagnosticLogEntry`; keep sequence numbers, cursor
  reads, rollover accounting, and the 2,000-entry thread-safe buffer private.
  The official `run_tui` lifecycle captures Python warnings and the shared
  Loguru logger at `INFO` and above without writing to Textual from worker
  threads. Reinstall that sink immediately after lazy project/MTH5 imports,
  because MTH5 calls `logger.configure()` and replaces process-global handlers.
  Drain the buffer on the UI thread every 250 ms and render Rich `Text`, not
  markup supplied by log messages. Keep startup records buffered until the Logs
  tab is visible, then write them at the pane's explicit content width and
  rerender from the bounded buffer after a terminal resize; wrapping therefore
  follows the screen width rather than RichLog's 78-column default. Retain
  diagnostics across project switches for one
  application session only; do not add general persistence, filtering,
  clearing, exporting, or debug capture. Existing per-job files remain the
  durable processing record, and direct callers using string
  `startup_warnings` remain compatible through normalization at the app/screen
  boundary.
- `D048` (2026-07-21): Keep `plot_flow` and `plot_job` as public model adapters
  in `resistics.plot`, and move their shared ranked layout and Plotly rendering
  into the private `resistics.flow_graph` boundary. Use a frozen private
  dataclass for the immutable labels, hover text, dimensions, title, layout,
  and trace-metadata contract; do not expose a new Pydantic or public graph
  model. Both adapters validate and adapt their own flow/job models before the
  renderer sees them. Preserve trace kinds, grouping, ordering, colours,
  arrow/card geometry, axis ranges, title spacing, parameter-summary wording,
  and clean hover templates. Transfer unchanged private-helper docstring debt
  to the line/path-sensitive baseline while fully documenting the new render
  contract and materially changed functions. This leaves `plot.py` and the new
  renderer below 800 lines without changing the public plotting API.
- `D049` (2026-07-21): Keep `resistics.gather` as the public facade and the
  owner of `Gather` orchestration while separating criteria/validation,
  project and MTH5 discovery, immutable planning, and data assembly into
  `gather_criteria`, `gather_project`, `gather_plan`, and `gather_data`.
  Preserve existing import, YAML, pickle, autodoc, and process-catalog paths by
  explicitly retaining `resistics.gather` as the public models' canonical
  module. Use frozen private dataclasses for locators and plans; public models
  remain Pydantic processes/data contracts. Catch arbitrary failures only at
  persisted evaluation and mask reader boundaries, where station, run, level,
  and evaluation context can be added with exception chaining. Automatic
  remote discovery may skip expected `ValueError` candidate rejection but must
  propagate unexpected failures. Assemble only from the completed immutable
  plan so discovery and array construction cannot mutate one another.
- `D050` (2026-07-21): Treat MTH5 as the sole source hierarchy for projects
  and make each public `Project` or `MTH5File` object the explicit owner of its
  live handle. Keep third-party construction behind a private, lazily imported
  structural protocol so importing `resistics.project` does not initialize the
  MTH5 stack. Expose idempotent `close()`, `closed`, and context-manager
  semantics; cached summaries remain readable after closure, while live group
  and sample operations raise a clear `RuntimeError`. Close handles at every
  failed-open or failed-model-construction ownership boundary without masking
  the originating exception. Remove the obsolete public `close_mth5`,
  `dir_path`, `metadata`, `init(force=...)`, measurement/site path and naming
  helpers, legacy project exceptions, and stale gallery paths after repository
  searches confirm no maintained caller. Require `output_label` for the sole
  canonical results tree and document these intentional removals in the next
  release notes. Defer tracked notebook modernization to Checkpoint 5.7, whose
  governing scope explicitly owns notebook review.
- `D051` (2026-07-21): Keep `resistics.testing` as installed support for the
  small builders imported by executable API documentation, not as a home for
  the repository's regression factories or comparison assertions. Move the
  still-used linear-time, evaluation-data, random-solution, and solution
  comparison machinery to `tests.synthetic_data`; delete unused history,
  spectra-metadata, time comparison, and bulk evaluation fixtures. Remove the
  empty `resq` placeholder, fully commented `Join` prototype, unreferenced
  exploratory notebooks and request data, always-true timestamp gather hook,
  and legacy flow builders after call-site and documentation searches. Use
  `single_site_mt_flow` as the canonical maintained flow builder; document the
  deliberate public removals rather than retaining more aliases. Defer removal
  of the now-suspect notebook dependency group to Checkpoint 6.1, where all
  direct and optional dependencies are reviewed together.
- `D052` (2026-07-21): Treat a direct dependency as an owned compatibility
  promise, not a record of everything present in the environment. Remove
  Resistics declarations for ObsPy and scikit-learn because production imports
  neither; regressioninc continues to own scikit-learn transitively. Replace
  the sole prettyprinter use with a small JSON-compatible formatter that
  preserves the executable ``summary()`` layout. Remove the ownerless notebook
  group, pytest-html, IPython, nbformat, seedir, and emoji declarations. Keep
  Plotly and Textual in runtime metadata because public plotting and the
  installed CLI require them; keep Matplotlib in the shared docs/tests group.
  Rebase legacy runtime lower bounds on Python 3.11-era/current APIs, require
  the locally verified MTH5 0.6.8 boundary, and give the editable regressioninc
  source its actual alpha version constraint. Relax exact attotime and
  fast-sugiyama pins because no incompatibility justifies upper bounds. Retain
  only the reproduced ``tsdownsample<0.2`` cap from D022. Checkpoint 6.2 owns
  installation and execution of the complete declared minimum set.
- `D053` (2026-07-21, supersedes D052's lower-bound and tsdownsample-cap
  policy): Prefer a modern supported floor over preserving compatibility with
  old dependency releases. Raise Resistics to Python `>=3.12,<3.15` so its
  floors can be the current locked NumPy 2.5 and SciPy 1.18 releases; raise all
  runtime and uv-group floors to their current verified releases. Keep these as
  open `>=` requirements so newer compatible packages can resolve. Remove the
  speculative `tsdownsample<0.2` cap because D022 verified only 0.1.5.1 and no
  incompatible 0.2 release was reproduced. Retain exact Pyrefly 1.1.1 as a
  deliberately versioned static-analysis/baseline contract, not a runtime
  compatibility promise. Test Resistics' runtime, shared, and test floors with
  uv `lowest-direct` on Python 3.12, while building and installing both local
  wheels. Leave regressioninc's own transitive lower-bound audit to its
  separate repository review; Resistics neither pins nor claims those
  transitive versions.
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

### S026 - 2026-07-20 - Refine the explorer model boundary

- Checkpoint state at start: `4.3` was verified and committed by the owner;
  Checkpoint 4.4 remained `not_started`.
- Starting branch, HEAD, and worktree: `mth5` at owner commit `ff155bb`. Only
  the required empty `pyrefly-baseline.json` was untracked; it was preserved.
- Session objective: apply the agreed boundary in which explorer results shared
  with callers are Pydantic v2 models and dataclasses remain private
  implementation records, then reverify Checkpoint 4.3 before beginning 4.4.
- Work completed: converted `ExplorerFileIdentity`, `ExplorerIssue`,
  `IndexedResource`, `IndexedJob`, and `ProjectExplorerState` to frozen
  Pydantic v2 models. Added the private frozen `_ResourceCacheKey` dataclass so
  cache mechanics are not exposed in the public schema. The DTOs inherit
  directly from `BaseModel` because `ResisticsModel.summary()` conflicts with
  the existing explorer `summary` fields.
- Test coverage: added schema and JSON round-trip coverage for every public
  explorer result class, project state, and a parsed flow resource. Strengthened
  cache reuse coverage to prove an unchanged private key returns the exact same
  indexed record. Updated the TUI fake project to return the declared
  `MTH5FileSummary` contract; Pydantic validation exposed its earlier
  `SimpleNamespace` substitute.
- Files changed: `resistics/explorer.py`, `tests/test_explorer.py`,
  `tests/test_tui.py`, and this implementation record. The pre-existing
  untracked Pyrefly baseline was not changed.
- Decisions added or superseded: D040 records the public-Pydantic and
  private-dataclass boundary and the direct `BaseModel` choice.
- Verification commands and results:
  - The focused explorer suite passed 8 tests with warnings treated as errors;
    the combined explorer, job, and TUI suites passed 54 tests.
  - The full suite passed 397 tests in 22.12 seconds.
  - Ruff format checked 44 maintained Python files and Ruff lint passed.
    Pydoclint reported no unbaselined violations; Pyrefly reported zero
    warning-level diagnostics with the two established suppressions.
- Known failures or incomplete work: none within the Checkpoint 4.3 model
  refinement.
- Checkpoint state at end: `4.3` remains `verified`; Checkpoint 4.4 remains
  `not_started`.
- Commit readiness or commit id: the explorer model-boundary refinement,
  serialization tests, fixture correction, and S026 record are verified and
  ready for an owner-selected commit; no commit was requested or created.
- Exact next action: inventory synchronous project-opening and explorer loads,
  define immutable worker results and generation-based stale-result rejection,
  and add a responsiveness test proving the screen mounts before discovery
  completes.

### S027 - 2026-07-20 - Move project and explorer loads to workers

- Checkpoint state at start: `4.4` was `not_started`; Checkpoint 4.3 was
  verified in S025 and its public model boundary was refined in S026.
- Starting branch, HEAD, and worktree: `mth5` at owner commit `ff155bb`. The
  verified S026 explorer model refinement was uncommitted, and the required
  empty `pyrefly-baseline.json` remained untracked; both were preserved.
- Session objective: display a responsive project surface before synchronous
  discovery completes, move project/explorer loads off the UI thread, load
  inactive tabs lazily, and reject cancelled or stale results safely.
- Work completed: added an immediate `ProjectLoadingScreen`; Textual worker
  boundaries for project opening and each explorer section; private immutable
  project-open and explorer result messages; and UI-thread-only result
  handlers. Project overview loads first, while Data, Flows, Parameters,
  Criteria, and Jobs load only when activated. Data discovery preloads stable
  run summaries so later plot-target selection remains cache-only. Generation
  tokens reject stale refreshes and superseded projects, whose MTH5 handles are
  closed. A dedicated executor per Textual async worker avoids blocking event-
  loop shutdown on active synchronous work. Successful resource loads now
  replace their loading text with the correct selected/empty placeholder.
- Cache concurrency: made `ProjectExplorerIndex` safe for overlapping worker
  reads with a private lock and section/job epochs. An invalidation can proceed
  without waiting for an old filesystem or MTH5 call, and that old result can
  no longer repopulate current cache state. Parsed file-identity records remain
  reusable across invalidation as established in Checkpoint 4.3.
- Test coverage: added deterministic blocked-load tests proving that the
  project loading screen and overview loading state render before project and
  MTH5 discovery are released; inactive Flow YAML is not parsed before its tab
  activates; late explorer refreshes cannot replace newer UI state; an
  invalidated concurrent cache load cannot recapture stale project state; and
  a superseded project result is closed without replacing the current project.
  A held-open discovery also proves that explorer navigation defers MTH5
  closure until the worker exits rather than closing a handle in active use.
  Existing TUI tests were updated to await explicit lazy-load completion and to
  assert final empty-resource messages after asynchronous deletion refreshes.
- Files changed: `resistics/explorer.py`, `resistics/tui.py`,
  `tests/test_explorer.py`, `tests/test_tui.py`, and this implementation record.
  The first four also contain the verified uncommitted S026 refinement where
  applicable; the untracked Pyrefly baseline was not changed.
- Decisions added or superseded: D041 records the Textual worker, dedicated
  executor, immutable-message, lazy-loading, generation-rejection, project
  cleanup, and cache-epoch boundaries.
- Verification commands and results:
  - The explorer suite passed 9 tests; the TUI suite passed 35 tests.
  - The complete suite passed 403 tests in 24.82 seconds.
  - Ruff format checked 42 maintained Python files and Ruff lint passed.
    Pydoclint reported no unbaselined violations; Pyrefly reported zero
    error-level diagnostics with the two established suppressions.
  - All seven configured pre-commit hooks passed, and `git diff --check`
    passed.
- Measurements/artifacts: before implementation, project opening and overview
  discovery completed synchronously before their screens could respond. The
  deterministic S027 workload holds each call in a thread: the loading surface
  remains queryable while both releases are unset. The same workload records
  zero Flow parses during Project-tab loading and exactly one after Flow-tab
  activation. Timing was deliberately not made a brittle test threshold.
- Known failures or incomplete work: none within Checkpoint 4.4. Heavy import
  cost and first-screen timing remain assigned to Checkpoint 4.5.
- Checkpoint state at end: `4.4` is `verified`; Checkpoint 4.5 is next.
- Commit readiness or commit id: the S026 explorer boundary and S027 worker
  implementation, concurrency hardening, regression tests, and record are
  verified and ready for an owner-selected commit; no commit was requested or
  created.
- Exact next action: remeasure the five-sample cold `resistics.tui` import and
  first-screen baseline, profile the import graph, then inventory Plotly,
  Matplotlib, SciPy, ObsPy, and processing imports for lazy feature boundaries.

### S028 - 2026-07-21 - Defer feature-specific TUI imports

- Checkpoint state at start: `4.5` was `not_started`; Checkpoint 4.4 was
  verified in S027. The tracker was moved to `in_progress` before production
  edits.
- Starting branch, HEAD, and worktree: `mth5` at owner commit `ff155bb`. The
  verified S026-S027 explorer and worker changes were uncommitted, and the
  required empty `pyrefly-baseline.json` remained untracked; both were
  preserved.
- Session objective: remove feature-specific services from the TUI startup
  path, report missing dependencies at the action boundary, and exceed the
  Phase 4 cold-import reduction target without weakening feature coverage.
- Work completed: limited module startup to the standard library, Loguru, and
  Textual; moved project, MTH5, explorer, job, flow, spectra, regression,
  transfer-function, Plotly, and plot-service imports to their owning screens
  and actions; and kept annotation-only domain imports behind `TYPE_CHECKING`.
  Added one feature-error formatter that explains how to recover from a
  missing required dependency while preserving other error detail. Updated
  tests to patch services in their owning modules, removing dependence on the
  former eager TUI aliases.
- Test coverage: added a fresh-subprocess module-inventory regression covering
  fourteen heavy feature/module roots and a focused missing-dependency/error-
  preservation contract. All existing project opening, explorer, plotting,
  processing, and project-creation TUI paths continue to run through the full
  pilot suite.
- Files changed: `resistics/tui.py`, `tests/test_tui.py`, and this
  implementation record. Ignored before/after reports were written below
  `.artifacts/hardening/performance/`; the untracked Pyrefly baseline was not
  changed.
- Decisions added or superseded: D042 records the lightweight module boundary,
  feature-local import policy, actionable error handling, owning-module test
  seams, and non-brittle regression strategy.
- Verification commands and results:
  - The focused TUI suite passed 37 tests; the complete suite passed 405 tests
    in 25.82 seconds.
  - Ruff format checked 44 maintained Python files and Ruff lint passed.
    Pydoclint reported no unbaselined violations; Pyrefly reported zero
    error-level diagnostics with the two established suppressions.
  - All seven configured pre-commit hooks passed.
- Measurements/artifacts: on CPython 3.13.5 under WSL2, the same-session
  five-sample cold-import baseline had a 2.4280-second median before the change
  and a 0.1748-second median afterward (range 0.1583-0.2059 seconds). Against
  the governing verified 2.2659-second baseline, this is a 92.29% reduction,
  exceeding the 60% Phase 4 gate. The five-sample first-screen median fell from
  2.076730 to 0.235886 seconds (range 0.225711-0.239910 seconds), an 88.64%
  reduction. Reports are
  `.artifacts/hardening/performance/tui-import-4.5-before.json` and
  `tui-import-4.5-after.json` in the same directory. Import profiling confirms
  that Matplotlib, MTH5, mt-io, mt-metadata, Plotly, SciPy, and Resistics domain
  services are absent from a fresh `resistics.tui` import.
- Known failures or incomplete work: none within Checkpoint 4.5. The timing
  values are environment evidence rather than automated thresholds.
- Checkpoint state at end: `4.5` is `verified`; Checkpoint 4.6 is next.
- Commit readiness or commit id: the S026-S028 explorer boundary, worker
  implementation, deferred imports, regression tests, and records are verified
  and ready for an owner-selected commit; no commit was requested or created.
- Exact next action: inventory unconditional `tqdm` rendering in Resistics and
  regressioninc processing paths, map existing `JobProgressEvent` ordering and
  cancellation semantics, then add failing structured-progress tests at the
  lowest shared processing boundary.

### S029 - 2026-07-21 - Replace terminal rendering with structured progress

- Checkpoint state at start: `4.6` was `not_started`; Checkpoint 4.5 was
  verified in S028 and had been committed by the owner.
- Starting branch, HEAD, and worktree: `mth5` at owner commit `09d048b`. Only
  the required empty `pyrefly-baseline.json` was untracked; it was preserved.
- Session objective: remove unconditional terminal progress from processing
  used by Textual, expose one serializable progress/cancellation contract for
  standalone and UI consumers, and verify completion, cancellation, and
  failure ordering.
- Inventory and failing evidence: the sibling regressioninc package contained
  no `tqdm` or terminal renderer. Resistics contained exactly two unconditional
  `tqdm` loops, in gathered-data regression preparation and linear solving.
  Flow execution already emitted untyped lifecycle dictionaries and job
  execution translated them into Pydantic events, but inner numerical loops
  could neither report structured progress nor observe cancellation. Initial
  tests failed at collection because the shared event/cancellation contract did
  not exist.
- Work completed: added frozen Pydantic processing progress events with stable
  lifecycle state, task, current/total, message, flow identity, and error
  fields; a shared cancellation exception and callback aliases; and opt-in
  callback forwarding from `ResisticsProcess.execute`. Flow execution now
  emits and enriches typed node/process events. Job events retain the nested
  processing event, while the TUI displays native current/total counters and
  task status through its existing thread-safe activity surface. Gathered and
  spectra regression preparation plus linear solving use a private dataclass
  reporter and check cancellation before each evaluation frequency.
- Dependency cleanup: removed `tqdm` and `types-tqdm` from project metadata.
  Lock refresh also removed the now-orphaned `types-requests`; 174 packages
  remain resolved. Ruff now recognises pydoclint's external `DOC` codes so six
  narrow definition-line DOC105 suppressions remain valid for callback aliases
  that pydoclint cannot resolve. Explicit pydoclint regeneration removed 17
  stale entries, leaving 1,697 baseline lines.
- Test coverage: added schema/JSON round-trip and start-advance-complete tests;
  direct cancellation and failure tests; flow callback forwarding and identity
  enrichment; job-level cancellation/failure ordering and terminal events; and
  TUI current/total rendering. The focused flow, job, regression, and TUI
  suites passed 97 tests.
- Files changed: `resistics/common.py`, `resistics/flow.py`,
  `resistics/job.py`, `resistics/regression.py`, `resistics/tui.py`, their four
  focused test modules, `pyproject.toml`, `uv.lock`,
  `pydoclint-baseline.txt`, and this implementation record. Regressioninc was
  read-only because its inventory was already clean. The untracked Pyrefly
  baseline was not changed.
- Decisions added or superseded: D043 records the event model, callback
  forwarding, private reporter, dependency removal, sibling no-op, and narrow
  pydoclint/Ruff interoperability decision.
- Verification commands and results:
  - The complete suite passed 411 tests in 24.73 seconds.
  - Ruff format checked 44 maintained Python files and Ruff lint passed.
    Pydoclint reported no unbaselined violations; Pyrefly reported zero
    error-level diagnostics with the two established suppressions.
  - `uv lock --check` passed with 174 packages; source, project metadata, and
    both repository inventories contain no `tqdm` reference.
  - All seven configured pre-commit hooks passed, and `git diff --check`
    passed.
- Measurements/artifacts: two raw terminal renderers and three resolved
  renderer/stub packages were removed. The replacement contract is protected
  by six tests across direct numerical, flow, job, and TUI layers rather than
  timing thresholds.
- Known failures or incomplete work: none within Checkpoint 4.6. Standalone
  callers may adapt the renderer-neutral callback to any progress library;
  core deliberately provides no terminal renderer.
- Checkpoint state at end: `4.6` and the Phase 4 review gate are `verified`;
  Checkpoint 5.1 is next.
- Commit readiness or commit id: the S029 structured-progress implementation,
  dependency cleanup, lock/baseline refresh, regression tests, and record are
  verified and ready for an owner-selected commit; no commit was requested or
  created.
- Exact next action: inventory the public names, module globals, internal
  imports, and `resistics.tui:main` entry point that the Phase 5.1 package
  facade must preserve before moving code mechanically.

### S030 - 2026-07-21 - Introduce the TUI package facade

- Checkpoint state at start: `5.1` was `not_started`; Checkpoint 4.6 and the
  Phase 4 review gate were verified in S029.
- Starting branch, HEAD, and worktree: `mth5` at owner commit `09d048b`. The
  verified S029 structured-progress implementation was uncommitted, and the
  required empty `pyrefly-baseline.json` remained untracked; both were
  preserved.
- Session objective: convert the single TUI module into the planned package
  skeleton without changing application behavior, while preserving public
  imports and the installed `resistics.tui:main` launcher.
- Inventory and failing evidence: the supported module-owned surface comprised
  four constants/type aliases, fourteen screens/application records, and the
  two launch functions. Tests used two private helpers and one launcher
  monkeypatch seam. A new facade contract initially failed twice because
  `resistics.tui` was not yet a package.
- Work completed: mechanically moved the complete implementation to
  `resistics/tui/app.py`; added an explicit facade re-exporting only TUI-owned
  public names; and created documented `state.py`, `services.py`, and
  `screens/{dialogs,launcher,project}.py` extraction boundaries. Private-helper
  tests now import their owning module. The launcher test patches the owning
  app seam, while new contracts prove facade identity, package location, and
  installed console-entry-point resolution. Updated the exact Ruff per-file
  path and regenerated the line/path-sensitive pydoclint baseline without
  changing its 1,697-line count.
- Files changed by this checkpoint: deletion/move of `resistics/tui.py`, the
  eight new files below `resistics/tui/`, `tests/test_cli.py`,
  `tests/test_tui.py`, `pyproject.toml`, `pydoclint-baseline.txt`, and this
  implementation record. Other dirty source, test, metadata, and lock changes
  remain the verified S029 work. The Pyrefly baseline was not changed.
- Decisions added or superseded: D044 records the explicit facade, single-app
  implementation, placeholder boundaries, private-test ownership, and
  path-sensitive tooling policy.
- Verification commands and results:
  - The launcher and TUI suites passed 41 tests; the complete suite passed 413
    tests in 26.34 seconds.
  - Ruff format checked 51 maintained Python files and Ruff lint passed.
    Pydoclint reported no unbaselined violations; Pyrefly reported zero
    error-level diagnostics with the two established suppressions.
  - All seven configured pre-commit hooks passed, and `git diff --check`
    passed.
  - `uv build` produced the wheel and source distribution. Both contain the
    complete eight-file TUI skeleton and no legacy `resistics/tui.py`; the
    wheel entry-point metadata remains `resistics = resistics.tui:main`, and a
    direct wheel import resolved the facade and app classes successfully.
- Measurements/artifacts: five independent CPython 3.13.5/WSL2 imports ranged
  from 0.1544 to 0.2159 seconds with a 0.1606-second median, compared with the
  0.1748-second S028 median. The ignored report is
  `.artifacts/hardening/performance/tui-import-5.1-after.json`; the facade adds
  no measurable startup regression and the heavy-import test remains green.
- Known failures or incomplete work: none within Checkpoint 5.1. The skeleton
  modules intentionally contain no classes until their assigned extraction
  checkpoints.
- Checkpoint state at end: `5.1` is `verified`; Checkpoint 5.2 is next.
- Commit readiness or commit id: the combined verified S029-S030 structured
  progress, dependency cleanup, package facade, artifacts, tests, baselines,
  and records are ready for an owner-selected commit; no commit was requested
  or created.
- Exact next action: map dialog and launcher dependencies on app helpers and
  types, then move the lowest-coupling screens into `screens/dialogs.py` and
  `screens/launcher.py` while preserving widget ids, bindings, and facade
  exports.

### S031 - 2026-07-21 - Extract dialogs and launcher screens

- Checkpoint state at start: `5.2` was `not_started`; Checkpoint 5.1 had
  introduced and verified the package facade and extraction skeleton in S030.
- Starting branch, HEAD, and worktree: `mth5` at owner commit `09d048b` with
  the verified S029-S030 structured-progress and TUI package work uncommitted.
  The required empty `pyrefly-baseline.json` remained untracked and untouched.
- Session objective: move the low-coupling dialogs and pre-project launcher
  screens to their stable modules without changing Textual behavior or public
  imports.
- Work completed: moved eight modal dialogs, their filename validator, and the
  immutable deletion request to `screens/dialogs.py`; moved `TuiHeader`,
  `HomeScreen`, `CreateProjectScreen`, and `ProjectLoadingScreen` to
  `screens/launcher.py`; and moved shared feature-error, focus, and app-contract
  helpers to `services.py`. The app imports the new owners, the facade and
  transitional app aliases retain object identity, and the app type check uses
  a function-local import to avoid an app/screens import cycle. A failing-first
  module-ownership test protects the new boundary. The Textual-specific Ruff
  exceptions now cover screen modules, and the line/path-sensitive pydoclint
  baseline was regenerated from 1,697 to 1,701 lines.
- Files changed by this checkpoint: `resistics/tui/app.py`,
  `resistics/tui/services.py`, `resistics/tui/screens/dialogs.py`,
  `resistics/tui/screens/launcher.py`, `tests/test_cli.py`, `tests/test_tui.py`,
  `pyproject.toml`, `pydoclint-baseline.txt`, and this implementation record.
  Earlier verified dirty files and the Pyrefly baseline were preserved.
- Decisions added or superseded: D045 records screen ownership, shared-helper
  cycle avoidance, compatibility aliases, mechanical scope, and the cohesive
  dialogs-module size exception.
- Verification commands and results:
  - The new ownership contract initially failed because the target modules were
    placeholders. The focused CLI/TUI suites then passed 42 tests, and the
    complete suite passed 414 tests in 25.73 seconds.
  - Ruff format checked 51 maintained Python files and Ruff lint passed.
    Pydoclint reported no unbaselined violations; Pyrefly reported zero errors
    with the two established suppressions. `uv lock --check` resolved 174
    packages, all seven pre-commit hooks passed, and `git diff --check` passed.
  - `uv build` produced the wheel and source distribution. Both contain the
    extracted modules; wheel metadata retains
    `resistics = resistics.tui:main`, and a direct wheel import proved facade
    identity plus the new dialog and launcher `__module__` ownership.
- Measurements/artifacts: the app fell from 3,592 to 2,565 lines; dialogs are
  693 lines, launcher 309, and shared services 94. Five independent CPython
  3.13.5/WSL2 cold imports ranged from 0.1334 to 0.1675 seconds with a 0.1377
  second median, improving on S030's 0.1606-second median. The ignored report
  is `.artifacts/hardening/performance/tui-import-5.2-after.json`; build
  artifacts are under `/tmp/resistics-checkpoint-5-2-dist-20260721/`.
- Known failures or incomplete work: none within Checkpoint 5.2. The empty
  legacy notebook and unrelated notebook lint debt remain assigned to later
  hardening checkpoints and were not changed. Project-screen state/services
  extraction remains Checkpoint 5.3.
- Checkpoint state at end: `5.2` is `verified`; Checkpoint 5.3 is ready.
- Commit readiness or commit id: the combined verified S029-S031 structured
  progress, dependency cleanup, TUI package and screen-boundary work, tests,
  artifacts, baselines, and records are ready for an owner-selected commit; no
  commit was requested or created.
- Exact next action: inventory `ProjectExplorerScreen` state, indexing,
  validation, plotting-target, deletion-preview, and execution responsibilities
  and select the first UI-neutral service contract for failing-first unit tests.

### S032 - 2026-07-21 - Extract project screen state and services

- Checkpoint state at start: `5.3` was `in_progress`; Checkpoint 5.2 had been
  committed by the owner at `8b84d31` after its verified extraction work.
- Starting branch, HEAD, and worktree: `mth5` at `8b84d31` with only the
  required empty `pyrefly-baseline.json` untracked and untouched.
- Session objective: separate project discovery and mutations from Textual
  presentation, keep service APIs usable without constructing an application,
  and retain the project-screen pilot coverage.
- Work completed: introduced `ProjectExplorerService` as the owner of explorer
  indexing, project state, resource and job validation, YAML copy/write/delete,
  template restoration, deletion options and previews, plot-target resolution,
  plot construction, job execution, cancellation, and project lifecycle. Moved
  shared typed state to `state.py`; the public deletion request is now a frozen
  Pydantic model and private worker/action records remain dataclasses. Reduced
  `app.py` to application and project-open orchestration; made
  `screens.project` the concrete screen owner; and separated private data, job,
  and resource presentation mixins without changing widget ids, bindings,
  focus, lazy-worker, or generation-rejection behaviour. Removed unsupported
  internal screen index/job aliases while preserving the public facade and
  transitional app identities. Added service-only tests plus explicit DTO and
  module-ownership contracts, and retained the Textual pilot tests.
- Files changed by this checkpoint: `resistics/tui/app.py`,
  `resistics/tui/state.py`, `resistics/tui/services.py`,
  `resistics/tui/screens/dialogs.py`, `resistics/tui/screens/project.py`, the
  new `project_base.py`, `project_data.py`, `project_jobs.py`, and
  `project_resources.py` screen modules, `tests/test_cli.py`,
  `tests/test_tui.py`, the new `tests/test_tui_services.py`,
  `pydoclint-baseline.txt`, and this implementation record.
- Decisions added or superseded: D046 records the UI-neutral service contract,
  public-Pydantic/private-dataclass boundary, narrowed D042 import rule,
  presentation ownership, compatibility surface, legacy internal-alias
  removal, and module-size limits.
- Verification commands and results:
  - Three direct service tests passed without constructing a Textual app; the
    combined service/CLI/TUI selection passed 45 tests, and the final complete
    suite passed 417 tests in 26.49 seconds.
  - Ruff format checked 56 maintained Python files and Ruff lint passed.
    Pydoclint passed against its regenerated 1,761-line baseline; Pyrefly
    reported zero errors with the two established suppressions. `uv lock
    --check` resolved 174 packages, all seven pre-commit hooks passed, and
    `git diff --check` passed.
  - `uv build` produced the wheel and source distribution. Both contain every
    new TUI module; wheel metadata retains
    `resistics = resistics.tui:main`, and a direct wheel import proved facade
    identity, concrete screen/service ownership, and the frozen Pydantic DTO
    schema contract.
- Measurements/artifacts: `app.py` fell from 2,565 to 388 lines; the concrete
  project screen is 492 lines, its private presentation modules range from 88
  to 756 lines, services are 642, and state is 118. All remain below the Phase
  5 ceiling. Five independent CPython 3.13.5/WSL2 cold imports ranged from
  0.2526 to 0.2908 seconds with a 0.2553-second median. The intentional eager
  public Pydantic DTO raises the S031 timing while remaining 88.7% faster than
  the 2.2659-second verified baseline and retaining deferred feature imports.
  The ignored report is
  `.artifacts/hardening/performance/tui-import-5.3-after.json`; build artifacts
  are under `/tmp/resistics-checkpoint-5-3-dist-20260721/`.
- Known failures or incomplete work: none within Checkpoint 5.3. The empty
  Pyrefly artifact remains intentionally untracked. Shared flow/job graph
  rendering remains Checkpoint 5.4.
- Checkpoint state at end: `5.3` is `verified`; Checkpoint 5.4 is ready.
- Commit readiness or commit id: the verified Checkpoint 5.3 service/screen
  extraction, tests, artifacts, baseline, and record are ready for an
  owner-selected commit; no commit was requested or created.
- Exact next action: compare `plot_flow` and `plot_job` in `resistics/plot.py`,
  capture shared rendering invariants in focused tests, and define the smallest
  shared graph-rendering contract before extracting it.

### S035 - 2026-07-21 - Split gather responsibilities

- Checkpoint state at start: Checkpoint 5.4 was committed and verified;
  Checkpoint 5.5 was ready.
- Starting branch, HEAD, and worktree: `mth5` at `c8b938f` with only the
  required empty `pyrefly-baseline.json` untracked. No unrelated tracked change
  was present or modified.
- Session objective: separate criteria/validation, MTH5/project discovery,
  planning, and data assembly; replace the monolithic gather flow with named
  domain operations; and narrow exception handling without changing its public
  API or selection semantics.
- Work completed: reduced `resistics.gather` to a public facade and 45-line
  `Gather.run` orchestration. Moved public criteria and selection contracts to
  `gather_criteria`, gathered-data models and assembly to `gather_data`,
  project artifact discovery and validation to `gather_project`, and immutable
  evaluation/window planning to `gather_plan`. Added `load_target`,
  `discover_remotes`, `admit_windows`, planning, and assembly operations.
  Reader failures now retain artifact role plus station/run/level/evaluation
  context; automatic remote discovery skips only expected candidate
  `ValueError`s and propagates unexpected failures. The facade explicitly
  preserves every established `resistics.gather.*` public identity and process
  descriptor path.
- Files changed by this checkpoint: `resistics/gather.py`, the new
  `resistics/gather_criteria.py`, `resistics/gather_data.py`,
  `resistics/gather_plan.py`, and `resistics/gather_project.py`,
  `tests/test_gather.py`, `pydoclint-baseline.txt`, and this implementation
  record.
- Decisions added or superseded: D049 records the stable public facade,
  responsibility owners, frozen internal planning contract, and exception
  boundary.
- Verification commands and results:
  - All 19 focused gather tests passed. The complete suite passed all 425 tests
    in 26.96 seconds, including new coverage for canonical process paths,
    contextual required-artifact failures, expected automatic-candidate
    rejection, and propagation of unexpected candidate failures.
  - Ruff formatting and lint passed for all changed modules. A dedicated C901
    check reported no findings and the former `Gather.run` suppression is gone.
    Pydoclint passed against a regenerated 1,728-line baseline, down from
    1,745; the new planner, source, and assembler operations add no debt.
    Pyrefly reported zero errors with the two established suppressions.
  - `uv lock --check` resolved 174 packages, `git diff --check` passed, and all
    pre-commit hooks passed: YAML, EOF, whitespace, Ruff lint/format,
    pydoclint, and Pyrefly.
  - The gather API HTML was generated and contains the canonical
    `resistics.gather.*` names. The repository-wide Sphinx invocation still
    exits on established gallery failures from removed legacy imports,
    unavailable intersphinx inventories, and Chrome-dependent Plotly examples;
    none originates in the gather API page or changed files.
  - The initial sandboxed build could not resolve Hatchling without network
    access. The approved retry built the wheel and sdist. Both contain all five
    gather modules, and an import directly from the wheel verified its facade
    exports and canonical process-catalog paths.
- Measurements/artifacts: the former 1,235-line gather module is now a
  183-line facade plus cohesive criteria (247), data/assembly (582), planning
  (312), and project-discovery (645) modules, each below the preferred 800-line
  ceiling. Production Python is 24,575 lines and tests are 8,101 lines. Build
  artifacts are under `/tmp/resistics-checkpoint-5-5-dist-20260721/` only.
- Known failures or incomplete work: none within Checkpoint 5.5. The full
  documentation build retains the established unrelated gallery/environment
  failures described above. The empty Pyrefly artifact remains intentionally
  untracked. MTH5 compatibility placeholders, file ownership, and remaining
  legacy paths remain Checkpoint 5.6.
- Checkpoint state at end: `5.5` is `verified`; Checkpoint 5.6 is ready.
- Commit readiness or commit id: the gather responsibility split, focused
  protections, baseline refresh, artifacts, and record are verified and ready
  for an owner-selected commit; no commit was requested or created.
- Exact next action: inventory `Measurement = None`, `Site = None`, every file
  ownership/close path, and all related callers, tests, docs, and release notes
  before defining the Checkpoint 5.6 cleanup boundary.

### S034 - 2026-07-21 - Unify flow and job graph rendering

- Checkpoint state at start: Checkpoint 5.3 and its diagnostics follow-up were
  committed and verified; Checkpoint 5.4 was ready.
- Starting branch, HEAD, and worktree: `mth5` at `675cfde` with only the
  required empty `pyrefly-baseline.json` untracked. No unrelated tracked change
  was present or modified.
- Session objective: remove the parallel `plot_flow` and `plot_job` Plotly
  construction paths while preserving their public APIs, model-specific
  adaptation, graph geometry, interaction, and presentation details.
- Work completed: added focused shared-renderer assertions for title spacing,
  node-text-over-edge layering, the 13-point data-type labels, parameter and
  work-plan wording, and hover templates without Plotly's extra trace label.
  Added the private frozen `_FlowGraphRenderSpec`; moved the ranked layout,
  routed edges, arrow/card geometry, trace metadata, graph construction, axes,
  legend, and figure layout into the new `resistics.flow_graph` module.
  `plot_flow` and `plot_job` now validate and adapt their own models into that
  immutable presentation contract and delegate rendering. The resolved-job
  adapter continues to own selected stages, expanded parameter/criteria card
  content, scope/work summaries, title wording, and job-specific dimensions.
- Files changed by this checkpoint: `resistics/plot.py`, the new
  `resistics/flow_graph.py`, `tests/test_plot.py`,
  `pydoclint-baseline.txt`, and this implementation record.
- Decisions added or superseded: D048 records the private-dataclass rendering
  boundary, model-adapter ownership, preserved output contract, and transferred
  line/path-sensitive helper docstring debt.
- Verification commands and results:
  - Focused plotting coverage passed all 13 tests; the complete suite passed
    all 421 tests in 27.13 seconds.
  - Ruff format and lint passed. Pydoclint passed against a regenerated
    1,745-line baseline, down from 1,757 before the checkpoint; unchanged moved
    helpers account for the new module entries while the render contract,
    materially changed layout, and both public adapters add no debt. Pyrefly
    passed all production modules with zero errors and the two established
    suppressions.
  - `uv lock --check` resolved 174 packages, `git diff --check` passed, and all
    pre-commit hooks passed: YAML, EOF, whitespace, Ruff lint/format,
    pydoclint, and Pyrefly.
  - The initial sandboxed build could not resolve cached Hatchling because
    network access was unavailable. The approved network-enabled retry built
    both distributions successfully. The wheel and sdist contain
    `resistics/{plot,flow_graph}.py`; direct imports proved both public adapters
    and the private dataclass contract.
- Measurements/artifacts: `plot.py` fell from 1,216 to 700 lines; the cohesive
  private renderer is 617 lines, so both are below the Phase 5 preferred
  ceiling. Production Python is 23,841 lines and tests are 7,943 lines. Build
  artifacts are under `/tmp/resistics-checkpoint-5-4-dist-20260721/` only.
- Known failures or incomplete work: none within Checkpoint 5.4. The empty
  Pyrefly artifact remains intentionally untracked. Gather responsibility
  boundaries remain Checkpoint 5.5.
- Checkpoint state at end: `5.4` is `verified`; Checkpoint 5.5 is ready.
- Commit readiness or commit id: the shared renderer, thin adapters, focused
  protection, baseline refresh, and record are verified and ready for an
  owner-selected commit; no commit was requested or created.
- Exact next action: inventory criteria, validation, MTH5/project discovery,
  planning, and data-assembly responsibilities in `resistics/gather.py`, then
  map their callers and focused tests before defining Checkpoint 5.5 boundaries.

### S033 - 2026-07-21 - Add session diagnostic logs to the TUI

- Checkpoint state at start: Checkpoint 5.3 was verified in S032 and its
  service/screen extraction remained uncommitted; Checkpoint 5.4 was ready.
- Starting branch, HEAD, and worktree: `mth5` at `8b84d31` with the verified
  S032 changes plus the required empty `pyrefly-baseline.json` untracked. The
  existing work was preserved and this feature was implemented as a focused
  5.3 follow-up.
- Session objective: retain MTH5/project-open warnings and the Loguru output
  suppressed while Textual owns the terminal, without mixing diagnostics into
  structured job progress or requiring filesystem persistence.
- Work completed: added the frozen public `DiagnosticLogEntry` contract and a
  private thread-safe 2,000-entry session buffer with incremental cursor reads
  and rollover accounting. The official launcher now redirects Python warnings
  and Loguru `INFO+` records into that buffer, restores the established stderr
  behavior on exit, and reinstalls capture after lazy project/MTH5 imports that
  reconfigure global Loguru handlers. Project-open workers preserve warning
  category and source location in immutable results; accepted results alone add
  their Python warnings to the app session, and direct string-warning callers
  remain compatible. Added a separate focusable Logs tab, safely rendered on
  the UI thread by a 250 ms drain. Startup records remain buffered until the
  Logs tab has a real layout width; writes use that explicit width, and a
  terminal resize triggers a bounded-buffer rerender. Activity remains
  job-specific and can be cleared independently. Diagnostics survive project
  switches but are not persisted; existing per-job files remain unchanged.
- Files changed by this follow-up: `resistics/tui/__init__.py`,
  `resistics/tui/app.py`, `resistics/tui/state.py`, the new
  `resistics/tui/logging.py`, `resistics/tui/screens/launcher.py`,
  `resistics/tui/screens/project.py`, `project_base.py`, the new
  `project_logs.py`, `tests/test_cli.py`, `tests/test_tui.py`, the new
  `tests/test_tui_logging.py`, `pydoclint-baseline.txt`, and this record.
- Decisions added or superseded: D047 records the separate Logs/Activity
  surfaces, public-Pydantic/private-buffer boundary, process-global capture and
  MTH5 reinstall rule, UI-thread rendering, bounded session lifetime, retained
  compatibility, and explicitly deferred persistence/filtering/export scope.
- Verification commands and results:
  - Focused logging, CLI, and TUI coverage passed 46 tests. The complete suite
    passed all 421 tests in 27.30 seconds. The Logs pilot additionally proves a
    diagnostic between 78 columns and the available content width remains on
    one rendered line.
  - Ruff formatting checked 59 maintained Python files and Ruff lint passed.
    Pydoclint passed against a regenerated baseline reduced from 1,761 to 1,757
    lines; Pyrefly reported zero errors with the two established suppressions.
    `uv lock --check` resolved 174 packages and `git diff --check` passed.
  - `uv build` produced the wheel and source distribution. Both contain the
    diagnostic capture and Logs presentation modules; wheel metadata retains
    `resistics = resistics.tui:main`, and a direct wheel import proved the
    frozen DTO schema, facade identity, and usable empty buffer.
- Measurements/artifacts: the concrete project screen is 499 lines, its Logs
  mixin is 63, logging is 322, app is 436, state is 156, and every new module
  remains below the Phase 5 ceiling. Five final CPython 3.13.5/WSL2 cold imports
  ranged from 0.2494 to 0.2856 seconds with a 0.2715-second median, 88.0% faster
  than the 2.2659-second verified baseline. The ignored report is
  `.artifacts/hardening/performance/tui-import-5.3-logs-after.json`; artifacts
  are under `/tmp/resistics-checkpoint-5-3-logs-dist-20260721/`.
- Known failures or incomplete work: none within the diagnostic-logging
  follow-up. General session persistence, filtering, clearing, exporting,
  debug records, and unrelated standard-library logging remain intentionally
  out of scope. Shared flow/job graph rendering remains Checkpoint 5.4.
- Checkpoint state at end: the `5.3` follow-up is `verified`; Checkpoint 5.4
  remains ready.
- Commit readiness or commit id: the combined verified Checkpoint 5.3 service,
  screen, and diagnostics work is ready for an owner-selected commit; no commit
  was requested or created.
- Exact next action: compare `plot_flow` and `plot_job` in `resistics/plot.py`,
  capture shared rendering invariants in focused tests, and define the smallest
  shared graph-rendering contract before extracting it.

### S036 - 2026-07-21 - Complete the MTH5-only project boundary

- Checkpoint state at start: Checkpoint 5.5 was committed and verified;
  Checkpoint 5.6 was ready.
- Starting branch, HEAD, and worktree: `mth5` at `34707a8` with only the
  required empty `pyrefly-baseline.json` untracked. No unrelated tracked change
  was present or modified.
- Session objective: remove verified legacy project compatibility surfaces,
  make MTH5 handle ownership deterministic, and ensure importing project
  metadata does not eagerly initialize the third-party MTH5 stack.
- Work completed: extracted the private `_MTH5Handle` protocol, lazy handle
  construction, failed-open cleanup, and shared ownership lifecycle into
  `project_mth5.py`. `Project` and `MTH5File` now expose `closed`, idempotent
  `close()`, and context-manager semantics. Cached summaries remain available
  after closure; every live group or sample operation requires an open handle.
  Both partial file-open and model-construction failures close acquired handles
  while preserving their original exception. Migrated all maintained TUI and
  service callers to `close()`. Removed the old `close_mth5`, `dir_path`,
  `metadata`, `init(force=...)`, measurement/site path helpers, mask/solution
  naming helpers, unused project exceptions, and the noncanonical results-path
  branch; `output_label` is now required. Removed stale gallery and API pages
  for already-retired quick/config/ASCII/legacy-project flows, updated the
  retained MTH5 examples and project documentation, and added migration notes
  to the changelog. Tracked legacy notebooks remain deliberately owned by
  Checkpoint 5.7.
- Files changed by this checkpoint: `resistics/project.py`, the new private
  `resistics/project_mth5.py`, `resistics/errors.py`, four TUI project owners,
  five focused test modules, `pydoclint-baseline.txt`, `CHANGELOG.rst`, project
  and read examples, Sphinx configuration/API pages, stale gallery deletions,
  and this implementation record.
- Decisions added or superseded: D050 records the sole MTH5 hierarchy, lazy
  structural boundary, public ownership contract, failure cleanup, deliberate
  removals, and Checkpoint 5.7 notebook deferral.
- Verification commands and results:
  - The focused project/error/explorer/TUI set passed all 81 tests. The complete
    suite passed all 423 tests in 43.07 seconds.
  - Ruff formatting and lint passed for the maintained package, tests, and
    gallery sources. Pydoclint passed with its production baseline reduced from
    1,728 to 1,672 lines; the new boundary adds no debt. Pyrefly reported zero
    errors with the two established suppressions. `uv lock --check` resolved
    174 packages and `git diff --check` passed.
  - The repository-wide pre-commit gate passed YAML, EOF, whitespace, Ruff
    lint/format, pydoclint, and Pyrefly.
  - The standard gallery-disabled Sphinx build exited zero. A full gallery run
    during the checkpoint eliminated the old removed-import failures and was
    limited to the two established Chrome/Kaleido-dependent calibration
    examples; offline intersphinx and duplicate-object warnings remain outside
    this checkpoint.
  - `uv build` produced the wheel and source distribution under
    `/tmp/resistics-checkpoint-5-6-dist-20260721/`. Both contain `project.py`,
    `project_mth5.py`, and `py.typed`. A direct wheel import proved the MTH5
    stack remains deferred, new lifecycle API is present, removed compatibility
    APIs are absent, and `output_label` is required.
- Measurements/artifacts: production Python is 24,784 lines and tests are
  8,192 lines. The former 1,224-line project module is now a 1,209-line public
  project module plus a cohesive 291-line private MTH5 boundary. Build and docs
  artifacts exist only under `/tmp`.
- Known failures or incomplete work: none within Checkpoint 5.6. Repository-root
  Ruff continues to expose pre-existing tracked notebook formatting, legacy
  import, and zero-byte JSON findings; those notebooks, empty `resq.py`, and
  test-helper cleanup are explicitly Checkpoint 5.7. The required empty
  `pyrefly-baseline.json` remains intentionally untracked.
- Checkpoint state at end: `5.6` is `verified`; Checkpoint 5.7 is ready.
- Commit readiness or commit id: the MTH5 ownership boundary, removals, tests,
  examples, documentation, baseline, artifacts, and record are verified and
  ready for an owner-selected commit; no commit was requested or created.
- Exact next action: inventory the empty `resq.py`, zero-byte notebook,
  commented time-processing blocks, unused helpers, and every public-doctest
  versus assertion-factory caller in `resistics/testing.py` before defining
  the Checkpoint 5.7 deletion and relocation boundary.

### S037 - 2026-07-21 - Remove verified dead and test-only code

- Checkpoint state at start: Checkpoint 5.6 was committed and verified;
  Checkpoint 5.7 was ready.
- Starting branch, HEAD, and worktree: `mth5` at `155d53b` with only the
  required empty `pyrefly-baseline.json` untracked. No unrelated tracked change
  was present or modified.
- Session objective: remove or relocate every governing Checkpoint 5.7
  candidate only after repository-wide code, test, documentation, notebook,
  history, and release-note searches established its owner or absence.
- Work completed: deleted the empty `resq` namespace, all eight unreferenced
  exploratory/empty/obsolete notebooks, their unused FDSN request CSV, the
  unreferenced 673-line evaluation fixture module, and the 153-line commented
  `Join` prototype. Reduced installed `resistics.testing` to the stable builders
  imported by executable documentation. Moved still-used linear time data,
  evaluation data, random transfer-function/solution factories, and solution
  comparison into the new test-owned `tests.synthetic_data`; removed unused
  history, multilevel-spectra, and time-comparison helpers. Removed four unused
  flow builders plus `standard_mt_flow`, migrated maintained tests and fixtures
  to canonical `single_site_mt_flow`, and removed the always-true criteria hook
  and its allocation/copy pass from persisted evaluation gathering. Added
  explicit boundary tests, testing-module documentation, and next-release
  removal notes. The obsolete package-specific Ruff exception disappeared with
  its assertion/random machinery.
- Files changed by this checkpoint: production flow, gather criteria/data,
  testing, and time modules; deleted `resq.py`, all tracked notebook assets, and
  `tests/testing_data_evals.py`; the new `tests/synthetic_data.py`; seven test
  modules plus migrated explorer/TUI fixture callers; Ruff configuration,
  pydoclint baseline, changelog, testing API page, and this implementation
  record.
- Decisions added or superseded: D051 records the installed-doctest versus
  suite-only testing boundary, canonical flow builder, verified deletions, and
  dependency-group deferral to Checkpoint 6.1.
- Verification commands and results:
  - Focused testing/numerical/gather/regression/flow coverage passed 71 tests;
    the flow/explorer/TUI migration set passed 66 tests. The complete suite
    passed all 422 tests in 27.28 seconds.
  - Repository-root Ruff formatting and lint passed all 64 maintained Python
    files with no notebook exclusions or failures. Pydoclint passed with its
    production baseline reduced from 1,672 to 1,637 lines. Pyrefly reported
    zero errors with the two established suppressions; `uv lock --check`
    resolved 174 packages and `git diff --check` passed.
  - The standard gallery-disabled Sphinx build exited zero and generated the
    reduced testing API. Established offline intersphinx, duplicate-object, and
    ignored stale-gallery warnings remain unrelated to this checkpoint.
  - The sandboxed build initially failed while resolving isolated Hatchling;
    the approved network-enabled retry produced wheel and sdist under
    `/tmp/resistics-checkpoint-5-7-dist-20260721/`. The sdist contains the
    test-owned synthetic factory; neither artifact contains `resq`, notebooks,
    or obsolete assets. A direct wheel import proved retained doctest builders
    work and every removed flow/testing API is absent.
- Measurements/artifacts: production Python is 24,071 lines and tests are
  7,757 lines. `resistics.testing` fell from 1,296 to 816 lines, its cohesive
  suite-only replacement is 195 lines, and `time.py` fell from 2,718 to 2,565
  lines. The checkpoint removes 3,336 tracked lines overall before its focused
  tests and release documentation. Build and docs artifacts exist only under
  `/tmp`.
- Known failures or incomplete work: none within Checkpoint 5.7 or the Phase 5
  review gate. With no tracked notebooks remaining, the `notebooks` dependency
  group is a deliberate Checkpoint 6.1 audit candidate rather than a metadata
  change hidden in this cleanup. The required empty `pyrefly-baseline.json`
  remains intentionally untracked.
- Checkpoint state at end: `5.7` and the Phase 5 review gate are `verified`;
  Checkpoint 6.1 is ready.
- Commit readiness or commit id: the dead/test-only cleanup, canonical caller
  migrations, focused protections, documentation, baseline, artifacts, and
  record are verified and ready for an owner-selected commit; no commit was
  requested or created.
- Exact next action: map every direct runtime dependency to production imports
  and every optional/development dependency to its docs, test, notebook, or
  tooling owner before proposing Checkpoint 6.1 metadata changes.

### S038 - 2026-07-21 - Audit and reduce direct dependencies

- Checkpoint state at start: Checkpoint 5.7 and Gate 5 were committed and
  verified; Checkpoint 6.1 was ready.
- Starting branch, HEAD, and worktree: `mth5` at `d86c59a` with only the
  required empty `pyrefly-baseline.json` untracked. No unrelated tracked change
  was present or modified.
- Session objective: map every direct and grouped dependency to a maintained
  owner, remove declarations without one, make runtime lower bounds credible
  for Python 3.11 and current APIs, and verify the smaller built distributions.
- Work completed: mapped all 14 retained runtime dependencies and all four uv
  groups. Removed direct ObsPy, prettyprinter, and scikit-learn declarations;
  scikit-learn remains correctly owned by regressioninc. Replaced the sole
  prettyprinter call with a small internal JSON-compatible renderer that
  preserves every established `summary()` doctest. Removed the now-ownerless
  notebook group plus pytest-html, IPython, nbformat, seedir, and emoji tooling
  declarations, including the corresponding unused entries in the transitional
  documentation requirements file. Kept Plotly/Textual as public runtime
  behavior and Matplotlib as shared docs/test support. Rebased legacy NumPy,
  SciPy, Pandas, Loguru, PyYAML, MTH5, Xarray, regressioninc, and Matplotlib
  lower bounds; relaxed unsupported exact pins for attotime and fast-sugiyama;
  retained only the reproduced tsdownsample cap. Refreshed the lock and release
  notes and recorded the permanent ownership map above.
- Files changed: `pyproject.toml`, `uv.lock`, `resistics/common.py`,
  `docs/requirements.txt`, `CHANGELOG.rst`, and this implementation record.
- Decisions added or superseded: D052 records direct-dependency ownership,
  runtime versus group boundaries, lower-bound scope, the preserved summary
  output, and the sole retained evidence-backed upper constraint.
- Verification commands and results:
  - `uv lock` and `uv sync --locked --all-groups` succeeded; the lock resolves
    128 packages. `uv tree --locked --depth 2 --no-dev` confirms all retained
    runtime/group owners and scikit-learn only below regressioninc.
  - The six summary-heavy doctest modules passed all 59 tests. The normal full
    suite passed all 422 tests in 33.61 seconds.
  - Ruff format/lint, pydoclint, and Pyrefly passed; Pyrefly reported zero
    errors with the two established suppressions. The package-legacy checker
    and `git diff --check` passed. The repository-wide pre-commit gate passed
    after the approved retry allowed its EOF hook to open read-only `.agents`
    records; the initial managed-sandbox run had already passed every
    substantive Python and whitespace hook.
  - The gallery-disabled Sphinx build succeeded with the established 85
    offline-intersphinx, duplicate-object, and stale-gallery warnings.
  - The OSV-backed `uv audit --locked` found no known vulnerabilities or
    adverse project statuses in 127 audited packages.
  - `uv build --no-sources` produced the wheel and sdist. Artifact inspection
    proved all 14 intended `Requires-Dist` entries, absence of the three removed
    direct dependencies, presence of `py.typed`, and preserved wheel-loaded
    model-summary behavior.
- Measurements/artifacts: the lock fell from 174 to 128 packages (26.4%); the
  synced `.venv` fell from 910 MB to 744 MB (18.2%). Build artifacts are under
  `/tmp/resistics-checkpoint-6-1-dist-20260721/`; generated documentation and
  coverage reports remain ignored under `.artifacts/hardening/`.
- Known failures or incomplete work: the exact coverage command exceeded its
  75.95% threshold at 80.59%, but two existing TUI tests fail only under full
  coverage instrumentation: the 5,200-check synthetic timing assertion takes
  about 3 seconds against a 2-second non-instrumented limit, and one async
  teardown assertion observes project closure too early. Both pass in the
  normal suite and are unrelated to dependency behavior; this instrumentation
  sensitivity remains visible for the final local-gate review. Exact declared
  minimum-version execution is deliberately Checkpoint 6.2, not claimed here.
- Checkpoint state at end: `6.1` is `verified`; Checkpoint 6.2 is ready.
- Commit readiness or commit id: the dependency map, metadata cleanup, lock,
  compatibility-preserving summary renderer, release notes, artifacts, and
  record are verified and ready for an owner-selected commit; no commit was
  requested or created.
- Exact next action: design the repeatable Checkpoint 6.2 command that resolves
  the lowest practical direct versions on Python 3.11 and tests paired built
  resistics/regressioninc distributions rather than editable source checkouts.

### S039 - 2026-07-21 - Verify modern dependency floors from paired wheels

- Checkpoint state at start: Checkpoint 6.1 was committed and verified;
  Checkpoint 6.2 was ready.
- Starting branch, HEAD, and worktree: `mth5` at `dfa45fe` with only the
  required empty `pyrefly-baseline.json` untracked. No unrelated tracked change
  was present or modified.
- Session objective: add a repeatable local compatibility command that proves
  Resistics' declared direct floors on its minimum Python using paired built
  distributions without mutating the normal locked environment.
- Work completed: prototyped uv `lowest-direct` resolution and demonstrated
  that the former `pydantic>=2.0` declaration could not import the current
  `JsonValue` API. Following owner direction to prefer current dependencies
  rather than old-version compatibility, moved Resistics to Python 3.12-3.14
  and raised every runtime and uv-group floor to its current locked release.
  Removed the speculative tsdownsample upper cap and refreshed the legacy docs
  requirements to the same current documentation stack. Updated Ruff and
  Pyrefly to target Python 3.12 and migrated the type aliases and generic
  functions newly exposed by that target to native Python 3.12 syntax. Added
  `scripts/check_minimum_dependencies.py`: it builds both local wheels, compiles
  runtime/shared/test requirements with `lowest-direct`, rejects any floor that
  silently resolves higher, creates a disposable Python 3.12 environment,
  installs both wheels, checks dependency consistency and source isolation, and
  executes installed-package doctests. Documented the command and policy in the
  README and refreshed the lock and release notes.
- Files changed: `pyproject.toml`, `uv.lock`, `README.md`, `CHANGELOG.rst`,
  `docs/requirements.txt`, the new minimum-dependency checker, five production
  modules using Python 3.12 typing syntax, and this implementation record.
- Decisions added or superseded: D053 supersedes D052's old-version floors and
  tsdownsample cap while retaining direct/transitive ownership and paired-wheel
  isolation.
- Verification commands and results:
  - The documented Python 3.12 command built both wheels, resolved 72 packages,
    selected all 17 checked runtime/shared/test requirements at their exact
    declared floors, reported a consistent 73-package installed environment,
    imported both packages from site-packages, and passed all 89 installed
    Resistics/RegressionInC doctests in 3.78 seconds.
  - The normal locked environment synced successfully and the complete source
    suite passed all 422 tests in 27.18 seconds.
  - Ruff format/lint, pydoclint, and Python-3.12-targeted Pyrefly passed with
    zero errors and the two established suppressions. The repository-wide
    pre-commit gate passed every YAML, EOF, whitespace, Ruff, pydoclint, and
    Pyrefly hook; the legacy-packaging and diff checks also passed.
  - The gallery-disabled Sphinx build succeeded with the established offline
    inventory/stale-gallery warnings. The normal wheel and sdist built under
    `/tmp/resistics-checkpoint-6-2-dist-20260721/`; wheel metadata declares
    Python 3.12-3.14, all modern runtime floors, no tsdownsample upper bound,
    and the `py.typed` marker. The final isolated build required the approved
    network retry when Hatchling was absent from the sandbox-visible cache.
- Measurements/artifacts: removing Python 3.11 lock variants reduced the normal
  lock from 128 to 124 packages. The compatibility command uses a temporary
  directory and leaves no environment, requirements file, or distribution in
  the repository; retained package artifacts are under `/tmp` only.
- Known failures or incomplete work: resolving regressioninc's own declared
  direct minima as first-class inputs exposed its stale `statsmodels>=0.13.2`
  floor, which cannot build on Python 3.11 with current isolated build tooling.
  That is evidence for the separate regressioninc repository dependency review,
  not a transitive version Resistics should claim. The prior coverage-only TUI
  timing/teardown sensitivity is unchanged and unrelated to this checkpoint.
- Checkpoint state at end: `6.2` is `verified`; Checkpoint 6.3 is ready.
- Commit readiness or commit id: the modern floors, Python 3.12 contract,
  compatibility command, annotations, documentation, lock, artifacts, and
  record are verified and ready for an owner-selected commit; no commit was
  requested or created.
- Exact next action: define Checkpoint 6.3's local vulnerability policy and add
  repeatable locked audits for every supported Python variant represented by
  the normal lock.

### S040 - 2026-07-22 - Add local dependency security auditing

- Checkpoint state at start: Checkpoint 6.2 was verified but uncommitted;
  Checkpoint 6.3 was ready.
- Starting branch, HEAD, and worktree: `mth5` at `dfa45fe` with the complete
  verified Checkpoint 6.2 dependency-floor and Python-baseline changes still
  uncommitted. The required empty `pyrefly-baseline.json` remained intentionally
  untracked; no existing work was reverted or overwritten.
- Session objective: make vulnerability discovery a repeatable local gate for
  every supported Python variant, define a strict policy for unavailable fixes
  and accepted risks, and durably assign the hosted security follow-ups without
  prematurely recreating hosted workflows.
- Work completed: added `scripts/audit_dependencies.py`, which reads the
  minor-specific Python classifiers, invokes `uv audit --locked` for Python
  3.12, 3.13, and 3.14, and audits the complete default runtime/development/
  documentation/test environment without installing those interpreters. Added
  an empty-by-default accepted-risk registry whose records require an advisory
  ID, mitigation rationale, owner, and review date; duplicate, incomplete, and
  expired records or reviews more than 90 days away fail before auditing, and
  accepted advisories use only `--ignore-until-fixed`. Replaced the
  single-version audit in the documented production gate, explained why the
  networked check is not a pre-commit hook,
  recorded the zero-exception policy, added the release note, and assigned
  Dependabot, action pinning, minimum permissions, and protected trusted
  publishing to the existing owner checkpoints with their consequences.
- Files changed: new `scripts/audit_dependencies.py`, `README.md`,
  `CHANGELOG.rst`, and this implementation record. These changes are stacked on
  the still-uncommitted Checkpoint 6.2 file set.
- Decisions added or superseded: D054 establishes the metadata-derived locked
  audit matrix, narrow accepted-risk mechanism, local-versus-hosted boundary,
  and explicit hosted-security ownership.
- Verification commands and results:
  - The documented audit command passed for Python 3.12, 3.13, and 3.14. Each
    target resolved the 124-package lock and reported no known vulnerability or
    adverse project status across 123 audited packages. No accepted risk was
    configured.
  - A focused policy probe recovered exactly the three supported classifiers
    and proved expired and more-than-90-day accepted-risk records are rejected.
  - The complete source suite passed all 422 tests in 30.68 seconds.
  - Repository-wide pre-commit passed YAML, EOF, whitespace, Ruff format/lint,
    pydoclint, and Pyrefly. The lock check resolved 124 packages.
- Measurements/artifacts: the audit creates no repository artifact or alternate
  environment. OSV results are deliberately live rather than committed and the
  three commands reuse the normal lock resolution.
- Known failures or incomplete work: uv's audit command remains explicitly
  preview-gated, so the wrapper opts into its named preview feature and will
  expose a future CLI change as a failing maintained command. OSV access is
  network-dependent. Hosted monitoring and publication protection do not exist
  yet and remain the owned Checkpoint 1.6/1.7 deferrals listed above. A final
  targeted pre-commit invocation that explicitly named this read-only
  ``.agents`` record hit the established managed-workspace ``rb+`` restriction
  in the external EOF hook; byte inspection and ``git diff --check`` confirmed
  its correct single newline, and the rerun across every normal repository file
  changed by this checkpoint passed all applicable hooks.
- Checkpoint state at end: `6.3` is `verified`; Checkpoint 6.4 is ready.
- Commit readiness or commit id: the local audit command, policy,
  documentation, follow-up ownership, release note, and record are verified and
  ready with the preceding Checkpoint 6.2 work for an owner-selected commit; no
  commit was requested or created.
- Exact next action: inventory every observed uv/PyPI resolution, build, wheel,
  and platform failure from Phases 1 and 6; separate sandbox/network and stale
  metadata failures from native-package compatibility limitations before
  deciding whether Checkpoint 6.4 needs a bounded Pixi trial.

### S041 - 2026-07-22 - Retain uv from compatibility evidence

- Checkpoint state at start: Checkpoint 6.3 was verified and the owner had
  committed Checkpoints 6.2-6.3 at `70a84f2`; Checkpoint 6.4 was ready.
- Starting branch, HEAD, and worktree: `mth5` at `70a84f2` with only the
  required empty `pyrefly-baseline.json` untracked. No unrelated tracked change
  was present or modified.
- Session objective: classify every observed resolution, build, native-wheel,
  and platform failure; verify the current supported matrix strongly enough to
  decide whether a bounded Pixi experiment is justified; and close the local
  Phase 6 gate.
- Failure classification: the old lttbc/NumPy ABI mismatch was the sole prior
  reproduced native-wheel incompatibility and disappeared when S008 replaced
  lttbc with tsdownsample. Repeated Hatchling failures occurred only when the
  managed sandbox could not reach PyPI and every approved network retry built
  successfully. Regressioninc's Python 3.11 statsmodels minimum-build failure
  is stale metadata in the separately owned sibling review. Coverage-sensitive
  TUI assertions and legacy gallery failures are application/test/docs issues,
  not dependency-manager evidence.
- Current platform boundary: a binary-only manylinux 2.28 resolution correctly
  failed because fast-sugiyama 0.5.3 publishes Linux wheels at manylinux 2.34.
  Repeating the matrix at that actual wheel floor passed. Older glibc systems
  remain outside the verified wheel-only target and would require separately
  proving a Rust source build or revisiting the supported-platform decision.
- Work completed: retained uv 0.11.25 as the sole environment manager and did
  not add Pixi, a second lock, or another contributor path because the
  governing trial threshold was not met. Added permanent README guidance for
  the decision and its re-evaluation trigger, recorded the failure taxonomy and
  matrix evidence above, and completed the Phase 6 review gate.
- Files changed: `README.md` and this implementation record only. No package
  source, dependency declaration, lock, or build configuration changed.
- Decisions added or superseded: D055 retains uv, documents the manylinux 2.34
  boundary, distinguishes package-manager evidence from environmental and
  application failures, and defines the future Pixi-trial trigger.
- Verification commands and results:
  - Built the local RegressionInC wheel, then used uv's binary-only resolver
    against Resistics runtime plus every default group and ignored local source
    overrides. All nine Python 3.12/3.13/3.14 by x86-64 manylinux 2.34/macOS/
    Windows cells resolved. Linux/macOS selected 120 distributions and Windows
    selected 123 because of platform support packages. The exact 124-package
    lock also passed dry-run all-group syncs for the same nine cells.
  - Lock artifact inspection confirmed supported-minor Linux, macOS, and
    Windows wheels for the critical NumPy, SciPy, Pandas, h5py, tsdownsample,
    PyProj, Pydantic Core, scikit-learn, and statsmodels native stack.
  - The Python 3.14 paired-wheel floor check built both local wheels, selected
    all 17 declared runtime/shared/test floors exactly, installed a consistent
    73-package environment, imported both projects only from site-packages, and
    passed all 89 installed doctests in 4.23 seconds. Together with S039, both
    supported Python endpoints now have the same installed-wheel evidence.
  - Locked all-group sync resolved 124 packages. The complete source suite
    passed all 422 tests in 29.57 seconds. Repository-wide pre-commit passed
    YAML, EOF, whitespace, Ruff format/lint, pydoclint, and Pyrefly.
  - The Python 3.12-3.14 OSV audit again found no known vulnerability or adverse
    status across 123 audited packages per target and retained zero accepted
    risks. The source distribution and universal wheel rebuilt successfully
    under `/tmp/resistics-checkpoint-6-4-3YY99wIk/resistics/`.
- Measurements/artifacts: the nine resolver outputs and temporary sibling wheel
  are under `/tmp/resistics-checkpoint-6-4-3YY99wIk/`; no generated evidence is
  retained in the repository. The experiment added no environment manager,
  dependency, configuration, or lockfile.
- Known failures or incomplete work: the macOS and Windows evidence proves
  resolution and binary availability, not runtime behavior on those hosts;
  hosted cross-platform execution remains the owned Checkpoint 1.6 deferral.
  ARM wheel metadata exists for the critical native stack but was not promoted
  to an executed support claim. Older-than-glibc-2.34 Linux is not a verified
  wheel-only target. The first sandboxed Python 3.14 endpoint build again
  demonstrated the known network restriction before its approved retry passed.
- Checkpoint state at end: `6.4` and the Phase 6 review gate are `verified`;
  Checkpoint 7.1 is ready.
- Commit readiness or commit id: the environment-manager decision, permanent
  guidance, Phase 6 evidence, artifacts, and record are verified and ready for
  an owner-selected commit; no commit was requested or created.
- Exact next action: add MyST-NB and autodoc2 for a bounded Checkpoint 7.1
  prototype, then compare representative Pydantic, inheritance, alias,
  overload, signature, ``__all__``, cross-reference, and source-link output
  against the current API before selecting one generator.

### S042 - 2026-07-22 - Establish the MyST foundation

- Checkpoint state at start: Checkpoint 6.4 and Gate 6 were verified in S041;
  the owner had not yet committed that work, and Checkpoint 7.1 was ready.
- Starting branch, HEAD, and worktree: `mth5` at `70a84f2` with the verified
  S041 changes to `README.md` and this record uncommitted, plus the required
  empty `pyrefly-baseline.json` intentionally untracked. Those changes were
  preserved.
- Session objective: install and prototype the MyST documentation stack,
  compare static autodoc2 output with the current dynamic API generator, select
  one generator, and prove the mixed MyST/RST path with executable content and
  references before bulk conversion.
- Prototype result: autodoc2 statically found normal production objects and
  produced source links, but it did not reach parity. It omitted facade
  re-exports such as `resistics.gather.GatherCriteria`, collapsed two overloads
  to the implementation signature, rendered the representative Pydantic model
  as ``(/, **data: Any)``, and could not run Matplotlib's fenced ``{plot}``
  because MyST's mock state machine lacks ``insert_input``. Standard autodoc
  preserved the facade identity, both overloads, the aliased Pydantic
  constructor, inherited members, ``__all__`` filtering, and source links.
- Work completed: added MyST-NB 1.4.0 and sphinx-autodoc2 0.5.0 to the uv docs
  group and refreshed the lock. Selected standard autodoc and left autodoc2
  inactive for removal with the transitional stack in Checkpoint 7.5. Added a
  docs-local adapter that wraps regex-selected autodoc content in a nested MyST
  parser while leaving unconverted production docstrings as RST. Its bounded
  Matplotlib bridge parses RST emitted by fenced ``{plot}`` blocks. Added a
  minimal labelled MyST page and a docs-only representative module covering a
  Python 3.12 type alias, Pydantic field alias, inheritance, positional- and
  keyword-only parameters, overloads, ``__all__``, and MyST docstring syntax.
- Files changed: `pyproject.toml`, `uv.lock`, `docs/requirements.txt`,
  `docs/source/conf.py`, `docs/source/index.rst`, new
  `docs/source/myst-foundation.md`, new `docs/_ext/myst_autodoc.py` package,
  new `docs/prototype_api` package, and this implementation record. The S041
  README and record edits remain part of the preceding checkpoint.
- Decisions added or superseded: D056 selects standard autodoc, defines the
  docs-environment import boundary, records why autodoc2 failed parity, and
  makes both transitional parser routing and the Matplotlib compatibility hook
  explicit.
- Verification commands and results:
  - `uv lock` resolved 169 packages and locked MyST-NB 1.4.0, MyST-Parser
    5.1.0, and sphinx-autodoc2 0.5.0; locked all-group sync completed.
  - The network-enabled focused HTML build fetched all five configured
    intersphinx inventories and completed. Inspection proved the internal
    function and stable label targets, external `pathlib.Path` target, Pydantic
    alias constructor, both overload signatures, inherited method, explicit
    type alias, source links, field lists, doctest rendering, valid plot image,
    and exclusion of the object omitted by ``__all__``.
  - The focused Sphinx doctest builder ran five examples with five passing and
    no failures. Matplotlib executed the MyST-fenced plot and emitted PNG,
    high-resolution PNG, PDF, and source artifacts under the temporary build.
  - Explicit Ruff format and lint checks passed for the otherwise excluded
    docs-local Python extension and prototype package. The complete source
    suite passed all 422 tests in 40.35 seconds; repository-wide pre-commit
    passed YAML, EOF, whitespace, Ruff format/lint, pydoclint, and Pyrefly.
    The lock check and `git diff --check` also passed.
- Measurements/artifacts: focused HTML, doctest output, and the generator
  comparison are under `/tmp/resistics-checkpoint-7-1-*`; no generated build
  output is retained in the repository. The docs-only prototype package and
  parser regex are transitional evidence to remove in Checkpoint 7.5 after the
  production migration carries the same coverage.
- Known failures or incomplete work: the existing RST site still emits its
  pre-existing duplicate-object, generated-gallery navigation, and unpickleable
  gallery-configuration warnings; warning cleanup is owned by Checkpoints
  7.2-7.7. Standard autodoc remains dynamic and therefore imports requested
  modules inside the locked documentation environment. The local Matplotlib
  bridge is required only because upstream MyST does not yet implement the
  directive state-machine insertion method.
- Checkpoint state at end: `7.1` is `verified`; Checkpoint 7.2 is ready.
- Commit readiness or commit id: the foundation, generator decision,
  dependencies, lock, prototype evidence, and record are ready with the
  preceding uncommitted S041 work for an owner-selected commit; no commit was
  requested or created.
- Exact next action: convert the site structure and maintained narrative/API
  entry pages to MyST with stable labels while retaining the explicit
  per-object parser boundary for unconverted production docstrings.

### S043 - 2026-07-22 - Convert site structure and narrative pages to MyST

- Checkpoint state at start: Checkpoint 7.1 was verified in S042 and its
  foundation was uncommitted; Checkpoint 7.2 was ready.
- Starting branch, HEAD, and worktree: `mth5` at `70a84f2` with the verified
  S041 uv guidance and S042 documentation-foundation changes uncommitted, plus
  the required empty `pyrefly-baseline.json` intentionally untracked. Those
  changes were preserved.
- Session objective: convert maintained site structure, narrative, navigation,
  and API-entry pages to MyST, provide stable labels, and make the landing page
  describe the current MTH5-backed processing contracts.
- Work completed: replaced the maintained top-level RST sources with 28 MyST
  pages; converted all navigation to suffixless MyST toctrees; added explicit
  stable page and section labels; deleted the obsolete `modules.rst`; and
  exposed current explorer, flow, job, mask, templates, and TUI module entry
  pages. Rewrote the landing and getting-started content around read-only MTH5
  projects, derived outputs, flows, parameter sets, jobs, trusted process
  plugins, structured execution, the Textual application, and the independent
  library API. Updated the lower-level, custom-process, docstring, and
  literature pages without changing production behaviour.
- Files changed: `docs/source/conf.py`; the former top-level narrative,
  navigation, and API-entry `.rst` files and their `.md` replacements; six new
  current-module API pages; `docs/source/myst-foundation.md`; and this
  implementation record. Earlier README, dependency, lock, extension, and
  prototype changes remain owned by S041-S042.
- Decisions added or superseded: D057 records the all-MyST maintained-page
  boundary, deletion of the obsolete module list, current-module navigation,
  and the two explicitly temporary fenced RST compatibility cases.
- Verification commands and results:
  - A fresh network-enabled `sphinx-build -b html -E -D plot_gallery=0` fetched
    all five intersphinx inventories and built 62 source pages successfully.
    Inspection confirmed the current landing headings and navigation links to
    getting started plus the flow, job, and TUI API pages. Stable-label audit
    found no duplicates; there are no maintained top-level `.rst` pages.
  - The complete source suite passed all 422 tests in 28.11 seconds.
    Repository-wide pre-commit passed YAML, EOF, whitespace, Ruff format/lint,
    pydoclint, and Pyrefly. The lock check and `git diff --check` also passed.
- Measurements/artifacts: final HTML is under
  `/tmp/resistics-checkpoint-7-2-final-html`; no generated gallery timing source
  is retained in the repository. The fresh build completed with 219 warnings,
  down from the 224-warning offline build after all intersphinx inventories
  loaded.
- Known failures or incomplete work: the successful transitional build is not
  yet warning-clean. Its remaining warnings come from the legacy gallery,
  duplicate autodoc/Pydantic object descriptions, and RST parsing of
  unconverted inherited or external docstrings; exposing the current public
  modules makes more of that existing API debt visible. Docstring migration,
  gallery replacement, transitional-stack removal, and the strict warning gate
  are explicitly owned by Checkpoints 7.3-7.6. There are no missing-document,
  unresolved citation, toctree-target, or MyST page-structure failures from
  this conversion.
- Checkpoint state at end: `7.2` is `verified`; Checkpoint 7.3 is ready.
- Commit readiness or commit id: the maintained MyST pages, current landing and
  navigation, configuration adjustment, and record are verified with the
  preceding uncommitted S041-S042 work for an owner-selected commit; no commit
  was requested or created.
- Exact next action: migrate `common`, `sampling`, and `transfunc` docstrings to
  MyST in place, preserve their examples and plots, and split pydoclint into
  explicit NumPy-style and Sphinx-style module invocations using the governing
  type-check settings.
