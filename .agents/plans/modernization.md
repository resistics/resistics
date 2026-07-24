# Resistics Standalone and App-Backend Modernization Record

Status: completed and reconciled
Last updated: 2026-07-23
Implementation branch: `mth5`
Reconciled by: code-hardening Checkpoint 8.1

## Purpose and authority

Resistics remains a standalone Python package for scripts, notebooks, batch
processing, and its Textual terminal application. The library owns scientific
processing, validation, serialization, MTH5 project discovery, job execution,
and result metadata. Presentation layers own interaction design and any future
transport or process-management choices.

This file began as the implementation plan for the Pydantic 2, MTH5-only,
flow/job, and RegressionInC migrations. Those migrations have now been
reconciled against the code. This document records the implemented architecture
and the disposition of the original phases; it is no longer an active task
queue. The code-hardening implementation record owns the remaining audit work.

When this record and the implementation disagree, treat the implementation and
its tests as the current state, then update this record explicitly. Do not
revive a superseded compatibility path merely because it appears in repository
history.

## Implemented architecture

### Standalone library boundary

The package can be used without starting the TUI. Its public library contracts
support:

- creating, opening, inspecting, and deterministically closing an MTH5-backed
  project;
- listing surveys, stations, runs, sampling frequencies, time spans, channels,
  and concurrent recordings without loading full time-series arrays;
- creating, loading, validating, and saving flows, parameter sets, gather
  criteria, and jobs;
- resolving human-authored jobs into concrete batches before execution;
- executing the same jobs from Python with structured progress and
  cancellation callbacks;
- archiving resolved job metadata and writing derived project artifacts; and
- plotting intermediate data, transfer functions, flow graphs, and job plans.

The `resistics` command is a Textual adapter over these library contracts.
Future GUIs, services, or transports may consume the same models and services,
but no web framework, IPC framework, or external application package belongs
in the core contract.

### Pydantic and dataclass boundary

Pydantic v2 owns public validated and serialized contracts: processing and data
models, project summaries, explorer results, process schemas, flow resources,
job validation, lifecycle events, and diagnostic log entries. These models
provide YAML/JSON serialization and JSON schema for presentation-layer forms.

Dataclasses are reserved for private implementation records where validation or
serialization is not part of the boundary, such as cache keys, worker
messages, render plans, and regression progress helpers. Public explorer and
service results must remain frozen Pydantic models; private immutable
bookkeeping may remain dataclasses.

New and migrated code uses Pydantic v2 validators and `model_validate`,
`model_dump`, `model_dump_json`, and `model_copy`. `ResisticsModel` still
provides bounded `dict()` and `json()` aliases for external callers. They are
compatibility aliases implemented with v2 semantics, not a Pydantic v1
dependency. Repository code must not call them; their final removal is assigned
to the 2.0 compatibility review.

### MTH5-only projects and time data

`resistics.project.Project` is the canonical project API.
`resistics.project.MTH5File` provides the matching standalone inspection
contract. Both own read-only MTH5 handles and support `close()` and context
manager cleanup. Failed construction closes any partially opened handle.

The canonical project structure is:

```text
project/
├── resistics.json
├── processing/
│   ├── flows/
│   ├── parameters/
│   ├── criteria/
│   └── jobs/
├── data/
│   └── [survey]/[station]/
│       ├── [run]/
│       └── results/
│           └── [output_label]/[sampling_frequency]/
├── logs/
└── plugins/
```

The source MTH5 file is opened read-only and is not used as a derived-artifact
store. Project metadata records its path and reference time. Spectra, masks,
transfer functions, job archives, and logs belong in the project tree.

`MTH5TimeReader` is the only public time-series source reader.
`Project.read_run()` and `MTH5File.read_run()` resolve an MTH5 run and return
the channel-labelled `TimeData` contract used by the numerical pipeline.
`TimeData` remains an in-memory processing model backed by a labelled array;
accepting NumPy values to construct that in-memory model does not make NumPy
files a supported input format.

The old directory-reader project API, `resp.py`, `letsgo.py`, reader selection,
and ASCII/NumPy time readers have been removed. Historical ASCII, bz2, and
NumPy files still present below `data/time/` are not referenced by public code
or documentation. Checkpoint 8.2 classified them as removable migration data;
F004 requires a provenance check before deletion. They must not return to
defaults, documentation, or tutorials.

### Flows, parameters, criteria, and jobs

The implemented names and responsibilities are:

`FlowDefinition`
: A serializable staged directed acyclic graph. Each `FlowStage` has `run` or
  `station_rate` scope, and each `FlowNode` names a stable ID, a qualified
  concrete `ResisticsProcess` class path, input edges, and whether its
  configuration comes from the parameter set or gather criteria. A flow
  contains no ordinary process parameter values or UI layout.

`ParameterSet`
: Reusable process configuration keyed by qualified process class path. This
  deliberately supersedes the original proposal to key values by node ID:
  repeated uses of one process class share its reusable configuration.

`GatherCriteria`
: Station- and sampling-frequency-specific gathering and remote-reference
  policy. Criteria are independent project resources rather than parameter-set
  fields.

`JobDefinition`
: The human-authored YAML contract. It references project-local flow,
  parameter, and optional criteria files and supplies survey, station,
  sampling-frequency, and stage scope, output label, and overwrite policy.

`ResolvedJob`
: A validated job containing loaded resources, selected stages, planned
  station/rate batches, resolved paths, and the executable binding.

`ProcessingJob`
: The in-memory flow/parameters/runtime/output binding used by validation and
  execution. It is not the persisted human-authored job format.

`ProjectJobs`
: The project-local repository and validation boundary. Normal configuration
  errors return `JobValidation` with complete errors and warnings rather than
  escaping into a presentation layer.

`JobRunner`
: The synchronous library executor. It expands selected batches, runs stages,
  supports cancellation, emits structured `JobProgressEvent` values, cleans
  partial outputs after failure or cancellation, captures warnings in project
  logs, and archives `job_info.json` beside completed results.

YAML is the human-authored resource format. JSON is used for archived execution
metadata and Pydantic schema/data interchange. Output labels are validated as
single safe path components.

### Process discovery and trusted plugins

Flow nodes use qualified Python class paths rather than the original global
configuration registry. `ProcessCatalog` discovers built-in processes and
trusted project plugin modules, while `resolve_process_class()` validates a
requested concrete class directly. Unknown or invalid paths produce actionable
validation errors.

The canonical `project/plugins/` package is discovered automatically. Plugin
code is executable trusted code, not an untrusted data format, and must
subclass `ResisticsProcess` and declare the same input, output, runtime,
validation, and schema contracts as built-in processes.

Checkpoint 8.2 removed the serialized but inert
`ProjectMetadata.plugin_paths` field and matching `init()` argument. The
canonical `project/plugins/` package remains the sole project-local discovery
boundary. Existing metadata containing the obsolete key remains loadable
because unknown metadata fields are ignored.

This qualified-path design supersedes the proposal to preserve monolithic
`Configuration`-style registry deserialization. Existing data and
transfer-function polymorphism remain owned by their model families; flow
execution has no built-in dispatch table.

### RegressionInC boundary

Resistics imports `LeastSquares` from `regressioninc.linear` through the small
`get_least_squares_regressor()` construction boundary and relies on a private
fit protocol rather than exposing RegressionInC internals to callers.
Resistics owns MT-specific gathering, predictor/observation preparation,
progress reporting, transfer-function packaging, and solution metadata.
RegressionInC owns the numerical linear estimator.

Presentation code does not call RegressionInC directly for normal Resistics
workflows. A future RegressionInC layout change should affect the construction
boundary and its focused tests rather than project, flow, job, or TUI code.

### TUI and explorer boundary

The TUI is split into a small public facade, application owner, stable screen
modules, UI-neutral services, cached explorer models, and logging support.
`ProjectExplorerService` can be exercised without constructing a Textual
application and returns frozen Pydantic DTOs from `ProjectExplorerIndex`.

Project opening, discovery, job execution, plotting preparation, and other
blocking work run outside the UI thread. Cache hits and footer action
predicates perform no project or filesystem I/O. Cache invalidation belongs to
the mutation that changes the corresponding state, and stale worker results
cannot replace newer selections.

The Logs tab captures Resistics and dependency diagnostics without allowing a
worker thread to write Textual widgets. It preserves warnings that are hidden
from the terminal while Textual owns the display. Plotly figures remain a
library result and are opened by the presentation adapter.

### Documentation and examples

Maintained documentation is Markdown/MyST only. Standard Sphinx autodoc,
through the single MyST-aware adapter, is the only API generator. Six small
MyST-NB tutorials cover MTH5 project creation and discovery, flows and
parameters, job validation and execution, calibration and remote reference,
and plotting results and plans.

Public examples use MTH5 input only. Rich examples and plots remain with the
objects they document when that is where users will find them most useful.
The strict local documentation command builds warning-fatal nitpicky HTML,
executes all tutorials, verifies protected plots, and runs the complete fenced
doctest inventory. HTML is the only supported published output.

## Original phase disposition

| Original phase | Disposition | Implemented result |
| --- | --- | --- |
| 1. Pydantic v2 foundation | Completed with bounded alias cleanup recorded above | Pydantic v2 models, validators, serialization, Python 3.12-3.14 |
| 2. Registry serialization | Superseded; inert external plugin paths removed in 8.2 | Qualified process paths, direct validation, built-in and canonical-project process catalogue |
| 3. MTH5-only input | Completed | Canonical `Project`/`MTH5File`, owned read-only handles, `MTH5TimeReader` |
| 4. Flow, parameters, and jobs | Completed with corrected final naming | Staged DAGs, class-keyed parameters, independent criteria, `JobDefinition`/`ResolvedJob` |
| 5. Regression boundary | Completed | Narrow RegressionInC construction/protocol boundary; MT preparation remains local |
| 6. Docs and examples | Completed | MTH5-only MyST site, six executable tutorials, current API reference |

## Superseded starting-state claims

The following statements described the repository before implementation and
must not guide new work:

- `flow.py` and its tests are tracked production code, not an untracked
  prototype.
- Full test collection passes under Pydantic v2; the old private-attribute and
  RegressionInC import failures are closed.
- `project.py` is the sole project direction; `resp.py` and `letsgo.py` no
  longer exist.
- Monolithic `Configuration`, reader selection, and ASCII/NumPy public input
  workflows are removed rather than compatibility targets.
- Parameter values are keyed by qualified process path, not flow node ID.
- `JobDefinition` is the authored job and `ResolvedJob` is its validated
  execution plan; `ProcessingJob` is only the loaded in-memory binding.
- The implemented TUI is a supported presentation adapter. “App backend”
  means presentation-neutral library/service contracts, not a dependency on a
  separate `resistics-app` package.

## Maintained verification

From the paired Resistics/RegressionInC checkout, use the locked environment:

```console
uv sync --locked --all-groups
uv run --locked --no-sync pytest
uv run --locked --no-sync python scripts/check_coverage.py
uv run --locked --no-sync ruff format --check resistics tests scripts
uv run --locked --no-sync ruff check resistics tests scripts
uv run --locked --no-sync pydoclint --config=pyproject.toml resistics
uv run --locked --no-sync pyrefly check
uv run --locked --no-sync python scripts/check_documentation.py
```

After changes to TUI state, actions, discovery, or service boundaries, also run:

```console
uv run --locked --no-sync pytest -q tests/test_tui.py -k "cached_action_checks_are_fast or binding_refreshes_follow_owned_state_transitions"
```

Release candidate verification and remote follow-ups are documented in
`docs/source/releasing.md`. Hosted CI, protected tags, Trusted Publishing,
hosted documentation deployment, and standalone registry installation remain
repository-owner work; they are not active or verified merely because the
local architecture is complete.

## Final audit outcome

Checkpoint 8.2 owns measurement and risk recording, not another architecture
migration. It migrated all repository callers from the bounded
`ResisticsModel.dict()`/`json()` aliases to their Pydantic v2 names, removed
the inert external `plugin_paths` field, classified the unreferenced legacy
time-data files, and measured current module size and complexity, performance,
coverage, dependencies, typing, documentation, and package build/install
results. Remaining issues are narrow, owned follow-ups rather than reopened
architecture phases.
