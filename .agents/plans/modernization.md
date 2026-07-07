# Resistics Standalone and App Backend Modernization Plan

## Purpose

Resistics should remain a standalone Python package for scripts, notebooks, and
batch processing, while also providing the stable backend contract used by
`resistics-app`. The core package should own processing, validation,
serialization, project discovery, and result metadata. The app should own UI,
interaction design, and any transport or process-management decisions.

This plan focuses on four connected changes:

- Move all Pydantic models and serialization to Pydantic v2.
- Make MTH5 the only public input format.
- Replace monolithic processing configuration with flows, configurations, and
  runs.
- Keep numerical complex-domain regression in `regressioninc`, with resistics
  owning MT-specific preparation and result packaging.

## Current State

- The environment is already resolving Pydantic `2.13.4`.
- `resistics/flow.py` and `tests/test_flow.py` are active untracked work and
  already prototype flow, parameter, and run concepts.
- `tests/test_flow.py` passes with `UV_CACHE_DIR=/tmp/uv-cache uv run pytest
  tests/test_flow.py -q`, with one Pydantic v1 validator deprecation warning.
- Full test collection currently fails early because Pydantic v2 treats
  `_types` on `ResisticsProcess` as a private model attribute, so subclass
  registration breaks during import.
- Full test collection also finds a `regressioninc` import mismatch:
  `resistics.regression` imports `regressioninc.linear.models`, but the sibling
  package currently exposes `regressioninc.base` and `regressioninc.linear`.
- There are two project directions in the code:
  - `resistics/project.py` and `resistics/letsgo.py` still model directory-based
    projects with configured time readers.
  - `resistics/resp.py` starts an MTH5-backed project model with Pydantic v2
    datetime handling and a proposed project structure.
- Existing docs and examples still describe ASCII/NumPy readers,
  `resistics-readers`, and old `Configuration` behavior.

## Target Architecture

### Standalone Package

Resistics must remain useful without `resistics-app`.

Required standalone capabilities:

- Open and inspect MTH5-backed projects from Python.
- List surveys, stations, runs, sample rates, time spans, and concurrent
  recordings.
- Build, load, validate, and save processing flows, configurations, and runs.
- Execute runs from scripts and notebooks with progress callbacks.
- Write reproducible result metadata and solution files.
- Plot and inspect intermediate and final products where the existing package
  already supports that.

### App Backend

The same core APIs should be app-safe.

Required backend capabilities:

- Pydantic DTOs for project summaries, survey/station/run listings, flow
  definitions, parameter schemas, run validation, progress events, run status,
  and result summaries.
- JSON schema generation for app forms and validation.
- Deterministic serialization to YAML/JSON for user-authored configs and
  archived run metadata.
- No dependency on GUI frameworks, FastAPI, IPC libraries, or app-specific
  state.
- Clear cancellation, error, warning, and progress status surfaces for
  long-running processing.

## Migration Phases

### Phase 1: Pydantic v2 Foundation

1. Pin the project dependency to Pydantic v2 in `pyproject.toml`.
2. Convert base model helpers in `resistics.common`:
   - Replace class `Config` with `model_config = ConfigDict(...)`.
   - Replace `self.json()` with `self.model_dump_json()`.
   - Replace `self.dict()` with `self.model_dump()`.
   - Replace `.copy(deep=True)` with `.model_copy(deep=True)`.
   - Replace `.parse_file(path)` with explicit `Path.read_text()` plus
     `model_validate_json`.
3. Make subclass registries explicit:
   - Declare registries as `ClassVar[dict[str, type[...]]]`.
   - Do not store registries as Pydantic model fields or private attrs.
4. Replace validators:
   - Use `@field_validator(..., mode="before")` for input coercion.
   - Use `@field_validator(..., mode="after")` or `@model_validator` for derived
     fields that depend on multiple fields.
   - Avoid mutating input dictionaries inside validators.
5. Move the Pydantic v2 `RSDateTime` serializer/validator pattern from
   `resistics/resp.py` into `resistics/sampling.py`.
6. Add compatibility helper functions only where they reduce mechanical churn;
   new code should use v2 APIs directly.

Acceptance criteria:

- Importing all resistics modules no longer fails on Pydantic private attrs.
- Pydantic deprecation warnings are removed from core modules.
- Existing metadata JSON round trips still work where the public model remains.

### Phase 2: Explicit Registry-Based Serialization

Keep the old capability of initializing the right class from saved JSON/YAML,
but make it explicit and v2-native.

1. Introduce a small registry utility for registered model families:
   - Processes
   - Transfer functions
   - Regression solver adapters
   - Flow step handlers, where needed
2. Use a discriminator field consistently:
   - Prefer `name` for backward compatibility with existing saved process
     metadata.
   - Prefer `type` or `type_id` for flow step definitions.
3. Validate unknown names with clear errors that list known registered types.
4. Support plugin registration without importing app code.
5. Document that deserializing plugin-defined classes requires importing or
   loading the plugin first.

Acceptance criteria:

- `Configuration`-style JSON/YAML can still recreate registered process classes.
- Transfer functions round trip through JSON/YAML.
- Unknown process and transfer-function names fail with actionable errors.

### Phase 3: MTH5-Only Public Input

1. Make the MTH5-backed project model the canonical public project API.
2. Merge useful work from `resistics/resp.py` into `resistics/project.py` or a
   clearly named project backend module.
3. Fix current `resp.py` issues during merge:
   - `check_project` currently returns inside the subdirectory loop.
   - MTH5 file lifecycle needs explicit open/close behavior.
   - MTH5 path serialization should be stable relative to project directory
     where practical.
4. Define canonical project structure:

```text
project/
├── resistics.json
├── configs/
│   ├── flows/
│   ├── configurations/
│   └── runs/
├── data/
│   └── [survey]/[station]/
│       ├── [run]/
│       └── results/
│           └── [processing_run]/
├── logs/
└── plugins/
```

5. Add an MTH5-to-`TimeData` adapter so existing decimation, windowing,
   spectral, calibration, gathering, and regression internals can be reused.
6. Remove non-MTH5 readers from public defaults.
7. Retire or hide old directory-reader project loading from public docs and
   examples.
8. Keep old reader classes only if they are required temporarily for tests or
   conversion utilities; they should not be part of default workflows.

Acceptance criteria:

- A standalone user can initialize and load a project from an MTH5 file.
- The app can list surveys, stations, runs, sample rates, and time ranges
  without loading full time-series data.
- Public docs no longer present ASCII/NumPy as supported input workflows.

### Phase 4: Flow, Configuration, and Run Model

Use three separate concepts.

`FlowDefinition`:

- Defines processing order and connectivity.
- Contains step ids and step types.
- Does not contain user parameter values beyond structural defaults needed by
  the flow.

`ProcessingConfiguration`:

- Defines parameter values for a flow.
- Stores values by flow node id.
- Does not define input sites, stations, runs, or time ranges.

`ProcessingRun`:

- Binds a flow reference, configuration reference, runtime inputs, and output
  label.
- Runtime inputs include survey, station, run, sampling frequency, time range,
  remote reference, and project path as needed.
- Batch runs are expanded into individual resolved runs before execution.

Implementation steps:

1. Keep the useful pieces of `resistics/flow.py`.
2. Rename or alias `ParameterSet` to `ProcessingConfiguration`.
3. Introduce unresolved and resolved run models:
   - File-authored run references flow/configuration YAML paths.
   - Resolved run contains the loaded flow/configuration models for execution.
4. Make YAML the preferred human-authored format and JSON schema the preferred
   app-form contract.
5. Ensure execution writes resolved run metadata beside outputs.
6. Keep `FlowExecutor` pure Python with callbacks for progress events.
7. Add app-safe validation results instead of raising exceptions for normal user
   configuration mistakes.

Acceptance criteria:

- Flows, configurations, and runs round trip through YAML.
- The app can validate a run before execution and show field-level errors.
- Standalone users can execute the same run from Python.

### Phase 5: Regression Boundary

1. Update imports to current `regressioninc` layout:
   - `Regressor` from `regressioninc.base`
   - `LeastSquares` and other linear models from `regressioninc.linear`
2. Add a small resistics adapter layer for solver selection.
3. Keep MT-specific data preparation in resistics:
   - `RegressionInputMetadata`
   - `RegressionInputData`
   - `RegressionPreparerGathered`
   - `RegressionPreparerSpectra`
   - `Solution`
4. Move or keep numerical regression algorithms in `regressioninc`.
5. Do not let app code call `regressioninc` directly for standard resistics
   workflows; use resistics solver adapters.

Acceptance criteria:

- Existing synthetic regression tests pass through the adapter.
- Solver choices are serializable in processing configurations.
- Future `regressioninc` layout changes affect only the adapter layer.

### Phase 6: Docs and Examples

1. Update README and getting-started docs:
   - MTH5 is the public input format.
   - Resistics is both standalone and app-backend-ready.
   - Flow/configuration/run replaces monolithic processing configuration.
2. Rewrite examples:
   - Project initialization from MTH5.
   - Listing project contents.
   - Creating a flow, configuration, and run.
   - Running standalone processing.
   - Loading results and plotting transfer functions.
3. Remove or archive old read examples for ASCII, bz2, and NumPy.
4. Update API docs to include flow/project backend modules.
5. Ensure examples and docs do not mention unsupported public input formats.

Acceptance criteria:

- Docs build without deprecated Pydantic warnings from resistics.
- Public examples use only MTH5 input.
- Standalone and app-backend usage are both documented.

## Testing Strategy

Run commands with a writable uv cache in this environment:

```console
UV_CACHE_DIR=/tmp/uv-cache uv run pytest
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_flow.py -q
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_regression.py -q
```

Add focused tests for:

- Pydantic v2 model round trips.
- Registry and discriminator behavior.
- MTH5 project initialization, loading, summary tables, and close behavior.
- MTH5-to-`TimeData` adapter reads.
- Flow/configuration/run YAML round trips.
- Run validation and batch expansion.
- Progress events and failure events.
- Regression adapter behavior.
- App-facing DTO JSON schema generation.

## Implementation Order

1. Fix Pydantic v2 import blockers.
2. Fix `regressioninc` import boundary enough to collect tests.
3. Convert remaining v1 Pydantic APIs module by module.
4. Establish canonical MTH5 project API.
5. Wire MTH5 read adapter into existing processing internals.
6. Finalize flow/configuration/run models.
7. Add app-safe DTOs and progress/status surfaces.
8. Update docs and examples.
9. Remove or archive unsupported public reader workflows.

## Non-Goals

- Do not add GUI, web, or IPC dependencies to core resistics.
- Do not make `resistics-app` a dependency of resistics.
- Do not rewrite numerical processing kernels unless required by tests or the
  `regressioninc` split.
- Do not preserve public ASCII/NumPy input workflows after the MTH5 migration.

## Assumptions

- Existing `uv.lock`, `resistics/flow.py`, and `tests/test_flow.py` changes are
  active user work and should not be reverted.
- Public input support should become MTH5-only.
- Registry-based deserialization remains useful for standalone configs,
  plugins, and app-authored YAML/JSON.
- Resistics should expose stable backend contracts, but transport remains
  outside the core package.
