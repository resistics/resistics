# Resistics Project Rules

## Product Rules

- Resistics remains a standalone Python package.
- Resistics also provides stable backend contracts for `resistics-app`.
- Core resistics must not depend on `resistics-app`.
- Core resistics must not import GUI, web, IPC, or app-state dependencies.
- Transport decisions belong outside core resistics.

## Input Rules

- MTH5 is the only public input format.
- Do not add new public workflows for ASCII, NumPy, bz2, or vendor-specific
  direct readers.
- Legacy readers may remain temporarily only as internal migration helpers or
  conversion utilities.
- Public docs and examples must not present legacy readers as supported input
  paths.

## Pydantic Rules

- New and migrated code must use Pydantic v2 APIs.
- Registries must be explicit `ClassVar` registries.
- Mutable defaults must use `Field(default_factory=...)`.
- App-facing models must be serializable and able to produce JSON schema.
- Saved YAML/JSON should use stable discriminator fields for registered types.

## Flow and Execution Rules

- A flow defines process order and connectivity.
- A parameter set defines parameter values.
- A processing job binds a flow, parameter set, runtime inputs, and output
  label.
- Runtime inputs include MTH5 selection and job-specific scope, not static
  process parameters.
- Execution must archive resolved job metadata with results.
- Execution should emit structured progress events.
- Validation errors intended for users should be structured, not buried in raw
  tracebacks.

## App Backend Rules

- App-facing APIs should be pure Python and usable from notebooks.
- DTOs should avoid exposing mutable MTH5 internals.
- Project summary APIs should be cheap and avoid loading full time-series data.
- Long-running execution should support progress, warnings, failures, and a
  future cancellation path.
- The app should be able to validate flows/parameters/jobs before execution.

## Regression Rules

- Resistics owns MT-specific regression preparation and solution metadata.
- `regressioninc` owns numerical complex-domain regression algorithms.
- Resistics should access `regressioninc` through an adapter boundary.
- Solver choices in configurations must be serializable.

## Documentation Rules

- Docs must describe the current public API, not aspirational or removed APIs.
- Standalone usage and app-backend usage should both be represented.
- Examples should be runnable or clearly marked as conceptual.
- Terminology must match MTH5: survey, station, run.

## Code Hygiene Rules

- Preserve user changes in a dirty worktree.
- Do not revert unrelated changes.
- Keep migrations scoped by subsystem.
- Prefer existing project patterns unless they conflict with Pydantic v2,
  MTH5-only input, or the standalone/app-backend split.
- Avoid broad rewrites without tests.

## Test Rules

- Run tests through uv with a writable cache in this environment:

```console
UV_CACHE_DIR=/tmp/uv-cache uv run pytest
```

- Add focused tests before or with risky migrations.
- Keep standalone workflows covered by tests or examples.
- Keep app-facing DTO schema and serialization covered by tests.

## Dependency Rules

- Do not add app-only dependencies to core resistics.
- Do not add a web service framework to core resistics unless the project
  explicitly decides that transport belongs in core.
- Keep `regressioninc` as a numerical dependency behind a small adapter.
- Avoid dependency changes that force unrelated lockfile churn.
