# Coding Standards

## Project Role

Resistics is both:

- A standalone Python package for scripts, notebooks, and batch processing.
- The backend contract for `resistics-app`.

Core resistics code must not depend on UI frameworks, web frameworks, app state,
or transport-specific concepts. App-facing models should be plain Python and
Pydantic objects that can be used by any caller.

## Formatting

- Use Ruff formatting.
- Keep line length at 88 characters.
- Use ASCII unless a file already uses non-ASCII or the content requires it.
- Prefer clear imports over dense one-line imports.
- Avoid unrelated formatting churn when touching legacy modules.

## Type Hints

- Add type hints to new public functions, service classes, DTOs, and model
  methods.
- Prefer modern built-in generics in new code where the supported Python version
  allows it.
- Use `Path` for filesystem paths.
- Use concrete domain aliases where they clarify intent, for example
  `RSDateTime`.
- Avoid `Any` in app-facing DTOs unless the data is intentionally open-ended.

## Pydantic

- Use Pydantic v2 APIs only in new and migrated code.
- Use `model_config = ConfigDict(...)`, not class `Config`.
- Use `@field_validator` and `@model_validator`, not `@validator`.
- Use `model_dump`, `model_dump_json`, `model_validate`, and
  `model_validate_json`.
- Do not use `.dict()`, `.json()`, `.parse_file()`, or `.copy()` in migrated
  code.
- Use `Field(default_factory=...)` for mutable defaults.
- Mark registries and constants as `ClassVar`.
- Keep model serialization deterministic for YAML/JSON files used by users and
  the app.

## Public API Design

- Public APIs should work from standalone Python without `resistics-app`.
- App-facing APIs should return serializable Pydantic models or simple Python
  values.
- Normal user mistakes in flows/parameters/jobs should return validation
  results, not raw tracebacks.
- Unexpected internal failures can raise exceptions, but execution wrappers
  should convert them into structured failure events for app use.
- Do not expose MTH5 object internals as required app-facing contracts unless
  there is no stable alternative.

## Processing Code

- Keep processing classes small and composable.
- Keep static parameters on Pydantic models.
- Pass runtime data to `run(...)` or execution contexts.
- Avoid hidden global state except explicit registries.
- Make filesystem writes explicit and localized to writer/executor classes.
- Write job metadata with outputs so results are reproducible.

## MTH5

- MTH5 is the only public input format.
- Project discovery should read metadata and summaries without loading full
  time-series arrays.
- MTH5 file lifecycle should be explicit: open, use, close.
- Prefer adapters from MTH5 data to resistics internal data containers over
  rewriting all processing internals at once.

## Regression

- Keep MT-specific preparation and result packaging in resistics.
- Keep numerical complex-domain regression algorithms in `regressioninc`.
- Access `regressioninc` through a small adapter boundary.
- Do not import `regressioninc` directly from app-facing code.

## Testing

- Use pytest.
- Use focused unit tests for validators, registries, serializers, and adapters.
- Use synthetic data for numerical behavior where possible.
- Add integration tests for MTH5 project discovery and processing execution.
- Run tests with a writable uv cache in this environment:

```console
UV_CACHE_DIR=/tmp/uv-cache uv run pytest
```

## Documentation

- Public docs and examples must match supported public APIs.
- Do not document legacy ASCII/NumPy input workflows as supported.
- Include standalone examples and app-backend-oriented schema/validation
  examples.
- Prefer short, executable examples over long narrative-only examples.

## Git Hygiene

- Preserve user work in dirty files.
- Do not revert `uv.lock`, untracked flow files, or unrelated changes unless
  explicitly asked.
- Keep migrations split into reviewable commits by subsystem where practical.
