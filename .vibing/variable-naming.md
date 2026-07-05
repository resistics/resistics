# Variable Naming

## General Principles

- Prefer domain names over generic names.
- Keep names stable across models, YAML files, docs, and app-facing DTOs.
- Use snake_case for Python variables, fields, and YAML keys.
- Use short names only where they are established in magnetotellurics or the
  existing codebase.

## Project and MTH5 Names

Use these names consistently:

- `project`: A resistics project.
- `project_path`: Filesystem path to the project directory.
- `mth5_path`: Filesystem path to the MTH5 file.
- `survey`: MTH5 survey name.
- `station`: MTH5 station name.
- `run`: MTH5 run name.
- `survey_path`: Survey identifier if a path-like value is needed.
- `station_path`: Canonical `"survey/station"` identifier.
- `run_path`: Canonical `"survey/station/run"` identifier.
- `sample_rate`: MTH5 table column name when mirroring MTH5 metadata.
- `fs`: Sampling frequency in processing and numerical code.
- `ref_time`: Project reference time.
- `start_time`: Start timestamp of an MT recording, MTH5 run, or summary row.
- `end_time`: End timestamp of an MT recording, MTH5 run, or summary row.

Avoid these public names in new code:

- `site` when the value is specifically an MTH5 station.
- `meas` or `measurement` for MTH5 runs.
- `mth_data`; use `mth5_data`.

Legacy names may remain temporarily in old code while migrating, but new public
APIs should use the MTH5 terminology.

## Flow, Configuration, and Run Names

Use:

- `flow`: A loaded `FlowDefinition`.
- `flow_path`: Path to a flow YAML file.
- `flow_id`: Stable id for a known flow, if needed.
- `node`: A node instance in a flow.
- `node_id`: Stable id of a node inside a flow.
- `step_type`: Registered processing step type.
- `type_id`: Registry id for a step definition.
- `configuration`: A loaded processing configuration.
- `configuration_path`: Path to a configuration YAML file.
- `processing_run`: A loaded run definition.
- `run_config`: Acceptable only as a local shorthand where it does not conflict
  with MTH5 `run`.
- `runtime`: Runtime input dictionary or model.
- `output_label`: User-visible output label for result grouping.

Avoid:

- `params` in public model fields; use `parameters` or `configuration`.
- `config` for new persisted processing models when it can be confused with
  Python or package config. Use `configuration`.
- `run` alone for processing runs in MTH5-facing code. Use `processing_run`.

## Time and Windowing Names

Use:

- `first_time`: First sample timestamp.
- `last_time`: Last sample timestamp.
- `start_time`: Start timestamp of the original available data interval.
- `end_time`: End timestamp of the original available data interval.
- `from_time`: User/runtime requested processing start time.
- `to_time`: User/runtime requested processing end time.
- `from_sample`: Inclusive processing start sample index.
- `to_sample`: Inclusive processing end sample index.
- `n_samples`: Number of samples.
- `n_chans`: Number of channels.
- `win_size`: Window size in samples.
- `olap_size`: Window overlap in samples.
- `local_win`: Window index relative to a recording.
- `global_win`: Window index relative to project `ref_time`.

Use `from_time` and `to_time` for processing subsets, reads, trims, and runtime
inputs. Use `start_time` and `end_time` for the original recording or dataset
bounds available before processing selection is applied.

## Channel and Transfer Function Names

Use:

- `chan`: Single channel name.
- `chans`: Ordered list of channel names.
- `out_chans`: Output channels in a transfer function.
- `in_chans`: Input channels in a transfer function.
- `cross_chans`: Cross-power channels.
- `tf`: Transfer function object in numerical/internal code.
- `transfer_function`: Public DTO or docs name where clarity is more important.
- `component`: Transfer-function component.
- `component_key`: String key such as `ExHy`.

## Regression Names

Use:

- `regression_input`: Prepared regression data.
- `observations` or `obs`: Observed values.
- `predictors` or `preds`: Predictor matrix.
- `coefficients` or `coef`: Regression coefficients.
- `solver`: Resistics solver adapter.
- `regressor`: Numerical model from `regressioninc`.
- `solution`: Resistics transfer-function solution.

Use short names like `X`, `y`, and `coef` only inside local numerical routines
where the linear algebra convention is clearer than verbose naming.

## Data Object Names

Use:

- `time_data`
- `dec_data`
- `win_data`
- `spec_data`
- `eval_data`
- `gathered_data`
- `reg_data`

These names are already established in the processing code and are acceptable
for internal functions. In app-facing DTOs, prefer fuller names such as
`spectra_summary` or `result_summary`.

## Filesystem Names

Use:

- `dir_path` for a directory argument in low-level code.
- `file_path` for a specific file.
- `metadata_path` for metadata JSON/YAML.
- `results_path` for a results directory or file path, depending on existing
  local convention.

For app-facing schemas, prefer explicit names:

- `project_path`
- `flow_path`
- `configuration_path`
- `run_path`
- `result_path`

## Boolean Names

Use positive names:

- `enabled`
- `overwrite`
- `apply_scalings`
- `metadata_only`
- `is_valid`

Avoid negative boolean names such as `skip_validation` unless they match an
existing third-party API.
