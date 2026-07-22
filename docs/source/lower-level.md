(lower-level)=
# Lower-level library use

The terminal interface is optional. Resistics separates source access, in-memory
data, processing definitions, execution, and persisted results so each layer can
be used independently.

(lower-level-data)=
## Data boundaries

`resistics.project` owns MTH5 discovery and handle lifetime. A loaded project
provides survey, station, run, channel, timing, and sampling-frequency summaries
without reading every sample. `Project.read_run()` converts a selected MTH5 run
into the `TimeData` and `TimeMetadata` containers used by processing code.

Use `resistics.sampling` for high-resolution timestamps and sample/time
conversion, and `resistics.time` for time-series containers and processors.
MTH5 remains the source-of-truth metadata boundary; Resistics metadata adds the
processing and persistence information required by derived artifacts.

(lower-level-processing)=
## Processing boundaries

Individual processors are Pydantic models derived from `ResisticsProcess`.
They can be constructed and run directly, or named by qualified class path in a
`FlowDefinition`. A `ParameterSet` supplies process configuration, while a
`JobDefinition` adds project scope, gather criteria, output naming, overwrite
policy, validation, progress, and cancellation.

This separation lets a notebook call one numerical processor, a service execute
a validated flow, and the TUI run a project job without maintaining different
processing implementations.

(lower-level-tutorials)=
## Data-container examples

```{toctree}
:maxdepth: 2

tutorial-datatypes/index
```
