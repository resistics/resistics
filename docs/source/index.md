(home)=
# Resistics

Resistics is a Python library and terminal application for processing
magnetotelluric data. The current line uses MTH5 as its time-series source and
keeps project discovery, processing definitions, execution, and derived
artifacts behind explicit, reusable contracts.

## Current processing model

### MTH5-backed projects

A project points to one existing MTH5 file and opens it read-only. Its cached
survey, station, run, channel, and sampling-frequency summaries support fast
inspection, while selected runs are loaded into Resistics data containers only
when processing needs them. Open projects own their MTH5 handle and support
deterministic cleanup through a context manager or `close()`.

Derived spectra, masks, transfer functions, logs, and job records live in the
project directory rather than being written back into the source MTH5 file.

### Flows, parameters, and jobs

A flow is a serializable directed acyclic graph of concrete Python process
classes. Parameter sets configure those classes independently of the graph.
Jobs bind a flow and parameter set to project scope, optional gather criteria,
an output label, and execution policy. Validation resolves those references
before a job runs, and execution emits structured progress, cancellation, and
failure events.

Projects may name trusted plugin directories, allowing custom process classes
to participate through the same qualified-path and validation contracts as
built-in processes.

### TUI and standalone library

The `resistics` command opens a Textual terminal interface for creating and
opening projects, exploring MTH5 and derived data, editing flow resources,
validating and running jobs, viewing plots, and inspecting captured logs. Slow
project and job work runs outside the UI thread.

The TUI is an adapter over the same library APIs. Scripts, notebooks, services,
and alternative interfaces can use project, flow, job, processing, and plotting
objects directly without starting Textual.

## Start here

```{toctree}
:maxdepth: 2
:caption: User guide

getting-started
lower-level
custom-process
```

```{toctree}
:maxdepth: 2
:caption: API reference

resistics
```

```{toctree}
:maxdepth: 2
:caption: Contributor guide

contributing
docstrings
releasing
literature
```

## Indices

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
