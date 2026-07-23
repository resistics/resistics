(getting-started)=
# Getting started

The current development line is built around MTH5 projects. Start with an
existing MTH5 file, create a Resistics project around it, and then use either
the terminal application or the Python library.

(getting-started-installation)=
## Installation

Resistics requires Python 3.12 through 3.14. This branch currently depends on
an adjacent RegressionInC checkout, so its verified development layout is:

```text
<development-directory>/
├── regressioninc/
└── resistics/
```

From the `resistics` directory, create the locked environment with:

```console
uv sync --locked --all-groups
```

This is the maintained installation path for the development branch. A
standalone registry installation is not yet part of its verified release
contract.

(getting-started-tui)=
## Start the terminal interface

Launch the TUI from the locked environment:

```console
uv run --locked --no-sync resistics
```

The launcher can create a project from an existing MTH5 file or open an
existing project directory. Project screens expose MTH5 structure, derived
artifacts, flows, parameter sets, gather criteria, jobs, plots, and diagnostic
logs.

(getting-started-library)=
## Start from Python

Create a project once, then load it with deterministic MTH5 handle ownership:

```python
from resistics.project import init, load

init(
    "example-project",
    "recordings.mth5",
    ref_time="2020-01-01T00:00:00",
)

with load("example-project") as project:
    print(project.surveys)
    print(project.fs())
    run = project.read_run("survey", "station", "run")
```

The project keeps source data read-only. Processing outputs are written beneath
the project directory and separated by output label.

(getting-started-tutorials)=
## Continue with the tutorials

The executable tutorials use current MTH5, flow, job, calibration, remote
reference, and plotting APIs. Every example creates deterministic synthetic
data locally and runs during the documentation build.

```{toctree}
:maxdepth: 2

tutorials/index
```
