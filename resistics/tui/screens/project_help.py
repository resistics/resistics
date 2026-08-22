"""Contextual help for the project explorer tabs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from rich.markdown import Markdown
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, RichLog, Static

type ProjectTab = Literal[
    "project",
    "data",
    "flows",
    "parameters",
    "criteria",
    "jobs",
    "activity",
    "logs",
]


@dataclass(frozen=True)
class _TabHelp:
    """Short and detailed help belonging to one project tab."""

    title: str
    summary: str
    details: str


_PROCESSING_MODEL = """
## How processing fits together

**Job = Flow + Parameters + optional Criteria + Scope + Output label**

```text
MTH5 recordings
       |
       v
+------------------------------+
| Job                          |
| scope + output label         |
|                              |
| + Flow       (what runs)     |
| + Parameters (how it runs)   |
| + Criteria   (which data,    |
|               optional)      |
+---------------+--------------+
                |
                v
         Project outputs
```

- A **Flow** defines the ordered stages, processing functions, and data dependencies:
  *what runs*.
- **Parameters** configure those processing functions independently of the Flow:
  *how each step runs*.
- **Criteria** are optional policies used by criteria-aware Flows to choose admitted
  windows, masks, and remote-reference stations: *which data are used*.
- The Job's **Scope** limits surveys, stations, sampling frequencies, or stages.
- The **Output label** keeps the resulting project data in a named namespace.

A Job validates these references together before it runs. This makes a Flow reusable
with different settings, selection policies, targets, and output labels.
"""


def _processing_details(before: str, after: str = "") -> str:
    """Compose tab-specific guidance around the shared processing model.

    :param before: Tab-specific guidance placed before the shared model.
    :param after: Optional guidance placed after the shared model.
    :return: Complete detailed help for one processing-related tab.
    """
    sections = [before.strip(), _PROCESSING_MODEL.strip()]
    if after.strip():
        sections.append(after.strip())
    return "\n\n".join(sections)


TAB_HELP: dict[ProjectTab, _TabHelp] = {
    "project": _TabHelp(
        title="Project",
        summary=(
            "Summarises the MTH5 source, project reference time, and available "
            "recordings."
        ),
        details=_processing_details("""
## What this is

The Project tab is the high-level view of the current Resistics project. It identifies
the read-only MTH5 source, the project reference time, and the surveys, stations, runs,
channels, and sampling frequencies available for processing.

Resistics keeps flows, parameters, criteria, jobs, logs, and derived results in the
project directory. It does not write those artifacts back into the source MTH5 file.

## What you can do here

- Review the source and recording summary.
- Press **P** to open the project recording timeline when data are available.
- Press **R** to refresh the project and resource catalogues.
"""),
    ),
    "data": _TabHelp(
        title="Data",
        summary=(
            "Browse read-only MTH5 recordings and outputs created by processing jobs."
        ),
        details="""
## What this is

The Data tab presents two sources: the original MTH5 hierarchy and artifacts created
inside the Resistics project. MTH5 time series remain read-only; derived spectra,
masks, transfer functions, and other outputs live under the project data tree.

Selecting an item displays its metadata on the right. Time-series channels from MTH5
and supported project artifacts can be plotted.

## What you can do here

- Move through the tree and select an item to inspect its metadata.
- Use **]** and **[** to expand or collapse the highlighted branch.
- Press **P** when the highlighted item supports plotting.
- Use the Delete action carefully to clear project-created data; the source MTH5 file
  and project setup are retained.

## How it connects

Jobs read selected MTH5 recordings and write their outputs beneath the project data
tree. The output label chosen by a Job separates different processing runs.
""",
    ),
    "flows": _TabHelp(
        title="Flows",
        summary=(
            "Define the ordered processing stages, functions, and data dependencies—"
            "what runs."
        ),
        details=_processing_details(
            """
## What this is

A Flow is a reusable directed graph of concrete Python processing classes. Each stage
has a scope such as `run` or `station_rate`; each node identifies a process and maps
its named inputs to earlier nodes. Validation checks imports, graph order, input and
output types, configuration, and required runtime context before execution.

## What you can do here

- Select a Flow to inspect or edit its YAML.
- Press **P** to plot a valid Flow graph.
- Copy, delete, or restore the built-in Flow definitions.
- Refer to built-in processes by paths such as `resistics.time.RemoveMean`.
""",
            """
## Add a project processing plugin

Project plugins are trusted executable Python code. Put a module below the project's
`plugins/` package—for example, `plugins/example.py`:

```python
from typing import ClassVar

from resistics.common import ResisticsProcess


class PassThrough(ResisticsProcess):
    input_types: ClassVar[dict[str, str]] = {"time_data": "time_data"}
    output_type: ClassVar[str] = "time_data"
    include_in_default_parameters: ClassVar[bool] = True

    label: str = "custom"

    def run(self, time_data):
        return time_data
```

Reference the class by its import path in a Flow node:

```yaml
- id: custom
  process: plugins.example.PassThrough
  inputs:
    time_data: read
```

Configure the same qualified path in a Parameters file:

```yaml
processes:
  plugins.example.PassThrough:
    label: custom
```

Plugin classes use the same descriptor and Flow validation as built-in processes.
Keep them inside the project so their qualified paths remain importable when the YAML
is loaded later.
""",
        ),
    ),
    "parameters": _TabHelp(
        title="Parameters",
        summary=(
            "Configure processing functions independently of a flow—how each step runs."
        ),
        details=_processing_details("""
## What this is

A Parameter set maps qualified process-class paths to their configuration values.
Keeping configuration separate lets several Jobs reuse one Flow with different
numerical or operational settings. A process omitted from the mapping uses its model
defaults.

## What you can do here

- Select a Parameter set to inspect or edit its YAML.
- Copy, delete, or restore the built-in Parameter sets.
- Match each configuration key exactly to the process path used by the Flow.
- Use model validation errors to identify unsupported fields or values.
"""),
    ),
    "criteria": _TabHelp(
        title="Criteria",
        summary=(
            "Optionally control admitted windows, masks, and remote-reference "
            "selection—which data are used."
        ),
        details=_processing_details("""
## What this is

Criteria are optional station- and sampling-frequency policies for gathering persisted
evaluation data. They can combine named masks and choose no remote reference, an
automatic remote reference, or an explicit list of remote stations.

Criteria affect only Flows whose nodes request criteria configuration. A single-site
Job can omit them.

## What you can do here

- Select a Criteria file to inspect or edit its YAML.
- Define policies by `survey/station` and sampling frequency.
- Copy, delete, or restore the built-in Criteria examples.
- Reference the file from a Job that uses a criteria-aware Flow.
"""),
    ),
    "jobs": _TabHelp(
        title="Jobs",
        summary=(
            "Bind a flow, parameters, optional criteria, scope, and output label into "
            "a runnable submission."
        ),
        details=_processing_details("""
## What this is

A Job is the runnable project-level definition. It names a Flow and Parameter set,
optionally names Criteria, chooses a processing scope, and assigns an output label.
Validation resolves all referenced YAML and process classes before the Job can run.

## What you can do here

- Press **N** to create a Job from available project resources.
- Select a Job to inspect, edit, copy, delete, or validate its YAML.
- Press **P** to plot a valid Job and **J** to review and submit it.
- Use scope fields to restrict surveys, stations, sampling frequencies, or stages.
- Set `overwrite` deliberately when an output label already contains results.
"""),
    ),
    "activity": _TabHelp(
        title="Activity",
        summary="Follow the progress and outcome of the current or most recent job.",
        details="""
## What this is

The Activity tab displays structured progress for the current or most recently run Job.
Entries identify lifecycle state, processing task, target station or run, sampling
frequency, elapsed time, and any reported failure.

## What you can do here

- Follow a running Job without blocking the rest of the terminal interface.
- Press **C** to request cancellation; the current processing step finishes first.
- Return to Jobs to inspect or rerun the definition after completion.

## How it connects

Submitting a validated Job switches here automatically. Completed work appears in the
Data tab, while diagnostic warnings and full exception details remain available in
Session logs.
""",
    ),
    "logs": _TabHelp(
        title="Logs",
        summary=(
            "Inspect session diagnostics, warnings, errors, and complete tracebacks."
        ),
        details="""
## What this is

Session logs retain INFO-and-higher diagnostics emitted while this terminal application
is running. They are separate from Job activity and include dependency warnings,
feature failures, source locations, and complete exception tracebacks when available.

## What you can do here

- Scroll through retained entries to diagnose metadata, loading, plotting, or processing
  problems.
- Use the source location and traceback when reporting or reproducing a failure.
- Review the status line for retained and dropped-entry counts.

## How it connects

Errors from other tabs should produce a short notification and preserve their debugging
detail here. Logs belong to the current application session and are not processing
outputs stored in the project data tree.
""",
    ),
}


class ProjectTabHelpScreen(ModalScreen[None]):
    """Show scrollable contextual help for one project tab.

    :param tab: Project tab whose help should be displayed.
    """

    AUTO_FOCUS = "#project-tab-help-scroll"
    BINDINGS = [
        Binding("h", "close_help", "Close", priority=True),
        Binding("escape", "close_help", "Close", priority=True),
    ]

    CSS = """
    ProjectTabHelpScreen { align: center middle; background: transparent; }
    #project-tab-help-dialog {
        width: 88;
        max-width: 94%;
        height: 84%;
        padding: 1 2;
        border: round #faa881;
        background: #101010;
        color: #f7f4f2;
    }
    #project-tab-help-title { height: auto; text-style: bold; margin-bottom: 1; }
    #project-tab-help-scroll {
        height: 1fr;
        padding-right: 1;
        overflow-y: auto;
        scrollbar-gutter: stable;
        scrollbar-size-vertical: 1;
        background: #101010;
        background-tint: 0%;
    }
    #project-tab-help-scroll:focus {
        background: #101010;
        background-tint: 0%;
    }
    #project-tab-help-actions {
        height: auto;
        align-horizontal: right;
        margin-top: 1;
        background: #101010;
    }
    """

    def __init__(self, tab: ProjectTab):
        super().__init__()
        self.tab = tab
        self.help = TAB_HELP[tab]

    def compose(self) -> ComposeResult:
        with Vertical(id="project-tab-help-dialog"):
            yield Static(f"{self.help.title} help", id="project-tab-help-title")
            yield RichLog(
                wrap=True,
                markup=False,
                auto_scroll=False,
                id="project-tab-help-scroll",
            )
            with Horizontal(id="project-tab-help-actions"):
                yield Button("Close", id="close-project-tab-help")

    def on_mount(self) -> None:
        help_log = self.query_one("#project-tab-help-scroll", RichLog)
        help_log.write(Markdown(self.help.details))

    @on(Button.Pressed, "#close-project-tab-help")
    def close(self) -> None:
        self.action_close_help()

    def action_close_help(self) -> None:
        """Close the contextual help dialog."""
        self.dismiss(None)
