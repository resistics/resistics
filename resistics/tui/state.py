"""Typed state shared by the terminal application and project screen."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from resistics.explorer import IndexedJob, IndexedResource, ProjectExplorerState
    from resistics.project import Project, RunSummary

TIME_PLOT_MAX_POINTS = 5_000

type TimePlotSelection = tuple[str, str, str, str | None]
type PlotTarget = (
    tuple[Literal["flow", "job", "spectra", "transfer_function"], Path]
    | tuple[Literal["project"], None]
    | tuple[Literal["time"], TimePlotSelection]
)
type ExplorerView = Literal[
    "project", "data", "flows", "parameters", "criteria", "jobs"
]


class DiagnosticLogEntry(BaseModel):
    """Serializable diagnostic captured while the terminal UI is running.

    **Attributes**

    - **model_config** — Pydantic frozen-model configuration.
    - **timestamp** — Timestamp attached to the diagnostic.
    - **level** — Logging severity name.
    - **source** — Module or warning category that emitted the entry.
    - **message** — Human-readable diagnostic message.
    - **location** — Source file and line when available.
    - **exception** — Formatted exception detail when available.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    timestamp: datetime = Field(description="Timestamp attached to the diagnostic.")
    level: str = Field(description="Logging severity name.")
    source: str = Field(
        description="Module or warning category that emitted the entry."
    )
    message: str = Field(description="Human-readable diagnostic message.")
    location: str | None = Field(
        default=None, description="Source file and line when available."
    )
    exception: str | None = Field(
        default=None, description="Formatted exception detail when available."
    )


class ProjectDataDeletionRequest(BaseModel):
    """User-selected scope for one destructive Data-tab action.

    **Attributes**

    - **model_config** — Pydantic frozen-model configuration.
    - **output_label** — Derived-data namespace to delete, or ``None`` for all project data.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    output_label: str | None = None


@dataclass(frozen=True)
class _ProjectOpenResult:
    """Immutable outcome returned by a project-opening thread worker.

    **Attributes**

    - **generation** — App generation that requested the project.
    - **project_path** — Project directory requested by the user.
    - **project** — Open project on success.
    - **diagnostics** — Python warnings captured while opening the project.
    - **error** — User-facing opening error on failure.
    """

    generation: int
    project_path: Path
    project: Project | None = None
    diagnostics: tuple[DiagnosticLogEntry, ...] = ()
    error: str | None = None


@dataclass(frozen=True)
class _ExplorerLoadResult:
    """Immutable discovery result passed from a worker to the UI thread.

    **Attributes**

    - **generation** — Screen generation that requested the discovery operation.
    - **section** — Explorer view owning the result.
    - **state** — Project catalogue result for Project and Data views.
    - **resources** — Parsed YAML resources for a resource view.
    - **jobs** — Validated jobs for the Jobs view.
    - **runs** — MTH5 run summaries preloaded for Data plot selection.
    - **error** — User-facing discovery error on failure.
    """

    generation: int
    section: ExplorerView
    state: ProjectExplorerState | None = None
    resources: tuple[IndexedResource, ...] = ()
    jobs: tuple[IndexedJob, ...] = ()
    runs: tuple[RunSummary, ...] = ()
    error: str | None = None


@dataclass
class _ProjectActionState:
    """Cache action eligibility that would otherwise require project I/O.

    **Attributes**

    - **plot_targets** — Cached plot targets for the Project and Data tabs.
    - **valid_flow_paths** — Flow files that passed validation during the latest tree population.
    - **has_project_data_to_delete** — Whether the latest data catalogue contains removable derived data.
    """

    plot_targets: dict[str, PlotTarget | None] = field(
        default_factory=lambda: {"project": None, "data": None}
    )
    valid_flow_paths: set[Path] = field(default_factory=set)
    has_project_data_to_delete: bool = False
