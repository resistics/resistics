"""Common state contract for project explorer presentation mixins."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from textual.screen import Screen

from resistics.tui.logging import _legacy_warning_entry, _TuiLogBuffer
from resistics.tui.services import ProjectExplorerService
from resistics.tui.state import ExplorerView, _ProjectActionState

if TYPE_CHECKING:
    from pydantic import BaseModel

    from resistics.job import (
        JobRunner,
        JobState,
        JobSummary,
        JobValidation,
    )
    from resistics.project import Project, ProjectDataItem


class _ProjectExplorerBase(Screen[None]):
    """Hold project explorer state shared across presentation concerns.

    :param project: Open project displayed by the explorer.
    :param startup_warnings: Compatibility warnings supplied by a direct screen caller.
    :param log_buffer: Application-session diagnostic buffer.
    """

    DATA_CATEGORIES = [
        ("Time data", "time"),
        ("Spectra/evaluations", "spectra"),
        ("Masks", "mask"),
        ("Transfer functions", "transfer_function"),
        ("Other", "other"),
    ]

    def __init__(
        self,
        project: Project,
        startup_warnings: list[str] | None = None,
        *,
        log_buffer: _TuiLogBuffer | None = None,
    ):
        super().__init__()
        from resistics.job import JobState

        self.project = project
        self.service = ProjectExplorerService(project)
        self.flow_paths: dict[str, Path] = {}
        self.parameter_paths: dict[str, Path] = {}
        self.criteria_paths: dict[str, Path] = {}
        self.selected_flow_path: Path | None = None
        self.selected_parameter_path: Path | None = None
        self.selected_criteria_path: Path | None = None
        self.job_summaries: dict[str, JobSummary] = {}
        self.selected_job_path: Path | None = None
        self.selected_validation: JobValidation | None = None
        self._pending_job_path: Path | None = None
        self.job_state: JobState | None = None
        self._job_state_type = JobState
        self.data_items: dict[str, ProjectDataItem] = {}
        self.action_state = _ProjectActionState()
        self.editing_yaml = False
        self.editing_path: Path | None = None
        self.editing_model: type[BaseModel] | None = None
        self.editing_editor_id: str | None = None
        self.startup_warnings = startup_warnings or []
        self.log_buffer = log_buffer or _TuiLogBuffer()
        if self.startup_warnings:
            self.log_buffer.extend(
                tuple(
                    _legacy_warning_entry(message) for message in self.startup_warnings
                )
            )
        self._log_cursor = 0
        self._dropped_logs = 0
        self._log_render_width = 0
        self._load_generation = 0
        self._loaded_sections: set[ExplorerView] = set()
        self._loading_sections: set[ExplorerView] = set()

    @property
    def job_runner(self) -> JobRunner | None:
        """Return the active runner owned by the UI-neutral service."""
        return self.service.job_runner

    @property
    def _active_discoveries(self) -> int:
        """Retain the internal discovery-count probe used by pilot tests."""
        return self.service.active_discoveries

    def _request_explorer_section(
        self, section: ExplorerView, *, force: bool = False
    ) -> None:
        raise NotImplementedError

    def _start_new_load_generation(self) -> None:
        raise NotImplementedError

    def _highlighted_yaml_file(self) -> tuple[Path, str] | None:
        raise NotImplementedError

    def _highlighted_job_path(self) -> Path | None:
        raise NotImplementedError

    def _submission_confirmed(self, confirmed: bool | None) -> None:
        raise NotImplementedError
