"""Project job execution and structured progress presentation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual import work
from textual.widgets import RichLog, Static, TabbedContent

from resistics.tui.screens.project_base import _ProjectExplorerBase

if TYPE_CHECKING:
    from resistics.job import JobProgressEvent


def _progress_details(event: JobProgressEvent) -> tuple[str, str]:
    """Return the counter suffix and activity status for a job event.

    :param event: Job event that may contain fine-grained process progress.

    :return: Counter suffix and complete activity-status text.
    """
    status = f"{event.job_name}: {event.state.value}"
    if event.progress is None:
        return "", status
    if event.progress.total is None:
        return f" [{event.progress.current}]", status
    counter = f"{event.progress.current}/{event.progress.total}"
    return f" [{counter}]", f"{event.job_name}: {event.progress.task} {counter}"


class _ProjectJobsMixin(_ProjectExplorerBase):
    """Execute validated jobs and present their structured lifecycle."""

    def _submission_confirmed(self, confirmed: bool | None) -> None:
        if confirmed:
            self._execute_selected_job()

    @work(thread=True, exclusive=True, group="processing-job")
    def _execute_selected_job(self) -> None:
        validation = self.selected_validation
        if validation is None or validation.resolved_job is None:
            raise ValueError("A resolved job is required for execution")
        self.app.call_from_thread(self._set_running)
        self.service.run_job(
            validation,
            lambda event: self.app.call_from_thread(self._show_progress, event),
        )

    def _set_running(self) -> None:
        self.job_state = self._job_state_type.running
        self.query_one("#activity-status", Static).update("Job running")
        self.query_one("#activity-log", RichLog).clear()
        tabs = self.query_one(TabbedContent)
        if tabs.active == "activity":
            self.refresh_bindings()
        else:
            tabs.active = "activity"

    def _show_progress(self, event: JobProgressEvent) -> None:
        self.job_state = event.state
        context = []
        if event.station:
            station = (
                f"{event.survey}/{event.station}" if event.survey else event.station
            )
            context.append(f"station {station}")
        elif event.survey:
            context.append(f"survey {event.survey}")
        if event.run:
            context.append(f"run {event.run}")
        if event.sample_rate is not None:
            context.append(f"sample rate {event.sample_rate:g} Hz")
        target = f" — {', '.join(context)}" if context else ""
        progress, status = _progress_details(event)
        line = (
            f"[{event.state.value}] {event.message}{progress}{target} "
            f"({event.elapsed_seconds:.1f}s)"
        )
        if event.error:
            line += f"\n[red]{event.error}[/red]"
        self.query_one("#activity-log", RichLog).write(line)
        self.query_one("#activity-status", Static).update(status)
        if event.state in {
            self._job_state_type.completed,
            self._job_state_type.failed,
            self._job_state_type.cancelled,
        }:
            reload_jobs = "jobs" in self._loaded_sections
            reload_data = "data" in self._loaded_sections
            self._start_new_load_generation()
            self.service.invalidate("project", "jobs")
            self._loaded_sections.discard("jobs")
            self._loaded_sections.discard("data")
            if reload_jobs:
                self._request_explorer_section("jobs", force=True)
            if reload_data:
                self._request_explorer_section("data", force=True)
        self.refresh_bindings()

    def _check_selected_job_action(self, active: str) -> bool | None:
        """Check job execution eligibility from cached validation summaries.

        :param active: Identifier of the active tab.

        :return: Whether a selected job can run, or ``None`` to disable it.
        """
        if active != "jobs":
            return False
        highlighted_path = self._highlighted_job_path()
        if highlighted_path is not None:
            summary = next(
                (
                    value
                    for value in self.job_summaries.values()
                    if value.path == highlighted_path
                ),
                None,
            )
            return True if summary is not None and summary.is_valid else None
        return (
            True
            if self.selected_validation is not None and self.selected_validation.ok
            else None
        )

    def action_cancel_job(self) -> None:
        if self.job_runner is None or self.job_state != self._job_state_type.running:
            self.notify("No active job")
            return
        self.service.cancel_job()
        self.notify("Cancellation requested; the current step will finish first")
