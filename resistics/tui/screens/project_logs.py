"""Session diagnostic-log presentation for the project explorer."""

from __future__ import annotations

from rich.text import Text
from textual.widgets import RichLog, Static, TabbedContent

from resistics.tui.screens.project_base import _ProjectExplorerBase
from resistics.tui.state import DiagnosticLogEntry

_LEVEL_STYLES = {
    "SUCCESS": "bold green",
    "WARNING": "bold yellow",
    "ERROR": "bold red",
    "CRITICAL": "bold red reverse",
}


def _diagnostic_text(entry: DiagnosticLogEntry) -> Text:
    """Render one diagnostic without interpreting its content as Rich markup.

    Parameters
    ----------
    entry : DiagnosticLogEntry
        Structured diagnostic to render.

    Returns
    -------
    Text
        Safely styled Rich text.
    """
    value = Text()
    timestamp = entry.timestamp.astimezone().strftime("%H:%M:%S")
    value.append(f"{timestamp} ", style="dim")
    value.append(f"{entry.level:<8}", style=_LEVEL_STYLES.get(entry.level, "bold cyan"))
    value.append(f" {entry.source}: ", style="bold")
    value.append(entry.message)
    if entry.location:
        value.append(f" ({entry.location})", style="dim")
    if entry.exception:
        value.append("\n")
        value.append(entry.exception, style="red")
    return value


class _ProjectLogsMixin(_ProjectExplorerBase):
    """Drain the thread-safe diagnostic buffer on the Textual UI thread."""

    def _start_log_updates(self) -> None:
        self.set_interval(0.25, self._drain_diagnostics)

    def _activate_log_view(self) -> None:
        self.call_after_refresh(self._drain_diagnostics)

    def _drain_diagnostics(self) -> None:
        if self.query_one(TabbedContent).active != "logs":
            return
        log = self.query_one("#logs-log", RichLog)
        render_width = log.scrollable_content_region.width
        if render_width < 1:
            return
        if render_width != self._log_render_width:
            log.clear()
            self._log_cursor = 0
            self._dropped_logs = 0
            self._log_render_width = render_width
        batch = self.log_buffer.read_after(self._log_cursor)
        for entry in batch.entries:
            log.write(_diagnostic_text(entry), width=render_width)
        self._log_cursor = batch.cursor
        self._dropped_logs += batch.dropped
        status = f"Session logs · INFO+ · {batch.retained} retained"
        if self._dropped_logs:
            status += f" · {self._dropped_logs} older entries dropped"
        self.query_one("#logs-status", Static).update(status)
