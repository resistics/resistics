"""Thread-safe diagnostic capture for the terminal application."""

from __future__ import annotations

import sys
import warnings
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from threading import Lock
from traceback import format_exception
from typing import TYPE_CHECKING

from loguru import logger

from resistics.tui.state import DiagnosticLogEntry

if TYPE_CHECKING:
    from types import TracebackType
    from typing import TextIO

    from loguru import Message


@dataclass(frozen=True)
class _SequencedDiagnostic:
    """Pair one diagnostic entry with its buffer sequence number.

    **Attributes**

    - **sequence** — Monotonically increasing buffer position.
    - **entry** — Serializable diagnostic stored at that position.
    """

    sequence: int
    entry: DiagnosticLogEntry


@dataclass(frozen=True)
class _DiagnosticRead:
    """Immutable incremental read returned to the presentation layer.

    **Attributes**

    - **entries** — Entries newer than the requested cursor and still retained.
    - **cursor** — Latest sequence observed by the read.
    - **retained** — Number of entries currently held by the buffer.
    - **dropped** — Entries lost between the requested cursor and the retained range.
    """

    entries: tuple[DiagnosticLogEntry, ...]
    cursor: int
    retained: int
    dropped: int


class _TuiLogBuffer:
    """Retain diagnostics safely for worker-thread writers.

    :param max_entries: Maximum diagnostics retained for one TUI session.

    :raises ValueError: If ``max_entries`` is less than one.
    """

    def __init__(self, max_entries: int = 2_000):
        if max_entries < 1:
            raise ValueError("max_entries must be at least one")
        self.max_entries = max_entries
        self._entries: deque[_SequencedDiagnostic] = deque(maxlen=max_entries)
        self._lock = Lock()
        self._sequence = 0

    def append(self, entry: DiagnosticLogEntry) -> int:
        """Append one entry and return its monotonically increasing sequence.

        :param entry: Diagnostic to retain.

        :return: Sequence assigned to the entry.
        """
        with self._lock:
            self._sequence += 1
            self._entries.append(_SequencedDiagnostic(self._sequence, entry))
            return self._sequence

    def extend(self, entries: tuple[DiagnosticLogEntry, ...]) -> None:
        """Append an immutable group of entries in order.

        :param entries: Diagnostics to append.
        """
        for entry in entries:
            self.append(entry)

    def read_after(self, cursor: int) -> _DiagnosticRead:
        """Return entries newer than a presentation cursor.

        :param cursor: Last sequence consumed by the presentation layer.

        :return: Incremental entries and the new cursor.
        """
        with self._lock:
            retained = len(self._entries)
            if not self._entries:
                return _DiagnosticRead((), self._sequence, 0, 0)
            oldest = self._entries[0].sequence
            dropped = max(0, oldest - cursor - 1)
            entries = tuple(
                item.entry for item in self._entries if item.sequence > cursor
            )
            return _DiagnosticRead(entries, self._sequence, retained, dropped)

    def write_loguru(self, message: Message) -> None:
        """Normalize one Loguru message into the session buffer.

        :param message: Loguru sink message carrying its structured record.
        """
        record = message.record
        exception = record["exception"]
        exception_text = None
        if exception is not None and exception.value is not None:
            exception_text = "".join(format_exception(exception.value)).rstrip()
        source = record["name"] or record["module"]
        self.append(
            DiagnosticLogEntry(
                timestamp=record["time"],
                level=record["level"].name,
                source=source,
                message=record["message"],
                location=f"{record['file'].path}:{record['line']}",
                exception=exception_text,
            )
        )

    def write_warning(
        self,
        message: Warning | str,
        category: type[Warning],
        filename: str,
        lineno: int,
        file: TextIO | None = None,
        line: str | None = None,
    ) -> None:
        """Normalize one uncaught Python warning without terminal output.

        :param message: Warning value or text.
        :param category: Warning class.
        :param filename: File that emitted the warning.
        :param lineno: Source line that emitted the warning.
        :param file: Unused warning-output stream required by ``warnings.showwarning``.
        :param line: Optional source line supplied by ``warnings.showwarning``.
        """
        del file, line
        self.append(
            DiagnosticLogEntry(
                timestamp=datetime.now(UTC),
                level="WARNING",
                source=category.__name__,
                message=str(message),
                location=f"{filename}:{lineno}",
            )
        )


def _warning_entry(warning: warnings.WarningMessage) -> DiagnosticLogEntry:
    """Convert a warning captured by ``catch_warnings`` into a diagnostic.

    :param warning: Captured Python warning.

    :return: Structured warning with its original category and location.
    """
    return DiagnosticLogEntry(
        timestamp=datetime.now(UTC),
        level="WARNING",
        source=warning.category.__name__,
        message=str(warning.message),
        location=f"{warning.filename}:{warning.lineno}",
    )


def _legacy_warning_entry(message: str) -> DiagnosticLogEntry:
    """Convert the established string-only project-open warning contract.

    :param message: Existing warning text supplied by a direct app caller.

    :return: Structured compatibility warning.
    """
    return DiagnosticLogEntry(
        timestamp=datetime.now(UTC),
        level="WARNING",
        source="ProjectOpenWarning",
        message=message,
    )


def _error_entry(source: str, message: str) -> DiagnosticLogEntry:
    """Return a structured error raised at a TUI feature boundary.

    :param source: Feature that failed.
    :param message: User-facing error detail.

    :return: Structured error entry.
    """
    return DiagnosticLogEntry(
        timestamp=datetime.now(UTC),
        level="ERROR",
        source=source,
        message=message,
    )


def _exception_entry(source: str, message: str, error: Exception) -> DiagnosticLogEntry:
    """Return a structured error with its complete traceback.

    :param source: Feature that failed.
    :param message: User-facing error detail.
    :param error: Exception raised at the feature boundary.

    :return: Structured error entry retaining debugging detail.
    """
    return DiagnosticLogEntry(
        timestamp=datetime.now(UTC),
        level="ERROR",
        source=source,
        message=message,
        exception="".join(format_exception(error)).rstrip(),
    )


class _TuiDiagnosticCapture:
    """Own process-global diagnostics during ``run_tui``.

    :param buffer: Session buffer receiving diagnostics.
    """

    def __init__(self, buffer: _TuiLogBuffer):
        self.buffer = buffer
        self._active = False
        self._original_showwarning = warnings.showwarning

    def __enter__(self) -> _TuiDiagnosticCapture:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        self.close()

    def start(self) -> None:
        """Replace terminal diagnostics with the in-memory session sink."""
        if self._active:
            return
        self._original_showwarning = warnings.showwarning
        warnings.showwarning = self.buffer.write_warning
        self._active = True
        self.reinstall()

    def reinstall(self) -> None:
        """Restore capture after a dependency reconfigures global Loguru sinks."""
        if not self._active:
            return
        logger.remove()
        logger.add(
            self.buffer.write_loguru,
            level="INFO",
            format="{message}",
            enqueue=False,
        )

    def close(self) -> None:
        """Restore the terminal logging behavior established before this feature."""
        if not self._active:
            return
        logger.remove()
        warnings.showwarning = self._original_showwarning
        logger.add(sys.stderr, level="INFO")
        self._active = False
