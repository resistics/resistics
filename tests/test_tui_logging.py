import sys
import warnings
from datetime import UTC, datetime
from threading import Thread

import pytest
from loguru import logger
from pydantic import BaseModel, ValidationError

from resistics.tui import DiagnosticLogEntry
from resistics.tui.logging import _TuiDiagnosticCapture, _TuiLogBuffer


def diagnostic(message: str) -> DiagnosticLogEntry:
    return DiagnosticLogEntry(
        timestamp=datetime.now(UTC),
        level="INFO",
        source=__name__,
        message=message,
    )


def test_diagnostic_log_entry_is_a_frozen_serializable_pydantic_model():
    entry = DiagnosticLogEntry(
        timestamp=datetime(2026, 7, 21, tzinfo=UTC),
        level="WARNING",
        source="UserWarning",
        message="MTH5 metadata is incomplete",
        location="example.py:42",
    )

    assert isinstance(entry, BaseModel)
    assert entry.model_config.get("frozen")
    assert DiagnosticLogEntry.model_validate_json(entry.model_dump_json()) == entry
    assert DiagnosticLogEntry.model_json_schema()["properties"]["source"]["description"]
    with pytest.raises(ValidationError):
        setattr(entry, "message", "changed")  # noqa: B010 - exercise frozen runtime


def test_diagnostic_buffer_is_thread_safe_incremental_and_bounded():
    buffer = _TuiLogBuffer(max_entries=3)
    threads = [
        Thread(target=buffer.append, args=(diagnostic(str(index)),))
        for index in range(8)
    ]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    first = buffer.read_after(0)
    assert first.retained == 3
    assert first.dropped == 5
    assert len(first.entries) == 3
    assert {entry.message for entry in first.entries}.issubset(
        {str(index) for index in range(8)}
    )
    assert buffer.read_after(first.cursor).entries == ()


def test_diagnostic_capture_filters_debug_and_recovers_after_loguru_reconfigure():
    buffer = _TuiLogBuffer()
    replacement_messages = []
    original_showwarning = warnings.showwarning

    with _TuiDiagnosticCapture(buffer) as capture:
        logger.debug("not retained")
        logger.info("before dependency import")
        warnings.showwarning(
            UserWarning("captured Python warning"),
            UserWarning,
            "mth5_reader.py",
            17,
        )
        logger.configure(
            handlers=[
                {
                    "sink": lambda message: replacement_messages.append(
                        message.record["message"]
                    ),
                    "level": "INFO",
                }
            ]
        )
        logger.warning("dependency-owned sink")
        capture.reinstall()
        try:
            raise RuntimeError("diagnostic failure")
        except RuntimeError:
            logger.exception("after sink reinstall")

    entries = buffer.read_after(0).entries
    messages = [entry.message for entry in entries]
    assert messages == [
        "before dependency import",
        "captured Python warning",
        "after sink reinstall",
    ]
    assert replacement_messages == ["dependency-owned sink"]
    assert entries[1].source == "UserWarning"
    assert entries[1].location == "mth5_reader.py:17"
    assert entries[2].exception is not None
    assert "RuntimeError: diagnostic failure" in entries[2].exception
    assert warnings.showwarning is original_showwarning
    logger.remove()
    logger.add(sys.stderr, level="INFO")
