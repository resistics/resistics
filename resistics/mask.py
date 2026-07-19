"""Per-run window masks used while gathering evaluation-frequency data.

Masks are deliberately persisted beside a run, independently of spectra and
evaluation-frequency artifacts.  A mask table is indexed by global window and
has one boolean column per evaluation-frequency index; ``True`` means that the
window is admissible.
"""

from __future__ import annotations

import re
from datetime import time
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd
from pydantic import ConfigDict, Field, field_validator, model_validator

from resistics.common import (
    History,
    ResisticsData,
    ResisticsModel,
    ResisticsProcess,
    ResisticsWriter,
    WriteableMetadata,
    validate_output_label,
)
from resistics.decimate import DecimationParameters
from resistics.sampling import HighResDateTime
from resistics.window import WindowedData, get_win_starts

SAFE_MASK_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def validate_mask_name(value: str) -> str:
    """Validate a mask artifact name that is safe as one path component."""
    if not SAFE_MASK_NAME.fullmatch(value):
        raise ValueError(
            "Mask names must contain only letters, numbers, '.', '_' and '-', "
            "and must start with a letter or number"
        )
    return value


class AbsoluteTimeRange(ResisticsModel):
    """One inclusive absolute UTC interval."""

    model_config = ConfigDict(extra="forbid")
    from_time: str
    to_time: str

    @model_validator(mode="after")
    def validate_order(self) -> AbsoluteTimeRange:
        if _utc_timestamp(self.from_time) > _utc_timestamp(self.to_time):
            raise ValueError("Absolute time range from_time must be <= to_time")
        return self


class DailyTimeRange(ResisticsModel):
    """One recurring inclusive UTC time-of-day interval."""

    model_config = ConfigDict(extra="forbid")
    from_time: time
    to_time: time


class WindowMaskLevelMetadata(ResisticsModel):
    """Window/evaluation layout recorded for one mask level."""

    level: int
    fs: float
    n_wins: int
    win_size: int
    olap_size: int
    index_offset: int
    n_evaluation_frequencies: int
    evaluation_frequencies: list[float]


class WindowMaskMetadata(WriteableMetadata):
    """Metadata needed to validate a mask against an evaluation artifact."""

    survey: str
    station: str
    run: str
    sample_rate: float
    ref_time: HighResDateTime
    levels: list[WindowMaskLevelMetadata]
    history: History = Field(default_factory=History)


class WindowMask(ResisticsData):
    """Boolean mask tables, keyed by decimation level."""

    def __init__(
        self, metadata: WindowMaskMetadata, tables: dict[int, pd.DataFrame]
    ) -> None:
        self.metadata = metadata
        self.tables = tables
        self._validate()

    def _validate(self) -> None:
        expected_levels = {item.level for item in self.metadata.levels}
        if set(self.tables) != expected_levels:
            raise ValueError(
                f"Mask table levels {sorted(self.tables)} do not match metadata "
                f"levels {sorted(expected_levels)}"
            )
        for level_meta in self.metadata.levels:
            table = self.tables[level_meta.level]
            expected_index = pd.RangeIndex(
                level_meta.index_offset,
                level_meta.index_offset + level_meta.n_wins,
            )
            expected_columns = list(range(level_meta.n_evaluation_frequencies))
            if not table.index.equals(expected_index):
                raise ValueError(
                    f"Mask level {level_meta.level} global-window index is invalid"
                )
            if list(table.columns) != expected_columns:
                raise ValueError(
                    f"Mask level {level_meta.level} columns must be {expected_columns}"
                )
            if any(not pd.api.types.is_bool_dtype(dtype) for dtype in table.dtypes):
                raise ValueError(
                    f"Mask level {level_meta.level} values must be boolean"
                )

    def get_keep(self, level: int, evaluation_frequency_index: int) -> pd.Series:
        """Return the keep decision indexed by global window."""
        try:
            return self.tables[level][evaluation_frequency_index]
        except KeyError as exc:
            raise ValueError(
                f"Mask has no level {level}, evaluation-frequency index "
                f"{evaluation_frequency_index}"
            ) from exc


def get_run_mask_path(
    project_path: Path,
    run_batch: dict[str, Any],
    name: str,
    output_label: str | None = None,
) -> Path:
    """Return the canonical path for a named run mask.

    ``output_label=None`` preserves the legacy unnamespaced path for direct
    library callers.  Processing jobs always supply their output label.
    """
    validate_mask_name(name)
    path = (
        Path(project_path)
        / "data"
        / run_batch["survey"]
        / run_batch["station"]
        / run_batch["run"]
        / "masks"
    )
    if output_label is not None:
        path = path / validate_output_label(output_label)
    return path / name


class WindowMaskWriter(ResisticsWriter):
    """Write mask metadata and compressed boolean level arrays."""

    def run(self, dir_path: Path, mask: WindowMask) -> None:
        """Write mask metadata and boolean tables beneath ``dir_path``."""
        from resistics.errors import WriteError

        if not self._check_dir(dir_path):
            raise WriteError(dir_path, "Unable to write mask directory")
        arrays = {
            str(level): table.to_numpy(dtype=bool)
            for level, table in mask.tables.items()
        }
        np.savez_compressed(dir_path / "data", **arrays)
        metadata = mask.metadata.model_copy(deep=True)
        metadata.history.add_record(self._get_record(dir_path, type(mask)))
        metadata.write(dir_path / "metadata.json")


class WindowMaskReader(ResisticsProcess):
    """Read a persisted per-run mask."""

    def run(self, dir_path: Path) -> WindowMask:
        """Read mask metadata and boolean tables from ``dir_path``."""
        from resistics.errors import ReadError

        if not dir_path.is_dir():
            raise ReadError(dir_path, "Mask directory does not exist")
        metadata = WindowMaskMetadata.model_validate_json(
            (dir_path / "metadata.json").read_bytes()
        )
        with np.load(dir_path / "data.npz") as stored:
            arrays = {int(level): stored[level].astype(bool) for level in stored.files}
        tables = {}
        for level_meta in metadata.levels:
            tables[level_meta.level] = pd.DataFrame(
                arrays[level_meta.level],
                index=pd.RangeIndex(
                    level_meta.index_offset,
                    level_meta.index_offset + level_meta.n_wins,
                ),
                columns=range(level_meta.n_evaluation_frequencies),
                dtype=bool,
            )
        metadata.history.add_record(
            self._get_record([f"Window mask read from {dir_path}"])
        )
        return WindowMask(metadata, tables)


class WindowMaskProcess(ResisticsProcess):
    """Base for pure window-mask calculations with a self-writing executor."""

    input_types: ClassVar[dict[str, str]] = {
        "win_data": "windowed_data",
        "dec_params": "decimation_parameters",
    }
    output_type: ClassVar[str] = "mask_result"
    runtime_requirements: ClassVar[list[str]] = ["project_path", "run_batch"]
    model_config = ConfigDict(extra="forbid")

    def execute(self, inputs: dict[str, Any], context: Any) -> dict[str, str]:
        """Calculate, persist, and return the path of a named mask."""
        mask = self.run(inputs["win_data"], inputs["dec_params"], context["run_batch"])
        path = get_run_mask_path(
            Path(context["project_path"]),
            context["run_batch"],
            self.name,
            context.get("output_label"),
        )
        WindowMaskWriter().run(path, mask)
        return {"mask_path": str(path)}

    def _metadata(
        self,
        win_data: WindowedData,
        dec_params: DecimationParameters,
        run_batch: dict[str, Any] | None,
    ) -> WindowMaskMetadata:
        run_batch = run_batch or {"survey": "", "station": "", "run": ""}
        levels = []
        for level, source in enumerate(win_data.metadata.levels_metadata):
            if level >= dec_params.n_levels:
                raise ValueError(
                    f"Window level {level} is absent from decimation parameters"
                )
            levels.append(
                WindowMaskLevelMetadata(
                    level=level,
                    fs=source.fs,
                    n_wins=source.n_wins,
                    win_size=source.win_size,
                    olap_size=source.olap_size,
                    index_offset=source.index_offset,
                    n_evaluation_frequencies=dec_params.per_level,
                    evaluation_frequencies=dec_params.get_eval_freqs(level),
                )
            )
        history = win_data.metadata.history.model_copy(deep=True)
        history.add_record(self._get_record([f"Calculated window mask {self.name}"]))
        return WindowMaskMetadata(
            survey=run_batch["survey"],
            station=run_batch["station"],
            run=run_batch["run"],
            sample_rate=dec_params.fs,
            ref_time=win_data.metadata.ref_time,
            levels=levels,
            history=history,
        )

    def _repeat_decision(
        self, metadata: WindowMaskMetadata, decisions: dict[int, np.ndarray]
    ) -> WindowMask:
        tables = {}
        for item in metadata.levels:
            values = np.repeat(
                np.asarray(decisions[item.level], dtype=bool)[:, None],
                item.n_evaluation_frequencies,
                axis=1,
            )
            tables[item.level] = pd.DataFrame(
                values,
                index=pd.RangeIndex(item.index_offset, item.index_offset + item.n_wins),
                columns=range(item.n_evaluation_frequencies),
                dtype=bool,
            )
        return WindowMask(metadata, tables)


class TimeMask(WindowMaskProcess):
    """Mask windows by inclusive absolute and recurring UTC start times."""

    name: ClassVar[str] = "TimeMask"
    include_in_default_parameters: ClassVar[bool] = True

    absolute_include: list[AbsoluteTimeRange] = Field(default_factory=list)
    absolute_exclude: list[AbsoluteTimeRange] = Field(default_factory=list)
    daily_include: list[DailyTimeRange] = Field(
        default_factory=lambda: [DailyTimeRange(from_time=time(20), to_time=time(6))]
    )
    daily_exclude: list[DailyTimeRange] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_criterion(self) -> TimeMask:
        if not any(
            (
                self.absolute_include,
                self.absolute_exclude,
                self.daily_include,
                self.daily_exclude,
            )
        ):
            raise ValueError("TimeMask requires at least one time criterion")
        return self

    def run(
        self,
        win_data: WindowedData,
        dec_params: DecimationParameters,
        run_batch: dict[str, Any] | None = None,
    ) -> WindowMask:
        """Build a mask from absolute and recurring UTC time constraints."""
        metadata = self._metadata(win_data, dec_params, run_batch)
        decisions = {}
        absolute_include = [
            (_utc_timestamp(item.from_time), _utc_timestamp(item.to_time))
            for item in self.absolute_include
        ]
        absolute_exclude = [
            (_utc_timestamp(item.from_time), _utc_timestamp(item.to_time))
            for item in self.absolute_exclude
        ]
        for item in metadata.levels:
            starts = get_win_starts(
                metadata.ref_time,
                item.win_size,
                item.olap_size,
                item.fs,
                item.n_wins,
                item.index_offset,
            )
            decisions[item.level] = np.array(
                [
                    self._includes(
                        _utc_timestamp(value), absolute_include, absolute_exclude
                    )
                    for value in starts
                ],
                dtype=bool,
            )
        return self._repeat_decision(metadata, decisions)

    def _includes(
        self,
        value: pd.Timestamp,
        absolute_include: list[tuple[pd.Timestamp, pd.Timestamp]],
        absolute_exclude: list[tuple[pd.Timestamp, pd.Timestamp]],
    ) -> bool:
        if absolute_include and not any(a <= value <= b for a, b in absolute_include):
            return False
        if any(a <= value <= b for a, b in absolute_exclude):
            return False
        current = value.time().replace(tzinfo=None)
        if self.daily_include and not any(
            _time_in_range(current, item.from_time, item.to_time)
            for item in self.daily_include
        ):
            return False
        return not any(
            _time_in_range(current, item.from_time, item.to_time)
            for item in self.daily_exclude
        )


class ChannelAmplitudeLimits(ResisticsModel):
    """Inclusive peak absolute-amplitude limits for one channel."""

    model_config = ConfigDict(extra="forbid")
    minimum: float | None = None
    maximum: float | None = None

    @model_validator(mode="after")
    def validate_limits(self) -> ChannelAmplitudeLimits:
        if self.minimum is None and self.maximum is None:
            raise ValueError("At least one of minimum or maximum is required")
        if (
            self.minimum is not None
            and self.maximum is not None
            and self.minimum > self.maximum
        ):
            raise ValueError("minimum must be <= maximum")
        return self


class AbsoluteAmplitudeMask(WindowMaskProcess):
    """Keep windows whose per-channel peak absolute amplitudes are in range."""

    name: ClassVar[str] = "AbsoluteAmplitudeMask"
    include_in_default_parameters: ClassVar[bool] = True

    limits: dict[str, ChannelAmplitudeLimits] = Field(
        default_factory=lambda: {
            channel: ChannelAmplitudeLimits(maximum=100_000)
            for channel in ("Ex", "Ey", "Hx", "Hy")
        }
    )

    @field_validator("limits")
    @classmethod
    def require_limits(
        cls, value: dict[str, ChannelAmplitudeLimits]
    ) -> dict[str, ChannelAmplitudeLimits]:
        if not value:
            raise ValueError("AbsoluteAmplitudeMask requires at least one channel")
        return value

    def run(
        self,
        win_data: WindowedData,
        dec_params: DecimationParameters,
        run_batch: dict[str, Any] | None = None,
    ) -> WindowMask:
        """Build a mask by applying each channel's amplitude limits."""
        missing = sorted(set(self.limits) - set(win_data.metadata.chans))
        if missing:
            raise ValueError(f"Amplitude mask channels not found: {missing}")
        metadata = self._metadata(win_data, dec_params, run_batch)
        channel_indices = {
            channel: win_data.metadata.chans.index(channel) for channel in self.limits
        }
        decisions = {}
        for item in metadata.levels:
            level_data = win_data.get_level(item.level)
            keep = np.ones(item.n_wins, dtype=bool)
            for channel, limits in self.limits.items():
                peaks = np.max(
                    np.abs(level_data[:, channel_indices[channel], :]), axis=-1
                )
                valid = np.isfinite(peaks)
                if limits.minimum is not None:
                    valid &= peaks >= limits.minimum
                if limits.maximum is not None:
                    valid &= peaks <= limits.maximum
                keep &= valid
            decisions[item.level] = keep
        return self._repeat_decision(metadata, decisions)


def _utc_timestamp(value: Any) -> pd.Timestamp:
    result = pd.Timestamp(value)
    return (
        result.tz_localize("UTC") if result.tzinfo is None else result.tz_convert("UTC")
    )


def _time_in_range(value: time, start: time, end: time) -> bool:
    return start <= value <= end if start <= end else value >= start or value <= end
