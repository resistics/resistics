"""Deterministic, offline MTH5 fixtures for the executable tutorials."""

from pathlib import Path

import numpy as np
from mt_metadata.timeseries import Electric, Magnetic, Run, Station, Survey
from mth5.mth5 import MTH5


_SAMPLE_RATE = 16.0
_N_SAMPLES = 512
_START = "2020-01-01T00:00:00+00:00"
_END = "2020-01-01T00:00:31.9375+00:00"


def create_demo_mth5(root: Path) -> Path:
    """Create a small two-station MTH5 file beneath ``root``."""
    root.mkdir(parents=True, exist_ok=True)
    path = root / "recordings.mth5"
    mth5 = MTH5(file_version="0.2.0")
    mth5.open_mth5(path, mode="w")
    try:
        survey = Survey(id="demo", name="Offline documentation survey")
        mth5.add_survey("demo", survey)
        _add_station(mth5, "target", latitude=51.0, phase=0.0)
        _add_station(mth5, "remote", latitude=51.1, phase=0.25)
    finally:
        mth5.close_mth5()
    return path


def _add_station(
    mth5: MTH5, station_id: str, *, latitude: float, phase: float
) -> None:
    """Add one deterministic four-channel run to the tutorial survey."""
    station = Station(id=station_id)
    station.location.latitude = latitude
    station.location.longitude = -1.0
    mth5.add_station(station_id, station, survey="demo")

    run = Run(id="run001", sample_rate=_SAMPLE_RATE)
    run.time_period.start = _START
    run.time_period.end = _END
    mth5.add_run(station_id, "run001", run, survey="demo")

    samples = np.arange(_N_SAMPLES, dtype=np.float64)
    channels = (
        ("ex", "electric", Electric, "mV/km"),
        ("ey", "electric", Electric, "mV/km"),
        ("hx", "magnetic", Magnetic, "nT"),
        ("hy", "magnetic", Magnetic, "nT"),
    )
    for index, (component, channel_type, metadata_type, units) in enumerate(
        channels, start=1
    ):
        metadata = metadata_type(
            component=component,
            sample_rate=_SAMPLE_RATE,
            units=units,
        )
        metadata.time_period.start = _START
        metadata.time_period.end = _END
        data = np.sin(2 * np.pi * index * samples / _N_SAMPLES + phase).astype(
            np.float32
        )
        mth5.add_channel(
            station_id,
            "run001",
            component,
            channel_type,
            data,
            channel_dtype="float32",
            channel_metadata=metadata,
            survey="demo",
        )
