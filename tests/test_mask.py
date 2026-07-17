"""Tests for persisted, evaluation-index-aware per-run masks."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from resistics.decimate import DecimationSetup
from resistics.mask import (
    AbsoluteAmplitudeMask,
    AbsoluteTimeRange,
    DailyTimeRange,
    TimeMask,
    WindowMaskReader,
    WindowMaskWriter,
    get_run_mask_path,
)
from resistics.testing import time_metadata_general
from resistics.window import WindowedData, WindowedLevelMetadata, WindowedMetadata


def make_windowed(data, *, fs=1.0, win_size=4, olap_size=1, index_offset=0):
    chans = ["Ex", "Hx"]
    source = time_metadata_general(chans, fs=fs).model_dump()
    source.update(
        fs=[fs],
        n_levels=1,
        levels_metadata=[
            WindowedLevelMetadata(
                fs=fs,
                n_wins=len(data),
                win_size=win_size,
                olap_size=olap_size,
                index_offset=index_offset,
            )
        ],
        ref_time=source["first_time"],
    )
    return WindowedData(WindowedMetadata(**source), {0: np.asarray(data)})


def decimation_parameters(fs=1.0):
    return DecimationSetup(n_levels=1, per_level=2, eval_freqs=[fs / 4, fs / 8]).run(fs)


def test_concrete_mask_names_are_fixed_class_properties():
    assert TimeMask.name == "TimeMask"
    assert AbsoluteAmplitudeMask.name == "AbsoluteAmplitudeMask"
    assert "name" not in TimeMask.model_fields
    assert "name" not in AbsoluteAmplitudeMask.model_fields
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        TimeMask(name="custom")


def test_mask_paths_are_namespaced_by_output_label():
    path = get_run_mask_path(
        Path("project"),
        {"survey": "survey", "station": "station", "run": "run"},
        "TimeMask",
        "field",
    )

    assert path == Path("project/data/survey/station/run/masks/field/TimeMask")


def test_absolute_amplitude_mask_uses_peak_all_channels_and_nonfinite_fails():
    windows = make_windowed(
        [
            [[-1, 2, -3, 4], [1, 1, 1, 1]],
            [[1, 5, 1, 1], [1, 1, 1, 1]],
            [[1, 2, 3, 4], [1, np.nan, 1, 1]],
        ]
    )
    process = AbsoluteAmplitudeMask(
        limits={"Ex": {"minimum": 3, "maximum": 4}, "Hx": {"maximum": 2}},
    )

    mask = process.run(windows, decimation_parameters())

    assert mask.tables[0].to_numpy().tolist() == [
        [True, True],
        [False, False],
        [False, False],
    ]


def test_time_mask_absolute_and_daily_cross_midnight_are_inclusive():
    data = np.ones((4, 2, 3601))
    windows = make_windowed(data, win_size=3601, olap_size=1)
    process = TimeMask(
        absolute_include=[
            AbsoluteTimeRange(
                from_time="2020-01-01T00:00:00Z",
                to_time="2020-01-01T02:00:00Z",
            )
        ],
        daily_include=[DailyTimeRange(from_time="23:00", to_time="01:00")],
    )

    mask = process.run(windows, decimation_parameters())

    assert mask.tables[0][0].tolist() == [True, True, False, False]


def test_mask_npz_round_trip_preserves_global_indices_and_bool_tables(tmp_path):
    windows = make_windowed([[[1, 1, 1, 1], [1, 1, 1, 1]]], index_offset=12)
    mask = AbsoluteAmplitudeMask(limits={"Ex": {"maximum": 2}}).run(
        windows,
        decimation_parameters(),
        {"survey": "survey", "station": "station", "run": "run"},
    )
    path = Path(tmp_path) / AbsoluteAmplitudeMask.name

    WindowMaskWriter().run(path, mask)
    restored = WindowMaskReader().run(path)

    pd.testing.assert_frame_equal(restored.tables[0], mask.tables[0])
    assert restored.metadata.run == "run"
