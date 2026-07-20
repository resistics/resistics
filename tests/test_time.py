"""
Test time data and processors
"""

from typing import Any

import numpy as np
import pandas as pd
import pytest

from resistics.errors import ChannelNotFoundError, ProcessRunError
from resistics.sampling import to_datetime
from resistics.testing import (
    time_data_random,
    time_data_simple,
)
from resistics.time import ChanMetadata, TimeData, TimeMetadata


def test_chan_metadata():
    """Test initialising a channel metadata"""
    chan_dict = {
        "name": "cat",
        "data_files": ["cat.exe"],
        "gain2": 5,
        "chopper": True,
        "dipole_dist": 80,
    }
    with pytest.raises(ValueError, match="Failed setting type"):
        ChanMetadata(**chan_dict)
    chan_dict["chan_type"] = "electric"
    chan_metadata = ChanMetadata(**chan_dict)
    assert chan_metadata.name == chan_dict["name"]
    assert chan_metadata.data_files == chan_dict["data_files"]
    assert chan_metadata.chan_type == chan_dict["chan_type"]
    assert chan_metadata.gain2 == chan_dict["gain2"]
    assert chan_metadata.chopper == chan_dict["chopper"]
    assert chan_metadata.dipole_dist == chan_dict["dipole_dist"]


@pytest.mark.parametrize(
    "time_metadata, chans_metadata, electric_chans, magnetic_chans",
    [
        (
            {
                "fs": 10,
                "n_samples": 1_000,
                "first_time": "2020-01-01 00:00:00",
                "last_time": "2020-01-01 00:01:39.9",
            },
            {
                "C1": ChanMetadata(name="C1", data_files=["a_file"], chan_type="test"),
                "croc": ChanMetadata(
                    name="croc", data_files=["cat.exe"], chan_type="danger"
                ),
            },
            [],
            [],
        ),
        (
            {
                "fs": 10,
                "n_samples": 1_000,
                "first_time": "2020-01-01 00:00:00",
                "last_time": "2020-01-01 00:01:39.9",
            },
            {
                "Ex": ChanMetadata(name="Ex", data_files=["a_file"]),
                "C1": ChanMetadata(name="C1", data_files=["a_file"], chan_type="test"),
                "cat": ChanMetadata(
                    name="cat", data_files=["cat.exe"], chan_type="electric"
                ),
                "dog": ChanMetadata(
                    name="dog", data_files=["dogs.txt"], chan_type="magnetic"
                ),
                "Bx": ChanMetadata(name="Bx", data_files=["Bx.txt"]),
                "Hy": ChanMetadata(name="Hy", data_files=["Hy.txt"]),
            },
            ["Ex", "cat"],
            ["dog", "Bx", "Hy"],
        ),
    ],
)
def test_time_metadata(
    time_metadata: dict[str, Any],
    chans_metadata: dict[str, dict[str, Any]],
    electric_chans: list[str],
    magnetic_chans: list[str],
):
    """Test initialising time metadata"""
    time_metadata["chans"] = list(chans_metadata.keys())
    time_metadata["chans_metadata"] = chans_metadata
    metadata = TimeMetadata(**time_metadata)
    # test attributes
    assert metadata.fs == time_metadata["fs"]
    assert metadata.chans == list(chans_metadata.keys())
    assert metadata.n_chans == len(chans_metadata)
    assert metadata.n_samples == time_metadata["n_samples"]
    assert metadata.first_time == to_datetime(time_metadata["first_time"])
    assert metadata.last_time == to_datetime(time_metadata["last_time"])
    # test methods
    assert metadata.dt == 1 / time_metadata["fs"]
    assert metadata.duration == to_datetime(time_metadata["last_time"]) - to_datetime(
        time_metadata["first_time"]
    )
    assert metadata.nyquist == time_metadata["fs"] / 2
    assert metadata.get_electric_chans() == electric_chans
    assert metadata.get_magnetic_chans() == magnetic_chans
    assert metadata.any_electric() == (len(electric_chans) > 0)
    assert metadata.any_magnetic() == (len(magnetic_chans) > 0)


@pytest.mark.parametrize(
    "fs, n_samples, first_time, time_data",
    [
        (1, 16, "2020-01-01 00:00:00", time_data_simple(1)),
        (10, 16, "2020-01-01 00:00:00", time_data_simple(10)),
        (0.1, 16, "2020-01-01 00:00:00", time_data_simple(0.1)),
        (1, 12000, "2020-01-01 00:00:00", time_data_random(1, n_samples=12000)),
        (10, 100000, "2020-01-01 00:00:00", time_data_random(10, n_samples=100000)),
        (0.1, 5000, "2020-01-01 00:00:00", time_data_random(0.1, n_samples=5000)),
    ],
)
def test_time_data(fs: float, n_samples: int, first_time: str, time_data: TimeData):
    """Test time data"""
    from resistics.sampling import to_datetime, to_timedelta

    chans = ["Ex", "Ey", "Hx", "Hy"]
    first_rstime = to_datetime(first_time)
    last_rstime = first_rstime + to_timedelta(1 / fs) * (n_samples - 1)

    # check metadata
    assert time_data.metadata.fs == fs
    assert time_data.metadata.chans == chans
    assert time_data.metadata.n_samples == n_samples
    assert time_data.metadata.n_chans == len(chans)
    assert time_data.metadata.first_time == first_rstime
    assert time_data.metadata.last_time == last_rstime
    # check the arrays and chan indexing
    for idx, chan in enumerate(time_data.metadata.chans):
        assert idx == time_data.get_chan_index(chan)
        np.testing.assert_equal(time_data[chan], time_data.data[idx, :])
        np.testing.assert_equal(time_data.get_chan(chan), time_data.data[idx, :])
    # try setting a channel
    time_data["Ex"] = np.ones(shape=(n_samples), dtype=np.float32)
    np.testing.assert_equal(time_data["Ex"], np.ones(shape=(n_samples)))
    # check the timestamps and plotting functions
    timestamps = pd.date_range(
        start=first_time, periods=n_samples, freq=pd.Timedelta(1 / fs, "s")
    )
    actual_timestamps = time_data.get_timestamps(estimate=True)
    assert isinstance(actual_timestamps, pd.DatetimeIndex)
    pd.testing.assert_index_equal(actual_timestamps, timestamps)
    selected_timestamps = time_data.get_timestamps(
        samples=np.array([1, 5]), estimate=True
    )
    assert isinstance(selected_timestamps, pd.DatetimeIndex)
    pd.testing.assert_index_equal(
        selected_timestamps,
        timestamps[np.array([1, 5])],
    )
    # copy
    time_data_copy = time_data.copy()
    assert time_data_copy.metadata == time_data.metadata
    np.testing.assert_equal(time_data_copy.data, time_data.data)
    # check to make sure getting an unknown chan raises an error
    with pytest.raises(ChannelNotFoundError):
        time_data["unknown"]
    with pytest.raises(ChannelNotFoundError):
        time_data["unknown"] = np.ones(shape=(n_samples), dtype=np.float32)


def test_mth5_reader_preserves_labels_and_source_metadata():
    """Real MTH5 ingestion keeps channel/time labels and source metadata."""
    import xarray as xr
    from mth5.timeseries import RunTS

    from resistics.time import MTH5TimeReader

    time = pd.date_range("2026-01-01", periods=4, freq="500ms")
    dataset = xr.Dataset(
        {
            "ex": xr.DataArray(
                np.arange(4, dtype=np.float32),
                dims=("time",),
                attrs={"type": "electric", "units": "mV/km"},
            ),
            "hx": xr.DataArray(
                np.arange(4, dtype=np.float32) + 10,
                dims=("time",),
                attrs={"type": "magnetic", "units": "nT"},
            ),
        },
        coords={"time": time},
    )
    run_ts = RunTS(
        dataset,
        run_metadata={"id": "run_1", "sample_rate": 2.0},
        station_metadata={
            "id": "station_1",
            "location.latitude": 50.0,
            "location.longitude": -1.0,
            "location.elevation": 100.0,
        },
        survey_metadata={"id": "survey_1"},
    )

    class FakeRunGroup:
        def to_runts(self, start=None, end=None, n_samples=None):
            assert start is None
            assert end is None
            assert n_samples is None
            return run_ts

    time_data = MTH5TimeReader().run(FakeRunGroup())

    assert time_data.metadata.chans == ["ex", "hx"]
    assert time_data.metadata.survey == "survey_1"
    assert time_data.metadata.station == "station_1"
    assert time_data.metadata.run == "run_1"
    assert time_data.metadata.wgs84_latitude == 50.0
    assert time_data.to_xarray().dims == ("channel", "time")
    assert list(time_data.dataset.data_vars) == ["ex", "hx"]
    np.testing.assert_array_equal(time_data["hx"], dataset["hx"].data)


@pytest.mark.parametrize(
    "time_data",
    [
        (time_data_random(n_samples=12000, dtype=np.float32)),
        (time_data_random(n_samples=12000, dtype=np.float64)),
        (time_data_random(n_samples=100000, dtype=np.float32)),
        (time_data_random(n_samples=100000, dtype=np.float64)),
        (time_data_random(n_samples=5000, dtype=np.float32)),
        (time_data_random(n_samples=5000, dtype=np.float64)),
    ],
)
def test_remove_mean(time_data: TimeData):
    """Test removing mean"""
    from resistics.time import RemoveMean

    time_data_new = RemoveMean().run(time_data)

    assert time_data_new.data.dtype == time_data.data.dtype
    mean = np.mean(time_data.data, axis=1)
    expected_data = time_data.data - mean[:, np.newaxis]
    np.testing.assert_array_equal(time_data_new.data, expected_data)


@pytest.mark.parametrize(
    "time_data, add_arg",
    [
        (time_data_random(n_samples=12000, dtype=np.float32), 5),
        (time_data_random(n_samples=12000, dtype=np.float64), 5),
        (time_data_random(n_samples=12000, dtype=np.float32), -7),
        (time_data_random(n_samples=12000, dtype=np.float64), -7),
        (time_data_random(n_samples=12000, dtype=np.float32), {"Ex": -3, "Hy": 15}),
        (time_data_random(n_samples=12000, dtype=np.float64), {"Ex": -3, "Hy": 15}),
    ],
)
def test_add(time_data: TimeData, add_arg: float | dict[str, float]):
    """Test adding"""
    from resistics.time import Add

    time_data_new = Add(add=add_arg).run(time_data)

    assert time_data_new.data.dtype == time_data.data.dtype
    to_add = add_arg
    if isinstance(to_add, (float, int)):
        to_add = dict.fromkeys(time_data.metadata.chans, to_add)
    for chan in time_data.metadata.chans:
        add_val = to_add.get(chan, 0)
        np.testing.assert_array_equal(time_data_new[chan], time_data[chan] + add_val)


@pytest.mark.parametrize(
    "time_data, mult_arg",
    [
        (time_data_random(n_samples=12000, dtype=np.float32), 5),
        (time_data_random(n_samples=12000, dtype=np.float64), 5),
        (time_data_random(n_samples=12000, dtype=np.float32), -7),
        (time_data_random(n_samples=12000, dtype=np.float64), -7),
        (time_data_random(n_samples=12000, dtype=np.float32), {"Ex": -3, "Hy": 15}),
        (time_data_random(n_samples=12000, dtype=np.float64), {"Ex": -3, "Hy": 15}),
    ],
)
def test_multiply(time_data: TimeData, mult_arg: float | dict[str, float]):
    """Test multiply"""
    from resistics.time import Multiply

    time_data_new = Multiply(multiplier=mult_arg).run(time_data)

    assert time_data_new.data.dtype == time_data.data.dtype
    to_mult = mult_arg
    if isinstance(to_mult, (float, int)):
        to_mult = dict.fromkeys(time_data.metadata.chans, to_mult)
    for chan in time_data.metadata.chans:
        mult_val = to_mult.get(chan, 1)
        np.testing.assert_array_equal(time_data_new[chan], time_data[chan] * mult_val)


@pytest.mark.parametrize(
    "time_data, cutoff",
    [
        (time_data_random(fs=128, n_samples=12000, dtype=np.float32), 32),
        (time_data_random(fs=128, n_samples=12000, dtype=np.float64), 32),
        (time_data_random(fs=128, n_samples=12000, dtype=np.float64), 75),
    ],
)
def test_lowpass_filter(time_data: TimeData, cutoff: float):
    """Test lowpass"""
    from resistics.time import LowPass

    lp = LowPass(cutoff=cutoff)
    if cutoff > time_data.metadata.nyquist:
        with pytest.raises(ProcessRunError):
            time_data_new = lp.run(time_data)
        return

    time_data_new = lp.run(time_data)
    assert time_data_new.data.dtype == time_data.data.dtype


@pytest.mark.parametrize(
    "time_data, cutoff",
    [
        (time_data_random(fs=128, n_samples=12000, dtype=np.float32), 32),
        (time_data_random(fs=128, n_samples=12000, dtype=np.float64), 32),
        (time_data_random(fs=128, n_samples=12000, dtype=np.float64), 75),
    ],
)
def test_highpass_filter(time_data: TimeData, cutoff: float):
    """Test highpass"""
    from resistics.time import HighPass

    hp = HighPass(cutoff=cutoff)
    if cutoff > time_data.metadata.nyquist:
        with pytest.raises(ProcessRunError):
            time_data_new = hp.run(time_data)
        return

    time_data_new = hp.run(time_data)
    assert time_data_new.data.dtype == time_data.data.dtype


@pytest.mark.parametrize(
    "time_data, low, high",
    [
        (time_data_random(fs=128, n_samples=12000, dtype=np.float32), 16, 32),
        (time_data_random(fs=128, n_samples=12000, dtype=np.float64), 16, 32),
        (time_data_random(fs=128, n_samples=12000, dtype=np.float64), 45, 20),
        (time_data_random(fs=128, n_samples=12000, dtype=np.float64), 15, 75),
    ],
)
def test_bandpass_filter(time_data: TimeData, low: float, high: float):
    """Test bandpass"""
    from resistics.time import BandPass

    bp = BandPass(cutoff_low=low, cutoff_high=high)
    if low > high:
        with pytest.raises(ProcessRunError):
            time_data_new = bp.run(time_data)
        return
    if high > time_data.metadata.nyquist:
        with pytest.raises(ProcessRunError):
            time_data_new = bp.run(time_data)
        return

    time_data_new = bp.run(time_data)
    assert time_data_new.data.dtype == time_data.data.dtype


@pytest.mark.parametrize(
    "time_data, notch, band",
    [
        (time_data_random(fs=128, n_samples=12000, dtype=np.float32), 50, 10),
        (time_data_random(fs=128, n_samples=12000, dtype=np.float64), 50, 10),
        (time_data_random(fs=128, n_samples=12000, dtype=np.float64), 50, None),
        (time_data_random(fs=128, n_samples=12000, dtype=np.float64), 80, 5),
    ],
)
def test_notch_filter(time_data: TimeData, notch: float, band: float | None):
    """Test notch"""
    from resistics.time import Notch

    notcher = Notch(notch=notch, band=band)
    if notch > time_data.metadata.nyquist:
        with pytest.raises(ProcessRunError):
            time_data_new = notcher.run(time_data)
        return

    time_data_new = notcher.run(time_data)
    assert time_data_new.data.dtype == time_data.data.dtype


@pytest.mark.parametrize(
    "time_data, new_fs",
    [
        (time_data_random(fs=128, n_samples=12000, dtype=np.float32), 32),
        (time_data_random(fs=128, n_samples=12000, dtype=np.float64), 32),
        (time_data_random(fs=128, n_samples=12000, dtype=np.float32), 256),
        (time_data_random(fs=128, n_samples=12000, dtype=np.float64), 256),
    ],
)
def test_resample(time_data: TimeData, new_fs: float):
    """Test resample"""
    from resistics.time import Resample

    resampler = Resample(new_fs=new_fs)
    time_data_new = resampler.run(time_data)
    assert time_data_new.metadata.fs == new_fs
    assert time_data_new.data.dtype == time_data.data.dtype


@pytest.mark.parametrize(
    "time_data, factor",
    [
        (time_data_random(fs=128, n_samples=12000, dtype=np.float32), 4),
        (time_data_random(fs=128, n_samples=12000, dtype=np.float64), 4),
        (time_data_random(fs=128, n_samples=12000, dtype=np.float32), 16),
        (time_data_random(fs=128, n_samples=12000, dtype=np.float64), 24),
    ],
)
def test_decimate(time_data: TimeData, factor: int):
    """Test decimate"""
    from resistics.time import Decimate

    decimator = Decimate(factor=factor)
    time_data_new = decimator.run(time_data)
    assert time_data_new.metadata.fs == time_data.metadata.fs / factor
    assert time_data_new.data.dtype == time_data.data.dtype


@pytest.mark.parametrize(
    "time_data, shift",
    [
        (time_data_random(fs=10, n_samples=12000, dtype=np.float32), 0.05),
        (time_data_random(fs=10, n_samples=12000, dtype=np.float64), 0.05),
        (time_data_random(fs=10, n_samples=12000, dtype=np.float64), 0.15),
    ],
)
def test_shift(time_data: TimeData, shift: float):
    """Test shifting of timestamps"""
    from resistics.sampling import to_timedelta
    from resistics.time import ShiftTimestamps

    shifter = ShiftTimestamps(shift=shift)
    if shift > time_data.metadata.dt:
        with pytest.raises(ProcessRunError):
            shifter.run(time_data)
        return
    time_data_new = shifter.run(time_data)
    expected_first_time = time_data.metadata.first_time + to_timedelta(0.05)
    assert time_data_new.metadata.first_time == expected_first_time
    assert time_data_new.data.dtype == time_data.data.dtype
