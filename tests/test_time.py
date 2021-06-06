from typing import Union, Dict
import pytest
import numpy as np

from resistics.errors import ChannelNotFoundError, ProcessRunError
from resistics.time import TimeData
from resistics.testing import time_data_simple, time_data_random


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
    # check to make sure getting an unknown chan raises an error
    with pytest.raises(ChannelNotFoundError):
        time_data["unknown"]
        return


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
def test_add(time_data: TimeData, add_arg: Union[float, Dict[str, float]]):
    """Test adding"""
    from resistics.time import Add

    time_data_new = Add(add=add_arg).run(time_data)

    assert time_data_new.data.dtype == time_data.data.dtype
    to_add = add_arg
    if isinstance(to_add, (float, int)):
        to_add = {x: to_add for x in time_data.metadata.chans}
    for chan in time_data.metadata.chans:
        add_val = to_add[chan] if chan in to_add else 0
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
def test_multiply(time_data: TimeData, mult_arg: Union[float, Dict[str, float]]):
    """Test multiply"""
    from resistics.time import Multiply

    time_data_new = Multiply(multiplier=mult_arg).run(time_data)

    assert time_data_new.data.dtype == time_data.data.dtype
    to_mult = mult_arg
    if isinstance(to_mult, (float, int)):
        to_mult = {x: to_mult for x in time_data.metadata.chans}
    for chan in time_data.metadata.chans:
        mult_val = to_mult[chan] if chan in to_mult else 1
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
def test_notch_filter(time_data: TimeData, notch: float, band: Union[float, None]):
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
    from resistics.time import ShiftTimestamps
    from resistics.sampling import to_timedelta

    shifter = ShiftTimestamps(shift=shift)
    if shift > time_data.metadata.dt:
        with pytest.raises(ProcessRunError):
            shifter.run(time_data)
        return
    time_data_new = shifter.run(time_data)
    expected_first_time = time_data.metadata.first_time + to_timedelta(0.05)
    assert time_data_new.metadata.first_time == expected_first_time
    assert time_data_new.data.dtype == time_data.data.dtype
