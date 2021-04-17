import pytest
import numpy as np

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

    assert time_data.chans == chans
    assert time_data.n_samples == n_samples
    assert time_data.n_chans == len(chans)
    assert time_data.fs == fs
    assert time_data.dt == 1 / fs
    assert time_data.nyquist == fs / 2
    assert time_data.first_time == first_rstime
    assert time_data.last_time == last_rstime
    assert time_data.duration == last_rstime - first_rstime
    # data frame
    for idx, chan in enumerate(time_data.chans):
        np.testing.assert_equal(
            time_data[chan], time_data.data[time_data.get_chan_index(chan)]
        )
        np.testing.assert_equal(time_data[chan], time_data.data[idx, :])
