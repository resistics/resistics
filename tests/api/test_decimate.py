"""
Test decimation
"""
import pytest
import numpy as np

from resistics.time import TimeData
from resistics.testing import time_data_linear


@pytest.mark.parametrize(
    "time_data, resample_flag",
    [
        (time_data_linear(fs=128, n_samples=50_000, dtype=np.float32), True),
        (time_data_linear(fs=128, n_samples=50_000, dtype=np.float64), True),
        (time_data_linear(fs=128, n_samples=50_000, dtype=np.float32), False),
        (time_data_linear(fs=128, n_samples=50_000, dtype=np.float64), False),
        (time_data_linear(fs=128, n_samples=50_000, dtype=np.float64), True),
    ],
)
def test_decimator(time_data: TimeData, resample_flag: bool):
    """Test shifting of timestamps"""
    from resistics.decimate import DecimationSetup, Decimator

    dec_params = DecimationSetup(n_levels=3, per_level=4).run(time_data.metadata.fs)
    dec_data = Decimator(resample=resample_flag).run(dec_params, time_data)

    np.testing.assert_equal(dec_data.metadata.fs, [128.0, 32.0, 8.0])
    for ilevel in range(dec_data.metadata.n_levels):
        assert dec_data.data[ilevel].dtype == time_data.data.dtype
