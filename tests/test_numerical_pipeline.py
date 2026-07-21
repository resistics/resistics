"""Integration tests for decimation, windowing, spectra, and persistence."""

from pathlib import Path

import numpy as np
import pytest

from resistics.decimate import (
    DecimatedData,
    DecimatedDataReader,
    DecimatedDataWriter,
    DecimationSetup,
    Decimator,
)
from resistics.spectra import (
    FourierTransform,
    SpectraData,
    SpectraDataReader,
    SpectraDataWriter,
)
from resistics.time import TimeData
from resistics.window import (
    WindowedData,
    WindowedDataReader,
    WindowedDataWriter,
    Windower,
    WindowParameters,
)
from tests.synthetic_data import time_data_linear


def test_labelled_time_data_survives_numerical_pipeline(tmp_path: Path) -> None:
    """Channel-labelled MTH5-style data remains ordered through every stage."""
    time_data = time_data_linear(fs=128, n_samples=4096, dtype=np.float32)
    dec_params = DecimationSetup(n_levels=2, per_level=2).run(time_data.metadata.fs)
    dec_data = Decimator().run(dec_params, time_data)
    win_params = WindowParameters(
        n_levels=2,
        min_n_wins=1,
        win_sizes=[128, 128],
        olap_sizes=[32, 32],
    )
    win_data = Windower().run(time_data.metadata.first_time, win_params, dec_data)
    spec_data = FourierTransform().run(win_data)

    assert dec_data.metadata.chans == time_data.metadata.chans
    assert win_data.metadata.chans == time_data.metadata.chans
    assert spec_data.metadata.chans == time_data.metadata.chans
    assert win_data.data[0].shape[1] == time_data.metadata.n_chans
    assert spec_data.data[0].shape[1] == time_data.metadata.n_chans

    dec_path = tmp_path / "decimated"
    win_path = tmp_path / "windowed"
    spec_path = tmp_path / "spectra"
    DecimatedDataWriter().run(dec_path, dec_data)
    WindowedDataWriter().run(win_path, win_data)
    SpectraDataWriter().run(spec_path, spec_data)

    dec_result = DecimatedDataReader().run(dec_path)
    win_result = WindowedDataReader().run(win_path)
    spec_result = SpectraDataReader().run(spec_path)
    assert isinstance(dec_result, DecimatedData)
    assert isinstance(win_result, WindowedData)
    assert isinstance(spec_result, SpectraData)
    for level in range(2):
        np.testing.assert_array_equal(dec_result.data[level], dec_data.data[level])
        np.testing.assert_array_equal(win_result.data[level], win_data.data[level])
        np.testing.assert_array_equal(spec_result.data[level], spec_data.data[level])


@pytest.mark.parametrize(
    "writer",
    [DecimatedDataWriter(), WindowedDataWriter(), SpectraDataWriter()],
)
def test_numerical_writers_reject_wrong_data(
    tmp_path: Path,
    writer: DecimatedDataWriter | WindowedDataWriter | SpectraDataWriter,
) -> None:
    """Numerical writers fail explicitly when handed another data contract."""
    time_data: TimeData = time_data_linear(fs=128, n_samples=32)
    with pytest.raises(TypeError, match="requires"):
        writer.run(tmp_path / writer.__class__.__name__, time_data)
