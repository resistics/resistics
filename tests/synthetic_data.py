"""Synthetic data factories and comparisons owned by the test suite."""

from __future__ import annotations

import string

import numpy as np
import pandas as pd

from resistics.common import History, get_record
from resistics.decimate import DecimationParameters, get_eval_freqs_size
from resistics.regression import Solution
from resistics.spectra import SpectraData, SpectraLevelMetadata, SpectraMetadata
from resistics.testing import (
    DEFAULT_TIME_DATA_DTYPE,
    _regression_input_metadata_single_site,
    time_metadata_general,
    time_metadata_mt,
)
from resistics.time import TimeData
from resistics.transfunc import Component, TransferFunction


def time_data_linear(
    fs: float = 10,
    first_time: str = "2020-01-01 00:00:00",
    n_samples: int = 10,
    dtype: type | None = None,
) -> TimeData:
    """Return four-channel time data containing a linear sample ramp."""
    if dtype is None:
        dtype = DEFAULT_TIME_DATA_DTYPE
    metadata = time_metadata_mt(fs, first_time, n_samples)
    data = np.empty(shape=(metadata.n_chans, n_samples), dtype=dtype)
    for idx in range(metadata.n_chans):
        data[idx, :] = np.arange(n_samples)
    record = get_record(
        {
            "name": "time_data_linear",
            "fs": fs,
            "first_time": first_time,
            "n_samples": n_samples,
        },
        ["Generated time data with linear values"],
    )
    metadata.history.add_record(record)
    return TimeData(metadata, data)


def _generate_evaluation_data(
    chans: list[str], solution: Solution, n_wins: int
) -> np.ndarray:
    """Return channel data that exactly satisfies a supplied solution."""
    n_evals = len(solution.freqs)
    in_chans = solution.tf.in_chans
    out_chans = solution.tf.out_chans
    cross_generate = set(solution.tf.cross_chans) - set(in_chans + out_chans)
    independent_chans = in_chans + list(cross_generate)
    data = np.empty((n_evals, len(chans), n_wins), dtype=np.complex128)
    for eval_idx in range(n_evals):
        tensor = solution.get_tensor(eval_idx)
        freq_data = {chan: np.random.randn(n_wins) for chan in independent_chans}
        for out_idx, out_chan in enumerate(out_chans):
            if out_chan in in_chans:
                continue
            products = [
                tensor[out_idx, in_idx] * freq_data[in_chan]
                for in_idx, in_chan in enumerate(in_chans)
            ]
            freq_data[out_chan] = np.sum(products, axis=0)
        for chan_idx, chan in enumerate(chans):
            data[eval_idx, chan_idx, ...] = freq_data[chan]
    return data.transpose()


def evaluation_data(
    dec_params: DecimationParameters, n_wins: int, solution: Solution
) -> SpectraData:
    """Return evaluation-frequency data that satisfies a supplied solution."""
    levels_fs = dec_params.dec_fs
    eval_freqs_for_levels = {
        level: dec_params.get_eval_freqs(level) for level in range(dec_params.n_levels)
    }
    chans = list(
        set(solution.tf.in_chans + solution.tf.out_chans + solution.tf.cross_chans)
    )
    data_array = _generate_evaluation_data(chans, solution, n_wins)
    data = {}
    for level in range(dec_params.n_levels):
        start = level * dec_params.per_level
        data[level] = data_array[..., start : start + dec_params.per_level]

    levels_metadata = [
        SpectraLevelMetadata(
            fs=level_fs,
            n_wins=n_wins,
            win_size=20,
            olap_size=5,
            index_offset=0,
            n_freqs=dec_params.per_level,
            freqs=eval_freqs_for_levels[level],
        )
        for level, level_fs in enumerate(levels_fs)
    ]
    metadata_values = time_metadata_general(chans).dict()
    metadata_values.update(
        {
            "chans": chans,
            "fs": levels_fs,
            "n_levels": len(levels_metadata),
            "levels_metadata": levels_metadata,
            "ref_time": metadata_values["first_time"],
        }
    )
    return SpectraData(SpectraMetadata(**metadata_values), data)


def transfer_function_random(
    n_in: int, n_out: int, n_cross: int = -1
) -> TransferFunction:
    """Return a deterministic transfer function with the requested dimensions."""
    if not 0 <= n_in <= 26 or not 0 <= n_out <= 26:
        raise ValueError("Input and output channel counts must be between 0 and 26")
    in_chans = list(string.ascii_lowercase[:n_in])
    out_chans = list(string.ascii_uppercase[:n_out])
    cross_chans = (
        [f"X{index:02d}" for index in range(n_cross)] if n_cross > 0 else in_chans
    )
    return TransferFunction(
        variation="random",
        in_chans=in_chans,
        out_chans=out_chans,
        cross_chans=cross_chans,
    )


def _solution_general(
    fs: float,
    transfer_function: TransferFunction,
    n_evals: int,
    components: dict[str, Component],
) -> Solution:
    """Build a solution around supplied component values."""
    freqs = get_eval_freqs_size(fs, n_evals).tolist()
    metadata = _regression_input_metadata_single_site(fs, freqs, transfer_function)
    return Solution(
        tf=transfer_function,
        freqs=freqs,
        components=components,
        history=History(),
        contributors=metadata.contributors,
    )


def solution_random_int(
    fs: float,
    transfer_function: TransferFunction,
    n_evals: int = 10,
    low: int = -10,
    high: int = 10,
) -> Solution:
    """Return a solution populated with random integer components."""
    components = {
        component: Component(
            real=np.random.randint(low, high, size=n_evals).tolist(),
            imag=np.random.randint(low, high, size=n_evals).tolist(),
        )
        for component in transfer_function.solution_components()
    }
    return _solution_general(fs, transfer_function, n_evals, components)


def solution_random_float(
    fs: float, transfer_function: TransferFunction, n_evals: int = 10
) -> Solution:
    """Return a solution populated with random floating-point components."""
    components = {
        component: Component(
            real=(np.random.randn(n_evals) * np.random.randint(0, 10)).tolist(),
            imag=(np.random.randn(n_evals) * np.random.randint(0, 10)).tolist(),
        )
        for component in transfer_function.solution_components()
    }
    return _solution_general(fs, transfer_function, n_evals, components)


def assert_solution_equal(actual: Solution, expected: Solution) -> None:
    """Assert that two solutions have numerically equivalent data frames."""
    pd.testing.assert_frame_equal(
        actual.to_dataframe(),
        expected.to_dataframe(),
        check_exact=False,
        rtol=1e-10,
        atol=1e-10,
    )
