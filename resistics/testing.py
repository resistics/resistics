"""
Module for producing testing data for resistics

This includes testing data for:

- Record
- History
- TimeMetadata
- TimeData
- DecimatedData
- SpectraData
"""

from typing import List, Dict, Optional, Type, Union
import numpy as np
import pandas as pd

from resistics.common import Record, History, get_record
from resistics.time import get_time_metadata, TimeMetadata, TimeData
from resistics.decimate import DecimatedMetadata, DecimatedData
from resistics.spectra import SpectraLevelMetadata, SpectraMetadata, SpectraData
from resistics.gather import SiteCombinedMetadata
from resistics.transfunc import Component, ImpedanceTensor
from resistics.regression import RegressionInputMetadata, Solution

DEFAULT_TIME_DATA_DTYPE = np.float32


def record_example1() -> Record:
    """Get an example Record"""
    from resistics.common import get_record

    return get_record(
        creator={"name": "example1", "a": 5, "b": -7.0},
        messages=["Message 1", "Message 2"],
    )


def record_example2() -> Record:
    """Get an example Record"""
    from resistics.common import get_record

    return get_record(
        creator={"name": "example2", "a": "parzen", "b": -21},
        messages=["Message 5", "Message 6"],
    )


def history_example() -> History:
    """Get a History example"""
    from resistics.common import History

    return History(records=[record_example1(), record_example2()])


def time_metadata_1chan(
    fs: float = 10, first_time: str = "2021-01-01 00:00:00", n_samples: int = 11
) -> TimeMetadata:
    """
    Get TimeMetadata for a single channel, "chan1"

    Parameters
    ----------
    fs : float, optional
        The sampling frequency, by default 10
    first_time : str, optional
        The first time, by default "2021-01-01 00:00:00"
    n_samples : int, optional
        The number of samples, by default 11

    Returns
    -------
    TimeMetadata
        TimeMetadata
    """
    first_time = pd.to_datetime(first_time)
    time_dict = {
        "chans": ["chan1"],
        "fs": fs,
        "n_samples": n_samples,
        "n_chans": 1,
        "first_time": first_time,
        "last_time": first_time + pd.Timedelta(1 / fs, "s") * (n_samples - 1),
    }
    chans_dict = {
        "chan1": {"data_files": "example1.ascii"},
    }
    return get_time_metadata(time_dict, chans_dict)


def time_metadata_2chan(
    fs: float = 10, first_time: str = "2021-01-01 00:00:00", n_samples: int = 11
) -> TimeMetadata:
    """
    Get a TimeMetadata instance with two channels, "chan1" and "chan2"

    Parameters
    ----------
    fs : float, optional
        The sampling frequency, by default 10
    first_time : str, optional
        The first time, by default "2021-01-01 00:00:00"
    n_samples : int, optional
        The number of samples, by default 11

    Returns
    -------
    TimeMetadata
        TimeMetadata
    """
    first_time = pd.to_datetime(first_time)
    time_dict = {
        "chans": ["chan1", "chan2"],
        "fs": fs,
        "n_samples": n_samples,
        "n_chans": 2,
        "first_time": first_time,
        "last_time": first_time + pd.Timedelta(1 / fs, "s") * (n_samples - 1),
    }
    chans_dict = {
        "chan1": {"data_files": "example1.ascii"},
        "chan2": {"data_files": "example2.ascii", "sensor": "MFS"},
    }
    return get_time_metadata(time_dict, chans_dict)


def time_metadata_mt(
    fs: float = 10, first_time: str = "2020-01-01 00:00:00", n_samples: int = 11
) -> TimeMetadata:
    """
    Get a magnetotelluric time metadata with four channels "Ex", "Ey", "Hx", "Hy"

    Parameters
    ----------
    fs : float, optional
        The sampling frequency, by default 10
    first_time : str, optional
        The first time, by default "2020-01-01 00:00:00"
    n_samples : int, optional
        The number of samples, by default 11

    Returns
    -------
    TimeMetadata
        TimeMetadata
    """
    first_time = pd.to_datetime(first_time)
    time_dict = {
        "chans": ["Ex", "Ey", "Hx", "Hy"],
        "fs": fs,
        "dt": 1 / fs,
        "n_chans": 4,
        "n_samples": n_samples,
        "first_time": first_time,
        "last_time": first_time + pd.Timedelta(1 / fs, "s") * (n_samples - 1),
    }
    chans_dict = {
        "Ex": {"data_files": "Ex.ascii"},
        "Ey": {"data_files": "Ex.ascii"},
        "Hx": {"data_files": "Ex.ascii"},
        "Hy": {"data_files": "Ex.ascii"},
    }
    return get_time_metadata(time_dict, chans_dict)


def time_data_ones(
    fs: float = 10,
    first_time: str = "2020-01-01 00:00:00",
    n_samples: int = 10,
    dtype: Optional[Type] = None,
) -> TimeData:
    """
    TimeData with all ones

    Parameters
    ----------
    fs : float, optional
        The sampling frequency, by default 10
    first_time : str, optional
        The time of the first sample, by default "2020-01-01 00:00:00"
    n_samples : int, optional
        The number of samples, by default 10
    dtype : Optional[Type], optional
        The data type for the values, by default None

    Returns
    -------
    TimeData
        The TimeData
    """
    if dtype is None:
        dtype = DEFAULT_TIME_DATA_DTYPE
    metadata = time_metadata_mt(fs, first_time, n_samples)
    data = np.ones(shape=(len(metadata.chans), n_samples), dtype=dtype)
    creator = {
        "name": "time_data_ones",
        "fs": fs,
        "first_time": first_time,
        "n_samples": n_samples,
    }
    messages = ["Generated time data with fixed values"]
    record = get_record(creator, messages)
    metadata.history.add_record(record)
    return TimeData(metadata, data)


def time_data_simple(
    fs: float = 10,
    first_time: str = "2020-01-01 00:00:00",
    dtype: Optional[Type] = None,
) -> TimeData:
    """
    Time data with 16 samples

    Parameters
    ----------
    fs : float, optional
        The sampling frequency, by default 10
    first_time : str, optional
        The time of the first sample, by default "2020-01-01 00:00:00"
    dtype : Optional[Type], optional
        The data type for the values, by default None

    Returns
    -------
    TimeData
        The TimeData
    """
    if dtype is None:
        dtype = DEFAULT_TIME_DATA_DTYPE
    data = np.array(
        [
            [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7],
            [1, 2, 3, 5, 1, 2, 3, 4, 2, 6, 7, 6, 5, 4, 3, 2],
            [2, 0, 0, 1, 2, 3, 0, 0, 0, 0, 0, 1, 3, 3, 3, 3],
        ],
        dtype=dtype,
    )
    n_samples = data.shape[1]
    metadata = time_metadata_mt(fs, first_time, n_samples)
    creator = {"name": "time_data_simple", "fs": fs, "first_time": first_time}
    messages = ["Generated time data with simple values"]
    record = get_record(creator, messages)
    metadata.history.add_record(record)
    return TimeData(metadata, data)


def time_data_with_nans(
    fs: float = 10,
    first_time: str = "2020-01-01 00:00:00",
    dtype: Optional[Type] = None,
) -> TimeData:
    """
    TimeData with 16 samples and some nan values

    Parameters
    ----------
    fs : float, optional
        Sampling frequency, by default 10
    first_time : str, optional
        The time of the first sample, by default "2020-01-01 00:00:00"
    dtype : Optional[Type], optional
        The data type for the values, by default None

    Returns
    -------
    TimeData
        The TimeData
    """
    if dtype is None:
        dtype = DEFAULT_TIME_DATA_DTYPE
    data = np.array(
        [
            [1, 1, 1, 0, np.nan, 0, 1, 1, 1, np.nan, 0, 0, 0, 0, 1, 1],
            [1, 2, np.nan, np.nan, 5, 6, np.nan, 8, 9, 1, 2, 3, 4, 5, 6, 7],
            [np.nan, 2, 3, 5, 1, 2, 3, 4, 2, 6, 7, np.nan, np.nan, 4, 3, 2],
            [2, 0, 0, 1, 2, 3, np.nan, np.nan, np.nan, 0, 0, 1, 3, 3, 3, 3],
        ],
        dtype=dtype,
    )
    n_samples = data.shape[1]
    metadata = time_metadata_mt(fs, first_time, n_samples)
    creator = {"name": "time_data_with_nans", "fs": fs, "first_time": first_time}
    messages = ["Generated time data with some nan values"]
    record = get_record(creator, messages)
    metadata.history.add_record(record)
    return TimeData(metadata, data)


def time_data_linear(
    fs: float = 10,
    first_time: str = "2020-01-01 00:00:00",
    n_samples: int = 10,
    dtype: Optional[Type] = None,
) -> TimeData:
    """
    Get TimeData with linear data

    Parameters
    ----------
    fs : float, optional
        The sampling frequency, by default 10
    first_time : str, optional
        Time of first sample, by default "2020-01-01 00:00:00"
    n_samples : int, optional
        The number of samples, by default 10
    dtype : Optional[Type], optional
        The data type for the values, by default None

    Returns
    -------
    TimeData
        TimeData with linear values
    """
    if dtype is None:
        dtype = DEFAULT_TIME_DATA_DTYPE
    metadata = time_metadata_mt(fs, first_time, n_samples)
    data = np.empty(shape=(metadata.n_chans, n_samples), dtype=dtype)
    for idx in range(metadata.n_chans):
        data[idx, :] = np.arange(n_samples)
    creator = {
        "name": "time_data_linear",
        "fs": fs,
        "first_time": first_time,
        "n_samples": n_samples,
    }
    messages = ["Generated time data with linear values"]
    record = get_record(creator, messages)
    metadata.history.add_record(record)
    return TimeData(metadata, data)


def time_data_random(
    fs: float = 10,
    first_time: str = "2020-01-01 00:00:00",
    n_samples: int = 10,
    dtype: Optional[Type] = None,
) -> TimeData:
    """
    TimeData with random values and specifiable number of samples

    Parameters
    ----------
    fs : float, optional
        The sampling frequency, by default 10
    first_time : str, optional
        Time of first sample, by default "2020-01-01 00:00:00"
    n_samples : int, optional
        The number of samples, by default 10
    dtype : Optional[Type], optional
        The data type for the values, by default None

    Returns
    -------
    TimeData
        The TimeData
    """
    if dtype is None:
        dtype = DEFAULT_TIME_DATA_DTYPE
    metadata = time_metadata_mt(fs, first_time, n_samples)
    data = np.random.normal(0, 3, size=(metadata.n_chans, n_samples)).astype(dtype)
    creator = {
        "name": "time_data_random",
        "fs": fs,
        "first_time": first_time,
        "n_samples": n_samples,
    }
    messages = ["Generated time data with random values"]
    record = get_record(creator, messages)
    metadata.history.add_record(record)
    return TimeData(metadata, data)


def time_data_periodic(
    frequencies: List[float],
    fs: float = 50,
    first_time: str = "2020-01-01 00:00:00",
    n_samples: int = 100,
    dtype: Optional[Type] = None,
) -> TimeData:
    """
    Get period TimeData

    Parameters
    ----------
    frequencies : List[float]
        Frequencies to include
    fs : float, optional
        Sampling frequency, by default 50
    first_time : str, optional
        The first time, by default "2020-01-01 00:00:00"
    n_samples : int, optional
        The number of samples, by default 100
    dtype : Optional[Type], optional
        The data type for the values, by default None

    Returns
    -------
    TimeData
        Periodic TimeData
    """
    if dtype is None:
        dtype = DEFAULT_TIME_DATA_DTYPE
    metadata = time_metadata_1chan(fs, first_time, n_samples)
    times = np.arange(0, n_samples) * (1 / fs)
    data = np.zeros(shape=(1, n_samples), dtype=dtype)
    for freq in frequencies:
        data += np.sin(times * 2 * np.pi * freq)
    creator = {
        "name": "time_data_periodic",
        "frequencies": frequencies,
        "fs": fs,
        "first_time": first_time,
        "n_samples": n_samples,
    }
    messages = ["Generated time data with periodic values"]
    record = get_record(creator, messages)
    metadata.history.add_record(record)
    return TimeData(metadata, data)


def time_data_with_offset(
    offset=0.05,
    fs: float = 10,
    first_time: str = "2020-01-01 00:00:00",
    n_samples: int = 11,
    dtype: Optional[Type] = None,
) -> TimeData:
    """
    Get TimeData with an offset on the sampling

    Parameters
    ----------
    offset : float, optional
        The offset on the sampling in seconds, by default 0.05
    fs : float, optional
        The sampling frequency, by default 10
    first_time : str, optional
        The first time of the TimeData, by default "2020-01-01 00:00:00"
    n_samples : int, optional
        The number of samples, by default 11
    dtype : Optional[Type], optional
        The data type for the values, by default None

    Returns
    -------
    TimeData
        The TimeData
    """
    if dtype is None:
        dtype = DEFAULT_TIME_DATA_DTYPE
    first_time = (pd.to_datetime(first_time) + pd.Timedelta(offset, "s")).isoformat()
    metadata = time_metadata_1chan(fs, first_time, n_samples)
    data = np.arange(0, n_samples).reshape(1, n_samples)
    creator = {
        "name": "time_data_with_offset",
        "offset": offset,
        "fs": fs,
        "first_time": first_time,
        "n_samples": n_samples,
    }
    messages = ["Generated time data with an offset"]
    record = get_record(creator, messages)
    metadata.history.add_record(record)
    return TimeData(metadata, data)


def decimated_metadata(
    fs: float = 0.25,
    first_time: str = "2021-01-01 00:00:00",
    n_samples: int = 1024,
    n_levels: int = 3,
    factor: int = 4,
) -> DecimatedMetadata:
    """
    Get example decimated metadata

    The final level has n_samples. The number of samples for all other
    levels is calculated using a decimation factor of 4.

    Similarly for the sampling frequencies, the final level is assumed to have
    a sample frequency of fs and all other levels sampling frequencies are
    calculated from there.

    Parameters
    ----------
    fs : float, optional
        The sampling frequency of the last level, by default 0.25
    first_time : str, optional
        The time of the first sample, by default "2021-01-01 00:00:00"
    n_samples : int, optional
        The number of samples, by default 1024
    n_levels : int, optional
        The number of decimation levels, by default 3
    factor : int, optional
        The decimation factor for each level, by default 4

    Returns
    -------
    DecimatedMetadata
        DecimatedMetadata
    """
    from resistics.decimate import DecimatedLevelMetadata
    from resistics.sampling import to_datetime, to_timedelta

    levels_metadata = []
    for ilevel in range(n_levels):
        level_factor = np.power(factor, (n_levels - 1 - ilevel))
        level_n_samples = n_samples * level_factor
        level_fs = fs * level_factor
        last_time = to_datetime(first_time) + to_timedelta(
            (level_n_samples - 1) * 1 / level_fs
        )
        metadata = DecimatedLevelMetadata(
            fs=level_fs,
            n_samples=level_n_samples,
            first_time=first_time,
            last_time=last_time,
        )
        levels_metadata.append(metadata)
    fs = levels_metadata[0].fs
    n_samples = levels_metadata[0].n_samples
    time_metadata = time_metadata_2chan(
        fs=fs, first_time=first_time, n_samples=n_samples
    )
    metadata_dict = time_metadata.dict()
    metadata_dict["fs"] = [x.fs for x in levels_metadata]
    metadata_dict["n_levels"] = n_levels
    metadata_dict["levels_metadata"] = levels_metadata
    return DecimatedMetadata(**metadata_dict)


def decimated_data_random(
    fs: float = 0.25,
    first_time: str = "2021-01-01 00:00:00",
    n_samples: int = 1024,
    n_levels: int = 3,
    factor: int = 4,
) -> DecimatedData:
    """
    Get random decimated data

    Parameters
    ----------
    fs : float, optional
        Sampling frequency, by default 10
    first_time : str, optional
        The time of the first sample, by default "2021-01-01 00:00:00"
    n_samples : int, optional
        The number of samples, by default 1024
    n_levels : int, optional
        The number of levels, by default 3
    factor : int, optional
        The decimation factor for each level, by default 4

    Returns
    -------
    DecimatedData
        The decimated data
    """
    metadata = decimated_metadata(
        fs, first_time, n_samples=n_samples, n_levels=n_levels, factor=factor
    )
    data = {}
    for ilevel in range(metadata.n_levels):
        level_samples = metadata.levels_metadata[ilevel].n_samples
        data[ilevel] = np.random.normal(0, 3, size=(metadata.n_chans, level_samples))
    creator = {
        "name": "decimated_data_random",
        "fs": fs,
        "first_time": first_time,
        "n_levels": n_levels,
    }
    record = get_record(creator, "Generated random decimated data")
    metadata.history.add_record(record)
    return DecimatedData(metadata, data)


def decimated_data_linear(
    fs: float = 0.25,
    first_time: str = "2021-01-01 00:00:00",
    n_samples: int = 1024,
    n_levels: int = 3,
    factor: int = 4,
):
    """
    Get linear decimated data

    Parameters
    ----------
    fs : float, optional
        Sampling frequency, by default 10
    first_time : str, optional
        The time of the first sample, by default "2021-01-01 00:00:00"
    n_samples : int, optional
        The number of samples, by default 1024
    n_levels : int, optional
        The number of levels, by default 3
    factor : int, optional
        The decimation factor for each level, by default 4

    Returns
    -------
    DecimatedData
        The decimated data
    """
    metadata = decimated_metadata(
        fs, first_time, n_samples=n_samples, n_levels=n_levels, factor=factor
    )
    data = {}
    for ilevel in range(metadata.n_levels):
        level_samples = metadata.levels_metadata[ilevel].n_samples
        level_data = np.array([np.arange(level_samples), -1 * np.arange(level_samples)])
        data[ilevel] = level_data
    creator = {
        "name": "decimated_data_linear",
        "fs": fs,
        "first_time": first_time,
        "n_levels": n_levels,
    }
    record = get_record(creator, "Generated linear decimated data")
    metadata.history.add_record(record)
    return DecimatedData(metadata, data)


def decimated_data_periodic(
    frequencies: Dict[str, List[float]],
    fs: float = 0.25,
    first_time: str = "2021-01-01 00:00:00",
    n_samples: int = 1024,
    n_levels: int = 3,
    factor: int = 4,
):
    """
    Get periodic decimated data

    Parameters
    ----------
    frequencies : Dict[str, List[float]]
        Mapping from channel to list of frequencies to add
    fs : float, optional
        Sampling frequency, by default 10
    first_time : str, optional
        The time of the first sample, by default "2021-01-01 00:00:00"
    n_samples : int, optional
        The number of samples, by default 1024
    n_levels : int, optional
        The number of levels, by default 3
    factor : int, optional
        The decimation factor for each level, by default 4

    Returns
    -------
    DecimatedData
        The decimated data
    """
    metadata = decimated_metadata(
        fs, first_time, n_samples=n_samples, n_levels=n_levels, factor=factor
    )
    data = {}
    for ilevel in range(metadata.n_levels):
        level_samples = metadata.levels_metadata[ilevel].n_samples
        level_fs = metadata.levels_metadata[ilevel].fs
        times = np.arange(0, level_samples) * (1 / level_fs)
        level_data = []
        for chan in metadata.chans:
            chan_data = np.zeros(shape=(level_samples))
            for freq in frequencies[chan]:
                if freq > level_fs / 2:
                    continue
                chan_data += np.sin(times * 2 * np.pi * freq)
            level_data.append(chan_data)
        data[ilevel] = np.array(level_data)
    creator = {
        "name": "decimated_data_periodic",
        "fs": fs,
        "first_time": first_time,
        "n_levels": n_levels,
    }
    record = get_record(creator, "Generated periodic decimated data")
    metadata.history.add_record(record)
    return DecimatedData(metadata, data)


def spectra_metadata_multilevel(
    fs: float = 128,
    n_levels: int = 3,
    n_wins: Union[List[int], int] = 2,
    index_offset: Union[List[int], int] = 0,
    chans: Optional[List[str]] = None,
) -> SpectraMetadata:
    """
    Get spectra metadata with multiple levels and two channels

    Parameters
    ----------
    fs : float, optional
        The original sampling frequency, by default 128
    n_levels : int, optional
        The number of levels, by default 3
    n_wins: Union[List[int], int]
        The number of windows for each level
    index_offset : Union[List[int], int], optional
        The index offset vs. the reference time, by default 0
    chans : Optional[List[str]]
        The channels in the data, by default None. If None, the channels will be
        chan1 and chan2

    Returns
    -------
    SpectraMetadata
        SpectraMetadata with n_levels

    Raises
    ------
    ValueError
        If the number of user input channels does not equal two
    """
    if isinstance(n_wins, int):
        n_wins = (n_wins * np.ones(shape=(n_levels))).tolist()
    if isinstance(index_offset, int):
        index_offset = (index_offset * np.ones(shape=(n_levels))).tolist()

    levels_metadata = []
    levels_fs = []
    for ilevel, offset in zip(range(n_levels), index_offset):
        factor = np.power(2, ilevel)
        fs = fs / factor
        levels_metadata.append(
            SpectraLevelMetadata(
                fs=fs,
                n_wins=n_wins[ilevel],
                win_size=20,
                olap_size=5,
                index_offset=offset,
                n_freqs=2,
                freqs=[fs / 4, fs / 8],
            )
        )
        levels_fs.append(fs)
    metadata_dict = time_metadata_2chan().dict()
    if chans is not None:
        if len(chans) != 2:
            raise ValueError(f"More than two channels {chans}")
        metadata_dict["chans"] = chans
    metadata_dict["fs"] = levels_fs
    metadata_dict["n_levels"] = len(levels_metadata)
    metadata_dict["levels_metadata"] = levels_metadata
    metadata_dict["ref_time"] = metadata_dict["first_time"]
    return SpectraMetadata(**metadata_dict)


def spectra_data_basic() -> SpectraData:
    """
    Spectra data with a single decimation level

    Returns
    -------
    SpectraData
        Spectra data with a single level, a single channel and two windows
    """

    data = {}
    # fmt:off
    data[0] = np.array(
        [
            [[0 + 0j, 1 + 1j, 2 + 2j, 3 + 3j, 4 + 4j, 5 + 5j, 6 + 6j, 7 + 7j, 8 + 8j, 9 + 9j]],
            [[-1 + 1j, 0 + 2j, 1 + 3j, 2 + 4j, 3 + 5j, 4 + 6j, 5 + 7j, 6 + 8j, 7 + 9j, 8 + 10j]],
        ]
    )
    # fmt:on
    freqs = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    level_metadata = SpectraLevelMetadata(
        fs=180,
        n_wins=2,
        win_size=20,
        olap_size=5,
        index_offset=0,
        n_freqs=10,
        freqs=freqs,
    )
    metadata_dict = time_metadata_1chan().dict()
    metadata_dict["fs"] = [180]
    metadata_dict["n_levels"] = 1
    metadata_dict["levels_metadata"] = [level_metadata]
    metadata_dict["ref_time"] = metadata_dict["first_time"]
    metadata = SpectraMetadata(**metadata_dict)
    creator = {
        "name": "spec_data_basic",
    }
    record = get_record(creator, "Generated spectra data with 1 channel and 1 level")
    metadata.history.add_record(record)
    return SpectraData(metadata, data)


def regression_input_metadata_mt(
    fs: float, freqs: List[float]
) -> RegressionInputMetadata:
    """
    Get example regression input metadata for single site mt

    Parameters
    ----------
    fs : float
        The sampling frequency
    freqs : List[float]
        The evaluation frequencies

    Returns
    -------
    RegressionInputMetadata
        Example regression input metadata with fs=128 and 5 evaluation
        frequencies
    """
    out_site = SiteCombinedMetadata(
        site_name="site1",
        fs=fs,
        measurements=["run1", "run2"],
        chans=["Ex", "Ey"],
        n_evals=len(freqs),
        eval_freqs=freqs,
        histories={"run1": History(), "run2": History()},
    )
    in_site = SiteCombinedMetadata(
        site_name="site1",
        fs=fs,
        measurements=["run1", "run2"],
        chans=["Hx", "Hy"],
        n_evals=len(freqs),
        eval_freqs=freqs,
        histories={"run1": History(), "run2": History()},
    )
    cross_site = SiteCombinedMetadata(**in_site.dict())
    creator = {
        "name": "regression_input_metadata_mt",
    }
    record = get_record(creator, "Generated testing regression input metadata for MT")
    return RegressionInputMetadata(
        contributors={
            "out_data": out_site,
            "in_data": in_site,
            "cross_data": cross_site,
        },
        history=History(records=[record]),
    )


def components_mt() -> Dict[str, Component]:
    """
    Get example components for the Impedance Tensor

    Returns
    -------
    Dict[str, Component]
        Dictionary of component values (ExHx, ExHy, EyHx, EyHy)
    """
    components = {
        "ExHx": Component(real=[1, 1, 2, 2, 3], imag=[5, 5, 4, 4, 3]),
        "ExHy": Component(real=[1, 2, 3, 4, 5], imag=[-5, -4, -3, -2, -1]),
        "EyHx": Component(real=[-1, -2, -3, -4, -5], imag=[5, 4, 3, 2, 1]),
        "EyHy": Component(real=[-1, -1, -2, -2, -3], imag=[-5, -5, -4, -4, -3]),
    }
    return components


def solution_mt() -> Solution:
    """
    Get an example impedance tensor solution

    Returns
    -------
    Solution
        The solution
    """
    tf = ImpedanceTensor()
    fs = 128
    freqs = [10.0, 20.0, 30.0, 40.0, 50.0]
    components = components_mt()
    metadata = regression_input_metadata_mt(fs, freqs)
    return Solution(
        tf=tf,
        freqs=freqs,
        components=components,
        history=History(),
        contributors=metadata.contributors,
    )
