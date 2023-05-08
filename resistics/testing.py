"""
Module for producing testing data for resistics and helper functions to compare
instances of the same object.

This includes testing data for:

- Record
- History
- TimeMetadata
- TimeData
- DecimatedData
- SpectraData
- Evaluation frequency SpectraData
- RegressionInputMetadata
- Solution
"""
from typing import List, Dict, Optional, Type, Union
import numpy as np
import pandas as pd

from resistics.common import Record, History, get_record, known_chan
from resistics.time import get_time_metadata, TimeMetadata, TimeData
from resistics.decimate import get_eval_freqs_size, DecimationParameters
from resistics.decimate import DecimatedMetadata, DecimatedData
from resistics.spectra import SpectraLevelMetadata, SpectraMetadata, SpectraData
from resistics.gather import SiteCombinedMetadata
from resistics.transfunc import Component, TransferFunction, ImpedanceTensor
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
        "chan1": {
            "name": "chan1",
            "chan_type": "electric",
            "data_files": "example1.ascii",
        },
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
        "chan1": {
            "name": "chan1",
            "chan_type": "electric",
            "data_files": "example1.ascii",
        },
        "chan2": {
            "name": "chan2",
            "chan_type": "magnetic",
            "data_files": "example2.ascii",
            "sensor": "MFS",
        },
    }
    return get_time_metadata(time_dict, chans_dict)


def time_metadata_general(
    chans: List[str],
    fs: float = 10,
    first_time: str = "2020-01-01 00:00:00",
    n_samples: int = 11,
    chan_type: Optional[str] = None,
) -> TimeMetadata:
    """
    Get general time metadata

    Parameters
    ----------
    chans : List[str]
        The channels in the time data
    fs : float, optional
        The sampling frequency, by default 10
    first_time : _type_, optional
        The time of the first sample, by default "2020-01-01 00:00:00"
    n_samples : int, optional
        The number of samples, by default 11
    chan_type : Optional[str], optional
        The channel type for channels with unknown type, by default None

    Returns
    -------
    TimeMetadata
        An instance of TimeMetadata with the approripate properties
    """
    first_time = pd.to_datetime(first_time)
    time_dict = {
        "chans": chans,
        "fs": fs,
        "dt": 1 / fs,
        "n_chans": len(chans),
        "n_samples": n_samples,
        "first_time": first_time,
        "last_time": first_time + pd.Timedelta(1 / fs, "s") * (n_samples - 1),
    }
    chans_dict = {chan: {"name": chan, "data_files": "Ex.ascii"} for chan in chans}
    for chan in chans:
        if not known_chan(chan):
            chans_dict[chan]["chan_type"] = "unknown"
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
    chans = ["Ex", "Ey", "Hx", "Hy"]
    return time_metadata_general(
        chans, fs=fs, first_time=first_time, n_samples=n_samples
    )


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


def generate_evaluation_data(
    chans: List[str], soln: Solution, n_wins: int
) -> np.ndarray:
    """
    Generate evaluation frequency data that satisfies a provided solution

    The returned array has the shape:
    n_wins x n_chans x n_evals
    Which is close to the shape required for spectra data

    There is an extra check provided to check if a channel appears in both the
    input and output channels, which could be a tricky scenario.

    The data is produced randomly using np.random.randn, meaning that it is
    sampled from a standard normal distribution

    Parameters
    ----------
    chans : List[str]
        The channels in the data
    soln : Solution
        The Solution that needs to be satisfied
    n_wins : int
        The number of windows to generate

    Returns
    -------
    np.ndarray
        The evaluation frequency data array
    """
    n_evals = len(soln.freqs)
    n_chans = len(chans)
    in_chans = soln.tf.in_chans
    out_chans = soln.tf.out_chans

    # create the data array to hold the data and generate the data
    data_array = np.empty((n_evals, n_chans, n_wins), dtype=np.complex128)
    for eval_idx in range(n_evals):
        freq_tensor = soln.get_tensor(eval_idx)
        # generate input channels
        freq_data = {in_chan: np.random.randn(n_wins) for in_chan in in_chans}
        # calculate output channels from input and solution
        for out_idx, out_chan in enumerate(out_chans):
            if out_chan in in_chans:
                # ignore if the channel already appears in the input data
                continue
            products = [
                freq_tensor[out_idx, in_idx] * freq_data[in_chan]
                for in_idx, in_chan in enumerate(in_chans)
            ]
            freq_data[out_chan] = np.sum(products, axis=0)
        # add the data to the data array
        for chan_idx, chan in enumerate(chans):
            data_array[eval_idx, chan_idx, ...] = freq_data[chan]
    return data_array.transpose()


def evaluation_data(
    fs: float, dec_params: DecimationParameters, n_wins: int, soln: Solution
) -> SpectraData:
    """
    Generate evaluation frequency data that will satisfy a given solution. This
    will generate random data between the low and high values

    Parameters
    ----------
    fs : float
        The sampling frequency of the original data
    dec_params : DecimationParameters
        The data decimation information
    n_wins : int
        The number of windows to generate
    soln : Solution
        The solution that the generated data should satisfy

    Returns
    -------
    SpectraData
        The evaluation frequency data

    Raises
    ------
    ValueError
        If the number of evaluation frequencies is not exactly divisible by the
        number of levels
    """
    # get information about the decimation levels
    n_levels = dec_params.n_levels
    per_level = dec_params.per_level
    levels_fs = dec_params.dec_fs
    eval_freqs_for_levels = {
        ilevel: dec_params.get_eval_freqs(ilevel) for ilevel in range(n_levels)
    }

    # create the data
    chans = list(set(soln.tf.in_chans + soln.tf.out_chans))
    data_array = generate_evaluation_data(chans, soln, n_wins)
    data = {}
    for ilevel in range(n_levels):
        istart = ilevel * per_level
        iend = istart + per_level
        data[ilevel] = data_array[..., istart:iend]

    # create the metadata
    levels_metadata = []
    for ilevel, level_fs in enumerate(levels_fs):
        levels_metadata.append(
            SpectraLevelMetadata(
                fs=level_fs,
                n_wins=n_wins,
                win_size=20,
                olap_size=5,
                index_offset=0,
                n_freqs=per_level,
                freqs=eval_freqs_for_levels[ilevel],
            )
        )
        levels_metadata[-1].summary()
    metadata_dict = time_metadata_general(chans).dict()
    metadata_dict["chans"] = chans
    metadata_dict["fs"] = levels_fs
    metadata_dict["n_levels"] = len(levels_metadata)
    metadata_dict["levels_metadata"] = levels_metadata
    metadata_dict["ref_time"] = metadata_dict["first_time"]
    spec_metadata = SpectraMetadata(**metadata_dict)
    return SpectraData(spec_metadata, data)


def transfer_function_random(n_in: int, n_out: int) -> TransferFunction:
    """
    Generate a random transfer function

    n_in and n_out must be less than or equal to 26 as the random samples are
    taken from the alphabet

    Parameters
    ----------
    n_in : int
        Number of input channels
    n_out : int
        Number of output channels

    Returns
    -------
    TransferFunction
        A randomly generated transfer function

    Raises
    ------
    ValueError
        If any of the channel names is duplicated
    """
    import random
    import string

    ins = string.ascii_lowercase
    outs = string.ascii_uppercase
    in_chans = random.sample(ins, n_in)
    out_chans = random.sample(outs, n_out)
    if len(set(ins + outs)) < len(ins) + len(outs):
        raise ValueError(f"There is a duplicate somewhere, {ins=}, {outs=}")

    return TransferFunction(
        name="testing", variation="random", in_chans=in_chans, out_chans=out_chans
    )


def regression_input_metadata_single_site(
    fs: float, freqs: List[float], tf: TransferFunction
) -> RegressionInputMetadata:
    """
    Given a transfer function, get example regression input metadata assuming a
    single site

    Parameters
    ----------
    fs : float
        The sampling frequency
    freqs : List[float]
        The evaluation frequencies
    tf : TransferFunction
        The transfer function for which to create RegressionInputMetadata

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
        chans=tf.out_chans,
        n_evals=len(freqs),
        eval_freqs=freqs,
        histories={"run1": History(), "run2": History()},
    )
    in_site = SiteCombinedMetadata(
        site_name="site1",
        fs=fs,
        measurements=["run1", "run2"],
        chans=tf.in_chans,
        n_evals=len(freqs),
        eval_freqs=freqs,
        histories={"run1": History(), "run2": History()},
    )
    cross_site = SiteCombinedMetadata(**in_site.dict())
    creator = {
        "name": "regression_input_metadata",
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
    return {
        "ExHx": Component(real=[1, 1, 2, 2, 3, 3], imag=[5, 5, 4, 4, 3, 3]),
        "ExHy": Component(real=[1, 2, 3, 4, 5, 6], imag=[-5, -4, -3, -2, -1, 1]),
        "EyHx": Component(real=[-1, -2, -3, -4, -5, -6], imag=[5, 4, 3, 2, 1, 2]),
        "EyHy": Component(real=[-1, -1, -2, -2, -3, -2], imag=[-5, -5, -4, -4, -3, -4]),
    }


def solution_mt() -> Solution:
    """
    Get an example impedance tensor solution

    Returns
    -------
    Solution
        The solution for an MT dataset
    """
    tf = ImpedanceTensor()
    fs = 256
    freqs = [100, 80, 60, 40, 20, 10]
    components = components_mt()
    metadata = regression_input_metadata_single_site(fs, freqs, tf)
    return Solution(
        tf=tf,
        freqs=freqs,
        components=components,
        history=History(),
        contributors=metadata.contributors,
    )


def solution_general(
    fs: float, tf: TransferFunction, n_evals: int, components: Dict[str, Component]
) -> Solution:
    """
    Create a Solution instance from the specified components

    Parameters
    ----------
    fs : float
        The sampling frequency of the original data
    tf : TransferFunction
        The transfer function to be solved
    n_evals : int
        The number of evaluation frequencies
    components : Dict[str, Component]
        The components of the solution

    Returns
    -------
    Solution
        The Solution instance
    """
    freqs = get_eval_freqs_size(fs, n_evals).tolist()
    metadata = regression_input_metadata_single_site(fs, freqs, tf)
    return Solution(
        tf=tf,
        freqs=freqs,
        components=components,
        history=History(),
        contributors=metadata.contributors,
    )


def solution_random_int(
    fs: float, tf: TransferFunction, n_evals=10, low: int = -10, high: int = 10
) -> Solution:
    """
    Generate a set of random integer components for a solution

    Parameters
    ----------
    fs : float
        The original sampling frequency of the data
    tf : TransferFunction
        The transfer function
    n_evals : int, optional
        The number of evaluation frequencies, by default 10
    low : int, optional
        A low value for the integers, by default -10
    high : int, optional
        A high value for the integers, by default 10

    Returns
    -------
    Solution
        A randomly generated solution for the transfer function
    """
    soln_components = tf.solution_components()
    # generate the components with values for each evaluation frequency
    components = {
        comp: Component(
            real=np.random.randint(low, high, size=n_evals).tolist(),
            imag=np.random.randint(low, high, size=n_evals).tolist(),
        )
        for comp in soln_components
    }
    return solution_general(fs, tf, n_evals, components)


def solution_random_float(fs: float, tf: TransferFunction, n_evals=10) -> Solution:
    """
    Generate a set of random float components for a solution

    This uses the numpy np.random.randn which generates numbers on a standard
    distribution and then multiplies that with a random integer between 0 and
    10.

    Parameters
    ----------
    fs : float
        The original sampling frequency of the data
    tf : TransferFunction
        The transfer function
    n_evals : int, optional
        The number of evaluation frequencies, by default 10

    Returns
    -------
    Solution
        A randomly generated solution for the transfer function
    """
    soln_components = tf.solution_components()
    # generate the components with values for each evaluation frequency
    components = {
        comp: Component(
            real=(np.random.randn(n_evals) * np.random.randint(0, 10)).tolist(),
            imag=(np.random.randn(n_evals) * np.random.randint(0, 10)).tolist(),
        )
        for comp in soln_components
    }
    return solution_general(fs, tf, n_evals, components)


def remove_record_times(records: Dict) -> Dict:
    """
    Remove timestamps from records

    Timestamps can make comparision of two data objects harder as processes need
    to have been run at exactly the same time for equality, which is unlikely to
    be the case in tests

    Parameters
    ----------
    records : Dict
        The history records

    Returns
    -------
    Dict
        The history records with timestamps removed
    """
    for rec in records:
        rec.pop("time_local")
        rec.pop("time_utc")
    return records


def assert_time_data_equal(
    time_data1: TimeData, time_data2: TimeData, history_times: bool = True
):
    """
    Assert that two time data instances are equal

    Parameters
    ----------
    time_data1 : TimeData
        Time data 1
    time_data2 : TimeData
        Time data 2
    history_times : bool, optional
        Flag to include history timestamps in the comparison, by default True.
        Including timestamps will cause a failure if processes were not run at
        exactly the same time.
    """
    metadata1 = time_data1.metadata.dict()
    history1 = metadata1.pop("history")
    metadata2 = time_data2.metadata.dict()
    history2 = metadata2.pop("history")
    # compare core metadata
    assert metadata1 == metadata2
    # compare histories
    if not history_times:
        history1["records"] = remove_record_times(history1["records"])
        history2["records"] = remove_record_times(history2["records"])
    assert history1 == history2
    # compare data
    np.testing.assert_array_equal(time_data1.data, time_data2.data)


def assert_soln_equal(soln1: Solution, soln2: Solution):
    """
    Check that two solutions are nearly the same

    Parameters
    ----------
    soln1 : Solution
        The first solution
    soln2 : Solution
        The second solution
    """
    df1 = soln1.to_dataframe()
    df2 = soln2.to_dataframe()
    pd.testing.assert_frame_equal(df1, df2)
