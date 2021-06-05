from typing import List
import numpy as np
import pandas as pd

from resistics.common import Record, History, get_record
from resistics.time import get_time_metadata, TimeMetadata, TimeData
from resistics.decimate import DecimatedMetadata, DecimatedData

# from resistics.project import Measurement, Site, Project


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
        "n_chans": 2,
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
    fs: float = 10, first_time: str = "2020-01-01 00:00:00", n_samples: int = 10
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

    Returns
    -------
    TimeData
        The TimeData
    """
    metadata = time_metadata_mt(fs, first_time, n_samples)
    data = np.ones(shape=(len(metadata.chans), n_samples), dtype=np.float32)
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
    fs: float = 10, first_time: str = "2020-01-01 00:00:00"
) -> TimeData:
    """
    Time data with 16 samples

    Parameters
    ----------
    fs : float, optional
        The sampling frequency, by default 10
    first_time : str, optional
        The time of the first sample, by default "2020-01-01 00:00:00"

    Returns
    -------
    TimeData
        The TimeData
    """
    data = np.array(
        [
            [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7],
            [1, 2, 3, 5, 1, 2, 3, 4, 2, 6, 7, 6, 5, 4, 3, 2],
            [2, 0, 0, 1, 2, 3, 0, 0, 0, 0, 0, 1, 3, 3, 3, 3],
        ]
    )
    n_samples = data.shape[1]
    metadata = time_metadata_mt(fs, first_time, n_samples)
    creator = {"name": "time_data_simple", "fs": fs, "first_time": first_time}
    messages = ["Generated time data with simple values"]
    record = get_record(creator, messages)
    metadata.history.add_record(record)
    return TimeData(metadata, data)


def time_data_with_nans(
    fs: float = 10, first_time: str = "2020-01-01 00:00:00"
) -> TimeData:
    """
    TimeData with 16 samples and some nan values

    Parameters
    ----------
    fs : float, optional
        Sampling frequency, by default 10
    first_time : str, optional
        The time of the first sample, by default "2020-01-01 00:00:00"

    Returns
    -------
    TimeData
        The TimeData
    """
    data = np.array(
        [
            [1, 1, 1, 0, np.nan, 0, 1, 1, 1, np.nan, 0, 0, 0, 0, 1, 1],
            [1, 2, np.nan, np.nan, 5, 6, np.nan, 8, 9, 1, 2, 3, 4, 5, 6, 7],
            [np.nan, 2, 3, 5, 1, 2, 3, 4, 2, 6, 7, np.nan, np.nan, 4, 3, 2],
            [2, 0, 0, 1, 2, 3, np.nan, np.nan, np.nan, 0, 0, 1, 3, 3, 3, 3],
        ]
    )
    n_samples = data.shape[1]
    metadata = time_metadata_mt(fs, first_time, n_samples)
    creator = {"name": "time_data_with_nans", "fs": fs, "first_time": first_time}
    messages = ["Generated time data with some nan values"]
    record = get_record(creator, messages)
    metadata.history.add_record(record)
    return TimeData(metadata, data)


def time_data_linear(
    fs: float = 10, first_time: str = "2020-01-01 00:00:00", n_samples: int = 10
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

    Returns
    -------
    TimeData
        TimeData with linear values
    """
    metadata = time_metadata_mt(fs, first_time, n_samples)
    data = np.empty(shape=(metadata.n_chans, n_samples))
    for idx in range(metadata.n_chans):
        data[idx, :] = np.arange(n_samples)
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


def time_data_random(
    fs: float = 10, first_time: str = "2020-01-01 00:00:00", n_samples: int = 10
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

    Returns
    -------
    TimeData
        The TimeData
    """
    metadata = time_metadata_mt(fs, first_time, n_samples)
    data = np.random.normal(0, 3, size=(metadata.n_chans, n_samples))
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

    Returns
    -------
    TimeData
        Periodic TimeData
    """
    metadata = time_metadata_1chan(fs, first_time, n_samples)
    times = np.arange(0, n_samples) * (1 / fs)
    data = np.zeros(shape=(1, n_samples))
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

    Returns
    -------
    TimeData
        The TimeData
    """
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
        frequencies = [[4, 7], [5, 8]]
        level_data = []
        for i_chan in range(metadata.n_chans):
            chan_data = np.zeros(shape=(level_samples))
            for freq in frequencies[i_chan]:
                chan_data += np.sin(times * 2 * np.pi * (level_fs / freq))
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


# def test_measurement(self) -> Measurement:


# def calibration_headers(
#     n_samples: int, serial: int = 10, sensor: str = "test sensor"
# ) -> Headers:
#     from resistics.calibrate import get_calibration_headers

#     headers = {
#         "data_file": "test.json",
#         "n_samples": n_samples,
#         "serial": serial,
#         "sensor": sensor,
#     }
#     return get_calibration_headers(headers)


# def calibration_data_ones(
#     n_samples: int = 10, first_freq: float = 0.1, last_freq: float = 10
# ) -> CalibrationData:
#     from resistics.common import get_record

#     headers = calibration_headers(n_samples)
#     freqs = np.linspace(start=first_freq, stop=last_freq, num=n_samples)
#     df = pd.DataFrame(data=freqs, columns=["frequencies"])
#     df["magnitude"] = 1
#     df["phase"] = 1
#     df = df.set_index("frequencies")
#     messages = ["Generating calibration data all 1s"]
#     parameters = {
#         "n_samples": n_samples,
#         "first_freq": first_freq,
#         "last_freq": last_freq,
#     }
#     record = get_record("calibration_data_ones", parameters, messages)
#     return CalibrationData(headers, df, History([record]))


# def calibration_data_linear(
#     n_samples: int = 10, first_freq: float = 0.1, last_freq: float = 10
# ) -> CalibrationData:
#     from resistics.common import get_record

#     headers = calibration_headers(n_samples)
#     freqs = np.linspace(start=first_freq, stop=last_freq, num=n_samples)
#     df = pd.DataFrame(data=freqs, columns=["frequencies"])
#     df["magnitude"] = np.arange(0, n_samples)
#     df["phase"] = np.arange(0, -n_samples, -1)
#     df = df.set_index("frequencies")
#     messages = ["Generating calibration with linear trends"]
#     parameters = {
#         "n_samples": n_samples,
#         "first_freq": first_freq,
#         "last_freq": last_freq,
#     }
#     record = get_record("calibration_data_ones", parameters, messages)
#     return CalibrationData(headers, df, History([record]))
