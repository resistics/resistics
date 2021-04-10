from typing import List
import numpy as np
import pandas as pd

from resistics.common import Headers, DatasetHeaders, ProcessHistory
from resistics.time import TimeData
from resistics.calibrate import CalibrationData


def time_headers(fs: float, first_time: pd.Timestamp, n_samples: int) -> DatasetHeaders:
    """
    Get some time headers

    Parameters
    ----------
    fs : float
        Sampling frequency
    first_time : pd.Timestamp
        Timestamp of first sample
    n_samples : int
        Number of samples

    Returns
    -------
    DatasetHeaders
        DatasetHeaders to use with TimeData
    """
    from resistics.time import get_time_headers

    dataset_headers = {
        "fs": fs,
        "dt": 1 / fs,
        "n_chans": 4,
        "n_samples": n_samples,
        "first_time": first_time,
        "last_time": first_time + pd.Timedelta(1 / fs, "s") * (n_samples - 1),
    }
    chan_headers = {
        "Ex": {"data_files": "Ex.ascii"},
        "Ey": {"data_files": "Ex.ascii"},
        "Hx": {"data_files": "Ex.ascii"},
        "Hy": {"data_files": "Ex.ascii"},
    }
    return get_time_headers(dataset_headers, chan_headers)


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
    from resistics.common import serialize, get_process_record

    first_time = pd.Timestamp(first_time)
    chans = ["Ex", "Ey", "Hx", "Hy"]
    data = np.ones(shape=(len(chans), n_samples), dtype=np.float32)
    headers = time_headers(fs, first_time, n_samples)
    parameters = {"fs": fs, "first_time": serialize(first_time), "n_samples": n_samples}
    messages = ["Generated time data with fixed values"]
    record = get_process_record("time_data_ones", parameters, messages)
    return TimeData(headers, chans, data, ProcessHistory([record]))


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
    from resistics.common import serialize, get_process_record

    first_time = pd.Timestamp(first_time)
    chans = ["Ex", "Ey", "Hx", "Hy"]
    data = np.array(
        [
            [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7],
            [1, 2, 3, 5, 1, 2, 3, 4, 2, 6, 7, 6, 5, 4, 3, 2],
            [2, 0, 0, 1, 2, 3, 0, 0, 0, 0, 0, 1, 3, 3, 3, 3],
        ]
    )
    n_samples = data.shape[1]
    headers = time_headers(fs, first_time, n_samples)
    parameters = {"fs": fs, "first_time": serialize(first_time)}
    messages = ["Generated time data with simple values"]
    record = get_process_record("time_data_simple", parameters, messages)
    return TimeData(headers, chans, data, ProcessHistory([record]))


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
    from resistics.common import serialize, get_process_record

    first_time = pd.Timestamp(first_time)
    chans = ["Ex", "Ey", "Hx", "Hy"]
    data = np.array(
        [
            [1, 1, 1, 0, np.nan, 0, 1, 1, 1, np.nan, 0, 0, 0, 0, 1, 1],
            [1, 2, np.nan, np.nan, 5, 6, np.nan, 8, 9, 1, 2, 3, 4, 5, 6, 7],
            [np.nan, 2, 3, 5, 1, 2, 3, 4, 2, 6, 7, np.nan, np.nan, 4, 3, 2],
            [2, 0, 0, 1, 2, 3, np.nan, np.nan, np.nan, 0, 0, 1, 3, 3, 3, 3],
        ]
    )
    n_samples = data.shape[1]
    headers = time_headers(fs, first_time, n_samples)
    parameters = {"fs": fs, "first_time": serialize(first_time)}
    messages = ["Generated time data with some nan values"]
    record = get_process_record("time_data_with_nans", parameters, messages)
    return TimeData(headers, chans, data, ProcessHistory([record]))


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
    from resistics.common import serialize, get_process_record

    first_time = pd.Timestamp(first_time)
    chans = ["Ex", "Ey", "Hx", "Hy"]
    data = np.random.normal(0, 3, size=(len(chans), n_samples))
    headers = time_headers(fs, first_time, n_samples)
    parameters = {"fs": fs, "first_time": serialize(first_time), "n_samples": n_samples}
    messages = ["Generated time data with random values"]
    record = get_process_record("time_data_random", parameters, messages)
    return TimeData(headers, chans, data, ProcessHistory([record]))


# def time_data_periodic(frequencies: List[float], fs = 10, first_time: str = "2020-01-01 00:00:00", n_samples: int = 1000):
#     from resistics.common import serialize, get_process_record

#     first_time = pd.Timestamp(first_time)
#     chans = ["Ex", "Ey", "Hx", "Hy"]
#     data = np.random.normal(0, 3, size=(len(chans), n_samples))
#     headers = time_headers(fs, first_time, n_samples)
#     parameters = {"fs": fs, "first_time": serialize(first_time), "n_samples": n_samples}
#     messages = ["Generated time data with random values"]
#     record = get_process_record("time_data_random", parameters, messages)
#     return TimeData(headers, chans, data, ProcessHistory([record]))


def calibration_headers(
    n_samples: int, serial: int = 10, sensor: str = "test sensor"
) -> Headers:
    from resistics.calibrate import get_calibration_headers

    headers = {
        "data_file": "test.json",
        "n_samples": n_samples,
        "serial": serial,
        "sensor": sensor,
    }
    return get_calibration_headers(headers)


def calibration_data_ones(
    n_samples: int = 10, first_freq: float = 0.1, last_freq: float = 10
) -> CalibrationData:
    from resistics.common import get_process_record

    headers = calibration_headers(n_samples)
    freqs = np.linspace(start=first_freq, stop=last_freq, num=n_samples)
    df = pd.DataFrame(data=freqs, columns=["frequencies"])
    df["magnitude"] = 1
    df["phase"] = 1
    df = df.set_index("frequencies")
    messages = ["Generating calibration data all 1s"]
    parameters = {
        "n_samples": n_samples,
        "first_freq": first_freq,
        "last_freq": last_freq,
    }
    record = get_process_record("calibration_data_ones", parameters, messages)
    return CalibrationData(headers, df, ProcessHistory([record]))


def calibration_data_linear(
    n_samples: int = 10, first_freq: float = 0.1, last_freq: float = 10
) -> CalibrationData:
    from resistics.common import get_process_record

    headers = calibration_headers(n_samples)
    freqs = np.linspace(start=first_freq, stop=last_freq, num=n_samples)
    df = pd.DataFrame(data=freqs, columns=["frequencies"])
    df["magnitude"] = np.arange(0, n_samples)
    df["phase"] = np.arange(0, -n_samples, -1)
    df = df.set_index("frequencies")
    messages = ["Generating calibration with linear trends"]
    parameters = {
        "n_samples": n_samples,
        "first_freq": first_freq,
        "last_freq": last_freq,
    }
    record = get_process_record("calibration_data_ones", parameters, messages)
    return CalibrationData(headers, df, ProcessHistory([record]))