from typing import Dict

from resistics.time.data import TimeData
from resistics.time.reader import TimeReader
from resistics.calibrate.calibrator import Calibrator


def applyPolarisationReversalOptions(options: Dict, timeData: TimeData) -> TimeData:
    """Polarity reverse time data

    Parameters
    ----------
    options : Dict[str, bool]
        User specified options for polarity reversal
    timeData : TimeData
        Time data to polarity reverse

    Returns
    -------
    TimeData
        Polarity reversed time data
    """
    if isinstance(options["polreverse"], bool):
        # no polarity reversal to be performed
        return timeData
    if not isinstance(options["polreverse"], dict):
        # not specified in the right way
        return timeData

    from resistics.time.math import polarityReversal

    timeData = polarityReversal(timeData, options["polreverse"])
    return timeData


def applyScaleOptions(options: Dict, timeData: TimeData) -> TimeData:
    """Scale the time data

    Parameters
    ----------
    options : Dict
        User specified options for scaling
    timeData : TimeData
        Time data to polarity reverse

    Returns
    -------
    TimeData
        Polarity reversed time data
    """
    if isinstance(options["scale"], bool):
        # no scale to be performed
        return timeData
    if not isinstance(options["scale"], dict):
        # not specified in the right way
        return timeData

    from resistics.time.math import scale

    timeData = scale(timeData, options["scale"])
    return timeData


def applyCalibrationOptions(
    options: Dict, cal: Calibrator, timeData: TimeData, reader: TimeReader
) -> TimeData:
    """Calibrate time data with user options

    To calibrate, specify
    options["calibrate"] = True

    Parameters
    ----------
    options : Dict
        User specified options for calibrating
    cal : Calibrator
        A calibrator instance
    timeData : TimeData
        Time data to filter
    reader: DataReader
        A data reader object for the data

    Returns
    -------
    TimeData
        Calibrated time data
    """
    if options["calibrate"]:
        sensors = reader.getSensors(timeData.chans)
        serials = reader.getSerials(timeData.chans)
        choppers = reader.getChoppers(timeData.chans)
        timeData = cal.calibrate(timeData, sensors, serials, choppers)
    return timeData


def applyFilterOptions(options: Dict, timeData: TimeData) -> TimeData:
    """Filter time data

    To low pass filter, specify
    options["filter"]["lpfilt"] = high cutoff frequency 
    To high pass filter, specify 
    options["filter"]["hpfilt"] = low cutoff frequency
    To bandass filter, specify
    options["filter"]["bpfilt"] = [low cutoff frequency, high cutoff frequency]

    Parameters
    ----------
    options : Dict
        User specified options for filtering
    timeData : TimeData
        Time data to filter

    Returns
    -------
    TimeData
        Filtered time data
    """
    if "lpfilt" in options["filter"]:
        from resistics.time.filter import lowPass

        timeData = lowPass(timeData, options["filter"]["lpfilt"])
    if "hpfilt" in options["filter"]:
        from resistics.time.filter import highPass

        timeData = highPass(timeData, options["filter"]["hpfilt"])
    if "bpfilt" in options["filter"]:
        from resistics.time.filter import bandPass

        timeData = bandPass(
            timeData, options["filter"]["bpfilt"][0], options["filter"]["bpfilt"][1]
        )
    return timeData


def applyNotchOptions(options: Dict, timeData: TimeData) -> TimeData:
    """Notch filter time data

    To notch filter, options["notch"] needs to be a list of frequencies to filter out. For example, to notch filter at 16.6Hz, this would be:
    options["notch"] = [16.6]
    For two frequencies, for example 16.6Hz and 50Hz, then:
    options["notch"] = [16.6, 50]

    Parameters
    ----------
    options : Dict
        User specified options for notching
    timeData : TimeData
        Time data to filter

    Returns
    -------
    TimeData
        Notch filtered time data
    """
    from resistics.time.filter import notchFilter

    if len(options["notch"]) > 0:
        for n in options["notch"]:
            timeData = notchFilter(timeData, n)
    return timeData


def applyNormaliseOptions(options: Dict, timeData: TimeData) -> TimeData:
    """Normalise time data

    To normalise, options["normalise"] needs to be set to True

    Parameters
    ----------
    options : Dict
        User specified options for normalising
    timeData : TimeData
        Time data to filter

    Returns
    -------
    TimeData
        Normalised time data
    """
    from resistics.time.filter import normalise

    if options["normalise"]:
        timeData = normalise(timeData)
    return timeData


def applyInterpolationOptions(options: Dict, timeData: TimeData) -> TimeData:
    """Interpolate time data

    Interpolate time data to make sure all the data starts on a full second. This is best to do before resampling.

    To interpolate, options["interp"] needs to be set to True

    Parameters
    ----------
    options : Dict
        Interpolation options in a dictionary
    timeData : TimeData
        Time data object
    
    Returns
    -------
    TimeData
        Interpolated time data

    Notes
    -----
    This will fail with longer sample periods (i.e. greater than a second)    
    """
    from resistics.time.interp import interpolateToSecond

    if options["interp"]:
        if timeData.startTime.microsecond != 0:
            timeData = interpolateToSecond(timeData)
    return timeData


def applyResampleOptions(options: Dict, timeData: TimeData) -> TimeData:
    """Resample time data

    The resampling options in the options dictionary should be specified as:
    options["resample"][frequency to resample from] = frequency to resample to

    Parameters
    ----------
    options : Dict
        Interpolation options in a dictionary
    timeData : TimeData
        Time data object
    
    Returns
    -------
    TimeData
        Resampled time data
    """
    from resistics.time.filter import resample

    if timeData.sampleFreq in options["resamp"]:
        # then need to resample this data
        timeData = resample(timeData, options["resamp"][timeData.sampleFreq])
    return timeData
