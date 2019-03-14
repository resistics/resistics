import inspect
from datetime import datetime
from typing import Dict, List, Union, Any

# import from package
from resistics.dataObjects.configData import ConfigData
from resistics.ioHandlers.dataReader import DataReader
from resistics.calculators.calibrator import Calibrator
from resistics.calculators.decimationParameters import DecimationParams
from resistics.calculators.windowParameters import WindowParams
from resistics.calculators.windowSelector import WindowSelector
from resistics.calculators.processorSingleSite import ProcessorSingleSite
from resistics.calculators.processorRemoteReference import ProcessorRemoteReference
from resistics.dataObjects.timeData import TimeData
from resistics.utilities.utilsPrint import (
    generalPrint,
    warningPrint,
    errorPrint,
    blockPrint,
)
from resistics.utilities.utilsFilter import (
    lowPass,
    highPass,
    bandPass,
    notchFilter,
    normalise,
    resample,
)
from resistics.utilities.utilsInterp import interpolateToSecond


def projectText(infoStr: str) -> None:
    """General print to terminal

    Parameters
    ----------
    infoStr : str
        The string to print to the console
    """

    generalPrint("{} Info".format(inspect.stack()[1][3]), infoStr)


def projectBlock(textLst: List[str]) -> None:
    """Class information as a list of strings

    Parameters
    ----------
    textLst : list[str]
        List of strings with information
    """

    blockPrint(inspect.stack()[1][3], textLst)


def projectWarning(warnStr: str) -> None:
    """Warning print to terminal

    Parameters
    ----------
    warnStr : str
        The string to print to the console
    """

    warningPrint("{} Warning".format(inspect.stack()[1][3]), warnStr)


def projectError(errorStr: str, quitRun: bool = False) -> None:
    """Error print to terminal and possibly quit

    Parameters
    ----------
    errorStr : str
        The string to print to the console
    quitRun : bool, optional (False)
        If True, the code will exit
    """

    errorPrint("{} Warning".format(inspect.stack()[1][3]), errorStr, quitRun=quitRun)


def getCalibrator(calPath: str, config: Union[ConfigData, None] = None) -> Calibrator:
    """Create a Calibnator object from calibration path and configuration information

    Parameters
    ----------
    calPath : str
        Path to calibration directory
    config : ConfigData
        Configuration data
    
    Returns
    -------
    Calibrator
        A calibrator object
    """

    cal = Calibrator(calPath)
    if config is None:
        return cal

    cal.useTheoretical = config.configParams["Calibration"]["usetheoretical"]
    return cal


def getDecimationParameters(
    sampleFreq: float, config: Union[ConfigData, None] = None
) -> DecimationParams:
    """Create a DecimationParams object from sampling frequency and configuration information

    Parameters
    ----------
    sampleFreq : float
        Sampling frequency of the data
    config : ConfigData
        Configuration data

    Returns
    -------
    DecimationParams
        A decimation parameters object        
    """

    decParams = DecimationParams(sampleFreq)
    if config is None:
        return decParams

    if config.flags["customfrequencies"]:
        decParams.setFrequencyParams(
            config.configParams["Frequencies"]["frequencies"],
            config.configParams["Decimation"]["numlevels"],
            config.configParams["Frequencies"]["perlevel"],
        )
    else:
        decParams.setDecimationParams(
            config.configParams["Decimation"]["numlevels"],
            config.configParams["Frequencies"]["perlevel"],
        )

    return decParams


def getWindowParameters(
    decParams: DecimationParams, config: Union[ConfigData, None] = None
) -> WindowParams:
    """Create a WindowParams object from decimationParams and configuration data

    Parameters
    ----------
    decParams : DecimationParams
        DecimationParams object to hold the decimation parameters
    config : ConfigData
        Configuration data
    
    Returns
    -------
    WindowParams
        A window parameters object        
    """

    winParams = WindowParams(decParams)
    if config is None:
        return winParams

    if config.flags["customwindows"]:
        winParams.setWindowParameters(
            config.configParams["Window"]["windowsizes"],
            config.configParams["Window"]["overlapsizes"],
        )
    else:
        winParams.setMinParams(
            config.configParams["Window"]["minwindowsize"],
            config.configParams["Window"]["minoverlapsize"],
        )

    return winParams


def getSingleSiteProcessor(
    winSelector: WindowSelector, outPath: str, config: Union[ConfigData, None] = None
) -> ProcessorSingleSite:
    """Create a ProcessorSingleSite object from a windowSelector object, outPath and config data

    Parameters
    ----------
    winSelector : WindowSelector
        Window selector with sites, masks etc already specified
    tfPath : str
        Path to output transfer function data
    config : ConfigData, optional
        Configuration data
    
    Returns
    -------
    ProcessorSingleSite
        A single site processor        
    """

    processor = ProcessorSingleSite(winSelector, outPath)
    if config is None:
        return processor

    return processor


def getRemoteReferenceProcessor(
    winSelector: WindowSelector, outPath: str, config: Union[ConfigData, None] = None
):
    """Create a ProcessorRemoteReference object from a windowSelector object, outPath and config data

    Parameters
    ----------
    winSelector : WindowSelector
        Window selector with sites, masks etc already specified
    outPath : str
        Path to output transfer function data
    config : ConfigData, optional
        Configuration data
    
    Returns
    -------
    ProcessorRemoteReference
        A remote reference processor        
    """

    processor = ProcessorRemoteReference(winSelector, outPath)
    if config is None:
        return processor
    
    return processor


def checkDateOptions(options: Dict, timeStart: datetime, timeStop: datetime) -> bool:
    """Check to see if data contributes to user specified date range

    Parameters
    ----------
    options : Dict
        Options dictionary with start and stop options specified by user (if specified at all)
    timeStart : datetime
        Start time of data
    timeStop : datetime
        Stop time of data

    Returns
    -------
    bool
        True if data contributes to the date range
    """

    # now check the user provided dates
    if options["start"] and options["start"] > timeStop:
        # this data has nothing to contribute in the optional date range
        return False
    if options["stop"] and options["stop"] < timeStart:
        # this data has nothing to contribute in the optional date range
        return False
    return True


def applyCalibrationOptions(
    options: Dict, cal: Calibrator, timeData: TimeData, reader: DataReader
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
        timeData = lowPass(timeData, options["filter"]["lpfilt"])
    if "hpfilt" in options["filter"]:
        timeData = highPass(timeData, options["filter"]["hpfilt"])
    if "bpfilt" in options["filter"]:
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

    if options["normalise"]:
        projectText("Normalising data")
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

    if timeData.sampleFreq in options["resamp"]:  # then need to resample this data
        timeData = resample(timeData, options["resamp"][timeData.sampleFreq])
    return timeData
