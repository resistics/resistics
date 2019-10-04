from typing import Union

from resistics.config.data import ConfigData
from resistics.calibrate.calibrator import Calibrator
from resistics.decimate.parameters import DecimationParameters
from resistics.window.parameters import WindowParameters
from resistics.window.selector import WindowSelector
from resistics.time.reader import TimeReader
from resistics.time.data import TimeData
from resistics.regression.local import LocalRegressor
from resistics.regression.remote import RemoteRegressor
from resistics.project.data import ProjectData


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
) -> DecimationParameters:
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
    decParams = DecimationParameters(sampleFreq)
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
    decParams: DecimationParameters, config: Union[ConfigData, None] = None
) -> WindowParameters:
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
    winParams = WindowParameters(decParams)
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


def getWindowSelector(
    projData: ProjectData, decParams: DecimationParameters, winParams: WindowParameters
) -> WindowSelector:
    """Create a WindowSelector object from projectData, decimationParams, windowParams

    Parameters
    ----------
    projData : ProjectData
        A project data object
    decParams : DecimationParams
        DecimationParams object to hold the decimation parameters
    winParams : Windowparams
        WindowParams object to hold the windowing parameters
    
    Returns
    -------
    WindowSelector
        A window selector object        
    """
    selector = WindowSelector(
        projData,
        decParams.sampleFreq,
        decParams,
        winParams,
        specdir=projData.config.configParams["Spectra"]["specdir"],
    )
    return selector


def getLocalRegressor(
    winSelector: WindowSelector, outPath: str, config: Union[ConfigData, None] = None
) -> LocalRegressor:
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
    LocalRegressor
        local, single site regression        
    """
    processor = LocalRegressor(winSelector, outPath)
    if config is None:
        return processor
    return processor


def getRemoteRegressor(
    winSelector: WindowSelector, outPath: str, config: Union[ConfigData, None] = None
) -> RemoteRegressor:
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
    RemoteRegression
        Regression including a remote reference        
    """
    processor = RemoteRegressor(winSelector, outPath)
    if config is None:
        return processor
    return processor
