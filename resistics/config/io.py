import os
from configobj import ConfigObj
from validate import Validator
from typing import Any

from resistics.common.print import generalPrint, errorPrint
from resistics.common.io import checkFilepath


def getDefaultConfigFilepath() -> str:
    """Get the default global configuration option

    Returns
    -------
    str
        Path to global config file
    """
    # use relative path from here
    path = os.path.split(__file__)[0]
    globalConfigFile = os.path.join(path, "..", "resisticsConfig.ini")
    if not checkFilepath(globalConfigFile):
        errorPrint(
            "getDefaultConfig",
            "Default configuration file could not be found",
            quitRun=True,
        )
    return globalConfigFile


def loadConfig(filepath: str = "") -> ConfigObj:
    """Get configuration information

    Parameters
    ----------
    filepath : str, optional
        The path to the configuration file

    Returns
    -------
    config : ConfigObj
        ConfigObj with global configuration parameters
    """
    configFile = getDefaultConfigFilepath()
    if filepath == "" or not checkFilepath(filepath):
        config = ConfigObj(configspec=configFile)
    else:
        config = ConfigObj(filepath, configspec=configFile)
        generalPrint("loadConfig", "Loading configuration file {:s}".format(filepath))
    validator = Validator()
    result = config.validate(validator)
    if not result:
        errorPrint("loadConfigFile", "Config file validation failed", quitRun=True)
    return config
