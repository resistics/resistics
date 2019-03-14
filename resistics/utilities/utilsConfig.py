import os
from configobj import ConfigObj
from validate import Validator
from typing import Any

# import from package
from resistics.utilities.utilsPrint import generalPrint, errorPrint
from resistics.utilities.utilsIO import checkFilepath


def getDefaultConfigFilepath() -> str:
    """Get the default global configuration option

    Returns
    -------
    str
        Path to global config file
    """

    # globalConfigFile = os.path.join("..", "config.ini")
    globalConfigFile = os.path.join(
        "e:/", "magnetotellurics", "code", "resistics", "resistics", "config.ini"
    )
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


def getDefaultParameter(param: str) -> Any:
    """Get the default global configuration option

    Parameters
    ----------
    param : str
        The config parameter
    """

    config = loadConfig()
    return config[param]


def copyDefaultConfig(filepath: str, name: str = "global copy") -> None:
    """Create copy of the global configuration file 
    
    Parameters
    ----------
    filepath : str
        The path to write the copy of the config file
    """

    config = ConfigObj(configspec=getDefaultConfigFilepath())
    validator = Validator()
    config.validate(validator, copy=True)
    # change the name of the configuration
    config["name"] = name
    with open(filepath, "wb") as f:
        config.write(f)
