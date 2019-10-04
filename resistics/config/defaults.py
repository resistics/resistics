from configobj import ConfigObj
from validate import Validator
from typing import Any


def getDefaultParameter(param: str) -> Any:
    """Get the default global configuration option

    Parameters
    ----------
    param : str
        The config parameter
    """
    from resistics.config.io import loadConfig

    config = loadConfig()
    return config[param]


def copyDefaultConfig(filepath: str, name: str = "global copy") -> None:
    """Create copy of the global configuration file 
    
    Parameters
    ----------
    filepath : str
        The path to write the copy of the config file
    """
    from resistics.config.io import getDefaultConfigFilepath

    config = ConfigObj(configspec=getDefaultConfigFilepath())
    validator = Validator()
    config.validate(validator, copy=True)
    # change the name of the configuration
    config["name"] = name
    with open(filepath, "wb") as f:
        config.write(f)
