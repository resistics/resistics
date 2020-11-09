"""Functions for configuring the logging"""
import logging
import logging.config
from typing import Dict


def logging_format() -> Dict[str, str]:
    """Return logging formatting options

    Returns
    -------
    Dict[str, str]
        Logging format dictionary
    """
    format_dict = {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(filename)s:%(lineno)s - %(funcName)20s(): %(message)s",
            "datefmt": "%Y/%m/%d %I:%M:%S %p",
        }
    }
    return format_dict


def logging_handlers(
    resistics_level: str = "INFO", root_level: str = "WARNING"
) -> Dict[str, str]:
    """Return logging handler options

    Returns
    -------
    Dict[str, str]
        Logging handling dictionary
    """
    handler_dict = {
        "root_handler": {
            "class": "logging.FileHandler",
            "level": root_level,
            "formatter": "standard",
            "filename": "resistics.log",
            "encoding": "utf8",
        },
        "resistics_handler": {
            "class": "logging.FileHandler",
            "level": resistics_level,
            "formatter": "standard",
            "filename": "resistics.log",
            "encoding": "utf8",
        },
    }
    return handler_dict


def configure_logging(
    resistics_level: str = "INFO", root_level: str = "WARNING"
) -> None:
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": logging_format(),
        "handlers": logging_handlers(resistics_level, root_level),
        "loggers": {
            "": {
                "handlers": ["root_handler"],
                "level": root_level,
                "propagate": False,
            },
            "resistics": {
                "handlers": ["resistics_handler"],
                "level": resistics_level,
                "propagate": False,
            },
        },
    }
    logging.config.dictConfig(logging_config)


def configure_default_logging() -> None:
    """Configure default logging"""
    configure_logging("INFO")
    # logging.getLogger("resistics").setLevel("INFO")


def configure_warning_logging() -> None:
    """Configure default logging"""
    configure_logging("WARNING")
    # logging.getLogger("resistics").setLevel("WARNING")


def configure_debug_logging() -> None:
    """Configure default logging"""
    configure_logging("DEBUG")
    # logging.getLogger("resistics").setLevel("DEBUG")
