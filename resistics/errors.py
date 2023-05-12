"""
Module for custom resistics errors
"""
from typing import Collection, Optional, List, Union
from pathlib import Path


def path_to_string(path: Path) -> str:
    """
    Convert a path to a string in a OS agnostic way

    Parameters
    ----------
    path : Path
        The path to convert

    Returns
    -------
    str
        A string for the path
    """
    return f"'{path.as_posix()}'"


###
# general errors
###
class PathError(Exception):
    """Use for a general error with paths"""

    def __init__(self, path: Path):
        self.path = path

    def __str__(self) -> str:
        pathstr = path_to_string(self.path)
        return f"Error with path {pathstr}"


class PathNotFoundError(PathError):
    """Use if path does not exist"""

    def __str__(self) -> str:
        pathstr = path_to_string(self.path)
        return f"Path {pathstr} does not exist"


class NotFileError(PathError):
    """Use if expected a file and got a directory"""

    def __str__(self) -> str:
        pathstr = path_to_string(self.path)
        return f"Path {pathstr} is not a file"


class NotDirectoryError(PathError):
    """Use if expected a directory and got a file"""

    def __str__(self) -> str:
        pathstr = path_to_string(self.path)
        return f"Path {pathstr} is not a directory"


class WriteError(Exception):
    def __init__(self, path: Path, message: str = ""):
        self.path = path
        self.message = message

    def __str__(self) -> str:
        pathstr = path_to_string(self.path)
        return f"Error with path {pathstr}. {self.message}."


class ReadError(Exception):
    def __init__(self, path: Path, message: str = ""):
        self.path = path
        self.message = message

    def __str__(self) -> str:
        pathstr = path_to_string(self.path)
        return f"Unable to read from {pathstr}. {self.message}."


###
# metadata data errors
###
class MetadataReadError(Exception):
    """Use when failed to read a metadata"""

    def __init__(self, path: Path, message: Optional[str] = None):
        self.path = path
        self.message = message

    def __str__(self) -> str:
        pathstr = path_to_string(self.path)
        out = f"Failed to read metadata from file {pathstr}."
        if self.message is not None:
            out += f" {self.message}."
        return out


###
# project and site errors
###
class ProjectPathError(Exception):
    """Use for a general error with a project path"""

    def __init__(self, project_dir: Path, message: str):
        self.project_dir = project_dir
        self.message = message

    def __str__(self):
        pathstr = path_to_string(self.project_dir)
        return f"{pathstr}, {self.message}."


class ProjectCreateError(ProjectPathError):
    """Use if encounter an error creating a project"""

    def __str__(self) -> str:
        pathstr = path_to_string(self.project_dir)
        return f"Error creating project in {pathstr}. {self.message}."


class ProjectLoadError(ProjectPathError):
    """Use if error on project load"""

    def __str__(self) -> str:
        pathstr = path_to_string(self.project_dir)
        return f"Error loading project {pathstr}. {self.message}."


class MeasurementNotFoundError(Exception):
    """Use if unable to find a measurement"""

    def __init__(self, site_name: str, meas_name: str):
        self.site_name = site_name
        self.meas_name = meas_name

    def __str__(self) -> str:
        return f"Measurement '{self.meas_name}' not found in Site '{self.site_name}'."


class SiteNotFoundError(Exception):
    """Use if unable to find a site"""

    def __init__(self, site_name: str):
        self.site_name = site_name

    def __str__(self) -> str:
        return f"Site '{self.site_name}' not found in project."


###
# time data errors
###
class TimeDataReadError(Exception):
    """Use when encounter an error reading time series data"""

    def __init__(self, dir_path: Path, message: str):
        self.dir_path = dir_path
        self.message = message

    def __str__(self) -> str:
        pathstr = path_to_string(self.dir_path)
        return f"Failed to read time series data from {pathstr}\n{self.message}."


class ChannelNotFoundError(Exception):
    """Use when a channel is not found"""

    def __init__(self, chan: str, chans: Collection[str]):
        self.chan = chan
        self.chans = chans

    def __str__(self) -> str:
        chans_string = "', '".join(self.chans)
        return f"'{self.chan}' not found in channels '{chans_string}'."


###
# calibration data errors
###
class CalibrationFileNotFound(Exception):
    """Use when calibration files are not found"""

    def __init__(
        self, dir_path: Path, file_paths: Union[Path, List[Path]], message: str = ""
    ):
        self.dir_path = dir_path
        self.file_paths = file_paths if isinstance(file_paths, list) else [file_paths]
        self.message = message

    def __str__(self) -> str:
        files = "', '".join([x.name for x in self.file_paths])
        pathstr = path_to_string(self.dir_path)
        outstr = f"Failed to find calibration files '{files}'"
        outstr += f" in calibration data folder {pathstr}."
        if self.message != "":
            outstr += f" {self.message}."
        return outstr


class CalibrationFileReadError(Exception):
    """Use if encounter an error reading a calibration file"""

    def __init__(self, calibration_path: Path, message: str = ""):
        self.calibration_path = calibration_path
        self.message = message

    def __str__(self) -> str:
        pathstr = path_to_string(self.calibration_path)
        outstr = f"Failed to read calibration file {pathstr}."
        if self.message != "":
            outstr += f" {self.message}"
        return outstr


###
# for running processes
###
class ProcessRunError(Exception):
    """Use when a error is encountered during a process run"""

    def __init__(self, process: str, message: str):
        self.process = process
        self.message = message

    def __str__(self) -> str:
        return f"Run error encounted in '{self.process}'. {self.message}."
