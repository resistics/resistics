"""
Module for custom resistics errors
"""
from typing import Collection, Optional, List, Union
from pathlib import Path


###
# general errors
###
class PathError(Exception):
    """Use for a general error with paths"""

    def __init__(self, path: Path):
        self.path = path

    def __str__(self) -> str:
        return f"Error with path {self.path}"


class PathNotFoundError(PathError):
    """Use if path does not exist"""

    def __str__(self) -> str:
        return f"Path {self.path} does not exist"


class NotFileError(PathError):
    """Use if expected a file and got a directory"""

    def __str__(self) -> str:
        return f"Path {self.path} is not a file"


class NotDirectoryError(PathError):
    """Use if expected a directory and got a file"""

    def __str__(self) -> str:
        return f"Path {self.path} is not a directory"


class WriteError(Exception):
    def __init__(self, path: Path, message: str = ""):
        self.path = path
        self.message = message

    def __str__(self) -> str:
        return f"Error with path {self.path}. {self.message}."


class ReadError(Exception):
    def __init__(self, path: Path, message: str = ""):
        self.path = path
        self.message = message

    def __str__(self) -> str:
        return f"Unable to read from {self.path}. {self.message}."


###
# metadata data errors
###
class MetadataReadError(Exception):
    """Use when failed to read a metadata"""

    def __init__(self, path: Path, message: Optional[str] = None):
        self.path = path
        self.message = message

    def __str__(self) -> str:
        out = f"Failed to read metadata from file {self.path}."
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
        return f"{self.project_dir}, {self.message}"


class ProjectCreateError(ProjectPathError):
    """Use if encounter an error creating a project"""

    def __str__(self) -> str:
        return f"Error creating project in {self.project_dir}. {self.message}."


class ProjectLoadError(ProjectPathError):
    """Use if error on project load"""

    def __str__(self) -> str:
        return f"Error loading project {self.project_dir}. {self.message}."


class MeasurementNotFoundError(Exception):
    """Use if unable to find a measurement"""

    def __init__(self, site_name: str, meas_name: str):
        self.site_name = site_name
        self.meas_name = meas_name

    def __str__(self) -> str:
        return f"Measurement {self.meas_name} not found in Site {self.site_name}"


class SiteNotFoundError(Exception):
    """Use if unable to find a site"""

    def __init__(self, site_name: str):
        self.site_name = site_name

    def __str__(self) -> str:
        return f"Site {self.site_name} not found in project"


###
# time data errors
###
class TimeDataReadError(Exception):
    """Use when encounter an error reading time series data"""

    def __init__(self, dir_path: Path, message: str):
        self.dir_path = dir_path
        self.message = message

    def __str__(self) -> str:
        return f"Failed to read time series data from {self.dir_path}\n{self.message}"


class ChannelNotFoundError(Exception):
    """Use when a channel is not found"""

    def __init__(self, chan: str, chans: Collection[str]):
        self.chan = chan
        self.chans = chans

    def __str__(self) -> str:
        chans_string = "', '".join(self.chans)
        return f"'{self.chan}' not found in channels '{chans_string}'"


###
# calibration data errors
###
class CalibrationFileNotFound(Exception):
    """Use when calibration files are not found"""

    def __init__(
        self, dir_path: Path, file_paths: Union[Path, List[Path]], message: str = ""
    ):
        self.dir_path = dir_path
        self.file_paths = file_paths
        self.message = message

    def __str__(self) -> str:
        outstr = f"Failed to find calibration files {self.file_paths}"
        outstr += f" in calibration data folder {self.dir_path}."
        if self.message != "":
            outstr += f" {self.message}"
        return outstr


class CalibrationFileReadError(Exception):
    """Use if encounter an error reading a calibration file"""

    def __init__(self, calibration_path: Path, message: str = ""):
        self.calibration_path = calibration_path
        self.message = message

    def __str__(self) -> str:
        outstr = f"Failed to read calibration file {self.calibration_path}."
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
        return f"Run error encounted in {self.process}. {self.message}."
