"""
Module for custom resistics errors
"""
from typing import Collection
from pathlib import Path

###
# general errors
###
class PathNotFoundError(Exception):
    """Use if path does not exist"""

    def __init__(self, path: Path):
        self.path = path

    def __str__(self):
        return f"Path {self.path} does not exist"


class NotFileError(PathNotFoundError):
    """Use if expected a file and got a directory"""

    def __str__(self):
        return f"Path {self.path} is not a file"


class NotDirectoryError(PathNotFoundError):
    """Use if expected a directory and got a file"""

    def __str__(self):
        return f"Path {self.path} is not a directory"


###
# project and site errors
###
class ProjectCreateError(Exception):
    """Use if encounter an error creating a project"""

    def __init__(self, project_dir: Path, message: str):
        self.project_dir = project_dir
        self.message = message

    def __str__(self):
        return f"Error creating project in {self.project_dir}. {self.message}."


class ProjectLoadError(Exception):
    """Use if error on project load"""

    def __init__(self, project_dir: Path, message: str):
        self.project_dir = project_dir
        self.message = message

    def __str__(self):
        return f"Error loading project {self.project_dir}. {self.message}."


class MeasurementNotFoundError(Exception):
    """Use if unable to find a measurement"""

    def __init__(self, site_name: str, meas_name: str):
        self.site_name = site_name
        self.meas_name = meas_name

    def __str__(self):
        return f"Measurement {self.meas_name} not found in Site {self.site_name}"


class SiteNotFoundError(Exception):
    """Use if unable to find a site"""

    def __init__(self, site_name: str):
        self.site_name = site_name

    def __str__(self):
        return f"Site {self.site_name} not found in project"


###
# time data errors
###
class HeaderReadError(Exception):
    """Use for an issue reading header files"""

    def __init__(self, path: Path, message: str = None):
        self.path = path
        self.message = message

    def __str__(self):
        outstr = f"Reading of header file {self.path} failed"
        if self.message is not None:
            outstr += f"\n{self.message}"
        return outstr


class TimeDataReadError(Exception):
    """Use when encounter an error reading time series data"""

    def __init__(self, dir_path: Path, message: str):
        self.dir_path = dir_path
        self.message = message

    def __str__(self):
        return f"Failed to read time series data from {self.dir_path}\n{self.message}"


class HeaderNotFoundError(Exception):
    """Use when a header key is not found"""

    def __init__(self, header_name: str, header_names: Collection[str]):
        self.header_name = header_name
        self.header_names = header_names

    def __str__(self):
        headers_string = "', '".join(self.header_names)
        return f"'{self.header_name}' not found in headers '{headers_string}'"


class ChannelNotFoundError(Exception):
    """Use when a channel is not found"""

    def __init__(self, chan: str, chans: Collection[str]):
        self.chan = chan
        self.chans = chans

    def __str__(self):
        chans_string = "', '".join(self.chans)
        return f"'{self.chan}' not found in channels '{chans_string}'"


###
# for running processes
###
class ProcessCheckError(Exception):
    """Use when a error is encountered during a process check"""

    def __init__(self, process: str, message: str):
        self.process = process
        self.message = message

    def __str__(self):
        return f"Check error encounted in {self.process}. {self.message}."


class ProcessRunError(Exception):
    """Use when a error is encountered during a process run"""

    def __init__(self, process: str, message: str):
        self.process = process
        self.message = message

    def __str__(self):
        return f"Run error encounted in {self.process}. {self.message}."