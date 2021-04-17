"""
Module for custom resistics errors
"""
from typing import Collection, Any, Type, Optional
from pathlib import Path


###
# general errors
###
class PathNotFoundError(Exception):
    """Use if path does not exist"""

    def __init__(self, path: Path):
        self.path = path

    def __str__(self) -> str:
        return f"Path {self.path} does not exist"


class NotFileError(PathNotFoundError):
    """Use if expected a file and got a directory"""

    def __str__(self) -> str:
        return f"Path {self.path} is not a file"


class NotDirectoryError(PathNotFoundError):
    """Use if expected a directory and got a file"""

    def __str__(self) -> str:
        return f"Path {self.path} is not a directory"


###
# Serialization errors
###
class SerializationError(Exception):
    def __init__(self, obj: Any):
        self.obj = obj

    def __str__(self) -> str:
        return f"Unable to serialize {self.obj} of type {type(self.obj)}"


class DeserializationError(Exception):
    def __init__(self, obj: Any, obj_type: Type[Any]):
        self.obj = obj
        self.obj_type = obj_type

    def __str__(self) -> str:
        return f"Unable to deserialize {self.obj} to expected type {self.obj_type}"


###
# metadata data errors
###
class MetadataKeyNotFound(Exception):
    """Use if a key is requested which does not exist in the metadata"""

    def __init__(self, key: str, keys: Collection[str]):
        self.key = key
        self.keys = keys

    def __str__(self) -> str:
        return f"Key {self.key} not found in Metadata with keys {self.keys}"


class MetadataEntryError(Exception):
    """Parent class for MetadataGroup entry errors"""

    def __init__(
        self, entry: str, entries: Collection[str], message: Optional[str] = None
    ):
        self.entry = entry
        self.entries = entries
        self.message = message


class MetadataEntryNotFound(MetadataEntryError):
    """Use if Metadata entry not found in a MetadataGroup"""

    def __str__(self) -> str:
        out = f"Entry {self.entry} not found in Group with entries {self.entries}."
        if self.message is not None:
            out += " {self.message}."
        return out


class MetadataEntryAlreadyExists(MetadataEntryError):
    """Use if Metadata entry already exists when trying to add"""

    def __str__(self) -> str:
        out = f"Entry {self.entry} already exists in Group with entries {self.entries}."
        if self.message is not None:
            out += " {self.message}."
        return out


class MetadataReadError(Exception):
    """Use when failed to read a metadata"""

    def __init__(self, path: Path, message: Optional[str] = None):
        self.path = path
        self.message = message

    def __str__(self) -> str:
        out = f"Failed to read metadata from file {self.path}"
        if self.message is not None:
            out += " {self.message}."
        return out


###
# project and site errors
###
class ProjectCreateError(Exception):
    """Use if encounter an error creating a project"""

    def __init__(self, project_dir: Path, message: str):
        self.project_dir = project_dir
        self.message = message

    def __str__(self) -> str:
        return f"Error creating project in {self.project_dir}. {self.message}."


class ProjectLoadError(Exception):
    """Use if error on project load"""

    def __init__(self, project_dir: Path, message: str):
        self.project_dir = project_dir
        self.message = message

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
# for running processes
###
class ProcessCheckError(Exception):
    """Use when a error is encountered during a process check"""

    def __init__(self, process: str, message: str):
        self.process = process
        self.message = message

    def __str__(self) -> str:
        return f"Check error encounted in {self.process}. {self.message}."


class ProcessRunError(Exception):
    """Use when a error is encountered during a process run"""

    def __init__(self, process: str, message: str):
        self.process = process
        self.message = message

    def __str__(self) -> str:
        return f"Run error encounted in {self.process}. {self.message}."
