"""Module for custom resistics errors"""

from collections.abc import Collection
from pathlib import Path


def path_to_string(path: Path) -> str:
    """Convert a path to a string in a OS agnostic way

    :param path: The path to convert

    :return: A string for the path
    """
    return f"'{path.as_posix()}'"


###
# general errors
###
class PathError(Exception):
    """Report a general filesystem-path failure.

    :param path: Filesystem path associated with the failure.
    """

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
    """Report a failure to write data at a filesystem path.

    :param path: Filesystem path that could not be written.
    :param message: Additional context for the write failure.
    """

    def __init__(self, path: Path, message: str = ""):
        self.path = path
        self.message = message

    def __str__(self) -> str:
        pathstr = path_to_string(self.path)
        return f"Error with path {pathstr}. {self.message}."


class ReadError(Exception):
    """Report a failure to read data from a filesystem path.

    :param path: Filesystem path that could not be read.
    :param message: Additional context for the read failure.
    """

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
    """Report a failure to read metadata.

    :param path: Metadata file that could not be read.
    :param message: Optional additional failure context.
    """

    def __init__(self, path: Path, message: str | None = None):
        self.path = path
        self.message = message

    def __str__(self) -> str:
        pathstr = path_to_string(self.path)
        out = f"Failed to read metadata from file {pathstr}."
        if self.message is not None:
            out += f" {self.message}."
        return out


class ChannelNotFoundError(Exception):
    """Report that a requested channel is unavailable.

    :param chan: Requested channel name.
    :param chans: Available channel names.
    """

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
    """Report missing calibration files.

    :param dir_path: Directory searched for calibration files.
    :param file_paths: Expected calibration filename or filenames.
    :param message: Additional context for the search failure.
    """

    def __init__(
        self, dir_path: Path, file_paths: Path | list[Path], message: str = ""
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
    """Report a failure to read a calibration file.

    :param calibration_path: Calibration file that could not be read.
    :param message: Additional context for the read failure.
    """

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
    """Report a process execution failure.

    :param process: Name of the process that failed.
    :param message: Failure detail from the process.
    """

    def __init__(self, process: str, message: str):
        self.process = process
        self.message = message

    def __str__(self) -> str:
        return f"Run error encounted in '{self.process}'. {self.message}."
