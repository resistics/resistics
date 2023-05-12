"""
Test that errors output the expected response
"""
from pathlib import Path
import pytest
from resistics.errors import (
    PathError,
    PathNotFoundError,
    NotFileError,
    NotDirectoryError,
    WriteError,
    ReadError,
    MetadataReadError,
    ProjectPathError,
    ProjectCreateError,
    ProjectLoadError,
    MeasurementNotFoundError,
    SiteNotFoundError,
    TimeDataReadError,
    ChannelNotFoundError,
    CalibrationFileNotFound,
    CalibrationFileReadError,
    ProcessRunError,
)


@pytest.mark.parametrize(
    "error, args, expected_message",
    [
        (
            PathError,
            [Path("test")],
            "Error with path 'test'",
        ),
        (
            PathError,
            [Path("test1", "test2")],
            "Error with path 'test1/test2'",
        ),
        (
            PathNotFoundError,
            [Path("test", "directory")],
            "Path 'test/directory' does not exist",
        ),
        (
            NotFileError,
            [Path("test", "directory")],
            "Path 'test/directory' is not a file",
        ),
        (
            NotDirectoryError,
            [Path("test", "file")],
            "Path 'test/file' is not a directory",
        ),
        (
            NotDirectoryError,
            [Path("/", "test", "file")],
            "Path '/test/file' is not a directory",
        ),
        (
            WriteError,
            [Path("/", "test", "file")],
            "Error with path '/test/file'.",
        ),
        (
            WriteError,
            [Path("/", "test", "file"), "example message"],
            "Error with path '/test/file'. example message.",
        ),
        (
            ReadError,
            [Path("/", "test", "file")],
            "Unable to read from '/test/file'.",
        ),
        (
            ReadError,
            [Path("/", "test", "file"), "example message"],
            "Unable to read from '/test/file'. example message.",
        ),
        (
            MetadataReadError,
            [Path("/", "test", "file")],
            "Failed to read metadata from file '/test/file'.",
        ),
        (
            MetadataReadError,
            [Path("/", "test", "file"), "example message"],
            "Failed to read metadata from file '/test/file'. example message.",
        ),
        (
            ProjectPathError,
            [Path("test", "project"), "dataset not found"],
            "'test/project', dataset not found.",
        ),
        (
            ProjectCreateError,
            [Path("test", "project"), "The project already exists."],
            "Error creating project in 'test/project'. The project already exists.",
        ),
        (
            ProjectLoadError,
            [Path("test", "project"), "Path does not exist"],
            "Error loading project 'test/project'. Path does not exist.",
        ),
        (
            MeasurementNotFoundError,
            ["siteA", "meas1"],
            "Measurement 'meas1' not found in Site 'siteA'.",
        ),
        (
            SiteNotFoundError,
            ["siteA"],
            "Site 'siteA' not found in project.",
        ),
        (
            TimeDataReadError,
            [
                Path("test", "project", "time", "siteA", "meas1"),
                "directory does not exist",
            ],
            "Failed to read time series data from 'test/project/time/siteA/meas1'\ndirectory does not exist.",
        ),
        (
            ChannelNotFoundError,
            ["Qx", ["Ex", "Hx"]],
            "'Qx' not found in channels 'Ex', 'Hx'.",
        ),
        (
            CalibrationFileNotFound,
            [
                Path("test", "project", "calibrate"),
                Path("test", "project", "calibrate", "calfile1"),
            ],
            "Failed to find calibration files 'calfile1' in calibration data folder 'test/project/calibrate'.",
        ),
        (
            CalibrationFileNotFound,
            [
                Path("test", "project", "calibrate"),
                Path("test", "project", "calibrate", "calfile1"),
                "Sorry",
            ],
            "Failed to find calibration files 'calfile1' in calibration data folder 'test/project/calibrate'. Sorry.",
        ),
        (
            CalibrationFileNotFound,
            [
                Path("test", "project", "calibrate"),
                [
                    Path("test", "project", "calibrate", "calfile1"),
                    Path("test", "project", "calibrate", "calfile2"),
                ],
                "Sorry",
            ],
            "Failed to find calibration files 'calfile1', 'calfile2' in calibration data folder 'test/project/calibrate'. Sorry.",
        ),
        (
            CalibrationFileReadError,
            [Path("test", "project", "calibrate", "calfile")],
            "Failed to read calibration file 'test/project/calibrate/calfile'.",
        ),
        (
            CalibrationFileReadError,
            [Path("test", "project", "calibrate", "calfile"), "File does not exist."],
            "Failed to read calibration file 'test/project/calibrate/calfile'. File does not exist.",
        ),
        (
            ProcessRunError,
            ["Add", "The value to add makes no sense."],
            "Run error encounted in 'Add'. The value to add makes no sense.",
        ),
    ],
)
def test_error(error, args, expected_message):
    with pytest.raises(error, match=expected_message):
        raise error(*args)
