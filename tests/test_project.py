"""
Testing project related functions and classes
"""
from typing import Union, Callable
from pathlib import Path
import pandas as pd
import pytest
from pydantic.error_wrappers import ValidationError

from resistics.project import PROJ_DIRS
from resistics.project import get_meas_spectra_path, get_meas_evals_path
from resistics.project import get_meas_features_path, get_mask_path, get_results_path
from resistics.testing import time_metadata_2chan
from resistics.time import TimeReaderAscii


@pytest.mark.parametrize(
    "proj_dir",
    [(Path("test")), (Path("this-is-my-project"))],
)
def test_calibration_path(proj_dir: Path):
    """Testing getting path to calibration files"""
    from resistics.project import get_calibration_path

    assert get_calibration_path(proj_dir) == proj_dir / PROJ_DIRS["calibration"]


@pytest.mark.parametrize(
    "proj_dir, site_name, meas_name",
    [
        (Path("test"), "site1", "all"),
        (Path("this-is-my-project"), "this-is-a-site", "meas"),
    ],
)
def test_meas_time_path(proj_dir: Path, site_name: str, meas_name: str):
    """Test getting paths to measurement time data"""
    from resistics.project import get_meas_time_path

    assert (
        get_meas_time_path(proj_dir, site_name, meas_name)
        == proj_dir / PROJ_DIRS["time"] / site_name / meas_name
    )


@pytest.mark.parametrize(
    "fnc, data_type, proj_dir, site_name, meas_name, config_name",
    [
        (get_meas_spectra_path, "spectra", Path("test"), "site1", "all", "example"),
        (get_meas_spectra_path, "spectra", Path("myproj"), "my-site", "all", "my_ex"),
        (get_meas_evals_path, "evals", Path("test"), "site1", "all", "example"),
        (get_meas_evals_path, "evals", Path("myproj"), "my-site", "all", "my_ex"),
        (get_meas_features_path, "features", Path("test"), "site1", "all", "example"),
        (get_meas_features_path, "features", Path("myproj"), "my-site", "all", "my_ex"),
    ],
)
def test_meas_other_data_paths(
    fnc: Callable,
    data_type: str,
    proj_dir: Path,
    site_name: str,
    meas_name: str,
    config_name: str,
):
    """Test getting paths to other data related to measurements"""
    assert (
        fnc(proj_dir, site_name, meas_name, config_name)
        == proj_dir / PROJ_DIRS[data_type] / site_name / config_name / meas_name
    )


@pytest.mark.parametrize(
    "fnc, data_type, proj_dir, site_name, config_name",
    [
        (get_mask_path, "masks", Path("test"), "site1", "example"),
        (get_mask_path, "masks", Path("myproj"), "my-site", "my_ex"),
        (get_results_path, "results", Path("test"), "site1", "example"),
        (get_results_path, "results", Path("myproj"), "my-site", "my_ex"),
    ],
)
def test_masks_results_data_paths(
    fnc: Callable,
    data_type: str,
    proj_dir: Path,
    site_name: str,
    config_name: str,
):
    """Test getting paths related to sampling frequencies"""
    assert (
        fnc(proj_dir, site_name, config_name)
        == proj_dir / PROJ_DIRS[data_type] / site_name / config_name
    )


@pytest.mark.parametrize(
    "fs, mask_name, expected",
    [
        (65536.0, "test_mask1", "test_mask1.dat"),
        (4096, "example", "example.dat"),
        (128, "this-is-a-mask", "this-is-a-mask.dat"),
        (0.0004, "what_a_mask", "what_a_mask.dat"),
    ],
)
def test_get_mask_name(fs: float, mask_name: str, expected: str):
    """Test getting the mask name"""
    from resistics.project import get_mask_name
    from resistics.common import fs_to_string

    fs_str = fs_to_string(fs)
    assert get_mask_name(fs, mask_name) == fs_str + "_" + expected


@pytest.mark.parametrize(
    "fs, tf_name, tf_var, postfix, expected",
    [
        (65536.0, "impedancetensor", "", None, "impedancetensor.json"),
        (
            4096,
            "impedancetensor",
            "var",
            "with_mask",
            "impedancetensor_var_with_mask.json",
        ),
        (128, "tipper", "", None, "tipper.json"),
        (
            0.0004,
            "tipper",
            "i am different",
            "with_mask",
            "tipper_i_am_different_with_mask.json",
        ),
    ],
)
def test_get_solution_name(
    fs: float, tf_name: str, tf_var: str, postfix: Union[str, None], expected
):
    """Test getting the solution name"""
    from resistics.project import get_solution_name
    from resistics.common import fs_to_string

    fs_str = fs_to_string(fs)
    assert get_solution_name(fs, tf_name, tf_var, postfix) == fs_str + "_" + expected


@pytest.mark.parametrize(
    "values, error",
    [
        (
            {
                "site_name": "test_site",
                "dir_path": Path("test", "project", "time", "test_site", "measX"),
                "metadata": time_metadata_2chan(),
                "reader": TimeReaderAscii(),
            },
            None,
        ),
        (
            {
                "site_name": "test_site",
                "dir_path": "test/project/time/test_site/measY",
                "metadata": time_metadata_2chan(),
                "reader": TimeReaderAscii(),
            },
            None,
        ),
        (
            {
                "site_name": "test_site",
                "dir_path": Path("test", "project", "time", "test_site", "measX"),
                "metadata": time_metadata_2chan(),
                "reader": -10,
            },
            ValidationError,
        ),
    ],
    ids=["One error", "Convert string to Path", "All correct"],
)
def test_measurement(values, error):
    """Test the Measurement class"""
    from resistics.project import Measurement

    if error is not None:
        with pytest.raises(error):
            assert Measurement(**values)
        return

    meas = Measurement(**values)
    assert meas.name == meas.dir_path.name


@pytest.mark.parametrize(
    "values, error",
    [
        ({}, ValidationError),
        ({"ref_time": -999}, ValidationError),
        ({"ref_time": "2021-01-01 00:00:00"}, None),
        ({"ref_time": pd.Timestamp("2021-01-01 00:00:00")}, None),
        (
            {
                "ref_time": "2021-01-01 00:00:00",
                "location": "Masai Mara",
                "country": "Kenya",
                "year": 2021,
                "description": "This is a dummy project",
                "contributors": ["A", "B", "C"],
            },
            None,
        ),
        (
            {
                "ref_time": "2021-01-01 00:00:00",
                "location": "Masai Mara",
                "country": 10,
                "year": 2021,
                "description": "This is a dummy project",
                "contributors": ["A", "B", "C"],
            },
            None,
        ),
    ],
    ids=[
        "Nothing",
        "Wrong type",
        "Just reference time",
        "Pandas datetime",
        "All correct",
        "Country wrong type",
    ],
)
def test_project_metadata(values, error):
    """Test the ProjectMetadata class"""
    from resistics.project import ProjectMetadata

    if error is not None:
        with pytest.raises(error):
            assert ProjectMetadata(**values)
        return
    assert ProjectMetadata(**values)
