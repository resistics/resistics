"""
Testing project related functions and classes
"""
from typing import Union, Callable
from pathlib import Path
import pytest

from resistics.project import PROJ_DIRS
from resistics.project import get_meas_spectra_path, get_meas_evals_path
from resistics.project import get_meas_features_path, get_fs_mask_path
from resistics.project import get_fs_results_path


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
    "fnc, data_type, proj_dir, site_name, config_name, fs",
    [
        (get_fs_mask_path, "masks", Path("test"), "site1", "example", 65536.0),
        (get_fs_mask_path, "masks", Path("test"), "site1", "example", 4096),
        (get_fs_mask_path, "masks", Path("test"), "site1", "example", 128.0),
        (get_fs_mask_path, "masks", Path("myproj"), "my-site", "my_ex", 0.0004),
        (get_fs_results_path, "results", Path("test"), "site1", "example", 65536.0),
        (get_fs_results_path, "results", Path("test"), "site1", "example", 4096),
        (get_fs_results_path, "results", Path("test"), "site1", "example", 128),
        (get_fs_results_path, "results", Path("myproj"), "my-site", "my_ex", 0.0004),
    ],
)
def test_fs_data_paths(
    fnc: Callable,
    data_type: str,
    proj_dir: Path,
    site_name: str,
    config_name: str,
    fs: Union[float, int],
):
    """Test getting paths related to sampling frequencies"""
    from resistics.common import fs_to_string

    data_dir = fs_to_string(fs)
    assert (
        fnc(proj_dir, site_name, config_name, fs)
        == proj_dir / PROJ_DIRS[data_type] / site_name / config_name / data_dir
    )
