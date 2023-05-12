"""
Testing project related functions and classes
"""
from typing import Union, Callable
from pathlib import Path
import pandas as pd
import pytest
from pydantic.error_wrappers import ValidationError

from resistics.project import PROJ_DIRS, Measurement, Site, Project, ProjectMetadata
from resistics.project import get_meas_spectra_path, get_meas_evals_path
from resistics.project import get_meas_features_path, get_mask_path, get_results_path
from resistics.testing import time_metadata_2chan, time_metadata_mt
from resistics.time import TimeReaderAscii, TimeReaderNumpy


MEAS1 = Measurement(
    site_name="SiteA",
    dir_path=Path("test", "project", "time", "siteA", "meas1"),
    metadata=time_metadata_2chan(
        fs=10, first_time="2022-01-05 12:37:22", n_samples=20_000
    ),
    reader=TimeReaderAscii(),
)
MEAS2 = Measurement(
    site_name="siteA",
    dir_path="test/project/time/siteA/meas2",
    metadata=time_metadata_mt(
        fs=20, first_time="2022-01-13 16:42:03", n_samples=30_000
    ),
    reader=TimeReaderNumpy(),
)
SITEA = Site(
    dir_path=Path("test_project", "time", "siteA"),
    measurements={"meas1": MEAS1, "meas2": MEAS2},
    begin_time=min(MEAS1.metadata.first_time, MEAS2.metadata.first_time),
    end_time=max(MEAS1.metadata.last_time, MEAS2.metadata.last_time),
)


RUN1 = Measurement(
    site_name="siteB",
    dir_path=Path("test", "project", "time", "siteB", "run1"),
    metadata=time_metadata_mt(
        fs=10, first_time="2022-01-05 12:40:48", n_samples=60_000
    ),
    reader=TimeReaderNumpy(),
)
RUN2 = Measurement(
    site_name="siteB",
    dir_path=Path("test", "project", "time", "siteB", "run2"),
    metadata=time_metadata_mt(
        fs=20, first_time="2022-01-13 16:31:36", n_samples=100_000
    ),
    reader=TimeReaderAscii(),
)
RUN3 = Measurement(
    site_name="siteB",
    dir_path=Path("test", "project", "time", "siteB", "run3"),
    metadata=time_metadata_mt(
        fs=50, first_time="2022-01-17 19:25:44", n_samples=50_000
    ),
    reader=TimeReaderNumpy(),
)
SITEB = Site(
    dir_path=Path("test_project", "time", "siteB"),
    measurements={"run1": RUN1, "run2": RUN2, "run3": RUN3},
    begin_time=min(
        RUN1.metadata.first_time, RUN2.metadata.first_time, RUN3.metadata.first_time
    ),
    end_time=max(
        RUN1.metadata.last_time, RUN2.metadata.last_time, RUN3.metadata.last_time
    ),
)


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
        (MEAS1.dict(), None),
        (MEAS2.dict(), None),
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


def test_site():
    """Test a Site instance"""
    site = SITEA
    assert site.name == "siteA"
    assert site.begin_time == MEAS1.metadata.first_time
    assert site.end_time == MEAS2.metadata.last_time
    assert site.n_meas == 2
    assert site.fs() == [10, 20]
    assert site.get_measurement("meas1") == MEAS1
    assert site.get_measurements() == {"meas1": MEAS1, "meas2": MEAS2}
    assert site.get_measurements(fs=20) == {"meas2": MEAS2}
    data = [
        {
            "name": "meas1",
            "fs": MEAS1.metadata.fs,
            "first_time": MEAS1.metadata.first_time.isoformat(),
            "last_time": MEAS1.metadata.last_time.isoformat(),
        },
        {
            "name": "meas2",
            "fs": MEAS2.metadata.fs,
            "first_time": MEAS2.metadata.first_time.isoformat(),
            "last_time": MEAS2.metadata.last_time.isoformat(),
        },
    ]
    df = pd.DataFrame.from_records(data=data).sort_values("name")
    df["first_time"] = pd.to_datetime(df["first_time"])
    df["last_time"] = pd.to_datetime(df["last_time"])
    df["site"] = "siteA"
    pd.testing.assert_frame_equal(site.to_dataframe().sort_values("name"), df)


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
    if error is not None:
        with pytest.raises(error):
            assert ProjectMetadata(**values)
        return
    assert ProjectMetadata(**values)


def test_project():
    """Test a project"""
    metadata = ProjectMetadata(
        ref_time="2022-01-03 00:00:00",
        location="Tsetserleg",
        country="Mongolia",
        year=2022,
        description="",
        contributors=["PersonA", "PersonB"],
    )
    proj = Project(
        dir_path=Path("test_project"),
        begin_time=min(SITEA.begin_time, SITEB.begin_time),
        end_time=max(SITEA.end_time, SITEB.end_time),
        metadata=metadata,
        sites={"siteA": SITEA, "siteB": SITEB},
    )
    assert proj.n_sites == 2
    assert proj.fs() == [10, 20, 50]
    assert proj.begin_time == SITEA.begin_time
    assert proj.end_time == SITEB.end_time
    assert proj.get_site("siteA") == SITEA
    assert proj.get_sites() == {"siteA": SITEA, "siteB": SITEB}
    assert proj.get_sites(fs=20) == {"siteA": SITEA, "siteB": SITEB}
    assert proj.get_sites(fs=50) == {"siteB": SITEB}
    assert proj.get_concurrent("siteA") == [SITEB]
    data = [
        {
            "name": "meas1",
            "fs": MEAS1.metadata.fs,
            "first_time": MEAS1.metadata.first_time.isoformat(),
            "last_time": MEAS1.metadata.last_time.isoformat(),
            "site": "siteA",
        },
        {
            "name": "meas2",
            "fs": MEAS2.metadata.fs,
            "first_time": MEAS2.metadata.first_time.isoformat(),
            "last_time": MEAS2.metadata.last_time.isoformat(),
            "site": "siteA",
        },
        {
            "name": "run1",
            "fs": RUN1.metadata.fs,
            "first_time": RUN1.metadata.first_time.isoformat(),
            "last_time": RUN1.metadata.last_time.isoformat(),
            "site": "siteB",
        },
        {
            "name": "run2",
            "fs": RUN2.metadata.fs,
            "first_time": RUN2.metadata.first_time.isoformat(),
            "last_time": RUN2.metadata.last_time.isoformat(),
            "site": "siteB",
        },
        {
            "name": "run3",
            "fs": RUN3.metadata.fs,
            "first_time": RUN3.metadata.first_time.isoformat(),
            "last_time": RUN3.metadata.last_time.isoformat(),
            "site": "siteB",
        },
    ]
    df = pd.DataFrame.from_records(data=data).sort_values(["name", "site"])
    df["first_time"] = pd.to_datetime(df["first_time"])
    df["last_time"] = pd.to_datetime(df["last_time"])
    # sort and reset the project data frame in case the ordering was different
    proj_df = proj.to_dataframe().sort_values(["name", "site"])
    proj_df = proj_df.reset_index(drop=True)
    pd.testing.assert_frame_equal(proj_df, df)
