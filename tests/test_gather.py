"""
Modules to test data gathering. This is quite complex testing as it requires
mock projects, sites, spectra metadata and spectra data to appropriately test
everything.
"""
from typing import Dict
from pathlib import Path
import pandas as pd
import pytest

from resistics.project import ProjectMetadata, Project
from resistics.project import get_meas_spectra_path
from resistics.decimate import DecimationSetup
from resistics.spectra import SpectraDataReader, SpectraMetadata
from resistics.testing import spectra_metadata_multilevel
import resistics.gather as gather

TEST_PROJECT_PATH = Path(".")
TEST_CONFIG_NAME = "test"


def get_spectra_metadata_site1(meas_name: str):
    """Get spectra metadata for site1"""
    if meas_name == "meas1":
        # level 0 windows: 4, 5, 6, 7, 8
        # level 1 windows: 2, 3, 4, 5
        # level 3 windows: 1, 2, 3
        return spectra_metadata_multilevel(
            n_levels=3, n_wins=[5, 4, 3], index_offset=[4, 2, 1]
        )
    if meas_name == "meas2":
        # level 0 windows: 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26
        # level 1 windows: 10, 11, 12, 13, 14, 15, 16, 17, 18
        # level 3 windows: 8, 9, 10, 11, 12, 13, 14
        return spectra_metadata_multilevel(
            n_levels=3, n_wins=[12, 9, 7], index_offset=[15, 10, 8]
        )
    if meas_name == "meas3":
        # level 0 windows: 41, 42, 43, 44, 45, 46, 47
        # level 1 windows: 38, 39, 40, 41
        # level 3 windows: 35, 36
        return spectra_metadata_multilevel(
            n_levels=3, n_wins=[7, 4, 2], index_offset=[41, 38, 35]
        )
    raise ValueError("Unknown measurement for site1")


def get_spectra_metadata_site2(meas_name: str):
    """Get spectra metadata for site2"""
    if meas_name == "run1":
        # level 0 windows: 3, 4, 5, 6, 7, 8
        # level 1 windows: 1, 2, 3
        return spectra_metadata_multilevel(
            n_levels=3, n_wins=[6, 3], index_offset=[3, 1]
        )
    if meas_name == "run2":
        # level 0 windows: 16, 17, 18, 19, 20, 21, 22, 23, 24, 25
        # level 1 windows: 9, 10, 11, 12, 13
        return spectra_metadata_multilevel(
            n_levels=2, n_wins=[10, 5], index_offset=[16, 9]
        )
    raise ValueError("Unknown measurement for site2")


def get_spectra_metadata_site3(meas_name: str):
    """Get spectra metadata for site3"""
    if meas_name == "data1":
        # level 0 windows: 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28
        # level 1 windows: 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19
        # level 3 windows: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12
        return spectra_metadata_multilevel(
            n_levels=2, n_wins=[25, 17, 12], index_offset=[4, 3, 1]
        )
    raise ValueError("Unknown measurement for site3")


def get_spectra_metadata(site_name: str, meas_name: str) -> SpectraMetadata:
    """Get example spectra metadata for testing"""
    if site_name == "site1":
        return get_spectra_metadata_site1(meas_name)
    if site_name == "site2":
        return get_spectra_metadata_site2(meas_name)
    if site_name == "site3":
        return get_spectra_metadata_site3(meas_name)
    raise ValueError(f"Site {site_name} not known")


def get_spectra_metadata_by_path(spectra_path: Path):
    """Get example spectra metadata for testing"""
    # site 1
    for meas_name in ["meas1", "meas2", "meas3"]:
        if spectra_path == get_meas_spectra_path(
            TEST_PROJECT_PATH, "site1", meas_name, TEST_CONFIG_NAME
        ):
            return get_spectra_metadata("site1", meas_name)
    # site 2
    for meas_name in ["run1", "run2"]:
        if spectra_path == get_meas_spectra_path(
            TEST_PROJECT_PATH, "site2", meas_name, TEST_CONFIG_NAME
        ):
            return get_spectra_metadata("site2", meas_name)
    # site 3
    for meas_name in ["data1"]:
        if spectra_path == get_meas_spectra_path(
            TEST_PROJECT_PATH, "site3", meas_name, TEST_CONFIG_NAME
        ):
            return get_spectra_metadata("site3", meas_name)
    raise ValueError("Spectra path not as expected")


def get_test_project(project_path) -> Project:
    """Get a testing project"""
    metadata = ProjectMetadata(ref_time="2021-01-01 00:00:00")
    dir_path = project_path
    begin_time = "2021-01-01 01:00:00"
    end_time = "2021-01-01 05:00:00"
    proj = Project(
        dir_path=dir_path, begin_time=begin_time, end_time=end_time, metadata=metadata
    )
    return proj


class MockMeas:
    def __init__(self, name: str):
        """Initialise with the measurement name"""
        self.name = name


class MockSite:
    def __init__(self, name: str):
        """Initialise"""
        self.name = name

    def __getitem__(self, meas_name: str) -> Dict[str, str]:
        """Get a mock measurement"""
        return MockMeas(meas_name)

    def get_measurements(self, fs: float):
        """Get the measurements for the site"""
        if self.name == "site1":
            return ["meas1", "meas2", "meas3"]
        if self.name == "site2":
            return ["run1", "run2"]
        if self.name == "site3":
            return ["data1"]


@pytest.fixture
def mock_project_site(monkeypatch):
    """Mock getting of site from project"""

    def mock_project_get_site(*args, **kwargs):
        """Mock for getting a site"""
        site_name = args[1]
        return MockSite(site_name)

    monkeypatch.setattr(Project, "get_site", mock_project_get_site)


@pytest.fixture
def mock_spec_reader_metadata_only(monkeypatch):
    """Mock fixture for reading spectra metadata"""

    def mock_spectra_data_reader_run(*args, **kwargs):
        """Mock for reading spectra metadata"""
        spectra_path = args[1]
        return get_spectra_metadata_by_path(spectra_path)

    monkeypatch.setattr(SpectraDataReader, "run", mock_spectra_data_reader_run)


def test_get_site_spectra_metadata(mock_project_site, mock_spec_reader_metadata_only):
    """Test gathering of spectra metadata"""
    proj = get_test_project(TEST_PROJECT_PATH)
    meas_metadata = gather.get_site_spectra_metadata(
        TEST_CONFIG_NAME, proj, "site1", 128
    )

    assert len(meas_metadata) == 3
    for meas_name, metadata in meas_metadata.items():
        assert get_spectra_metadata("site1", meas_name) == metadata


def test_get_site_level_wins():
    """Test getting site level windows for decimation level 0"""
    meas_metadata = {}
    for meas_name in ["meas1", "meas2", "meas3"]:
        meas_metadata[meas_name] = get_spectra_metadata("site1", meas_name)

    table = gather.get_site_level_wins(meas_metadata, 0)
    # fmt:off
    index = [4, 5, 6, 7, 8, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 41, 42, 43, 44, 45, 46, 47]
    data = ["meas1", "meas1", "meas1", "meas1", "meas1"]
    data += ["meas2", "meas2", "meas2", "meas2", "meas2", "meas2", "meas2", "meas2", "meas2", "meas2", "meas2", "meas2"]
    data += ["meas3", "meas3", "meas3", "meas3", "meas3", "meas3", "meas3"]
    # fmt:on
    pd.testing.assert_series_equal(table, pd.Series(data=data, index=index))


def test_get_site_wins(mock_project_site, mock_spec_reader_metadata_only):
    """Test getting site windows for all decimation levels"""
    proj = get_test_project(TEST_PROJECT_PATH)
    tables = gather.get_site_wins(TEST_CONFIG_NAME, proj, "site1", 128)
    # level 0
    # fmt:off
    index = [4, 5, 6, 7, 8, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 41, 42, 43, 44, 45, 46, 47]
    data = ["meas1", "meas1", "meas1", "meas1", "meas1"]
    data += ["meas2", "meas2", "meas2", "meas2", "meas2", "meas2", "meas2", "meas2", "meas2", "meas2", "meas2", "meas2"]
    data += ["meas3", "meas3", "meas3", "meas3", "meas3", "meas3", "meas3"]
    # fmt:on
    pd.testing.assert_series_equal(tables[0], pd.Series(data=data, index=index))
    # level 1
    # fmt:off
    index = [2, 3, 4, 5, 10, 11, 12, 13, 14, 15, 16, 17, 18, 38, 39, 40, 41]
    data = ["meas1", "meas1", "meas1", "meas1"]
    data += ["meas2", "meas2", "meas2", "meas2", "meas2", "meas2", "meas2", "meas2", "meas2"]
    data += ["meas3", "meas3", "meas3", "meas3"]
    # fmt:on
    pd.testing.assert_series_equal(tables[1], pd.Series(data=data, index=index))
    # level 2
    # fmt:off
    index = [1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 35, 36]
    data = ["meas1", "meas1", "meas1"]
    data += ["meas2", "meas2", "meas2", "meas2", "meas2", "meas2", "meas2"]
    data += ["meas3", "meas3"]
    # fmt:on
    pd.testing.assert_series_equal(tables[2], pd.Series(data=data, index=index))


def test_selector(mock_project_site, mock_spec_reader_metadata_only):
    """Test the Selector"""
    proj = get_test_project(TEST_PROJECT_PATH)
    site_names = ["site1", "site2", "site3"]
    dec_params = DecimationSetup(n_levels=4, per_level=2).run(128)
    selection = gather.Selector().run(TEST_CONFIG_NAME, proj, site_names, dec_params)
    selector_site_names = set([x.name for x in selection.sites])

    assert set(site_names) == selector_site_names
    assert selection.n_levels == 2
    assert selection.get_n_evals() == 4
    assert selection.get_n_wins(0, 0) == 15
    assert selection.get_n_wins(0, 1) == 15
    assert selection.get_n_wins(1, 0) == 5
    assert selection.get_n_wins(1, 1) == 5
    assert selection.get_measurements(MockSite("site1")) == ["meas1", "meas2"]
    assert selection.get_measurements(MockSite("site2")) == ["run1", "run2"]
    assert selection.get_measurements(MockSite("site3")) == ["data1"]

    # level 0
    index = [4, 5, 6, 7, 8, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
    data = {}
    # fmt:off
    data["site1"] = ["meas1", "meas1", "meas1", "meas1", "meas1", "meas2", "meas2", "meas2", "meas2", "meas2", "meas2", "meas2", "meas2", "meas2", "meas2"]
    data["site2"] = ["run1", "run1", "run1", "run1", "run1", "run2", "run2", "run2", "run2", "run2", "run2", "run2", "run2", "run2", "run2"]
    data["site3"] = ["data1", "data1", "data1", "data1", "data1", "data1", "data1", "data1", "data1", "data1", "data1", "data1", "data1", "data1", "data1"]
    data[0] = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
    data[1] = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
    # fmt:on
    pd.testing.assert_frame_equal(
        pd.DataFrame(data=data, index=index), selection.tables[0]
    )
    eval_wins = pd.DataFrame(data=data, index=index).drop([0, 1], axis=1)
    pd.testing.assert_frame_equal(eval_wins, selection.get_eval_wins(0, 0))
    pd.testing.assert_frame_equal(eval_wins, selection.get_eval_wins(0, 1))

    # level 1
    index = [3, 10, 11, 12, 13]
    data = {}
    data["site1"] = ["meas1", "meas2", "meas2", "meas2", "meas2"]
    data["site2"] = ["run1", "run2", "run2", "run2", "run2"]
    data["site3"] = ["data1", "data1", "data1", "data1", "data1"]
    data[0] = [True, True, True, True, True]
    data[1] = [True, True, True, True, True]
    pd.testing.assert_frame_equal(
        pd.DataFrame(data=data, index=index), selection.tables[1]
    )
    eval_wins = pd.DataFrame(data=data, index=index).drop([0, 1], axis=1)
    pd.testing.assert_frame_equal(eval_wins, selection.get_eval_wins(1, 0))
    pd.testing.assert_frame_equal(eval_wins, selection.get_eval_wins(1, 1))


def test_projectgather():
    """Test project gathering of data"""
    pass
