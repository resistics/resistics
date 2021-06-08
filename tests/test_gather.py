"""
Modules to test data gathering. This is quite complex testing as it requires
mock projects, sites and spectra metadata to appropriately test everything.
"""
from typing import Dict
from pathlib import Path
import pandas as pd
import pytest

from resistics.project import ProjectMetadata, Project
from resistics.project import get_meas_spectra_path
from resistics.spectra import SpectraDataReader, SpectraMetadata
from resistics.testing import spectra_metadata_multilevel
import resistics.gather as gather

TEST_PROJECT_PATH = Path(".")
TEST_CONFIG_NAME = "test"


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

    @staticmethod
    def get_measurements(fs: float):
        return ["meas1", "meas2", "meas3"]


def get_spectra_metadata(site_name: str, meas_name: str) -> SpectraMetadata:
    """Get example spectra metadata for testing"""
    if site_name == "site1" and meas_name == "meas1":
        return spectra_metadata_multilevel(
            n_levels=3, n_wins=[5, 4, 3], index_offset=[3, 2, 1]
        )
    if site_name == "site1" and meas_name == "meas2":
        return spectra_metadata_multilevel(
            n_levels=3, n_wins=[12, 9, 7], index_offset=[15, 10, 8]
        )
    if site_name == "site1" and meas_name == "meas3":
        return spectra_metadata_multilevel(
            n_levels=3, n_wins=[7, 4, 2], index_offset=[41, 38, 35]
        )
    raise ValueError(f"Site {site_name} and mesurement {meas_name} not known")


def get_spectra_metadata_by_path(spectra_path: Path):
    """Get example spectra metadata for testing"""
    # site 1
    for meas_name in ["meas1", "meas2", "meas3"]:
        if spectra_path == get_meas_spectra_path(
            TEST_PROJECT_PATH, "site1", meas_name, TEST_CONFIG_NAME
        ):
            return get_spectra_metadata("site1", meas_name)
    raise ValueError("Spectra path not as expected")


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
    index = [3, 4, 5, 6, 7, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 41, 42, 43, 44, 45, 46, 47]
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
    index = [3, 4, 5, 6, 7, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 41, 42, 43, 44, 45, 46, 47]
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
