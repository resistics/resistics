"""
Modules to test data gathering. This is quite complex testing as it requires
mock projects, sites, evals metadata and evals data to appropriately test
everything. Recall that evals data is simply spectra data reduced to the
evaluation frequencies. It is, however, separated into its own data folder to
avoid in potential future ambiguity in which data is being fetched.

The test works towards gathering data for three sites

- site1: measurements meas1, meas2, meas3
- site2: measurements run1, run2
- site3: measurements data1

The intention is to setup intersite processing with a remote reference. Each
site will have two channels, namely,

- site1: Ex, Ey (output site)
- site2: Hx, Hy (input site)
- site3: Hx, Hy (remote/cross site)

There are only two frequencies per decimation level, but multiple windows
"""
from typing import Dict
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from resistics.errors import ChannelNotFoundError
from resistics.project import ProjectMetadata, Project
from resistics.project import get_meas_evals_path
from resistics.decimate import DecimationSetup
from resistics.spectra import SpectraData, SpectraDataReader, SpectraMetadata
import resistics.gather as gather
from resistics.transfunc import TransferFunction, ImpedanceTensor
from testing_data_evals import get_evals_metadata_site1, get_evals_data_site1
from testing_data_evals import get_evals_metadata_site2, get_evals_data_site2
from testing_data_evals import get_evals_metadata_site3, get_evals_data_site3
from testing_data_evals import SITE1_COMBINED_DATA, SITE2_COMBINED_DATA
from testing_data_evals import SITE3_COMBINED_DATA
from testing_data_evals import SITE2_RUN2_QUICK_OUT, SITE2_RUN2_QUICK_IN
from testing_data_evals import SITE2_RUN2_QUICK_CROSS


TEST_PROJECT_PATH = Path(".")
TEST_CONFIG_NAME = "test"


def get_evals_metadata(site_name: str, meas_name: str) -> SpectraMetadata:
    """Get example evals metadata for testing"""
    if site_name == "site1":
        return get_evals_metadata_site1(meas_name)
    if site_name == "site2":
        return get_evals_metadata_site2(meas_name)
    if site_name == "site3":
        return get_evals_metadata_site3(meas_name)
    raise ValueError(f"Site {site_name} not known")


def get_evals_metadata_by_path(spectra_path: Path):
    """Get example evals metadata for testing"""
    # site 1
    for meas_name in ["meas1", "meas2", "meas3"]:
        if spectra_path == get_meas_evals_path(
            TEST_PROJECT_PATH, "site1", meas_name, TEST_CONFIG_NAME
        ):
            return get_evals_metadata("site1", meas_name)
    # site 2
    for meas_name in ["run1", "run2"]:
        if spectra_path == get_meas_evals_path(
            TEST_PROJECT_PATH, "site2", meas_name, TEST_CONFIG_NAME
        ):
            return get_evals_metadata("site2", meas_name)
    # site 3
    for meas_name in ["data1"]:
        if spectra_path == get_meas_evals_path(
            TEST_PROJECT_PATH, "site3", meas_name, TEST_CONFIG_NAME
        ):
            return get_evals_metadata("site3", meas_name)
    raise ValueError("Spectra path not as expected")


def get_evals_data(site_name: str, meas_name: str) -> SpectraData:
    """Get example evals data for testing"""
    if site_name == "site1":
        return get_evals_data_site1(meas_name)
    if site_name == "site2":
        return get_evals_data_site2(meas_name)
    if site_name == "site3":
        return get_evals_data_site3(meas_name)
    raise ValueError(f"Site {site_name} not known")


def get_evals_data_by_path(evals_path: Path):
    """Get example evals data for testing"""
    # site 1
    for meas_name in ["meas1", "meas2", "meas3"]:
        if evals_path == get_meas_evals_path(
            TEST_PROJECT_PATH, "site1", meas_name, TEST_CONFIG_NAME
        ):
            return get_evals_data("site1", meas_name)
    # site 2
    for meas_name in ["run1", "run2"]:
        if evals_path == get_meas_evals_path(
            TEST_PROJECT_PATH, "site2", meas_name, TEST_CONFIG_NAME
        ):
            return get_evals_data("site2", meas_name)
    # site 3
    for meas_name in ["data1"]:
        if evals_path == get_meas_evals_path(
            TEST_PROJECT_PATH, "site3", meas_name, TEST_CONFIG_NAME
        ):
            return get_evals_data("site3", meas_name)
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
        evals_path = args[1]
        if "metadata_only" in kwargs:
            return get_evals_metadata_by_path(evals_path)
        else:
            return get_evals_data_by_path(evals_path)

    monkeypatch.setattr(SpectraDataReader, "run", mock_spectra_data_reader_run)


def get_selection():
    """Get a selection as it is used in multiple places"""
    proj = get_test_project(TEST_PROJECT_PATH)
    site_names = ["site1", "site2", "site3"]
    dec_params = DecimationSetup(n_levels=4, per_level=2).run(128)
    selection = gather.Selector().run(TEST_CONFIG_NAME, proj, site_names, dec_params)
    return selection


def test_get_site_evals_metadata(mock_project_site, mock_spec_reader_metadata_only):
    """Test gathering of spectra metadata"""
    proj = get_test_project(TEST_PROJECT_PATH)
    meas_metadata = gather.get_site_evals_metadata(TEST_CONFIG_NAME, proj, "site1", 128)

    assert len(meas_metadata) == 3
    for meas_name, metadata in meas_metadata.items():
        assert get_evals_metadata("site1", meas_name) == metadata


def test_get_site_level_wins():
    """Test getting site level windows for decimation level 0"""
    meas_metadata = {}
    for meas_name in ["meas1", "meas2", "meas3"]:
        meas_metadata[meas_name] = get_evals_metadata("site1", meas_name)

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
    selection = get_selection()
    selector_site_names = [x.name for x in selection.sites]
    expected_site_names = ["site1", "site2", "site3"]

    assert set(expected_site_names) == set(selector_site_names)
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


def test_projectgather_get_empty_data(
    mock_project_site, mock_spec_reader_metadata_only
):
    """Test creating empty data"""
    selection = get_selection()
    chans = ["Hx", "Hy"]
    n_chans = len(chans)
    # now test gatherer._get_empty_data
    gatherer = gather.ProjectGather()
    empty_data = gatherer._get_empty_data(selection, chans)
    assert len(empty_data) == 4
    assert sorted(list(empty_data.keys())) == [0, 1, 2, 3]
    # check size of arrays
    for eval_idx, data in empty_data.items():
        # two decimation levels
        level = eval_idx // 2
        eval_level_idx = eval_idx - level * 2
        n_wins = selection.get_n_wins(level, eval_level_idx)
        assert data.shape == (n_wins, n_chans)
        assert data.dtype == np.complex128


def test_projectgather_get_indices_site1_meas1(
    mock_project_site, mock_spec_reader_metadata_only
):
    """Test project gathering of data"""
    # get required data
    site = MockSite("site1")
    meas_name = "meas1"
    metadata = get_evals_metadata(site.name, meas_name)
    selection = get_selection()
    # now test gatherer._get_indices
    gatherer = gather.ProjectGather()
    # site 1, meas1, level 0
    eval_wins = selection.get_eval_wins(0, 0)
    level_metadata = metadata.levels_metadata[0]
    spectra_indices, combined_indices = gatherer._get_indices(
        eval_wins, site, meas_name, level_metadata
    )
    np.testing.assert_array_equal(spectra_indices, np.array([0, 1, 2, 3, 4]))
    np.testing.assert_array_equal(combined_indices, np.array([0, 1, 2, 3, 4]))
    # site 1, meas1, level 1
    eval_wins = selection.get_eval_wins(1, 0)
    level_metadata = metadata.levels_metadata[1]
    spectra_indices, combined_indices = gatherer._get_indices(
        eval_wins, site, meas_name, level_metadata
    )
    np.testing.assert_array_equal(spectra_indices, np.array([1]))
    np.testing.assert_array_equal(combined_indices, np.array([0]))


def test_projectgather_get_indices_site2_run2(
    mock_project_site, mock_spec_reader_metadata_only
):
    """Test project gathering of data"""
    # get required data
    site = MockSite("site2")
    meas_name = "run2"
    metadata = get_evals_metadata(site.name, meas_name)
    selection = get_selection()
    # now test gatherer._get_indices
    gatherer = gather.ProjectGather()
    # site 2, run1, level 0
    eval_wins = selection.get_eval_wins(0, 0)
    level_metadata = metadata.levels_metadata[0]
    spectra_indices, combined_indices = gatherer._get_indices(
        eval_wins, site, meas_name, level_metadata
    )
    expected_spectra = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    expeceted_combined = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    np.testing.assert_array_equal(spectra_indices, np.array(expected_spectra))
    np.testing.assert_array_equal(combined_indices, np.array(expeceted_combined))
    # site 2, run1, level 1
    eval_wins = selection.get_eval_wins(1, 0)
    level_metadata = metadata.levels_metadata[1]
    spectra_indices, combined_indices = gatherer._get_indices(
        eval_wins, site, meas_name, level_metadata
    )
    np.testing.assert_array_equal(spectra_indices, np.array([1, 2, 3, 4]))
    np.testing.assert_array_equal(combined_indices, np.array([1, 2, 3, 4]))


def test_projectgather_get_indices_site3_data1(
    mock_project_site, mock_spec_reader_metadata_only
):
    """Test project gathering of data"""
    # get required data
    site = MockSite("site3")
    meas_name = "data1"
    metadata = get_evals_metadata(site.name, meas_name)
    selection = get_selection()
    # now test gatherer._get_indices
    gatherer = gather.ProjectGather()
    # site 3, data1, level 0
    eval_wins = selection.get_eval_wins(0, 0)
    level_metadata = metadata.levels_metadata[0]
    spectra_indices, combined_indices = gatherer._get_indices(
        eval_wins, site, meas_name, level_metadata
    )
    expected_spectra = [0, 1, 2, 3, 4, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
    expected_combined = np.arange(15)
    np.testing.assert_array_equal(spectra_indices, np.array(expected_spectra))
    np.testing.assert_array_equal(combined_indices, expected_combined)
    # site 3, data1, level 1
    eval_wins = selection.get_eval_wins(1, 0)
    level_metadata = metadata.levels_metadata[1]
    spectra_indices, combined_indices = gatherer._get_indices(
        eval_wins, site, meas_name, level_metadata
    )
    np.testing.assert_array_equal(spectra_indices, np.array([0, 7, 8, 9, 10]))
    np.testing.assert_array_equal(combined_indices, np.array([0, 1, 2, 3, 4]))


def test_projectgather_get_site_data_site1(
    mock_project_site, mock_spec_reader_metadata_only
):
    """Test combining data for site1"""
    # get required data
    proj = get_test_project(TEST_PROJECT_PATH)
    site_name = "site1"
    selection = get_selection()
    # now test gatherer._get_indices
    gatherer = gather.ProjectGather()
    combined_data = gatherer._get_site_data(
        TEST_CONFIG_NAME, proj, selection, site_name, ["Ex", "Ey"]
    )
    assert len(combined_data.data) == 4
    assert combined_data.metadata.chans == ["Ex", "Ey"]
    assert combined_data.metadata.n_evals == 4
    assert combined_data.metadata.measurements == ["meas1", "meas2"]
    np.testing.assert_equal(combined_data.data[0], SITE1_COMBINED_DATA[0])
    np.testing.assert_equal(combined_data.data[1], SITE1_COMBINED_DATA[1])
    np.testing.assert_equal(combined_data.data[2], SITE1_COMBINED_DATA[2])
    np.testing.assert_equal(combined_data.data[3], SITE1_COMBINED_DATA[3])


def test_projectgather_get_site_data_site2(
    mock_project_site, mock_spec_reader_metadata_only
):
    """Test combining data for site2"""
    # get required data
    proj = get_test_project(TEST_PROJECT_PATH)
    site_name = "site2"
    selection = get_selection()
    # now test gatherer._get_indices
    gatherer = gather.ProjectGather()
    combined_data = gatherer._get_site_data(
        TEST_CONFIG_NAME, proj, selection, site_name, ["Hx", "Hy"]
    )
    assert len(combined_data.data) == 4
    assert combined_data.metadata.chans == ["Hx", "Hy"]
    assert combined_data.metadata.n_evals == 4
    assert combined_data.metadata.measurements == ["run1", "run2"]
    np.testing.assert_equal(combined_data.data[0], SITE2_COMBINED_DATA[0])
    np.testing.assert_equal(combined_data.data[1], SITE2_COMBINED_DATA[1])
    np.testing.assert_equal(combined_data.data[2], SITE2_COMBINED_DATA[2])
    np.testing.assert_equal(combined_data.data[3], SITE2_COMBINED_DATA[3])


def test_projectgather_get_site_data_site3(
    mock_project_site, mock_spec_reader_metadata_only
):
    """Test combining data for site1"""
    # get required data
    proj = get_test_project(TEST_PROJECT_PATH)
    site_name = "site3"
    selection = get_selection()
    # now test gatherer._get_indices
    gatherer = gather.ProjectGather()
    combined_data = gatherer._get_site_data(
        TEST_CONFIG_NAME, proj, selection, site_name, ["Hx", "Hy"]
    )
    assert len(combined_data.data) == 4
    assert combined_data.metadata.chans == ["Hx", "Hy"]
    assert combined_data.metadata.n_evals == 4
    assert combined_data.metadata.measurements == ["data1"]
    np.testing.assert_equal(combined_data.data[0], SITE3_COMBINED_DATA[0])
    np.testing.assert_equal(combined_data.data[1], SITE3_COMBINED_DATA[1])
    np.testing.assert_equal(combined_data.data[2], SITE3_COMBINED_DATA[2])
    np.testing.assert_equal(combined_data.data[3], SITE3_COMBINED_DATA[3])


def test_projectgather_run(mock_project_site, mock_spec_reader_metadata_only):
    """Test gathering data for all sites"""
    # get required data
    proj = get_test_project(TEST_PROJECT_PATH)
    tf = ImpedanceTensor()
    selection = get_selection()
    gathered_data = gather.ProjectGather().run(
        TEST_CONFIG_NAME,
        proj,
        selection,
        tf,
        out_name="site1",
        in_name="site2",
        cross_name="site3",
    )
    # output data
    assert gathered_data.out_data.metadata.name == "site1"
    assert gathered_data.out_data.metadata.chans == ["Ex", "Ey"]
    assert gathered_data.out_data.metadata.fs == 128
    assert gathered_data.out_data.metadata.n_evals == 4
    np.testing.assert_equal(gathered_data.out_data.data[0], SITE1_COMBINED_DATA[0])
    np.testing.assert_equal(gathered_data.out_data.data[1], SITE1_COMBINED_DATA[1])
    np.testing.assert_equal(gathered_data.out_data.data[2], SITE1_COMBINED_DATA[2])
    np.testing.assert_equal(gathered_data.out_data.data[3], SITE1_COMBINED_DATA[3])
    # input data
    assert gathered_data.in_data.metadata.name == "site2"
    assert gathered_data.in_data.metadata.chans == ["Hx", "Hy"]
    assert gathered_data.in_data.metadata.fs == 128
    assert gathered_data.in_data.metadata.n_evals == 4
    np.testing.assert_equal(gathered_data.in_data.data[0], SITE2_COMBINED_DATA[0])
    np.testing.assert_equal(gathered_data.in_data.data[1], SITE2_COMBINED_DATA[1])
    np.testing.assert_equal(gathered_data.in_data.data[2], SITE2_COMBINED_DATA[2])
    np.testing.assert_equal(gathered_data.in_data.data[3], SITE2_COMBINED_DATA[3])
    # cross data
    assert gathered_data.cross_data.metadata.name == "site3"
    assert gathered_data.cross_data.metadata.chans == ["Hx", "Hy"]
    assert gathered_data.cross_data.metadata.fs == 128
    assert gathered_data.cross_data.metadata.n_evals == 4
    np.testing.assert_equal(gathered_data.cross_data.data[0], SITE3_COMBINED_DATA[0])
    np.testing.assert_equal(gathered_data.cross_data.data[1], SITE3_COMBINED_DATA[1])
    np.testing.assert_equal(gathered_data.cross_data.data[2], SITE3_COMBINED_DATA[2])
    np.testing.assert_equal(gathered_data.cross_data.data[3], SITE3_COMBINED_DATA[3])


def test_quickgather_run():
    """Test quick gathering with some spectra data"""
    dir_path = Path("test")
    dec_params = DecimationSetup(n_levels=4, per_level=2).run(128)
    tf = TransferFunction(
        name="quick", out_chans=["Hy"], in_chans=["Hx"], cross_chans=["Ex", "Ey"]
    )
    eval_data = get_evals_data("site2", "run2")
    with pytest.raises(ChannelNotFoundError):
        # there are no electronic channels in the data
        gathered_data = gather.QuickGather().run(dir_path, dec_params, tf, eval_data)

    tf = TransferFunction(
        name="quick2", in_chans=["Hx"], out_chans=["Hy"], cross_chans=["Hx", "Hy"]
    )
    gathered_data = gather.QuickGather().run(dir_path, dec_params, tf, eval_data)
    # output data
    assert gathered_data.out_data.metadata.name == "test"
    assert gathered_data.out_data.metadata.chans == ["Hy"]
    assert gathered_data.out_data.metadata.fs == 128
    assert gathered_data.out_data.metadata.n_evals == 4
    np.testing.assert_equal(gathered_data.out_data.data[0], SITE2_RUN2_QUICK_OUT[0])
    np.testing.assert_equal(gathered_data.out_data.data[1], SITE2_RUN2_QUICK_OUT[1])
    np.testing.assert_equal(gathered_data.out_data.data[2], SITE2_RUN2_QUICK_OUT[2])
    np.testing.assert_equal(gathered_data.out_data.data[3], SITE2_RUN2_QUICK_OUT[3])
    # input data
    assert gathered_data.in_data.metadata.name == "test"
    assert gathered_data.in_data.metadata.chans == ["Hx"]
    assert gathered_data.in_data.metadata.fs == 128
    assert gathered_data.in_data.metadata.n_evals == 4
    np.testing.assert_equal(gathered_data.in_data.data[0], SITE2_RUN2_QUICK_IN[0])
    np.testing.assert_equal(gathered_data.in_data.data[1], SITE2_RUN2_QUICK_IN[1])
    np.testing.assert_equal(gathered_data.in_data.data[2], SITE2_RUN2_QUICK_IN[2])
    np.testing.assert_equal(gathered_data.in_data.data[3], SITE2_RUN2_QUICK_IN[3])
    # cross data
    assert gathered_data.cross_data.metadata.name == "test"
    assert gathered_data.cross_data.metadata.chans == ["Hx", "Hy"]
    assert gathered_data.cross_data.metadata.fs == 128
    assert gathered_data.cross_data.metadata.n_evals == 4
    np.testing.assert_equal(gathered_data.cross_data.data[0], SITE2_RUN2_QUICK_CROSS[0])
    np.testing.assert_equal(gathered_data.cross_data.data[1], SITE2_RUN2_QUICK_CROSS[1])
    np.testing.assert_equal(gathered_data.cross_data.data[2], SITE2_RUN2_QUICK_CROSS[2])
    np.testing.assert_equal(gathered_data.cross_data.data[3], SITE2_RUN2_QUICK_CROSS[3])
