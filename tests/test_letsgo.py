from pathlib import Path
import os
from datetime import datetime
import pandas as pd
import pytest
from pydantic.error_wrappers import ValidationError

from resistics.errors import ProjectCreateError
from resistics.common import WriteableMetadata
from resistics.time import Add, Multiply
from resistics.decimate import DecimationSetup
from resistics.transfunc import TransferFunction, ImpedanceTensor
from resistics.regression import Solution, Solver, SolverScikitOLS
from resistics.testing import time_data_ones, solution_mt, solution_random_float
from resistics.testing import transfer_function_random


def mock_mkdir(*args, **kwargs):
    """Mock the Path mkdir method"""
    return True


def mock_write(*args):
    """Mock write from WriteableMetadata"""
    return


@pytest.mark.parametrize(
    "dir_path, proj_info, dir_exists, raises",
    [
        (
            Path("test", "project"),
            {},
            False,
            ValidationError,
        ),
        (
            Path("test", "project"),
            {"ref_time": "2021-01-01 00:00:00"},
            False,
            None,
        ),
        (
            Path("test", "project"),
            {"ref_time": "2021-01-01 00:00:00"},
            True,
            None,
        ),
        (
            os.path.join("test", "project"),
            {"ref_time": "2021-01-01 00:00:00"},
            False,
            None,
        ),
        (
            os.path.join("test", "project"),
            {"ref_time": pd.Timestamp("2021-01-01 00:00:00")},
            False,
            None,
        ),
        (
            os.path.join("test", "project"),
            {"ref_time": datetime(2021, 1, 1, 0, 0, 0)},
            False,
            None,
        ),
    ],
)
def test_new_project_validation(monkeypatch, dir_path, proj_info, dir_exists, raises):
    """Test creating a new project"""
    from resistics.letsgo import new

    def mock_exists(*args, **kwargs):
        """Mock the Path exists function to return True"""
        return dir_exists

    def mock_assert_dir(*args):
        """Accept that the directory is real"""
        return

    monkeypatch.setattr(Path, "exists", mock_exists)
    monkeypatch.setattr("resistics.common.assert_dir", mock_assert_dir)
    monkeypatch.setattr(Path, "mkdir", mock_mkdir)
    monkeypatch.setattr(WriteableMetadata, "write", mock_write)

    if raises is not None:
        with pytest.raises(raises):
            assert new(dir_path, proj_info)
        return
    assert new(dir_path, proj_info)


def test_new_project_already_exists(monkeypatch):
    """Test creating a new project when the project already exists"""
    from resistics.letsgo import new

    def mock_exists(*args, **kwargs):
        """Mock the Path exists function to return True"""
        return True

    def mock_is_dir(*args):
        """Get is directory to return True"""
        return True

    monkeypatch.setattr(Path, "exists", mock_exists)
    monkeypatch.setattr("resistics.common.is_dir", mock_is_dir)

    dir_path = Path("test", "project")
    proj_info = {"ref_time": "2021-01-01 00:00:00"}

    with pytest.raises(ProjectCreateError):
        new(dir_path, proj_info)


# def test_load_project():
#     """Test loading of a project"""
#     assert True


@pytest.mark.parametrize(
    "time_data, time_processors",
    [
        (time_data_ones(), [Add(add=4)]),
        (time_data_ones(), [Add(add=5), Multiply(multiplier=7)]),
    ],
)
def test_run_time_processors(time_data, time_processors):
    """Test running of time processors"""
    from resistics.config import get_default_configuration
    from resistics.letsgo import run_time_processors
    from resistics.testing import assert_time_data_equal

    config = get_default_configuration()
    config.time_processors = time_processors
    time_data_new = run_time_processors(config, time_data)
    # expected
    for processor in time_processors:
        time_data = processor.run(time_data)
    # compare
    assert_time_data_equal(time_data_new, time_data, history_times=False)


RANDOM_TF_EXAMPLE = transfer_function_random(3, 11)


@pytest.mark.parametrize(
    "fs, tf, expected_soln, solver, n_levels, n_wins",
    [
        (
            256,
            ImpedanceTensor(),
            solution_mt(),
            SolverScikitOLS(),
            1,
            50,
        ),
        (
            512,
            RANDOM_TF_EXAMPLE,
            solution_random_float(512, RANDOM_TF_EXAMPLE, 25),
            SolverScikitOLS(),
            5,
            1000,
        ),
    ],
)
def test_run_preparer_solver(
    fs: float,
    tf: TransferFunction,
    expected_soln: Solution,
    solver: Solver,
    n_levels: int,
    n_wins: int,
):
    """Test regression using synthetic evaluation frequency data"""
    from pathlib import Path
    from resistics.config import get_default_configuration
    from resistics.gather import QuickGather
    from resistics.letsgo import run_regression_preparer, run_solver
    from resistics.testing import evaluation_data, assert_soln_equal

    n_evals = len(expected_soln.freqs)
    if n_evals % n_levels != 0:
        raise ValueError(f"{n_evals=} not divisible by {n_levels=}")
    per_level = n_evals // n_levels
    dec_setup = DecimationSetup(
        n_levels=n_levels, per_level=per_level, eval_freqs=expected_soln.freqs
    )
    dec_params = dec_setup.run(fs)
    eval_data = evaluation_data(fs, dec_params, n_wins, expected_soln)

    # solve
    config = get_default_configuration()
    config.solver = solver
    config.tf = tf
    gathered_data = QuickGather().run(Path(), dec_params, config.tf, eval_data)
    reg_data = run_regression_preparer(config, gathered_data)
    soln = run_solver(config, reg_data)
    assert_soln_equal(soln, expected_soln)
