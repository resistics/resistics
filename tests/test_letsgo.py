from pathlib import Path
import pytest

from resistics.time import Add, Multiply
from resistics.decimate import DecimationSetup
from resistics.transfunc import TransferFunction, ImpedanceTensor
from resistics.regression import Solution, Solver, SolverOLS
from resistics.testing import time_data_ones, solution_mt, solution_random_float
from resistics.testing import transfer_function_random


def test_new_mth5_project(tmp_path):
    """Test creating an MTH5-backed project through letsgo."""
    from resistics.letsgo import new

    mth5_path = tmp_path / "data.h5"
    mth5_path.write_text("")
    project_path = tmp_path / "project"

    assert new(project_path, mth5_path, "2021-01-01 00:00:00")
    assert (project_path / "resistics.json").exists()


def test_new_mth5_project_already_exists(tmp_path):
    """Test creating an existing MTH5-backed project fails."""
    from resistics.letsgo import new

    mth5_path = tmp_path / "data.h5"
    mth5_path.write_text("")
    project_path = tmp_path / "project"
    assert new(project_path, mth5_path, "2021-01-01 00:00:00")

    with pytest.raises(ValueError, match="Project already exists"):
        new(project_path, mth5_path, "2021-01-01 00:00:00")


def test_quick_read_requires_mth5_selection(tmp_path):
    """Test legacy path-only quick_read calls fail at the Python signature."""
    from resistics.letsgo import quick_read

    with pytest.raises(TypeError):
        quick_read(tmp_path)


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


RANDOM_TF1 = transfer_function_random(3, 11)
RANDOM_TF2 = transfer_function_random(3, 7, n_cross=5)


@pytest.mark.parametrize(
    "fs, tf, expected_soln, solver, n_levels, n_wins",
    [
        (
            256,
            ImpedanceTensor(),
            solution_mt(),
            SolverOLS(),
            1,
            50,
        ),
        (
            512,
            RANDOM_TF1,
            solution_random_float(512, RANDOM_TF1, 25),
            SolverOLS(),
            5,
            1000,
        ),
        (
            512,
            RANDOM_TF2,
            solution_random_float(512, RANDOM_TF2, 25),
            SolverOLS(),
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
    eval_data = evaluation_data(dec_params, n_wins, expected_soln)

    # solve
    config = get_default_configuration()
    config.solver = solver
    config.tf = tf
    gathered_data = QuickGather().run(Path(), dec_params, config.tf, eval_data)
    reg_data = run_regression_preparer(config, gathered_data)
    soln = run_solver(config, reg_data)
    assert_soln_equal(soln, expected_soln)
