"""
Testing of regression fuctions and processors

GatheredData is the input into the regression. This is made up of three
SiteCombinedData instances, one for the output channels, one for the input
channels and one for the cross channels.

SiteCombinedData has a data dictionary which has a single entry for each
evaluation frequency. A single entry in the SiteCombinedData data attribute has
shape:

- n_wins x n_chans

The GatheredData is used to create the RegressionInputData. This has two key
attributes, the obs and the preds.

The obs has an array for each output channel with size:

n_wins x n_cross_chans x 2

And the corresponding predictor array has shape

(n_wins x n_cross_chans x 2) x (n_in_chans x 2)
"""
from typing import List
import numpy as np
import pytest

from resistics.common import History
from resistics.gather import SiteCombinedData, SiteCombinedMetadata, GatheredData
from resistics.decimate import DecimationSetup
from resistics.transfunc import TransferFunction, ImpedanceTensor
from resistics.regression import RegressionPreparerGathered, Solution
from resistics.regression import Solver, SolverOLS
from resistics.testing import solution_mt, solution_random_float, solution_random_int
from resistics.testing import transfer_function_random


# this first example has 2 windows, 1chan
# recall that the cross data will be conjugated, 5+3j will become 5-3j
TEST1_OUT_DATA = {0: np.array([[3 - 1j], [1 + 2j]])}
TEST1_IN_DATA = {0: np.array([[-1 - 1j], [-2 - 3j]])}
TEST1_CROSS_DATA = {0: np.array([[5 + 3j], [0 - 2j]])}
# expected output
TEST1_FREQS = np.array([10])
TEST1_OBS = {"Ex": np.array([12 - 14j, -4 + 2j])}
TEST1_PREDS = {0: np.array([[-8 - 2j], [6 - 4j]])}


# testing data with 2 windows, 2 chans
# recall that the cross data will be conjugated, 1-1j will become 1+1j
TEST2_OUT_DATA = {0: np.array([[3 - 1j, 4 + 3j], [1 + 2j, 2 + 1j]])}
TEST2_IN_DATA = {0: np.array([[-1 - 1j, 0 + 3j], [-2 - 3j, 4 - 1j]])}
TEST2_CROSS_DATA = {0: np.array([[5 + 3j, 2 + 0j], [0 - 2j, 1 - 1j]])}
# expected output
TEST2_FREQS = np.array([10])
TEST2_OBS = {
    "Ex": np.array([12 - 14j, 6 - 2j, -4 + 2j, -1 + 3j]),
    "Ey": np.array([29 + 3j, 8 + 6j, -2 + 4j, 1 + 3j]),
}
TEST2_PREDS = {
    0: np.array(
        [[-8 - 2j, 9 + 15j], [-2 - 2j, 0 + 6j], [6 - 4j, 2 + 8j], [1 - 5j, 5 + 3j]]
    )
}


def get_combined_metadata(
    site_name: str, measurements: List[str], chans: List[str]
) -> SiteCombinedMetadata:
    """Get metadata for SiteCombinedData"""
    histories = {x: History() for x in measurements}
    return SiteCombinedMetadata(
        site_name=site_name,
        fs=128,
        measurements=measurements,
        chans=chans,
        n_evals=1,
        eval_freqs=[10],
        histories=histories,
    )


def test_regression_preparer_1chan():
    """Test regression preparer"""
    out_metadata = get_combined_metadata("site1", ["meas1"], ["Ex"])
    out_data = SiteCombinedData(out_metadata, TEST1_OUT_DATA)
    in_metadata = get_combined_metadata("site2", ["run1"], ["Hy"])
    in_data = SiteCombinedData(in_metadata, TEST1_IN_DATA)
    cross_metadata = get_combined_metadata("site3", ["data1"], ["Hx"])
    cross_data = SiteCombinedData(cross_metadata, TEST1_CROSS_DATA)
    # generate the gathered data
    tf = TransferFunction(out_chans=["Ex"], in_chans=["Hy"], cross_chans=["Hx"])
    gathered_data = GatheredData(
        out_data=out_data, in_data=in_data, cross_data=cross_data
    )
    reg_data = RegressionPreparerGathered().run(tf, gathered_data)
    np.testing.assert_equal([10], reg_data.freqs)
    np.testing.assert_array_equal(reg_data.obs[0]["Ex"], TEST1_OBS["Ex"])
    np.testing.assert_array_equal(reg_data.preds[0], TEST1_PREDS[0])


def test_regression_preparer_2chan():
    """Test regression preparer"""
    out_metadata = get_combined_metadata("site1", ["meas1"], ["Ex", "Ey"])
    out_data = SiteCombinedData(out_metadata, TEST2_OUT_DATA)
    in_metadata = get_combined_metadata("site2", ["run1"], ["Hx", "Hy"])
    in_data = SiteCombinedData(in_metadata, TEST2_IN_DATA)
    cross_metadata = get_combined_metadata("site3", ["data1"], ["Hx", "Hy"])
    cross_data = SiteCombinedData(cross_metadata, TEST2_CROSS_DATA)
    # generate the gathered data
    tf = ImpedanceTensor()
    gathered_data = GatheredData(
        out_data=out_data, in_data=in_data, cross_data=cross_data
    )
    reg_data = RegressionPreparerGathered().run(tf, gathered_data)
    np.testing.assert_equal([10], reg_data.freqs)
    np.testing.assert_array_equal(reg_data.obs[0]["Ex"], TEST2_OBS["Ex"])
    np.testing.assert_array_equal(reg_data.obs[0]["Ey"], TEST2_OBS["Ey"])
    np.testing.assert_array_equal(reg_data.preds[0], TEST2_PREDS[0])


RANDOM_TF1 = transfer_function_random(5, 7)
RANDOM_TF2 = transfer_function_random(12, 4)
RANDOM_TF3 = transfer_function_random(6, 8, n_cross=1)
RANDOM_TF4 = transfer_function_random(3, 4, n_cross=3)


@pytest.mark.parametrize(
    "fs, tf, expected_soln, solver, n_levels, n_wins",
    [
        (
            256,
            ImpedanceTensor(),
            solution_mt(),
            SolverOLS(),
            1,
            2,
        ),
        (
            256,
            ImpedanceTensor(),
            solution_mt(),
            SolverOLS(),
            1,
            10,
        ),
        (
            256,
            ImpedanceTensor(),
            solution_mt(),
            SolverOLS(),
            1,
            50,
        ),
        (
            256,
            ImpedanceTensor(),
            solution_mt(),
            SolverOLS(),
            2,
            30,
        ),
        (
            512,
            RANDOM_TF1,
            solution_random_float(512, RANDOM_TF1, 25),
            SolverOLS(),
            5,
            1_000,
        ),
        (
            512,
            RANDOM_TF2,
            solution_random_int(512, RANDOM_TF2, 20),
            SolverOLS(),
            5,
            800,
        ),
        (
            512,
            RANDOM_TF3,
            solution_random_float(512, RANDOM_TF3, 25),
            SolverOLS(),
            5,
            400,
        ),
        (
            512,
            RANDOM_TF4,
            solution_random_int(512, RANDOM_TF4, 20),
            SolverOLS(),
            5,
            200,
        ),
    ],
    ids=[
        "Impedance tensor 1levels 2wins",
        "Impedance tensor 1levels 10wins",
        "Impedance tensor 1levels 50wins",
        "Impedance tensor 2levels 30wins",
        "Random transfer function 5levels 6000wins",
        "Random transfer function 5levels 800wins",
        "Random transfer function with cross chans 400wins",
        "Random transfer function with cross chans 200wins",
    ],
)
def test_regression_solution_single_site(
    fs: float,
    tf: TransferFunction,
    expected_soln: Solution,
    solver: Solver,
    n_levels: int,
    n_wins: int,
):
    """Test regression using synthetic evaluation frequency data"""
    from pathlib import Path
    from resistics.gather import QuickGather
    from resistics.regression import RegressionPreparerGathered
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
    gathered_data = QuickGather().run(Path(), dec_params, tf, eval_data)
    reg_data = RegressionPreparerGathered().run(tf, gathered_data)
    soln = solver.run(reg_data)
    assert_soln_equal(soln, expected_soln)


@pytest.mark.parametrize(
    "fs, tf, expected_soln, solver, n_levels, n_wins",
    [
        (
            256,
            ImpedanceTensor(),
            solution_mt(),
            SolverOLS(),
            1,
            2,
        ),
        (
            256,
            ImpedanceTensor(),
            solution_mt(),
            SolverOLS(),
            1,
            10,
        ),
        (
            256,
            ImpedanceTensor(),
            solution_mt(),
            SolverOLS(),
            1,
            50,
        ),
        (
            256,
            ImpedanceTensor(),
            solution_mt(),
            SolverOLS(),
            2,
            30,
        ),
        (
            512,
            RANDOM_TF1,
            solution_random_float(512, RANDOM_TF1, 25),
            SolverOLS(),
            5,
            1_000,
        ),
        (
            512,
            RANDOM_TF2,
            solution_random_int(512, RANDOM_TF2, 20),
            SolverOLS(),
            5,
            800,
        ),
        (
            512,
            RANDOM_TF3,
            solution_random_float(512, RANDOM_TF3, 25),
            SolverOLS(),
            5,
            400,
        ),
        (
            512,
            RANDOM_TF4,
            solution_random_int(512, RANDOM_TF4, 20),
            SolverOLS(),
            5,
            200,
        ),
    ],
    ids=[
        "Impedance tensor 1levels 2wins",
        "Impedance tensor 1levels 10wins",
        "Impedance tensor 1levels 50wins",
        "Impedance tensor 2levels 30wins",
        "Random transfer function 5levels 6000wins",
        "Random transfer function 5levels 800wins",
        "Random transfer function with cross chans 400wins",
        "Random transfer function with cross chans 200wins",
    ],
)
def test_regression_solution_spectra_input(
    fs: float,
    tf: TransferFunction,
    expected_soln: Solution,
    solver: Solver,
    n_levels: int,
    n_wins: int,
):
    """Test regression using synthetic evaluation frequency data"""
    from resistics.regression import RegressionPreparerSpectra
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
    reg_data = RegressionPreparerSpectra().run(tf, eval_data)
    soln = solver.run(reg_data)
    assert_soln_equal(soln, expected_soln)
