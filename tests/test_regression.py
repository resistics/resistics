"""
Testing of regression fuctions and processors
"""
from typing import List
import numpy as np

from resistics.common import History
from resistics.gather import SiteCombinedData, SiteCombinedMetadata, GatheredData
from resistics.transfunc import TransferFunction, ImpedanceTensor
from resistics.regression import RegressionPreparerGathered

OUT_DATA = {0: np.array([[3 - 1j], [1 + 2j]])}
IN_DATA = {0: np.array([[-1 - 1j], [-2 - 3j]])}
CROSS_DATA = {0: np.array([[5 + 3j], [0 - 2j]])}

OUT_DATA2 = {0: np.array([[3 - 1j, 4 + 3j], [1 + 2j, 2 + 1j]])}
IN_DATA2 = {0: np.array([[-1 - 1j, 0 + 3j], [-2 - 3j, 4 - 1j]])}
CROSS_DATA2 = {0: np.array([[5 + 3j, 2 + 0j], [0 - 2j, 1 - 1j]])}

# OUT_DATA = {0: np.array([[3 - 1j]])}
# IN_DATA = {0: np.array([[-1 - 1j]])}
# CROSS_DATA = {0: np.array([[5 + 3j]])}
# PREDS = {0}
# OBS = {"Ex": }


def get_combined_metadata(
    site_name: str, measurements: List[str], chans: List[str]
) -> SiteCombinedMetadata:
    """Get metadata for SiteCombinedData"""
    histories = {x: History() for x in measurements}
    return SiteCombinedMetadata(
        name=site_name,
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
    out_data = SiteCombinedData(out_metadata, OUT_DATA)
    in_metadata = get_combined_metadata("site2", ["run1"], ["Hy"])
    in_data = SiteCombinedData(in_metadata, IN_DATA)
    cross_metadata = get_combined_metadata("site3", ["data1"], ["Hx"])
    cross_data = SiteCombinedData(cross_metadata, CROSS_DATA)
    # generate the gathered data
    tf = TransferFunction(out_chans=["Ex"], in_chans=["Hy"], cross_chans=["Hx"])
    gathered_data = GatheredData(
        out_data=out_data, in_data=in_data, cross_data=cross_data
    )
    reg_data = RegressionPreparerGathered().run(tf, gathered_data)
    np.testing.assert_equal([10], reg_data.freqs)
    print(reg_data.obs[0])
    print(reg_data.preds[0])
    assert False


def test_regression_preparer_2chan():
    """Test regression preparer"""
    out_metadata = get_combined_metadata("site1", ["meas1"], ["Ex", "Ey"])
    out_data = SiteCombinedData(out_metadata, OUT_DATA2)
    in_metadata = get_combined_metadata("site2", ["run1"], ["Hx", "Hy"])
    in_data = SiteCombinedData(in_metadata, IN_DATA2)
    cross_metadata = get_combined_metadata("site3", ["data1"], ["Hx", "Hy"])
    cross_data = SiteCombinedData(cross_metadata, CROSS_DATA2)
    # generate the gathered data
    tf = ImpedanceTensor()
    gathered_data = GatheredData(
        out_data=out_data, in_data=in_data, cross_data=cross_data
    )
    reg_data = RegressionPreparerGathered().run(tf, gathered_data)
    np.testing.assert_equal([10], reg_data.freqs)
    print(reg_data.obs[0])
    print(reg_data.preds[0])
    assert False
