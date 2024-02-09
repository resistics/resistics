"""
The regression module provides functions and classes for the following:

- Preparing gathered data for regression
- Performing the regression to calculate the components of the transfer function

Resistics has a few built in solvers, but makes it possible to define custom
solvers as required
"""
from loguru import logger
from typing import List, Dict, Tuple, Union
from tqdm import tqdm
import numpy as np
import pandas as pd
from regressioninc.linear.models import Regressor, LeastSquares

from resistics.common import Metadata, WriteableMetadata, History
from resistics.common import ResisticsData, ResisticsProcess
from resistics.transfunc import Component, get_component_key
from resistics.transfunc import TransferFunction
from resistics.spectra import SpectraMetadata, SpectraData
from resistics.gather import SiteCombinedMetadata, GatheredData


class RegressionInputMetadata(Metadata):
    """Metadata for regression input data, mainly to track processing history"""

    contributors: Dict[str, Union[SiteCombinedMetadata, SpectraMetadata]]
    """Details about the data contributing to the regression input data"""
    history: History = History()
    """The processing history"""


class RegressionInputData(ResisticsData):
    """
    Class to hold data that will be input into a solver

    The purpose of regression input data is to provision for many different
    solvers and user written solvers.

    The regression input data has the following key attributes:

    - freqs
    - obs
    - preds

    The freqs attribute is a 1-D array of evaluation frequencies.

    The obs attribute is a dictionary of dictionaries. The parent dictionary has
    a key of the evaluation frequency index. The secondary dictionary has key of
    output channel. The values in the secondary dictionary are the observations
    for that output channel and have 1-D size:

    [n_wins x n_cross_chans].

    The preds attribute is a single level dictionary with key of evaluation
    frequency index and value of the predictors for the evaluation frequency.
    The predictors have 2-D shape:

    [n_wins x n_cross_chans, n_input_channels].


    For an example, consider the impedance tensor. This has:

    - output channels Ex, Ey
    - input channels Hx, Hy

    Call the cross channels

    - cross channels C1, C2

    For single site processing, the cross channels are often Hx and Hy, though
    this does not have to be the case and the source of this data changes for
    remote reference processing.

    In this case, the observations and predictors for the output channel Ex are:

    Observations Ex

    - win1 C1: <Ex_win1, conj(C1_win1)>
    - win1 C2: <Ex_win1, conj(C2_win1)>
    - win2 C1: <Ex_win2, conj(C1_win2)>
    - win2 C2: <Ex_win2, conj(C2_win1)>
    - ...

    Predictors Ex

    - win1 C1:  Zxx <Hx_win1, conj(C1_win1)>   Zxy <Hy_win1, conj(C1_win1)>
    - win1 C2:  Zxx <Hx_win1, conj(C2_win1)>   Zxy <Hy_win1, conj(C2_win1)>
    - win2 C1:  Zxx <Hx_win2, conj(C1_win2)>   Zxy <Hy_win2, conj(C1_win2)>
    - win2 C2:  Zxx <Hx_win2, conj(C2_win1)>   Zxy <Hy_win2, conj(C2_win1)>
    - ...

    Note that the predictors are the same regardless of the output channel,
    only the observations change.
    """

    def __init__(
        self,
        metadata: RegressionInputMetadata,
        tf: TransferFunction,
        freqs: List[float],
        obs: List[Dict[str, np.ndarray]],
        preds: List[np.ndarray],
    ):
        """
        Initialisation of regression input data

        Parameters
        ----------
        metadata : RegressionInputMetadata
            The metadata, mainly to hold the various processing histories that
            have been combined to produce the data
        tf : TransferFunction
            The transfer function that is to be solver
        freqs : List[float]
            The evaluation frequencies
        obs : List[Dict[str, np.ndarray]]
            The observations (output channels). This is a list of dictionaries.
            The entries into the list are on for each evaluation evaluation
            frequency. They keys of the dictionary are the output channels.
        preds : List[np.ndarray]
            The predictions, an entry for each evaluation frequenvy
        """
        self.metadata = metadata
        self.tf = tf
        self.freqs = freqs
        self.obs = obs
        self.preds = preds

    @property
    def n_freqs(self) -> int:
        """Get the number of frequencies"""
        return len(self.freqs)

    def get_inputs(self, freq_idx: int, out_chan: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get observations and predictions

        Parameters
        ----------
        freq_idx : int
            The evaluation frequency index
        out_chan : str
            The output channel

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Observations and predictons
        """
        return self.obs[freq_idx][out_chan], self.preds[freq_idx]


class RegressionPreparerGathered(ResisticsProcess):
    """
    Regression preparer for gathered data

    In nearly all cases, this is the regresson preparer to use. As input, it
    requires GatheredData.
    """

    def run(
        self, tf: TransferFunction, gathered_data: GatheredData
    ) -> RegressionInputData:
        """
        Create the RegressionInputData

        Parameters
        ----------
        tf : TransferFunction
            The transfer function
        gathered_data : GatheredData
            The gathered data

        Returns
        -------
        RegressionInputData
            Data that can be used as input into a solver
        """
        logger.info("Preparing regression data")
        logger.info(f"Out chans site: {gathered_data.out_data.metadata.site_name}")
        logger.info(f"Out chans: {gathered_data.out_data.metadata.chans}")
        logger.info(f"In chans site: {gathered_data.in_data.metadata.site_name}")
        logger.info(f"In chans: {gathered_data.in_data.metadata.chans}")
        logger.info(f"Cross chans site: {gathered_data.cross_data.metadata.site_name}")
        logger.info(f"Cross chans: {gathered_data.cross_data.metadata.chans}")
        return self._get_regression_data(tf, gathered_data)

    def _get_regression_data(
        self, tf: TransferFunction, gathered_data: GatheredData
    ) -> RegressionInputData:
        """
        Get the regression input data

        Parameters
        ----------
        tf : TransferFunction
            The transfer function
        gathered_data : GatheredData
            The gathered data

        Returns
        -------
        RegressionInputData
            Data to be used as input to a solver
        """
        freqs = []
        obs = []
        preds = []
        metadata = gathered_data.out_data.metadata
        logger.info(f"Preparing regression data for {metadata.n_evals} frequencies")
        for idx, freq in enumerate(tqdm(metadata.eval_freqs)):
            out_powers, in_powers = self._get_cross_powers(tf, gathered_data, idx)
            freqs.append(freq)
            obs_freq = self._get_obs(tf, out_powers)
            preds_freq = self._get_preds(tf, in_powers)
            obs.append(obs_freq)
            preds.append(preds_freq)
        record = self._get_record(
            f"Produced regression input data for {metadata.n_evals} frequencies"
        )
        metadata = RegressionInputMetadata(
            contributors={
                "out_data": gathered_data.out_data.metadata,
                "in_data": gathered_data.in_data.metadata,
                "cross_data": gathered_data.cross_data.metadata,
            }
        )
        metadata.history.add_record(record)
        return RegressionInputData(metadata, tf, freqs, obs, preds)

    def _get_cross_powers(
        self, tf: TransferFunction, gathered_data: GatheredData, eval_idx: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get cross powers

        Gathered data for an evaluation frequency is:

        [n_wins, n_chans]

        To multiply each in/out channel with the cross channels, broadcasting is
        used. Using output channels as an example, this is what we have:

        out_data = [n_wins, n_out_chans]
        cross_data = [n_wins, n_cross_chans]

        The aim is to achieve an array that looks like this:

        cross_powers = [n_wins, n_out_chans, n_cross_chans]

        This can be achieved by numpy broadcasting the two arrays as follows

        out_data = [n_wins, n_out_chans, new_axis]
        cross_data = [n_wins, new_axis, n_cross_chans]

        Parameters
        ----------
        tf : TransferFunction
            Definition of transfer function
        gathered_data : GatheredData
            All the gathered data
        eval_idx : int
            The evaluation frequency index

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Cross powers with output channels and cross powers with input
            channels
        """
        # calculate the cross powers
        out_data = gathered_data.out_data.data[eval_idx]
        in_data = gathered_data.in_data.data[eval_idx]
        cross_data = gathered_data.cross_data.data[eval_idx]
        cross_data = np.conjugate(cross_data[:, np.newaxis, :])

        # multiply using broadcasting
        out_powers = out_data[..., np.newaxis] * cross_data
        in_powers = in_data[..., np.newaxis] * cross_data
        return out_powers, in_powers

    def _get_obs(
        self, tf: TransferFunction, out_powers: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Get observations for an output channel

        This is a single dimension array with shape

        [n_wins * n_cross_chans]

        Parameters
        ----------
        tf : TransferFunction
            Definition of transfer function
        out_powers : np.ndarray
            The cross powers for the output channels

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary with output channel as key and observations as value
        """
        return {
            out_chan: out_powers[:, idx, ...].flatten()
            for idx, out_chan in enumerate(tf.out_chans)
        }

    def _get_preds(self, tf: TransferFunction, in_powers: np.ndarray) -> np.ndarray:
        """
        Construct the predictors

        The in_powers is received with shape

        [n_wins, n_in_chans, n_cross_chans]

        The aim is to make this into

        [n_wins * n_cross_chans, n_in_chans]

        Parameters
        ----------
        tf : TransferFunction
            Transfer function definition
        in_powers : np.ndarray
            The cross powers for the input channels

        Returns
        -------
        np.ndarray
            The predictors
        """
        in_powers = np.swapaxes(in_powers, 1, 2)
        return in_powers.reshape(-1, in_powers.shape[-1])


class RegressionPreparerSpectra(RegressionPreparerGathered):
    """
    Prepare regression data directly from spectra data

    This can be useful for running a single measurement

    See Also
    --------
    RegressionPreparerGathered : Produce regression input data from gathered
    data
    """

    def run(self, tf: TransferFunction, spec_data: SpectraData) -> RegressionInputData:
        """Construct the linear equation for solving"""
        freqs = []
        obs = []
        preds = []
        for ilevel in range(spec_data.metadata.n_levels):
            level_metadata = spec_data.metadata.levels_metadata[ilevel]
            out_powers, in_powers = self._get_cross_powers(tf, spec_data, ilevel)
            for idx, freq in enumerate(level_metadata.freqs):
                logger.info(
                    f"Preparing regression data: level {ilevel}, freq. {idx} = {freq}"
                )
                freqs.append(freq)
                obs.append(self._get_obs(tf, out_powers[..., idx]))
                preds.append(self._get_preds(tf, in_powers[..., idx]))
        record = self._get_record("Produced regression input data for spectra data")
        metadata = RegressionInputMetadata(contributors={"data": spec_data.metadata})
        metadata.history.add_record(record)
        return RegressionInputData(metadata, tf, freqs, obs, preds)

    def _get_cross_powers(
        self, tf: TransferFunction, spec_data: SpectraData, level: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get cross powers

        Spectra data is:

        [n_wins, n_chans, n_freqs]

        To multiply each in/out channel with the cross channels, broadcasting is
        used. Using output channels as an example, this is what we have:

        out_data = [n_wins, n_out_chans, n_freqs]
        cross_data = [n_wins, n_cross_chans, n_freqs]

        The aim is to achieve an array that looks like this:

        cross_powers = [n_wins, n_out_chans, n_cross_chans, n_freqs]

        Parameters
        ----------
        tf : TransferFunction
            Definition of transfer function
        spec_data : SpectraTimeData
            Spectra data for a decimation level
        level : int
            The decimation level

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Cross powers with output channels and cross powers with input
            channels
        """
        # prepare to calculate the crosspowers
        out_data = spec_data.get_chans(level, tf.out_chans)
        in_data = spec_data.get_chans(level, tf.in_chans)
        cross_data = spec_data.get_chans(level, tf.cross_chans)
        cross_data = np.conj(cross_data[:, np.newaxis, ...])

        # multiply using broadcasting
        out_powers = out_data[..., np.newaxis, :] * cross_data
        in_powers = in_data[..., np.newaxis, :] * cross_data
        return out_powers, in_powers


class Solution(WriteableMetadata):
    """
    Class to hold a transfer function solution

    Examples
    --------
    >>> from resistics.testing import solution_mt
    >>> solution = solution_mt()
    >>> print(solution.tf.to_string())
    | Ex | = | Ex_Hx Ex_Hy | | Hx |
    | Ey |   | Ey_Hx Ey_Hy | | Hy |
    >>> solution.n_freqs
    6
    >>> solution.freqs
    [100.0, 80.0, 60.0, 40.0, 20.0, 10.0]
    >>> solution.periods.tolist()
    [0.01, 0.0125, 0.016666666666666666, 0.025, 0.05, 0.1]
    >>> solution.components["ExHx"]
    Component(real=[1.0, 1.0, 2.0, 2.0, 3.0, 3.0], imag=[5.0, 5.0, 4.0, 4.0, 3.0, 3.0])
    >>> solution.components["ExHy"]
    Component(real=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], imag=[-5.0, -4.0, -3.0, -2.0, -1.0, 1.0])

    To get the components as an array, either get_component or subscripting
    be used

    >>> solution["ExHy"]
    array([1.-5.j, 2.-4.j, 3.-3.j, 4.-2.j, 5.-1.j, 6.+1.j])
    >>> solution["ab"]
    Traceback (most recent call last):
    ...
    ValueError: Component ab not found in ['ExHx', 'ExHy', 'EyHx', 'EyHy']

    It is also possible to get the tensor values at a particular evaluation
    frequency

    >>> solution.get_tensor(2)
    array([[ 2.+4.j,  3.-3.j],
           [-3.+3.j, -2.-4.j]])
    """

    tf: TransferFunction
    """The transfer function that was solved"""
    freqs: List[float]
    """The evaluation frequencies"""
    components: Dict[str, Component]
    """The solution"""
    history: History
    """The processing history"""
    contributors: Dict[str, Union[SiteCombinedMetadata, SpectraMetadata]]
    """The contributors to the solution with their respective details"""

    def __getitem__(self, key: str) -> np.ndarray:
        """
        Solution for a single component for all evaluation frequencies

        The arguments should be output channel followed by input channel

        Parameters
        ----------
        key : str
            The component key

        Returns
        -------
        np.ndarray
            The component values as an array

        Raises
        ------
        ValueError
            If incorrect number of arguments
        """
        if not isinstance(key, str):
            raise ValueError("Subscripting takes only 1 argument != {len(arg)}")
        return self.get_component(key)

    @property
    def n_freqs(self):
        """Get the number of evaluation frequencies"""
        return len(self.freqs)

    @property
    def periods(self) -> np.ndarray:
        """Get the periods"""
        return np.reciprocal(self.freqs)

    def get_component(self, key: str) -> np.ndarray:
        """
        Get the solution for a single component for all the evaluation
        frequencies

        Parameters
        ----------
        key : str
            The component key

        Returns
        -------
        np.ndarray
            The component data in an array

        Raises
        ------
        ValueError
            If the component does not exist in the solution
        """
        if key not in self.components:
            raise ValueError(
                f"Component {key} not found in {list(self.components.keys())}"
            )
        return self.components[key].to_numpy()

    def get_tensor(self, eval_idx: int) -> np.ndarray:
        """
        Get the tensor at a single evaluation frequency. This has shape:

        n_out_chans x n_in_chans

        Parameters
        ----------
        eval_idx : int
            The index of the evaluation frequency

        Returns
        -------
        np.ndarray
            The tensor as a numpy array
        """
        tensor = np.zeros(shape=(self.tf.n_out, self.tf.n_in), dtype=np.complex128)
        for out_idx, out_chan in enumerate(self.tf.out_chans):
            for in_idx, in_chan in enumerate(self.tf.in_chans):
                key = get_component_key(out_chan, in_chan)
                tensor[out_idx, in_idx] = self.components[key].get_value(eval_idx)
        return tensor

    def to_dataframe(self) -> pd.DataFrame:
        """Get the solution as a pandas DataFrame"""
        soln_data = {comp: self.get_component(comp) for comp in self.components}
        index = self.freqs
        return pd.DataFrame(data=soln_data, index=index)


class Solver(ResisticsProcess):
    """General resistics solver"""

    def run(self, regression_input: RegressionInputData) -> Solution:
        """Every solver should have a run method"""
        raise NotImplementedError("Run not implemented in parent Solver class")


class SolverLinear(Solver):
    """Base class for linear solvers"""

    fit_intercept: bool = False
    """Flag for adding an intercept term"""

    def _solve(
        self, regression_input: RegressionInputData, model: Regressor
    ) -> Solution:
        """
        Get the regression solution for all evaluation frequencies

        Parameters
        ----------
        regression_input : RegressionInputData
            The regression input data
        model : BaseEstimator
            The model to use to solve the linear regressions

        Returns
        -------
        Solution
            The solution for the transfer function
        """
        n_freqs = regression_input.n_freqs
        tf = regression_input.tf
        tensors = np.ndarray((n_freqs, tf.n_out, tf.n_in), dtype=np.complex128)
        logger.info(f"Solving for {n_freqs} evaluation frequencies")
        for eval_idx in tqdm(range(n_freqs)):
            for iout, out_chan in enumerate(tf.out_chans):
                obs, preds = regression_input.get_inputs(eval_idx, out_chan)
                tensors[eval_idx, iout] = self._get_coef(model, obs, preds)
        return self._get_solution(tf, regression_input, tensors)

    def _get_coef(
        self, model: Regressor, obs: np.ndarray, preds: np.ndarray
    ) -> np.ndarray:
        """
        Get coefficients for a single evaluation frequency and output channel

        Parameters
        ----------
        model : BaseEstimator
            sklearn base estimator
        obs : np.ndarray
            The observations
        preds : np.ndarray
            The predictors

        Returns
        -------
        np.ndarray
            The coefficients
        """
        model.fit(preds, obs)
        return model.coef

    def _get_solution(
        self,
        tf: TransferFunction,
        regression_input: RegressionInputData,
        tensors: np.ndarray,
    ) -> Solution:
        """
        Get the solution

        Parameters
        ----------
        tf : TransferFunction
            The transfer function
        regression_input : RegressionInputData
            The regression input data
        tensors : np.ndarray
            The coefficients

        Returns
        -------
        Solution
            The transfer function solution
        """
        components = {}
        for out_idx, out_chan in enumerate(tf.out_chans):
            for in_idx, in_chan in enumerate(tf.in_chans):
                key = get_component_key(out_chan, in_chan)
                values = tensors[:, out_idx, in_idx]
                components[key] = Component(
                    real=values.real.tolist(), imag=values.imag.tolist()
                )
        history = History(**regression_input.metadata.history.dict())
        message = f"Solved {len(regression_input.freqs)} evaluation frequencies"
        history.add_record(self._get_record(message))
        return Solution(
            tf=tf,
            freqs=regression_input.freqs,
            components=components,
            history=history,
            contributors=regression_input.metadata.contributors,
        )


class SolverOLS(SolverLinear):
    n_jobs: int = -2
    """Number of jobs to run"""

    def run(self, regression_input: RegressionInputData) -> Solution:
        """Run ordinary least squares regression on the RegressionInputData"""
        model = LeastSquares()
        return self._solve(regression_input, model)
