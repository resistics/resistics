"""
The regression module provides functions and classes for the following:

- Preparing gathered data for regression
- Performing the linear regression

Resistics has built in solvers that use scikit learn models, namely

- Ordinary least squares
- RANSAC
- TheilSen

These will perform well in many scenarios. However, the functionality available
in resistics makes it possible to use custom solvers if required.
"""
from loguru import logger
from typing import List, Dict, Tuple, Union, Optional
from tqdm import tqdm
import numpy as np
from sklearn.base import BaseEstimator

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

    (n_wins x n_cross_chans x 2).

    The factor of 2 is because the real and complex parts of each equation are
    separated into two equations to allow use of solvers that work exclusively
    on real data.

    The preds attribute is a single level dictionary with key of evaluation
    frequency index and value of the predictors for the evaluation frequency.
    The predictors have 2-D shape:

    (n_wins x n_cross_chans x 2)  x (n_input_channels x 2).

    The number of windows is multiplied by 2 for the same reason as the
    observations. The doubling of the input channels is because one is the
    predictor for the real part of that transfer function component and one is
    the predictor for the complex part of the transfer function component.

    Considering the impedance tensor as an example with:

    - output channels Ex, Ey
    - input channels Hx, Hy
    - cross channels Hx, Hy

    The below shows the arrays for the 0 index evaluation frequency:

    Observations

    - Ex: [w1_crossHx_RE, w1_crossHx_IM, w1_crossHy_RE, w1_crossHy_IM]
    - Ey: [w1_crossHx_RE, w1_crossHx_IM, w1_crossHy_RE, w1_crossHy_IM]

    Predictors Ex

    - w1_crossHx_RE:    Zxx_RE  Zxx_IM  Zxy_RE  Zxy_IM
    - w1_crossHx_IM:    Zxx_RE  Zxx_IM  Zxy_RE  Zxy_IM
    - w1_crossHy_RE:    Zxx_RE  Zxx_IM  Zxy_RE  Zxy_IM
    - w1_crossHy_IM:    Zxx_RE  Zxx_IM  Zxy_RE  Zxy_IM

    Predictors Ey

    - w1_crossHx_RE:    Zyx_RE  Zyx_IM  Zyy_RE  Zyy_IM
    - w1_crossHx_IM:    Zyx_RE  Zyx_IM  Zyy_RE  Zyy_IM
    - w1_crossHy_RE:    Zyx_RE  Zyx_IM  Zyy_RE  Zyy_IM
    - w1_crossHy_IM:    Zyx_RE  Zyx_IM  Zyy_RE  Zyy_IM

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


class RegressionPreparerSpectra(ResisticsProcess):
    """
    Prepare regression data directly from spectra data

    This can be useful for running a single measurement
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
        return RegressionInputData(tf, freqs, obs, preds)

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

        This can be achieved by numpy broadcasting the two arrays as follows

        out_data = [n_wins, n_out_chans, new_axis, n_freqs]
        cross_data = [n_wins, new_axis, n_cross_chans, n_freqs]

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

    def _get_obs(
        self, tf: TransferFunction, out_powers: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Get observations for an output channel

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
        obs = {}
        for idx, out_chan in enumerate(tf.out_chans):
            flattened = out_powers[:, idx, ...].flatten()
            obs_chan = np.empty((flattened.size * 2), dtype=float)
            obs_chan[0::2] = flattened.real
            obs_chan[1::2] = flattened.imag
            obs[out_chan] = obs_chan
        return obs

    def _get_obs_chan(self, tf: TransferFunction, out_powers: np.ndarray) -> np.ndarray:
        """
        Get observations for a single output channel

        The data comes in as:

        [n_wins, n_cross_chans]

        The aim is to turn this into an array of obserations for this output
        channel. The shape should be:

        [n_wins * n_cross_chans]

        Next, make everything floats to allow use of a wider range of open
        source solvers. This means interleaving the real and imaginary
        components into a longer single array, giving a final shape of:

        [n_wins * n_cross_chans * 2 ]

        Considering a concrete example for two windows:

        [[<Ex1, Hx1>, <Ex1,Hy1>], [[<Ex2, Hx2>, <Ex2,Hy2>]]

        Should become:

        [<Ex1, Hx1>.Real, <Ex1, Hx1>.Imag, <Ex1,Hy1>.Real, <Ex1,Hy1>.Imag,
        <Ex2, Hx2>.Real, <Ex2, Hx2>.Imag, <Ex2,Hy2>.Real, <Ex2,Hy2>.Imag]

        Parameters
        ----------
        tf : TransferFunction
            The transfer function definition
        out_powers : np.ndarray
            The cross powers for the a single output channel

        Returns
        -------
        np.ndarray
            The observations as a float
        """
        flattened = out_powers.flatten()
        obs = np.empty((flattened.size * 2), dtype=float)
        obs[0::2] = flattened.real
        obs[1::2] = flattened.imag
        return obs

    def _get_preds(self, tf: TransferFunction, in_powers: np.ndarray) -> np.ndarray:
        """
        Construct the predictors

        The in_powers is received with shape

        [n_wins, n_in_chans, n_cross_chans]

        The aim is to make this into

        [n_wins * n_cross_chans * 2, n_in_chans * 2]

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
        np.swapaxes(in_powers, 1, 2)
        n_wins = in_powers.shape[0]
        entries_per_win = tf.n_cross * 2
        preds = np.empty((n_wins * entries_per_win, tf.n_in * 2), dtype=float)
        for iwin in range(0, n_wins):
            idx_from = iwin * entries_per_win
            idx_to = idx_from + entries_per_win
            preds[idx_from:idx_to, :] = self._get_preds_win(tf, in_powers[iwin])
        return preds

    def _get_preds_win(self, tf: TransferFunction, in_powers: np.ndarray) -> np.ndarray:
        """Get predictors for a window"""
        preds_win = np.empty((tf.n_cross * 2, tf.n_in * 2), dtype=float)
        in_powers = np.swapaxes(in_powers, 0, 1)
        in_real = np.real(in_powers)
        in_imag = np.imag(in_powers)
        preds_win[0::2, 0::2] = in_real
        preds_win[0::2, 1::2] = -in_imag
        preds_win[1::2, 0::2] = in_imag
        preds_win[1::2, 1::2] = in_real
        return preds_win


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
        cross_data = np.conj(cross_data[:, np.newaxis, ...])

        # multiply using broadcasting
        out_powers = out_data[..., np.newaxis] * cross_data
        in_powers = in_data[..., np.newaxis] * cross_data
        return out_powers, in_powers

    def _get_obs(
        self, tf: TransferFunction, out_powers: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Get observations for an output channel

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
        obs = {}
        for idx, out_chan in enumerate(tf.out_chans):
            flattened = out_powers[:, idx, ...].flatten()
            obs_chan = np.empty((flattened.size * 2), dtype=float)
            obs_chan[0::2] = flattened.real
            obs_chan[1::2] = flattened.imag
            obs[out_chan] = obs_chan
        return obs

    def _get_obs_chan(self, tf: TransferFunction, out_powers: np.ndarray) -> np.ndarray:
        """
        Get observations for a single output channel

        The data comes in as:

        [n_wins, n_cross_chans]

        The aim is to turn this into an array of obserations for this output
        channel. The shape should be:

        [n_wins * n_cross_chans]

        Next, make everything floats to allow use of a wider range of open
        source solvers. This means interleaving the real and imaginary
        components into a longer single array, giving a final shape of:

        [n_wins * n_cross_chans * 2 ]

        Considering a concrete example for two windows:

        [[<Ex1, Hx1>, <Ex1,Hy1>], [[<Ex2, Hx2>, <Ex2,Hy2>]]

        Should become:

        [<Ex1, Hx1>.Real, <Ex1, Hx1>.Imag, <Ex1,Hy1>.Real, <Ex1,Hy1>.Imag,
        <Ex2, Hx2>.Real, <Ex2, Hx2>.Imag, <Ex2,Hy2>.Real, <Ex2,Hy2>.Imag]

        Parameters
        ----------
        tf : TransferFunction
            The transfer function definition
        out_powers : np.ndarray
            The cross powers for the a single output channel

        Returns
        -------
        np.ndarray
            The observations as a float
        """
        flattened = out_powers.flatten()
        obs = np.empty((flattened.size * 2), dtype=float)
        obs[0::2] = flattened.real
        obs[1::2] = flattened.imag
        return obs

    def _get_preds(self, tf: TransferFunction, in_powers: np.ndarray) -> np.ndarray:
        """
        Construct the predictors

        The in_powers is received with shape

        [n_wins, n_in_chans, n_cross_chans]

        The aim is to make this into

        [n_wins * n_cross_chans * 2, n_in_chans * 2]

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
        np.swapaxes(in_powers, 1, 2)
        n_wins = in_powers.shape[0]
        entries_per_win = tf.n_cross * 2
        preds = np.empty((n_wins * entries_per_win, tf.n_in * 2), dtype=float)
        for iwin in range(0, n_wins):
            idx_from = iwin * entries_per_win
            idx_to = idx_from + entries_per_win
            preds[idx_from:idx_to, :] = self._get_preds_win(tf, in_powers[iwin])
        return preds

    def _get_preds_win(self, tf: TransferFunction, in_powers: np.ndarray) -> np.ndarray:
        """Get predictors for a window"""
        preds_win = np.empty((tf.n_cross * 2, tf.n_in * 2), dtype=float)
        in_powers = np.swapaxes(in_powers, 0, 1)
        in_real = np.real(in_powers)
        in_imag = np.imag(in_powers)
        preds_win[0::2, 0::2] = in_real
        preds_win[0::2, 1::2] = -in_imag
        preds_win[1::2, 0::2] = in_imag
        preds_win[1::2, 1::2] = in_real
        return preds_win


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
    5
    >>> solution.freqs
    [10.0, 20.0, 30.0, 40.0, 50.0]
    >>> solution.periods.tolist()
    [0.1, 0.05, 0.03333333333333333, 0.025, 0.02]
    >>> solution.components["ExHx"]
    Component(real=[1.0, 1.0, 2.0, 2.0, 3.0], imag=[5.0, 5.0, 4.0, 4.0, 3.0])
    >>> solution.components["ExHy"]
    Component(real=[1.0, 2.0, 3.0, 4.0, 5.0], imag=[-5.0, -4.0, -3.0, -2.0, -1.0])

    To get the components as an array, either get_component or subscripting
    be used

    >>> solution["ExHy"]
    array([1.-5.j, 2.-4.j, 3.-3.j, 4.-2.j, 5.-1.j])
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
        Get the tensor at a single evaluation frequency

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


class Solver(ResisticsProcess):
    """General resistics solver"""

    def run(self, regression_input: RegressionInputData):
        """Every solver should have a run method"""
        raise NotImplementedError("run should only be called from child classes")


class SolverScikit(Solver):
    """Base class for Scikit learn solvers"""

    fit_intercept: bool = False
    """Flag for adding an intercept term"""
    normalize: bool = False
    """Flag for normalizing, only used if fit_intercept is True"""

    def _solve(
        self, regression_input: RegressionInputData, model: BaseEstimator
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
                coef = self._get_coef(model, obs, preds)
                tensors[eval_idx, iout] = self._get_tensor(tf, coef)
        return self._get_solution(tf, regression_input, tensors)

    def _get_coef(
        self, model: BaseEstimator, obs: np.ndarray, preds: np.ndarray
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
        return model.coef_

    def _get_tensor(self, tf: TransferFunction, coef: np.ndarray) -> np.ndarray:
        """
        Rearrange the coefficients into a tensor

        Recall that the real and complex parts of the problem are separated out
        to allow use of a greater selection of solvers. Therefore, part of
        the job of this function is to reform the complex numbers.

        Parameters
        ----------
        tf : TransferFunction
            The transfer fuction
        coef : np.ndarray
            The coefficients

        Returns
        -------
        np.ndarray
            The coefficients in a tensor
        """
        values = np.empty((tf.n_in), dtype=np.complex128)
        for in_idx in range(tf.n_in):
            idx_coef = in_idx * 2
            values[in_idx] = coef[idx_coef] + 1j * coef[idx_coef + 1]
        return values

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


class SolverScikitOLS(SolverScikit):
    """
    Ordinary least squares solver

    This is simply a wrapper around the scikit learn least squares regression
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
    """

    n_jobs: int = -2
    """Number of jobs to run"""

    def run(self, regression_input: RegressionInputData) -> Solution:
        """Run ordinary least squares regression on the RegressionInputData"""
        from sklearn.linear_model import LinearRegression

        model = LinearRegression(
            fit_intercept=self.fit_intercept,
            normalize=self.normalize,
            n_jobs=self.n_jobs,
        )
        return self._solve(regression_input, model)


class SolverScikitHuber(SolverScikit):
    """
    Scikit Huber solver

    This is simply a wrapper around the scikit learn Huber Regressor. For
    more information, please see
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.HuberRegressor.html
    """

    epsilon: float = 1
    """The smaller the epsilon, the more robust it is to outliers."""

    def run(self, regression_input: RegressionInputData) -> Solution:
        """Run Huber Regressor regression on the RegressionInputData"""
        from sklearn.linear_model import HuberRegressor

        model = HuberRegressor(epsilon=self.epsilon)
        return self._solve(regression_input, model)


class SolverScikitTheilSen(SolverScikit):
    """
    Scikit Theil Sen solver

    This is simply a wrapper around the scikit learn Theil Sen Regressor. For
    more information, please see
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.TheilSenRegressor.html
    """

    n_jobs: int = -2
    """Number of jobs to run"""
    max_subpopulation: int = 2_000
    """Maximum population. Reduce this if the process is taking a long time"""
    n_subsamples: Optional[int] = None
    """Number of rows to use for each solution"""

    def run(self, regression_input: RegressionInputData) -> Solution:
        """Run TheilSen regression on the RegressionInputData"""
        from sklearn.linear_model import TheilSenRegressor

        model = TheilSenRegressor(
            fit_intercept=self.fit_intercept,
            max_subpopulation=self.max_subpopulation,
            n_subsamples=self.n_subsamples,
            n_jobs=self.n_jobs,
        )
        return self._solve(regression_input, model)


class SolverScikitRANSAC(SolverScikit):
    """
    Run a RANSAC solver with LinearRegression as Base Estimator

    This is a wrapper around the scikit learn RANSAC regressor. More information
    can be found here
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RANSACRegressor.html
    """

    min_samples: float = 0.1
    """Minimum number of samples in each solution as a proportion of total"""
    max_trials: int = 1000
    """The maximum number of trials to run"""

    def run(self, regression_input: RegressionInputData) -> Solution:
        """Run RANSAC regression on the RegressionInputData"""
        from sklearn.linear_model import LinearRegression, RANSACRegressor

        model = RANSACRegressor(
            LinearRegression(
                fit_intercept=self.fit_intercept, normalize=self.normalize
            ),
            min_samples=self.min_samples,
            max_trials=self.max_trials,
        )
        return self._solve(regression_input, model)

    def _get_coef(
        self, model: BaseEstimator, obs: np.ndarray, preds: np.ndarray
    ) -> np.ndarray:
        """Get coefficients for single evaluation frequeny and output channel"""

        model.fit(preds, obs)
        return model.estimator_.coef_


class SolverScikitWLS(SolverScikitOLS):
    """
    Weighted least squares solver

    .. warning::

        This is homespun and is currently only experimental

    This is simply a wrapper around the scikit learn least squares regression
    using the sample_weight option
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
    """

    n_jobs: int = -2
    """Number of jobs to run"""
    n_iter: int = 50

    def _get_coef(
        self, model: BaseEstimator, obs: np.ndarray, preds: np.ndarray
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
        from sklearn.preprocessing import RobustScaler

        weights = np.ones(shape=(obs.size))
        scalar = RobustScaler()
        iteration = 0
        while iteration < self.n_iter:
            model.fit(preds, obs, sample_weight=weights)
            obs_pred = model.predict(preds)
            resids = np.absolute(obs - obs_pred)
            resids_scaled = scalar.fit_transform(resids.reshape(-1, 1))
            weights = self.bisquare(resids_scaled)
            iteration += 1
        return model.coef_

    def bisquare(self, r: np.ndarray, k: float = 4.685) -> np.ndarray:
        """
        Bisquare location weights

        Parameters
        ----------
        r : np.ndarray
            Residuals
        k : float, None
            Tuning parameter. If None, a standard value will be used.

        Returns
        -------
        weights : np.ndarray
            The robust weights
        """
        r = r.reshape((r.shape[0]))
        ones = np.ones(shape=r.shape)
        thresh = np.minimum(ones, r / k)
        return np.power((1 - np.power(thresh, 2)), 2)

    def huber(self, r: np.ndarray, k: float = 1.345) -> np.ndarray:
        """Huber location weights

        Parameters
        ----------
        r : np.ndarray
            Residuals
        k : float
            Tuning parameter. If None, a standard value will be used.

        Returns
        -------
        weights : np.ndarray
            The robust weights
        """
        r = r.reshape((r.shape[0]))
        indices = np.where(r > k)
        weights = np.ones(shape=r.shape)
        weights[indices] = k / r[indices]
        return weights

    def trimmed_mean(self, r: np.ndarray, k: float = 2) -> np.ndarray:
        """
        Trimmed mean location weights

        Parameters
        ----------
        r : np.ndarray
            Residuals
        k : float
            Tuning parameter. If None, a standard value will be used.

        Returns
        -------
        weights : np.ndarray
            The robust weights
        """
        r = r.reshape((r.shape[0]))
        indices = np.where(r <= k)
        weights = np.zeros(shape=r.shape)
        weights[indices] = 1
        return weights.real
