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

    The regression input data has the following key attributes

    - freqs
    - obs
    - preds

    The freqs attribute is a 1-D array of evaluation frequencies.

    The obs attribute is a dictionary of dictionaries. The parent dictionary has
    a key of the evaluation frequency index. The secondary dictionary has key of
    output channel. The value in the secondary dictionary are the observations
    for that output channel and have size (n_wins x 2). The reason this is
    multiplied by 2 is because the real and complex parts of the equation are
    separated into separate equations.

    The preds attribute is a single level dictionary with key of evaluation
    frequency index and value of the predictors for the evaluation frequency.
    The predictors have shape (n_wins x 2) x n_input_channels. The reason for
    the factor of 2 is the same as for the observations. The same predictors can
    be used for all output channels.
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
        logger.info(f"Out channels site: {gathered_data.out_data.metadata.name}")
        logger.info(f"Out channels: {gathered_data.out_data.metadata.chans}")
        logger.info(f"In channels site: {gathered_data.in_data.metadata.name}")
        logger.info(f"In channels: {gathered_data.in_data.metadata.chans}")
        logger.info(f"Cross channels site: {gathered_data.cross_data.metadata.name}")
        logger.info(f"Cross channels: {gathered_data.cross_data.metadata.chans}")
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


class Component(Metadata):
    """
    Data class for a single component
    """

    real: List[float]
    """The real part of the component"""
    imag: List[float]
    """The complex part of the component"""

    def get_value(self, eval_idx: int) -> complex:
        """Get the value for an evaluation frequency"""
        return self.real[eval_idx] + 1j * self.imag[eval_idx]

    def to_array(self) -> np.array:
        """Get the component as a numpy complex array"""
        return np.array(self.real) + 1j * np.array(self.imag)


def get_component_key(out_chan: str, in_chan: str) -> str:
    """
    Get key for out channel and in channel combination in the solution

    Parameters
    ----------
    out_chan : str
        The output channel
    in_chan : str
        The input channel

    Returns
    -------
    str
        The component key

    Examples
    --------
    >>> from resistics.regression import get_component_key
    >>> get_component_key("Ex", "Hy")
    'ExHy'
    """
    return f"{out_chan}{in_chan}"


class Solution(WriteableMetadata):
    """
    Class to hold a transfer function solution
    """

    tf: TransferFunction
    """The transfer function that was solved"""
    freqs: List[float]
    """The evaluation frequencies"""
    components: Dict[str, Component]
    """The solution"""
    source: RegressionInputMetadata
    """The regression input metadata to provide traceability"""

    def __getitem___(self, out_chan: str, in_chan: str) -> np.ndarray:
        """Solution for a single component for all evaluation frequencies"""
        return self.get_component(out_chan, in_chan)

    @property
    def n_freqs(self):
        """Get the number of evaluation frequencies"""
        return len(self.freqs)

    @property
    def periods(self) -> np.ndarray:
        """Get the periods"""
        return np.reciprocal(self.freqs)

    def get_component(self, out_chan: str, in_chan: str) -> np.ndarray:
        """
        Get the solution for a single component for all the evaluation
        frequencies

        Parameters
        ----------
        out_chan : str
            The output channel
        in_chan : str
            The input channel

        Returns
        -------
        np.ndarray

        Raises
        ------
        ValueError
            out_chan not in the transfer function
        ValueError
            in_chan not in the transfer function
        """
        if out_chan not in self.tf.out_chans:
            raise ValueError(f"Out channel {out_chan} not in tf {self.tf.out_chans}")
        if in_chan not in self.tf.in_chans:
            raise ValueError(f"In channel {in_chan} not in tf {self.tf.in_chans}")
        key = get_component_key(out_chan, in_chan)
        return self.components[key].to_array()

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
        tensor = np.array(shape=(self.tf.n_out, self.tf.n_in), dtype=np.complex128)
        for out_idx, out_chan in enumerate(self.tf.out_chans):
            for in_idx, in_chan in enumerate(self.tf.in_chans):
                key = get_component_key(out_chan, in_chan)
                tensor[out_idx, in_idx] = self.solution[key].get_value(eval_idx)
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
        return Solution(
            tf=tf,
            freqs=regression_input.freqs,
            components=components,
            source=regression_input.metadata,
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
