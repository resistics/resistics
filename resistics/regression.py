"""
Module for collecting data for the regression
Includes collection of RegressionData and RegressionProcessors
TO BE IMPLEMENTED
"""
from loguru import logger
from typing import List, Optional, Dict, Tuple, Any, Union
from pydantic import validator
from tqdm import tqdm
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.linear_model import RegressionResultsWrapper
from statsmodels.robust.scale import HuberScale

from resistics.common import ResisticsData, ResisticsProcess, Metadata
from resistics.spectra import SpectraData


class TransferFunction(Metadata):
    """
    Define the transfer function

    This class has few methods and is a simple way of defining the transfer
    input and output channels for which to calculate the transfer function

    Examples
    --------
    A standard magnetotelluric transfer function

    >>> from resistics.regression import TransferFunction
    >>> tf = TransferFunction(["Hx", "Hy"], ["Ex", "Ey"])
    >>> print(tf.to_string())
    <class 'resistics.regression.TransferFunction'>
    | Ex | = | Ex_Hx Ex_Hy | | Hx |
    | Ey |   | Ey_Hx Ey_Hy | | Hy |

    Additionally including the Hz component

    >>> tf = TransferFunction(["Hx", "Hy", "Hz"], ["Ex", "Ey"])
    >>> print(tf.to_string())
    <class 'resistics.regression.TransferFunction'>
    | Ex |   | Ex_Hx Ex_Hy Ex_Hz | | Hx |
    | Ey | = | Ey_Hx Ey_Hy Ey_Hz | | Hy |
                                   | Hz |

    The magnetotelluric tipper

    >>> tf = TransferFunction(["Hx", "Hy"], ["Hz"])
    >>> print(tf.to_string())
    <class 'resistics.regression.TransferFunction'>
    | Hz | = | Hz_Hx Hz_Hy | | Hx |
                             | Hy |

    And a generic example

    >>> tf = TransferFunction(["hello", "hi_there"], ["bye", "see you", "ciao"])
    >>> print(tf.to_string())
    <class 'resistics.regression.TransferFunction'>
    | bye      |   | bye_hello         bye_hi_there      | | hello    |
    | see you  | = | see you_hello     see you_hi_there  | | hi_there |
    | ciao     |   | ciao_hello        ciao_hi_there     |
    """

    in_chans: List[str]
    out_chans: List[str]
    cross_chans: Optional[List[str]] = None
    n_in: Optional[int] = None
    n_out: Optional[int] = None
    n_cross: Optional[int] = None

    @validator("cross_chans", always=True)
    def validate_cross_chans(
        cls, value: Union[None, List[str]], values: Dict[str, Any]
    ) -> List[str]:
        """Validate cross spectra channels"""
        if value is None:
            return values["in_chans"]
        return value

    @validator("n_in", always=True)
    def validate_n_in(cls, value: Union[None, int], values: Dict[str, Any]) -> int:
        """Validate number of input channels"""
        if value is None:
            return len(values["in_chans"])
        return value

    @validator("n_out", always=True)
    def validate_n_out(cls, value: Union[None, int], values: Dict[str, Any]) -> int:
        """Validate number of output channels"""
        if value is None:
            return len(values["in_chans"])
        return value

    @validator("n_cross", always=True)
    def validate_n_cross(cls, value: Union[None, int], values: Dict[str, Any]) -> int:
        """Validate number of cross channels"""
        if value is None:
            return len(values["cross_chans"])
        return value

    def n_eqns_per_output(self) -> int:
        return len(self.cross_chans)

    def n_regressors(self) -> int:
        return self.n_in

    def to_string(self):
        n_lines = max(len(self.in_chans), len(self.out_chans))
        lens = [len(x) for x in self.in_chans] + [len(x) for x in self.out_chans]
        max_len = max(lens)
        line_equals = (n_lines - 1) // 2
        outstr = ""
        for il in range(n_lines):
            out_chan = self._out_chan_string(il, max_len)
            in_chan = self._in_chan_string(il, max_len)
            tensor = self._tensor_string(il, max_len)
            eq = "=" if il == line_equals else " "
            outstr += f"{out_chan} {eq} {tensor} {in_chan}\n"
        return outstr.rstrip("\n")

    def _out_chan_string(self, il: int, max_len: int) -> str:
        if il >= self.n_out:
            empty_len = max_len + 4
            return f"{'':{empty_len}s}"
        return f"| { self.out_chans[il]:{max_len}s} |"

    def _in_chan_string(self, il: int, max_len: int) -> str:
        if il >= self.n_in:
            return ""
        return f"| { self.in_chans[il]:{max_len}s} |"

    def _tensor_string(self, il: int, max_len: int) -> str:
        if il >= self.n_out:
            element_len = ((max_len * 2 + 1) + 1) * self.n_in + 3
            return f"{'':{element_len}s}"
        elements = "| "
        for chan in self.in_chans:
            component = f"{self.out_chans[il]}_{chan}"
            elements += f"{component:{2*max_len + 1}s} "
        elements += "|"
        return elements


class ImpedanceTensor(TransferFunction):
    """
    Standard magnetotelluric impedance tensor

    Examples
    --------
    >>> from resistics.regression import ImpedanceTensor
    >>> tf = ImpedanceTensor()
    >>> print(tf.to_string())
    <class 'resistics.regression.ImpedanceTensor'>
    | Ex | = | Ex_Hx Ex_Hy | | Hx |
    | Ey |   | Ey_Hx Ey_Hy | | Hy |
    """

    in_chans: List[str] = ["Hx", "Hy"]
    out_chans: List[str] = ["Ex", "Ey"]


class Tipper(TransferFunction):
    """
    Magnetotelluric tipper

    Examples
    --------
    >>> from resistics.regression import Tipper
    >>> tf = Tipper()
    >>> print(tf.to_string())
    <class 'resistics.regression.Tipper'>
    | Hz | = | Hz_Hx Hz_Hy | | Hx |
                             | Hy |
    """

    in_chans: List[str] = ["Hx", "Hy"]
    out_chans: List[str] = ["Hz"]


class RegressionInputData(ResisticsData):
    def __init__(
        self,
        tf: TransferFunction,
        freqs: List[float],
        obs: List[Dict[str, np.ndarray]],
        preds: List[np.ndarray],
    ):
        self.tf = tf
        self.freqs = freqs
        self.obs = obs
        self.preds = preds
        pass

    @property
    def n_freqs(self):
        return len(self.freqs)

    def get_inputs(self, freq_idx: int, out_chan: str) -> Tuple[np.ndarray, np.ndarray]:
        return self.obs[freq_idx][out_chan], self.preds[freq_idx]


class RegressionPreparer(ResisticsProcess):

    tf: TransferFunction
    coh_thresh: Optional[float] = None
    coh_chans: Optional[List[str]] = None

    def run(self, spec_data: SpectraData) -> RegressionInputData:
        """Construct the linear equation for solving"""
        freqs = []
        obs = []
        preds = []
        for ilevel in range(spec_data.metadata.n_levels):
            level_metadata = spec_data.metadata.levels_metadata[ilevel]
            out_powers, in_powers = self._get_cross_powers(self.tf, spec_data, ilevel)
            for idx, freq in enumerate(level_metadata.freqs):
                logger.info(
                    f"Preparing regression data: level {ilevel}, freq. {idx} = {freq}"
                )
                freqs.append(freq)
                obs.append(self._get_obs(self.tf, out_powers[..., idx]))
                preds.append(self._get_preds(self.tf, in_powers[..., idx]))
        return RegressionInputData(self.tf, freqs, obs, preds)

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
        # what cross powers do we need
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


class Solution(ResisticsData):
    def __init__(self, tf: TransferFunction, freqs: List[float], tensors: np.ndarray):
        self.tf = tf
        self.freqs = freqs
        self.tensors = tensors

    @property
    def n_freqs(self):
        return len(self.freqs)

    @property
    def periods(self) -> np.ndarray:
        return np.reciprocal(self.freqs)

    def get_freq(self, ifreq: int) -> np.ndarray:
        return self.tensors[ifreq]

    def get_component(self, out_chan: str, in_chan: str) -> np.ndarray:
        iout = self.tf.out_chans.index(out_chan)
        iin = self.tf.in_chans.index(in_chan)
        return self.tensors[:, iout, iin]

    def to_string(self) -> str:
        spc1 = 8
        spc2 = 24
        outstr = f"{self.type_to_string()}\n"
        for ifreq, freq in enumerate(self.freqs):
            outstr += f"Frequency Hz: {freq:.8f}\n"
            out_heading = "".join(
                [f"{in_chan:{spc2}s}" for in_chan in self.tf.in_chans]
            )
            outstr += f"{'':{spc1}s}{out_heading}\n"
            for i_out, out_chan in enumerate(self.tf.out_chans):
                out_values = self.tensors[ifreq][i_out].tolist()
                out_values = [f"{value:6f}" for value in out_values]
                out_values = "".join([f"{value:{spc2}s}" for value in out_values])
                outstr += f"{out_chan:{spc1}s}{out_values}\n"
        return outstr


class Solver(ResisticsProcess):
    """General resistics solver"""

    def run(self, regression_input: RegressionInputData):
        raise NotImplementedError("run should only be called from child classes")


class SolverStandard(Solver):
    """Standard linear solver"""

    def _get_tensor(self, tf: TransferFunction, params: np.ndarray):
        """Rearrange parameters into to help shape a tensor"""
        values = np.empty((tf.n_in), dtype=np.complex128)
        for iin in range(tf.n_in):
            idx_param = iin * 2
            values[iin] = params[idx_param] + 1j * params[idx_param + 1]
        return values


class SolverStatsmodels(SolverStandard):
    """Statsmodels solver"""

    def run(self, regression_input: RegressionInputData):
        n_freqs = regression_input.n_freqs
        tf = regression_input.tf
        tensors = np.ndarray((n_freqs, tf.n_out, tf.n_in), dtype=np.complex128)
        logger.info("Running solver over evaluation frequencies")
        for ifreq in tqdm(range(n_freqs)):
            for iout, out_chan in enumerate(tf.out_chans):
                obs, preds = regression_input.get_inputs(ifreq, out_chan)
                result = self.run_kernel(obs, preds)
                tensors[ifreq, iout] = self._get_tensor(tf, result.params)
        return Solution(tf, regression_input.freqs, tensors)

    def run_kernel(
        self, obs: np.ndarray, preds: np.ndarray
    ) -> RegressionResultsWrapper:
        raise NotImplementedError("run_kernel should be implemented in child classes")


class OLSSolver(SolverStatsmodels):
    """Statsmodels Ordinary Least Squares solver"""

    def run_kernel(
        self, obs: np.ndarray, preds: np.ndarray
    ) -> RegressionResultsWrapper:
        model = sm.OLS(obs, preds)
        return model.fit()


class RLMSolver(SolverStatsmodels):
    """Statsmodels Robust Least Squares solver"""

    cov: str = "H1"
    scale_est: str = "mad"

    def run_kernel(
        self, obs: np.ndarray, preds: np.ndarray
    ) -> RegressionResultsWrapper:
        model = sm.RLM(obs, preds, M=sm.robust.norms.HuberT())
        return model.fit(cov=self.cov, scale_est=HuberScale())


class MMSolver(SolverStatsmodels):
    """Statsmodels MM estimates solver"""

    cov: str = "H1"
    scale_est: str = "mad"

    def run_kernel(
        self, obs: np.ndarray, preds: np.ndarray
    ) -> RegressionResultsWrapper:
        model1 = sm.RLM(obs, preds, M=sm.robust.norms.HuberT())
        result = model1.fit(cov=self.cov, scale_est=self.scale_est)
        model2 = sm.RLM(obs, preds, M=sm.robust.norms.TukeyBiweight())
        return model2.fit(
            cov=self.cov, scale_est=self.scale_est, start_params=result.params
        )


class SolverScikitLinear(SolverStandard):
    """General solver for Scikit linear models"""

    model: Any

    # def __init__(self, model: LinearModel):
    #     self.model = model

    def parameters(self) -> Dict[str, Any]:
        params = {"model": type(self.model).__name__}
        params.update(self.model.get_params())
        return params

    def run(self, regression_input: RegressionInputData):
        n_freqs = regression_input.n_freqs
        tf = regression_input.tf
        tensors = np.ndarray((n_freqs, tf.n_out, tf.n_in), dtype=np.complex128)
        for ifreq in tqdm(range(n_freqs)):
            for iout, out_chan in enumerate(tf.out_chans):
                obs, preds = regression_input.get_inputs(ifreq, out_chan)
                params = self.run_kernel(obs, preds)
                tensors[ifreq, iout] = self._get_tensor(tf, params)
        return Solution(tf, regression_input.freqs, tensors)

    def run_kernel(self, obs: np.ndarray, preds: np.ndarray) -> np.ndarray:
        self.model.fit(preds, obs)
        return self.model.coef_


class SolverScikitRANSAC(SolverScikitLinear):
    def run_kernel(self, obs: np.ndarray, preds: np.ndarray) -> np.ndarray:
        from sklearn.linear_model import RANSACRegressor

        ransac = RANSACRegressor(self.model)
        ransac.fit(preds, obs)
        return ransac.estimator_.coef_


class SolverScikitMCD(SolverScikitLinear):

    max_iter: int = 2

    def run_kernel(self, obs: np.ndarray, preds: np.ndarray) -> np.ndarray:
        from sklearn.covariance import MinCovDet
        from sklearn.linear_model import LinearRegression

        model = LinearRegression()
        mcd = MinCovDet()
        weights = np.ones_like(obs)
        for it in range(self.max_iter):
            logger.info(f"Running iteration {it}")
            model.fit(preds, obs, sample_weight=weights)
            resids = np.absolute(obs - model.predict(preds))
            R = np.array((resids[0::2], resids[1::2])).T
            mcd.fit(R)
            weights = np.repeat(mcd.support_.astype(float), 2)
        return model.coef_
