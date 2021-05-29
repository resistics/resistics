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


class SiteSelector(ResisticsData):
    in_site: str = ""
    out_site: str = ""
    remote_site: str = ""


# class WindowDataFetcher(ResisticsData):
#     # class to get the windows we need for the processing
#     def __init__(self):
#         pass


# class WindowSelector(ResisticsProcess):
#     def __init__(self, proj: Project, sites: SiteSelector):
#         self.proj = proj
#         self.sites = sites

#     def run(self, out_site: str):
#         # select the windows
#         pass


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

    # def __init__(
    #     self,
    #     projData: ProjectData,
    #     sampleFreq: float,
    #     decParams: DecimationParameters,
    #     winParams: WindowParameters,
    #     **kwargs
    # ) -> None:
    #     """Initialise window selector

    #     Parameters
    #     ----------
    #     projData : ProjectData
    #         A ProjectData instance
    #     sampleFreq : float
    #         The sampling frequency of the raw time data
    #     decParams : DecimationParameters
    #         A decimation parameters instance detailing decimaion scheme
    #     winParams : WindowParameters
    #         A window parameters instance detailing window schemes
    #     specdir : str, optional
    #         The spectra directories to use
    #     """
    #     self.projData: ProjectData = projData
    #     self.sampleFreq: float = sampleFreq
    #     self.decParams = decParams
    #     self.winParams = winParams
    #     self.sites: List = []
    #     # shared indices
    #     self.sharedWindows: Dict = {}
    #     # the masks to use for each site - there can be multiple masks for each site
    #     self.siteMasks: Dict[str, List[str]] = {}
    #     # the spec files for each site at fs
    #     self.siteSpecFolders: Dict = {}
    #     self.siteSpecReaders: Dict = {}
    #     # global indices (ranges and sets)
    #     self.siteSpecRanges: Dict = {}
    #     self.siteGlobalIndices: Dict = {}
    #     # spectra directory information
    #     self.specdir = kwargs["specdir"] if "specdir" in kwargs else "spectra"
    #     self.prepend: str = "spectra"
    #     # time constraints: priority is datetimes > dates > times
    #     self.datetimes: Dict[int, List] = {}
    #     self.dates: Dict[int, List] = {}
    #     self.times: Dict[int, List] = {}
    #     # dictionary for datetime constraints
    #     self.datetimeConstraints: Dict[int, List] = {}
    #     # set all datetime constraints to empty
    #     self.resetDatetimeConstraints()

    # def getSharedWindowsLevel(self, declevel: int) -> Set:
    #     """Get the shared windows for a decimation level

    #     Parameters
    #     ----------
    #     declevel : int
    #         The decimation level (0 is the first level)

    #     Returns
    #     -------
    #     set
    #         The shared windows for the decimation level
    #     """
    #     return self.sharedWindows[declevel]

    # def getNumSharedWindows(self, declevel: int) -> int:
    #     """Get the number of shared windows for a decimation level

    #     Parameters
    #     ----------
    #     declevel : int
    #         The decimation level (0 is the first level)

    #     Returns
    #     -------
    #     int
    #         The number of shared windows for the decimation level
    #     """
    #     return len(self.sharedWindows[declevel])

    # def getWindowsForFreq(self, declevel: int, eIdx: int) -> Set:
    #     """Get the number of shared windows for a decimation level and evaluation frequency

    #     Parameters
    #     ----------
    #     declevel : int
    #         The decimation level (0 is the first level)
    #     eIdx : int
    #         The evaluation frequency index

    #     Returns
    #     -------
    #     set
    #         The shared windows for evaluation frequency eIdx at decimation level declevel
    #     """
    #     sharedWindows = self.getSharedWindowsLevel(declevel)
    #     # now mask for the particular frequency - mask for each given site
    #     for s in self.sites:
    #         for mask in self.siteMasks[s]:
    #             # remove the masked windows from shared indices
    #             sharedWindows = sharedWindows - mask.getMaskWindowsFreq(declevel, eIdx)
    #     return sharedWindows

    # def getUnmaskedWindowsLevel(self, declevel: int) -> Set:
    #     """Get unmasked windows for a decimation level

    #     Calculate the number of non masked windows for the decimation level. This should speed up processing when constraints are applied.

    #     Parameters
    #     ----------
    #     declevel : int
    #         The decimation level

    #     Returns
    #     -------
    #     set
    #         Unmasked windows for the decimation level
    #     """
    #     indices = set()
    #     evalFreq = self.decParams.getEvalFrequenciesForLevel(declevel)
    #     for eIdx, eFreq in enumerate(evalFreq):
    #         indices.update(self.getWindowsForFreq(declevel, eIdx))
    #     return indices

    # def getDatetimeConstraints(self) -> Dict:
    #     """Get the datetime constraints

    #     Returns
    #     -------
    #     Dict
    #         Dictionary of datetime constraints at all decimation levels
    #     """
    #     self.calcDatetimeConstraints()
    #     return self.datetimeConstraints

    # def getLevelDatetimeConstraints(self, declevel: int) -> List[List[datetime]]:
    #     """Get the datetime constraints for a decimation level

    #     Returns
    #     -------
    #     List[List[datetime]]
    #         Returns a list of datetime constraints, where each is a 2 element list with a start and stop
    #     """
    #     self.calcDatetimeConstraints()
    #     return self.datetimeConstraints[declevel]

    # def getMasks(self) -> Dict:
    #     """Get a dictionary with masks to use for each site in the window selector

    #     Returns
    #     -------
    #     Dict
    #         Dictionary with masks to use for each site in the window selector
    #     """
    #     return self.siteMasks

    # def getSpecReaderForWindow(self, site: str, declevel: int, iWin: int):
    #     """Get the spectrum reader for a window

    #     Parameters
    #     ----------
    #     site : str
    #         The name of the site to get the spectrum reader for
    #     declevel : int
    #         The decimation level
    #     iWin : int
    #         The window index

    #     Returns
    #     -------
    #     specFile : str, bool
    #         The name of the spectra file or False if the window is not found in any spectra file
    #     specReader : SpectrumReader, bool
    #         The spectrum reader or False if the window is not found in any spectra file
    #     """
    #     specRanges = self.siteSpecRanges[site][declevel]
    #     specReaders = self.siteSpecReaders[site][declevel]
    #     for specFile in specRanges:
    #         if iWin >= specRanges[specFile][0] and iWin <= specRanges[specFile][1]:
    #             return specFile, specReaders[specFile]

    #     # if here, no window found
    #     self.printWarning(
    #         "Shared window {}, decimation level {} does not appear in any files given the constraints applied".format(
    #             iWin, declevel
    #         )
    #     )
    #     return False, False

    # def getSpecReaderBatches(self, declevel: int) -> Tuple[List, List]:
    #     """Batch the readers into wndow groups that can be read as required in a more efficient way

    #     First sorts all the site spectra files by global window range, putting them in ascending order. Loops over all the sites and constructs spectra batches to read in.

    #     Parameters
    #     ----------
    #     declevel : int
    #         The decimation level

    #     Returns
    #     -------
    #     batches : List[Dict[str, All]]
    #         The list of batches
    #     """
    #     # sort spectra files for the decimation level
    #     files: Dict[str, List[str]] = dict()
    #     readers: Dict[str, List[SpectrumReader]] = dict()
    #     winStarts: Dict[str, List[int]] = dict()
    #     winStops: Dict[str, List[int]] = dict()
    #     for site in self.sites:
    #         tmpFiles: List = list()
    #         tmpReaders: List = list()
    #         tmpStarts: List = list()
    #         tmpStops: List = list()
    #         for specFile, specRange in self.siteSpecRanges[site][declevel].items():
    #             tmpFiles.append(specFile)
    #             tmpReaders.append(self.siteSpecReaders[site][declevel][specFile])
    #             tmpStarts.append(specRange[0])
    #             tmpStops.append(specRange[1])
    #         # now sort on winstarts using zip
    #         zipped = list(zip(tmpStarts, tmpFiles, tmpReaders, tmpStops))
    #         zipped.sort()
    #         winStarts[site], files[site], readers[site], winStops[site] = zip(*zipped)

    #     # create batches
    #     mainSite: str = self.sites[0]
    #     otherSites: List[str] = self.sites[1:]
    #     # for saving batches
    #     batches: List[Dict[str, Any]] = list()
    #     for mainIdx, mainStart in enumerate(winStarts[mainSite]):
    #         lastBatchWin = mainStart - 1
    #         while True:
    #             batch = dict()
    #             batch["globalrange"] = [lastBatchWin + 1, winStops[mainSite][mainIdx]]
    #             batch[mainSite] = readers[mainSite][mainIdx]
    #             for site in otherSites:
    #                 for otherIdx, (start, stop) in enumerate(
    #                     zip(winStarts[site], winStops[site])
    #                 ):
    #                     if (
    #                         start >= batch["globalrange"][1]
    #                         or stop <= batch["globalrange"][0]
    #                     ):
    #                         continue
    #                     # else there is an overlap and it is the first overlap we are interested in
    #                     # amend the batch range as required
    #                     if start > batch["globalrange"][0]:
    #                         batch["globalrange"][0] = start
    #                     if stop < batch["globalrange"][1]:
    #                         batch["globalrange"][1] = stop
    #                     batch[site] = readers[site][otherIdx]
    #                     break

    #             # now need to decide if the batch is complete - the batch dict has an entry for every site and one globalrange
    #             if len(batch.keys()) == len(self.sites) + 1:
    #                 batches.append(batch)
    #             lastBatchWin = batch["globalrange"][1]
    #             if lastBatchWin >= winStops[mainSite][mainIdx]:
    #                 break
    #             # otherwise, continue in the while loop batching up

    #     # print information
    #     self.printText("Spectra batches")
    #     for batchI, batch in enumerate(batches):
    #         self.printText(
    #             "SPECTRA BATCH {}, Window Range {}".format(batchI, batch["globalrange"])
    #         )
    #         for site in self.sites:
    #             self.printText(
    #                 "SPECTRA BATCH {}, Site {}, Path {}, Window Range {}".format(
    #                     batchI, site, batch[site].datapath, batch[site].getGlobalRange()
    #                 )
    #             )
    #     return batches

    # def getDataSize(self, declevel: int) -> int:
    #     """Get the spectrum reader for a window

    #     Parameters
    #     ----------
    #     declevel : str
    #         The decimation level

    #     Returns
    #     -------
    #     int
    #         The data size (number of points in the spectrum) at the decimation level
    #     """
    #     # return data size of first file
    #     site = self.sites[0]
    #     specReaders = self.siteSpecReaders[site][declevel]
    #     for sF in specReaders:
    #         return specReaders[sF].getDataSize()

    # def setSites(self, sites: List[str]) -> None:
    #     """Set the sites for which to find the shared windows

    #     Parameters
    #     ----------
    #     sites : List[str]
    #         List of sites
    #     """
    #     # first remove repeated sites
    #     sitesSet = set(sites)
    #     sites = list(sitesSet)
    #     # now continue
    #     self.sites = sites
    #     for s in self.sites:
    #         self.siteMasks[s] = []
    #         self.siteSpecFolders[s] = []
    #         self.siteSpecReaders[s] = {}
    #         self.siteSpecRanges[s] = {}
    #         # use sets to hold gIndices
    #         # optimised to find intersections
    #         self.siteGlobalIndices[s] = {}
    #     # at the end, calculate global indices
    #     self.calcGlobalIndices()

    # def addDatetimeConstraint(
    #     self, start: str, stop: str, declevel: Union[List[int], int, None] = None
    # ):
    #     """Add datetime constraints

    #     Parameters
    #     ----------
    #     start : str
    #         Datetime constraint start in format %Y-%m-%d %H:%M:%S
    #     stop : str
    #         Datetime constraint end in format %Y-%m-%d %H:%M:%S
    #     declevel : List[int], int, optional
    #         The decimation level. If left as default, will be applied to all decimation levels.
    #     """
    #     datetimeStart = datetime.strptime(start, "%Y-%m-%d %H:%M:%S")
    #     datetimeStop = datetime.strptime(stop, "%Y-%m-%d %H:%M:%S")

    #     # levels the constraint applies to
    #     if declevel is None:
    #         levels = range(0, self.decParams.numLevels)
    #     elif isinstance(declevel, list):
    #         levels = declevel
    #     else:
    #         levels = [declevel]
    #     # then add constraints as appropriate
    #     for declevel in levels:
    #         self.datetimes[declevel].append([datetimeStart, datetimeStop])

    # def addDateConstraint(
    #     self, dateC: str, declevel: Union[List[int], int, None] = None
    # ):
    #     """Add a date constraint

    #     Parameters
    #     ----------
    #     dateC : str
    #         Datetime constraint in format %Y-%m-%d
    #     declevel : List[int], int, optional
    #         The decimation level. If left as default, will be applied to all decimation levels.
    #     """
    #     datetimeC = datetime.strptime(dateC, "%Y-%m-%d").date()

    #     # levels the constraint applies to
    #     if declevel is None:
    #         levels = range(0, self.decParams.numLevels)
    #     elif isinstance(declevel, list):
    #         levels = declevel
    #     else:
    #         levels = [declevel]
    #     # then add constraints as appropriate
    #     for declevel in levels:
    #         self.dates[declevel].append(datetimeC)

    # def addTimeConstraint(
    #     self, start: str, stop: str, declevel: Union[List[int], int, None] = None
    # ):
    #     """Add a time constraint. This will recur on every day of recording.

    #     Parameters
    #     ----------
    #     start : str
    #         Time constraint start in format %H:%M:%S
    #     stop : str
    #         Time constraint end in format %H:%M:%S
    #     declevel : List[int], int, optional
    #         The decimation level. If left as default, will be applied to all decimation levels.
    #     """
    #     timeStart = datetime.strptime(start, "%H:%M:%S").time()
    #     timeStop = datetime.strptime(stop, "%H:%M:%S").time()

    #     # levels the constraint applies to
    #     if declevel is None:
    #         levels = range(0, self.decParams.numLevels)
    #     elif isinstance(declevel, list):
    #         levels = declevel
    #     else:
    #         levels = [declevel]
    #     # then add constraints as appropriate
    #     for declevel in levels:
    #         self.times[declevel].append([timeStart, timeStop])

    # def addWindowMask(self, site: str, maskName: str) -> None:
    #     """Add a window mask

    #     This is a mask with values for each evaluation frequency.

    #     Parameters
    #     ----------
    #     site : str
    #         The site for which to search for a mask
    #     maskName : str
    #         The name of the mask
    #     """
    #     siteData = self.projData.getSiteData(site)
    #     maskIO = MaskIO(siteData.getSpecdirMaskPath(self.specdir))
    #     maskData = maskIO.read(maskName, self.sampleFreq)
    #     self.siteMasks[site].append(maskData)

    # def resetDatetimeConstraints(self) -> None:
    #     """Reset datetime constraints"""
    #     # add a list for each decimation level
    #     for declevel in range(0, self.decParams.numLevels):
    #         self.datetimes[declevel] = []
    #         self.dates[declevel] = []
    #         self.times[declevel] = []
    #         self.datetimeConstraints[declevel] = []

    # def resetMasks(self) -> None:
    #     """Reset masks"""
    #     # reset to no masks for any site
    #     for site in self.siteMasks:
    #         self.siteMasks[site] = []

    # def calcSharedWindows(self):
    #     """Calculate shared windows between sites

    #     Calculates the shared windows between sites. Datetime constraints are applied. No masks are applied in this method. Masks are only applied when getting the windows for a particular evaluation frequency.
    #     """
    #     if len(self.sites) == 0:
    #         self.printWarning(
    #             "No sites given to Window Selector. At least one site needs to be given."
    #         )
    #         return False

    #     # calculate datetime constraints
    #     self.calcDatetimeConstraints()
    #     # initialise the sharedWindows with a set from one site
    #     sites = self.sites
    #     siteInit = sites[0]
    #     numLevels = self.decParams.numLevels
    #     for declevel in range(0, numLevels):
    #         self.sharedWindows[declevel] = self.siteGlobalIndices[siteInit][declevel]

    #     # now for each decimation level, calculate the shared windows
    #     for declevel in range(0, numLevels):
    #         for site in self.sites:
    #             self.sharedWindows[declevel] = self.sharedWindows[
    #                 declevel
    #             ].intersection(self.siteGlobalIndices[site][declevel])

    #     # apply time constraints
    #     # time constraints should be formulated as a set
    #     # and then, find the intersection again
    #     for declevel in range(0, numLevels):
    #         constraints = self.getLevelDatetimeConstraints(declevel)
    #         if len(constraints) != 0:
    #             datetimeIndices = set()
    #             for dC in constraints:
    #                 gIndexStart, firstWindowStart = datetime2gIndex(
    #                     self.projData.refTime,
    #                     dC[0],
    #                     self.decParams.getSampleFreqLevel(declevel),
    #                     self.winParams.getWindowSize(declevel),
    #                     self.winParams.getOverlap(declevel),
    #                 )
    #                 gIndexEnd, firstWindowEnd = datetime2gIndex(
    #                     self.projData.refTime,
    #                     dC[1],
    #                     self.decParams.getSampleFreqLevel(declevel),
    #                     self.winParams.getWindowSize(declevel),
    #                     self.winParams.getOverlap(declevel),
    #                 )
    #                 gIndexEnd = (
    #                     gIndexEnd - 1
    #                 )  # as the function returns the next window starting after time
    #                 if gIndexEnd < gIndexStart:
    #                     gIndexEnd = gIndexStart
    #                 datetimeIndices.update(list(range(gIndexStart, gIndexEnd)))
    #                 self.printText(
    #                     "Decimation level = {}. Applying date constraint {} - {}, global index constraint {} - {}".format(
    #                         declevel, dC[0], dC[1], gIndexStart, gIndexEnd
    #                     )
    #                 )
    #             self.sharedWindows[declevel] = self.sharedWindows[
    #                 declevel
    #             ].intersection(datetimeIndices)

    # def calcGlobalIndices(self) -> None:
    #     """Find all the global indices for the sites"""
    #     # get all the spectra files with the correct sampling frequency
    #     for site in self.sites:
    #         siteData = self.projData.getSiteData(site)
    #         timeFilesFs = siteData.getMeasurements(self.sampleFreq)
    #         # specFiles = self.proj.getSiteSpectraFiles(s)
    #         specFolders = siteData.spectra
    #         specFoldersFs = []
    #         for specFolder in specFolders:
    #             if specFolder in timeFilesFs:
    #                 specFoldersFs.append(specFolder)

    #         self.siteSpecFolders[site] = specFoldersFs

    #         # for each decimation level
    #         # loop through each of the spectra folders
    #         # and find the global indices ranges for each decimation level
    #         numLevels = self.decParams.numLevels
    #         for declevel in range(0, numLevels):
    #             # get the dictionaries ready
    #             self.siteSpecReaders[site][declevel] = {}
    #             self.siteSpecRanges[site][declevel] = {}
    #             self.siteGlobalIndices[site][declevel] = set()
    #             # loop through spectra folders and figure out global indices
    #             for specFolder in self.siteSpecFolders[site]:
    #                 # here, have to use the specdir option
    #                 specReader = SpectrumReader(
    #                     os.path.join(siteData.specPath, specFolder, self.specdir)
    #                 )
    #                 # here, use prepend to open the spectra file
    #                 check = specReader.openBinaryForReading(self.prepend, declevel)
    #                 # if file does not exist, continue
    #                 if not check:
    #                     continue
    #                 self.siteSpecReaders[site][declevel][specFolder] = specReader
    #                 globalRange = specReader.getGlobalRange()
    #                 self.siteSpecRanges[site][declevel][specFolder] = globalRange
    #                 # and save set of global indices
    #                 self.siteGlobalIndices[site][declevel].update(
    #                     list(range(globalRange[0], globalRange[1] + 1))
    #                 )

    # def calcDatetimeConstraints(self) -> None:
    #     """Calculate overall datetime constraints

    #     Priority order for datetime constraints is:
    #     1. datetime constraints
    #     2. date constraints
    #     3. time constraints
    #     """
    #     # calculate site dates if required
    #     siteDates = self.calcSiteDates()

    #     # datetime constraints are for each decimation level
    #     numLevels = self.decParams.numLevels
    #     for declevel in range(0, numLevels):
    #         # calculate date and time constraints for each level
    #         # begin with the datetime constraints - these have highest priority
    #         self.datetimeConstraints[declevel] = self.datetimes[declevel]

    #         # check to see whether any date and time constraints
    #         if len(self.dates[declevel]) == 0 and len(self.times[declevel]) == 0:
    #             continue

    #         dateConstraints = []
    #         if len(self.dates[declevel]) != 0:
    #             # apply time constraints only on specified days
    #             dateConstraints = self.dates[declevel]
    #         else:
    #             dateConstraints = siteDates

    #         # finally, add the time constraints to the dates
    #         # otherwise add the whole day
    #         dateAndTimeConstraints = []
    #         if len(self.times[declevel]) == 0:
    #             # add whole days
    #             for dC in dateConstraints:
    #                 start = datetime.combine(dC, time(0, 0, 0))
    #                 stop = datetime.combine(dC, time(23, 59, 59))
    #                 dateAndTimeConstraints.append([start, stop])
    #         else:
    #             # add each time for each day
    #             for tC in self.times[declevel]:
    #                 for dC in dateConstraints:
    #                     start = datetime.combine(dC, tC[0])
    #                     stop = datetime.combine(dC, tC[1])
    #                     # check if this goes over a day
    #                     if tC[1] < tC[0]:
    #                         # then need to increment the day
    #                         dCNext = dC + timedelta(days=1)
    #                         stop = datetime.combine(dCNext, tC[1])
    #                     # append to constraints
    #                     dateAndTimeConstraints.append([start, stop])

    #         # finally, combine datetimes and dateAndTimeConstraints
    #         self.datetimeConstraints[declevel] = (
    #             self.datetimeConstraints[declevel] + dateAndTimeConstraints
    #         )
    #         self.datetimeConstraints[declevel] = sorted(
    #             self.datetimeConstraints[declevel]
    #         )

    # def calcSiteDates(self) -> List[datetime]:
    #     """Calculate a list of days that all the sites were operating

    #     This uses the siteStart and siteEnd datetimes, so does not take into account the start and end of actual time series measurements, which is taken into account later.

    #     Returns
    #     -------
    #     List[datetime]
    #         A list of dates all the sites were operating
    #     """
    #     starts = []
    #     stops = []
    #     for site in self.sites:
    #         siteData = self.projData.getSiteData(site)
    #         starts.append(siteData.siteStart)
    #         stops.append(siteData.siteEnd)
    #     # need all the dates between
    #     d1 = max(starts).date()
    #     d2 = min(stops).date()
    #     if d1 > d2:
    #         self.printError(
    #             "A site passed to the window selector does not overlap with any other sites. There will be no shared windows",
    #             quitrun=True,
    #         )
    #     # now with d2 > d1
    #     siteDates = []
    #     delta = d2 - d1
    #     # + 1 because inclusive of stop and start days
    #     for i in range(delta.days + 1):
    #         siteDates.append(d1 + timedelta(days=i))
    #     return siteDates

    # def printList(self) -> List[str]:
    #     """Class information as a list of strings

    #     Returns
    #     -------
    #     out : list
    #         List of strings with information
    #     """
    #     textLst = []
    #     textLst.append("Sampling frequency [Hz] = {:.6f}".format(self.sampleFreq))
    #     textLst.append("Spectra directory = {}".format(self.specdir))
    #     textLst.append("Sites = {}".format(", ".join(self.sites)))
    #     textLst.append("Site information:")
    #     for site in self.sites:
    #         textLst = textLst + self.printSiteInfoList(site)
    #     return textLst

    # def printSiteInfo(self):
    #     """Print out information about the sites"""
    #     for site in self.sites:
    #         blockPrint("WindowSelector::site info", self.printSiteInfoList(site))

    # def printSiteInfoList(self, site: str) -> List[str]:
    #     """Return site window information as a list of strings

    #     Parameters
    #     ----------
    #     site : str
    #         The site name

    #     Returns
    #     -------
    #     List[str]
    #         Site window information as a list of strings
    #     """
    #     textLst = []
    #     textLst.append("Sampling frequency [Hz] = {:.6f}".format(self.sampleFreq))
    #     textLst.append("Site = {}".format(site))
    #     textLst.append("Site global index information")
    #     numLevels = self.decParams.numLevels
    #     for declevel in range(0, numLevels):
    #         textLst.append("Decimation Level = {:d}".format(declevel))
    #         ranges = self.siteSpecRanges
    #         for sF in sorted(list(ranges[site][declevel].keys())):
    #             startTime1, endTime1 = gIndex2datetime(
    #                 ranges[site][declevel][sF][0],
    #                 self.projData.refTime,
    #                 self.sampleFreq / self.decParams.getDecFactor(declevel),
    #                 self.winParams.getWindowSize(declevel),
    #                 self.winParams.getOverlap(declevel),
    #             )
    #             startTime2, endTime2 = gIndex2datetime(
    #                 ranges[site][declevel][sF][1],
    #                 self.projData.refTime,
    #                 self.sampleFreq / self.decParams.getDecFactor(declevel),
    #                 self.winParams.getWindowSize(declevel),
    #                 self.winParams.getOverlap(declevel),
    #             )
    #             textLst.append(
    #                 "Measurement file = {}\ttime range = {} - {}\tGlobal Indices Range = {:d} - {:d}".format(
    #                     sF,
    #                     startTime1,
    #                     endTime2,
    #                     ranges[site][declevel][sF][0],
    #                     ranges[site][declevel][sF][1],
    #                 )
    #             )
    #     return textLst

    # def printSharedWindows(self) -> None:
    #     """Print out the shared windows"""
    #     blockPrint("WindowSelector::shared windows", self.printSharedWindowsList())

    # def printSharedWindowsList(self) -> List[str]:
    #     """Shared window information as a list of strings

    #     Returns
    #     -------
    #     List[str]
    #         Shared window information as a list of strings
    #     """
    #     textLst = []
    #     numLevels = self.decParams.numLevels
    #     for declevel in range(0, numLevels):
    #         textLst.append("Decimation Level = {:d}".format(declevel))
    #         textLst.append(
    #             "\tNumber of shared windows = {:d}".format(
    #                 self.getNumSharedWindows(declevel)
    #             )
    #         )
    #         textLst.append(
    #             "\tShared window indices: {}".format(
    #                 list2ranges(self.getSharedWindowsLevel(declevel))
    #             )
    #         )
    #         textLst.append(
    #             "\tNumber of unmasked windows: {}".format(
    #                 len(self.getUnmaskedWindowsLevel(declevel))
    #             )
    #         )
    #     textLst.append(
    #         "NOTE: These are the shared windows at each decimation level. Windows for each evaluation frequency might vary depending on masks"
    #     )
    #     return textLst

    # def printDatetimeConstraints(self):
    #     """Print out the datetime constraints"""
    #     blockPrint(
    #         "WindowSelector::datetime constraints", self.printDatetimeConstraintsList()
    #     )

    # def printDatetimeConstraintsList(self) -> List[str]:
    #     """Datetime constraint information as a list of strings

    #     Returns
    #     -------
    #     List[str]
    #         Datetime constraint information as a list of strings
    #     """
    #     textLst = []
    #     # calculate datetime constraints
    #     self.calcDatetimeConstraints()
    #     # populate textLst
    #     textLst.append("Datetime constraints")
    #     numLevels = self.decParams.numLevels
    #     for declevel in range(0, numLevels):
    #         textLst.append("Decimation Level = {:d}".format(declevel))
    #         for d in self.getLevelDatetimeConstraints(declevel):
    #             textLst.append("\tConstraint {} - {}".format(d[0], d[1]))
    #     return textLst

    # def printWindowMasks(self) -> None:
    #     """Print mask information"""
    #     blockPrint("WindowSelector::window masks", self.printWindowMasksList())

    # def printWindowMasksList(self) -> List[str]:
    #     """Window mask information as a list of strings

    #     Returns
    #     -------
    #     List[str]
    #         Window mask information as a list of strings
    #     """
    #     textLst = []
    #     for s in self.sites:
    #         textLst.append("Site = {}".format(s))
    #         if len(self.siteMasks[s]) == 0:
    #             textLst.append("\t\tNo masks for this site")
    #         else:
    #             for mask in self.siteMasks[s]:
    #                 textLst.append("\t\tMask = {}".format(mask.maskName))
    #     return textLst

    # def printWindowsForFrequency(self, listwindows=False):
    #     """Print information about the windows for each evaluation frequency

    #     Parameters
    #     ----------
    #     listwindows : bool
    #         Boolean flag to actually write out all the windows. Default is False as this takes up a lot of space in the terminal
    #     """
    #     blockPrint(
    #         "WindowSelector::windows for frequency",
    #         self.printWindowsForFrequencyList(listwindows),
    #     )

    # def printWindowsForFrequencyList(self, listwindows=False) -> List[str]:
    #     """Information about windows for each evaluation frequency as a list of strings

    #     Parameters
    #     ----------
    #     listwindows : bool
    #         Boolean flag to actually write out all the windows. Default is False as this takes up a lot of space in the terminal

    #     Returns
    #     -------
    #     List[str]
    #         Windows for evaluation frequency information as a list of strings
    #     """
    #     textLst = []
    #     for declevel in range(0, self.decParams.numLevels):
    #         evalFreq = self.decParams.getEvalFrequenciesForLevel(declevel)
    #         unmaskedWindows = self.getNumSharedWindows(declevel)
    #         for eIdx, eFreq in enumerate(evalFreq):
    #             maskedWindows = self.getWindowsForFreq(declevel, eIdx)
    #             textLst.append(
    #                 "Evaluation frequency = {:.6f}, shared windows = {:d}, windows after masking = {:d}".format(
    #                     eFreq, unmaskedWindows, len(maskedWindows)
    #                 )
    #             )
    #             if listwindows:
    #                 textLst.append("{}".format(list2ranges(maskedWindows)))
    #     return textLst
