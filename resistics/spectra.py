"""
Module containing functions and classes related to Spectra calculation and
manipulation
"""
from resistics.decimate import DecimationParameters
from loguru import logger
from typing import Optional, Union, Tuple, Dict, List
import numpy as np

from resistics.common import ResisticsData, ResisticsProcess, ProcessHistory
from resistics.common import MetadataGroup
from resistics.window import WindowedTimeData, WindowedDecimatedData


class SpectraTimeData(ResisticsData):
    def __init__(
        self,
        metadata: MetadataGroup,
        chans: List[str],
        freqs: np.ndarray,
        spec: np.ndarray,
    ):
        """Initialise"""
        self.metadata = metadata
        self.chans = chans
        self.freqs = freqs
        self.spec = spec
        self.n_wins = spec.shape[0]
        self.size = spec.shape[-1]

    @property
    def fs(self):
        return self.metadata["common", "fs"]

    @property
    def n_chans(self):
        return len(self.chans)

    @property
    def n_freqs(self):
        return len(self.freqs)

    @property
    def nyquist(self) -> float:
        return self.fs / 2

    def get_chan(self, chan: str) -> np.ndarray:
        from resistics.errors import ChannelNotFoundError

        if chan not in self.chans:
            raise ChannelNotFoundError(chan, self.chans)
        idx = self.chans.index(chan)
        return self.spec[..., idx, :]

    def get_chans(self, chans: List[str]) -> np.ndarray:
        from resistics.errors import ChannelNotFoundError

        for chan in chans:
            if chan not in self.chans:
                raise ChannelNotFoundError(chan, self.chans)
        indices = [self.chans.index(chan) for chan in chans]
        return self.spec[..., indices, :]

    def get_freq(self, idx: int) -> np.ndarray:
        if idx < 0 or idx >= self.n_freqs:
            raise ValueError(f"Freq. index {idx} not 0 <= idx < {self.n_freqs}")
        return np.squeeze(self.spec[..., idx])

    def get_mag_phs(self, unwrap: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        if unwrap:
            return np.absolute(self.spec), np.unwrap(np.angle(self.spec))
        return np.absolute(self.spec), np.angle(self.spec)

    def plot(self):
        pass

    def stack(self):
        pass

    def section(self):
        pass


class TimeDataFFT(ResisticsProcess):
    """
    The processor is inspired by the scipy.signal.stft function which performs
    a similar process.
    """

    def __init__(
        self,
        win_fnc: Optional[Union[str, Tuple[str, float]]] = None,
        detrend: Optional[str] = "linear",
    ):
        """Initialise"""
        if win_fnc is None:
            self.win_fnc = ("kaiser", 14)
        else:
            self.win_fnc = win_fnc
        self.detrend = detrend

    def parameters(self):
        """Get the processor parameters"""
        from resistics.common import serialize

        params = {"detrend": self.detrend}
        if isinstance(self.win_fnc, tuple):
            params["win_fnc"] = self.win_fnc[0]
            params["win_fnc_arg"] = serialize(self.win_fnc[1])
        else:
            params["win_fnc"] = self.win_fnc
        return params

    def run(self, win_data: WindowedTimeData) -> SpectraTimeData:
        """Run the spectra calculation"""
        from resistics.math import get_freqs
        from scipy import signal
        from scipy.fft import rfft

        logger.info("Performing fourier transforms of windowed data")
        data = win_data.win_views
        if self.detrend is not None:
            data = signal.detrend(data, axis=-1, type=self.detrend)
        # apply window
        win_coeffs = self._get_window(win_data.win_size)
        data = data * win_coeffs
        # perform the fft on the last axis
        result = rfft(data, n=win_data.win_size, axis=-1, norm="ortho")
        freqs = get_freqs(win_data.fs, result.shape[-1])
        logger.info("Fourier transform completed")
        return SpectraTimeData(win_data.metadata, win_data.chans, freqs, result)

    def _get_window(self, win_size: int):
        """Get the window to apply to the data"""
        from scipy.signal import get_window
        from scipy.signal.windows import dpss

        if self.win_fnc == "dpss":
            return dpss(win_size, 5)
        return get_window(self.win_fnc, win_size)


class SpectraDecimatedData(ResisticsData):
    def __init__(
        self,
        chans: List[str],
        decspec: Dict[int, SpectraTimeData],
        history: ProcessHistory,
    ):
        self.chans = chans
        self.decspec = decspec
        self.history = history
        self.max_level = max(list(self.decspec.keys()))
        self.n_levels = self.max_level + 1

    @property
    def n_chans(self):
        return len(self.chans)

    def get_level(self, level: int) -> SpectraTimeData:
        if level > self.max_level:
            raise ValueError(f"Level {level} not <= max {self.max_level}")
        return self.decspec[level]


class DecimatedDataFFT(TimeDataFFT):
    def run(self, decwin_data: WindowedDecimatedData) -> SpectraDecimatedData:
        decspec = {}
        messages = []
        logger.info("Performing fourier transforms of windowed decimated data")
        for ilevel in range(decwin_data.n_levels):
            messages.append(f"Calculating spectra for level {ilevel}")
            decspec[ilevel] = super().run(decwin_data.get_level(ilevel))
        history = decwin_data.history.copy()
        history.add_record(self._get_process_record(messages))
        logger.info("Fourier transforms completed")
        return SpectraDecimatedData(decwin_data.chans, decspec, history)


class SpectraSmoother(ResisticsProcess):
    def __init__(self, sigma: int = 1):
        self.sigma = sigma

    def run(self, decspec_data: SpectraDecimatedData) -> SpectraDecimatedData:
        from scipy.ndimage import gaussian_filter1d

        smoothspec = {}
        messages = []
        for ilevel in range(decspec_data.n_levels):
            spec_data = decspec_data.get_level(ilevel)
            mag, phs = spec_data.get_mag_phs(unwrap=True)
            new_mag = gaussian_filter1d(mag, self.sigma, axis=-1)
            new_phs = gaussian_filter1d(phs, self.sigma, axis=-1)
            new_phs = (new_phs + np.pi) % (2 * np.pi) - np.pi
            data = new_mag * np.exp(1j * new_phs)
            smoothspec[ilevel] = SpectraTimeData(
                spec_data.metadata, spec_data.chans, spec_data.freqs, data
            )

            # import matplotlib.pyplot as plt

            # spec_data = decspec_data.get_level(ilevel)
            # n_chans = data.shape[-2]
            # print(n_chans)
            # window = 0
            # plt.figure(figsize=(12,20))
            # share_ax = None
            # for idx in range(0, n_chans):
            #     if idx == 0:
            #         share_ax = plt.subplot(n_chans, 2, 2*idx + 1)
            #     else:
            #         ax = plt.subplot(n_chans, 2, 2*idx + 1, sharex=share_ax)
            #     plt.semilogy(freqs, mag[window, idx], "bo-", label=f"FFT")
            #     plt.semilogy(freqs, new_mag[window, idx], "rs", label=f"Smooth")
            #     plt.title(f"{idx}")
            #     plt.legend(loc=1)
            #     ax = plt.subplot(n_chans, 2, 2*(idx + 1), sharex=share_ax)
            #     plt.plot(freqs, phs[window, idx], "bo-", label=f"FFT")
            #     plt.plot(freqs, new_phs[window, idx], "rs", label=f"Smooth")
            #     plt.title(f"{idx}")
            #     plt.legend(loc=1)
            # plt.tight_layout()
            # plt.show()

        messages.append("Spectra smooth using a 1-D spline")
        history = decspec_data.history.copy()
        history.add_record(self._get_process_record(messages))
        logger.info("Smoothing completed")
        return SpectraDecimatedData(decspec_data.chans, smoothspec, history)


class EvaluationFreqs(ResisticsProcess):
    def __init__(self, sigma: int = 3):
        self.win_fnc = "parzen"
        self.win_len = 11

    def run(
        self, dec_params: DecimationParameters, decspec_data: SpectraDecimatedData
    ) -> SpectraDecimatedData:
        evalspec = {}
        messages = []
        for ilevel in range(decspec_data.n_levels):
            spec_data = decspec_data.get_level(ilevel)
            freqs = spec_data.freqs
            index = np.arange(len(freqs))
            eval_freqs = dec_params.get_eval_freqs(ilevel).values
            # print(freqs)
            # print(eval_freqs)
            eval_index = np.round(np.interp(eval_freqs, freqs, index)).astype(int)
            eval_freqs = freqs[eval_index]
            eval_data = spec_data.spec[..., eval_index]
            # index_low = np.floor(eval_index).astype(int)
            # index_high = np.ceil(eval_index).astype(int)
            # index_off = eval_index - index_low
            # print(eval_index)
            # print(index_low)
            # print(index_high)
            # print(eval_index - index_low)
            # mag, phs = spec_data.get_mag_phs(unwrap=True)
            # low_mag = mag[..., index_low]
            # high_mag = mag[..., index_high]
            # low_phs = phs[..., index_low]
            # high_phs = phs[..., index_high]
            # new_mag = low_mag + (high_mag - low_mag) * index_off
            # new_phs = low_phs + (high_phs - low_phs) * index_off
            # eval_data = new_mag * np.exp(1j * new_phs)
            evalspec[ilevel] = SpectraTimeData(
                spec_data.metadata, spec_data.chans, eval_freqs, eval_data
            )

            # debug plot
            # import matplotlib.pyplot as plt

            # n_chans = eval_data.shape[-2]
            # window = 0
            # plt.figure(figsize=(12, 20))
            # share_ax = None
            # for idx in range(0, n_chans):
            #     if idx == 0:
            #         share_ax = plt.subplot(n_chans, 2, 2 * idx + 1)
            #     else:
            #         ax = plt.subplot(n_chans, 2, 2 * idx + 1, sharex=share_ax)
            #     plt.semilogy(
            #         freqs, np.absolute(spec_data.spec[window, idx]), "bo-", label=f"FFT"
            #     )
            #     plt.semilogy(
            #         eval_freqs, np.absolute(eval_data[window, idx]), "rs", label=f"Eval"
            #     )
            #     plt.title(f"{idx}")
            #     plt.legend(loc=1)
            #     ax = plt.subplot(n_chans, 2, 2 * (idx + 1), sharex=share_ax)
            #     plt.plot(
            #         freqs,
            #         np.unwrap(np.angle(spec_data.spec[window, idx])),
            #         "bo-",
            #         label=f"FFT",
            #     )
            #     plt.plot(
            #         eval_freqs,
            #         np.unwrap(np.angle(eval_data[window, idx])),
            #         "rs",
            #         label=f"Eval",
            #     )
            #     plt.title(f"{idx}")
            #     plt.legend(loc=1)
            # plt.tight_layout()
            # plt.show()

        messages.append("Spectra interpolated onto evaluation frequencies")
        history = decspec_data.history.copy()
        history.add_record(self._get_process_record(messages))
        logger.info("Fourier coefficients calculated at evaluation frequencies")
        return SpectraDecimatedData(decspec_data.chans, evalspec, history)
