"""
Module containing functions and classes related to Spectra calculation and
manipulation
"""
from loguru import logger
from typing import Union, Tuple, Dict, List, Any, Optional
from pydantic import validator
import numpy as np

from resistics.common import ResisticsData, ResisticsProcess, History
from resistics.common import Metadata, WriteableMetadata
from resistics.decimate import DecimationParameters
from resistics.window import WindowedData, WindowedLevelMetadata


class SpectraLevelMetadata(Metadata):
    """Metadata for spectra of a windowed decimation level"""

    fs: float
    nyquist: Optional[float] = None
    win_size: int
    olap_size: int
    n_freqs: int
    freqs: Optional[List[float]] = None

    @validator("nyquist", always=True)
    def validate_nyquist(cls, value: float, values: Dict[str, Any]) -> float:
        """Validate nyquist frequency"""
        if value is None:
            return values["fs"] / 2
        return value

    @validator("freqs", always=True)
    def validate_freqs(
        cls, value: Union[List[float], None], values: Dict[str, Any]
    ) -> List[float]:
        """Validate frequencies list"""
        if value is None:
            return np.linspace(0, values["fs"] / 2, values["n_freqs"]).tolist()
        return value


class SpectraMetadata(WriteableMetadata):
    """Metadata for spectra data"""

    fs: List[float]
    chans: List[str]
    n_chans: Optional[int] = None
    n_levels: int
    system: str = ""
    wgs84_latitude: float = -999.0
    wgs84_longitude: float = -999.0
    easting: float = -999.0
    northing: float = -999.0
    elevation: float = -999.0
    levels_metadata: List[SpectraLevelMetadata] = []
    history: History = History()

    class Config:

        extra = "ignore"


class SpectraData(ResisticsData):
    """Class for holding spectra data"""

    def __init__(self, metadata: SpectraMetadata, data: Dict[int, np.ndarray]):
        self.metadata = metadata
        self.data = data

    def get_level(self, level: int) -> np.ndarray:
        """Get the spectra data for a decimation level"""
        if level >= self.metadata.n_levels:
            raise ValueError(f"Level {level} not <= max {self.metadata.n_levels - 1}")
        return self.data[level]

    def get_chan(self, level: int, chan: str) -> np.ndarray:
        """Get the channel spectra data for a decimation level"""
        from resistics.errors import ChannelNotFoundError

        if chan not in self.metadata.chans:
            raise ChannelNotFoundError(chan, self.metadata.chans)
        idx = self.metadata.chans.index(chan)
        return self.data[level][..., idx, :]

    def get_chans(self, level: int, chans: List[str]) -> np.ndarray:
        """Get the channels spectra data for a decimation level"""
        from resistics.errors import ChannelNotFoundError

        for chan in chans:
            if chan not in self.metadata.chans:
                raise ChannelNotFoundError(chan, self.metadata.chans)
        indices = [self.metadata.chans.index(chan) for chan in chans]
        return self.data[level][..., indices, :]

    def get_freq(self, level: int, idx: int) -> np.ndarray:
        """Get the spectra data at a frequency index for a decimation level"""
        n_freqs = self.metadata.levels_metadata[level].n_freqs
        if idx < 0 or idx >= n_freqs:
            raise ValueError(f"Freq. index {idx} not 0 <= idx < {n_freqs}")
        return np.squeeze(self.data[level][..., idx])

    def get_mag_phs(
        self, level: int, unwrap: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get magnitude and phase for a decimation level"""
        spec = self.data[level]
        if unwrap:
            return np.absolute(spec), np.unwrap(np.angle(spec))
        return np.absolute(spec), np.angle(spec)

    def stack(self):
        pass

    def section(self):
        pass


class FourierTransform(ResisticsProcess):
    """
    The processor is inspired by the scipy.signal.stft function which performs
    a similar process.
    """

    win_fnc: Union[str, Tuple[str, float]] = ("kaiser", 14)
    detrend: str = "linear"

    def run(self, win_data: WindowedData) -> SpectraData:
        metadata_dict = win_data.metadata.dict()
        data = {}
        spectra_levels_metadata = []
        messages = []
        logger.info("Performing fourier transforms of windowed decimated data")
        for ilevel in range(win_data.metadata.n_levels):
            logger.info(f"Transforming level {ilevel}")
            level_metadata = win_data.metadata.levels_metadata[ilevel]
            data[ilevel] = self._get_level_data(
                level_metadata, win_data.get_level(ilevel)
            )
            n_freqs = data[ilevel].shape[-1]
            spectra_levels_metadata.append(
                self._get_level_metadata(level_metadata, n_freqs)
            )
            messages.append(f"Calculated spectra for level {ilevel}")
        metadata = self._get_metadata(metadata_dict, spectra_levels_metadata)
        metadata.history.add_record(self._get_record(messages))
        logger.info("Fourier transforms completed")
        return SpectraData(metadata, data)

    def _get_level_data(
        self, metadata: WindowedLevelMetadata, data: np.ndarray
    ) -> np.ndarray:
        """Run the spectra calculation for a single decimation level"""
        from scipy import signal
        from scipy.fft import rfft

        if self.detrend is not None:
            data = signal.detrend(data, axis=-1, type=self.detrend)
        # apply window
        win_coeffs = self._get_window(metadata.win_size)
        data = data * win_coeffs
        # perform the fft on the last axis
        return rfft(data, n=metadata.win_size, axis=-1, norm="ortho")

    def _get_window(self, win_size: int):
        """Get the window to apply to the data"""
        from scipy.signal import get_window
        from scipy.signal.windows import dpss

        if self.win_fnc == "dpss":
            return dpss(win_size, 5)
        return get_window(self.win_fnc, win_size)

    def _get_level_metadata(
        self, level_metadata: WindowedLevelMetadata, n_freqs: int
    ) -> SpectraLevelMetadata:
        """Get the spectra metadata for a decimation level"""
        return SpectraLevelMetadata(
            fs=level_metadata.fs,
            win_size=level_metadata.win_size,
            olap_size=level_metadata.olap_size,
            n_freqs=n_freqs,
        )

    def _get_metadata(
        self,
        metadata_dict: Dict[str, Any],
        levels_metadata: List[SpectraLevelMetadata],
    ) -> SpectraMetadata:
        """Get the metadata for the windowed data"""
        metadata_dict.pop("file_info")
        metadata_dict["levels_metadata"] = levels_metadata
        return SpectraMetadata(**metadata_dict)


class EvaluationFreqs(ResisticsProcess):
    def run(
        self, dec_params: DecimationParameters, spec_data: SpectraData
    ) -> SpectraData:
        """Get values close to the evaluation frequencies"""
        metadata_dict = spec_data.metadata.dict()
        data = {}
        spectra_levels_metadata = []
        messages = []
        for ilevel in range(spec_data.metadata.n_levels):
            logger.info(f"Reducing freqs to evaluation freqs for level {ilevel}")
            level_metadata = spec_data.metadata.levels_metadata[ilevel]
            freqs = np.array(level_metadata.freqs)
            index = np.arange(level_metadata.n_freqs)
            eval_freqs = dec_params.get_eval_freqs(ilevel)
            eval_index = np.round(np.interp(eval_freqs, freqs, index)).astype(int)
            eval_freqs = freqs[eval_index]
            data[ilevel] = spec_data.get_level(ilevel)[..., eval_index]
            spectra_levels_metadata.append(
                self._get_level_metadata(level_metadata, eval_freqs)
            )
        messages.append("Spectra reduced to evaluation frequencies")
        metadata = self._get_metadata(metadata_dict, spectra_levels_metadata)
        metadata.history.add_record(self._get_record(messages))
        logger.info("Fourier coefficients calculated at evaluation frequencies")
        return SpectraData(metadata, data)

    def _get_level_metadata(
        self, level_metadata: SpectraLevelMetadata, eval_freqs: np.ndarray
    ) -> SpectraLevelMetadata:
        """Get the metadata for the decimation level"""
        metadata_dict = level_metadata.dict()
        metadata_dict["n_freqs"] = len(eval_freqs)
        metadata_dict["freqs"] = eval_freqs.tolist()
        return SpectraLevelMetadata(**metadata_dict)

    def _get_metadata(
        self, metadata_dict: Dict[str, Any], levels_metadata: List[SpectraLevelMetadata]
    ) -> SpectraMetadata:
        """Get metadata for the dataset"""
        metadata_dict.pop("file_info")
        metadata_dict["levels_metadata"] = levels_metadata
        return SpectraMetadata(**metadata_dict)


# class SpectraSmoother(ResisticsProcess):
#     def __init__(self, sigma: int = 1):
#         self.sigma = sigma

#     def run(self, decspec_data: SpectraDecimatedData) -> SpectraDecimatedData:
#         from scipy.ndimage import gaussian_filter1d

#         smoothspec = {}
#         messages = []
#         for ilevel in range(decspec_data.n_levels):
#             spec_data = decspec_data.get_level(ilevel)
#             mag, phs = spec_data.get_mag_phs(unwrap=True)
#             new_mag = gaussian_filter1d(mag, self.sigma, axis=-1)
#             new_phs = gaussian_filter1d(phs, self.sigma, axis=-1)
#             new_phs = (new_phs + np.pi) % (2 * np.pi) - np.pi
#             data = new_mag * np.exp(1j * new_phs)
#             smoothspec[ilevel] = SpectraTimeData(
#                 spec_data.metadata, spec_data.chans, spec_data.freqs, data
#             )

#             # import matplotlib.pyplot as plt

#             # spec_data = decspec_data.get_level(ilevel)
#             # n_chans = data.shape[-2]
#             # print(n_chans)
#             # window = 0
#             # plt.figure(figsize=(12,20))
#             # share_ax = None
#             # for idx in range(0, n_chans):
#             #     if idx == 0:
#             #         share_ax = plt.subplot(n_chans, 2, 2*idx + 1)
#             #     else:
#             #         ax = plt.subplot(n_chans, 2, 2*idx + 1, sharex=share_ax)
#             #     plt.semilogy(freqs, mag[window, idx], "bo-", label=f"FFT")
#             #     plt.semilogy(freqs, new_mag[window, idx], "rs", label=f"Smooth")
#             #     plt.title(f"{idx}")
#             #     plt.legend(loc=1)
#             #     ax = plt.subplot(n_chans, 2, 2*(idx + 1), sharex=share_ax)
#             #     plt.plot(freqs, phs[window, idx], "bo-", label=f"FFT")
#             #     plt.plot(freqs, new_phs[window, idx], "rs", label=f"Smooth")
#             #     plt.title(f"{idx}")
#             #     plt.legend(loc=1)
#             # plt.tight_layout()
#             # plt.show()

#         messages.append("Spectra smooth using a 1-D spline")
#         history = decspec_data.history.copy()
#         history.add_record(self._get_process_record(messages))
#         logger.info("Smoothing completed")
#         return SpectraDecimatedData(decspec_data.chans, smoothspec, history)
