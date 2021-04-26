"""
Module containing functions and classes related to Spectra calculation and
manipulation
"""
from loguru import logger
from typing import Optional, Union, Tuple
import numpy as np

from resistics.common import ResisticsData, ResisticsProcess, ProcessHistory
from resistics.common import MetadataGroup
from resistics.window import WindowedTimeData


class SpectraData(ResisticsData):
    def __init__(
        self, metadata: MetadataGroup, data: np.ndarray, history: ProcessHistory
    ):
        """Initialise"""
        self.metadata = metadata
        self.data = data
        self.history = history


class Spectra(ResisticsProcess):
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

    def run(self, win_data: WindowedTimeData) -> np.ndarray:
        """Run the spectra calculation"""
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
        logger.info("Fourier transform completed")
        return result

    def _get_window(self, win_size: int):
        """Get the window to apply to the data"""
        from scipy.signal import get_window
        from scipy.signal.windows import dpss

        if self.win_fnc == "dpss":
            return dpss(win_size, 5)
        return get_window(self.win_fnc, win_size)
