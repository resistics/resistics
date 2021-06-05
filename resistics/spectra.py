"""
Module containing functions and classes related to Spectra calculation and
manipulation

Spectra are calculated from the windowed, decimated time data. The inbuilt
Fourier transform implementation is inspired by the implementation of the
scipy stft function.
"""
from loguru import logger
from pathlib import Path
from typing import Union, Tuple, Dict, List, Any, Optional
from pydantic import PositiveInt
import numpy as np

from resistics.common import ResisticsData, ResisticsProcess, History
from resistics.common import ResisticsWriter, Metadata, WriteableMetadata
from resistics.time import ChanMetadata
from resistics.decimate import DecimationParameters
from resistics.window import WindowedData, WindowedLevelMetadata


class SpectraLevelMetadata(Metadata):
    """Metadata for spectra of a windowed decimation level"""

    fs: float
    """The sampling frequency of the decimation level"""
    n_wins: int
    """The number of windows"""
    win_size: PositiveInt
    """The window size in samples"""
    olap_size: PositiveInt
    """The overlap size in samples"""
    index_offset: int
    """The global window offset for local window 0"""
    n_freqs: int
    """The number of frequencies in the frequency data"""
    freqs: List[float]
    """List of frequencies"""

    @property
    def nyquist(self) -> float:
        """Get the nyquist frequency"""
        return self.fs / 2


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
    chans_metadata: Dict[str, ChanMetadata]
    levels_metadata: List[SpectraLevelMetadata]
    history: History = History()

    class Config:

        extra = "ignore"


class SpectraData(ResisticsData):
    """
    Class for holding spectra data

    The spectra data is stored in the class as a dictionary mapping decimation
    level to numpy array. The shape of the array for each decimation level is:

    n_wins x n_chans x n_freqs
    """

    def __init__(self, metadata: SpectraMetadata, data: Dict[int, np.ndarray]):
        """
        Initialise spectra data

        Parameters
        ----------
        metadata : SpectraMetadata
            Metadata for the spectra data
        data : Dict[int, np.ndarray]
            Dictionary of data, one entry for each evaluation level
        """
        logger.debug(f"Creating SpectraData with data type {data[0].dtype}")
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

    def plot_stack(self):
        pass

    def plot_section(self):
        pass


class FourierTransform(ResisticsProcess):
    """
    Perform a Fourier transform of the windowed data

    The processor is inspired by the scipy.signal.stft function which performs
    a similar process and involves a Fourier transform along the last axis of
    the windowed data.

    Parameters
    ----------
    win_fnc : Union[str, Tuple[str, float]]
        The window to use before performing the FFT, by default ("kaiser", 14)
    detrend : Union[str, None]
        Type of detrending to apply before performing FFT, by default linear
        detrend. Setting to None will not apply any detrending to the data prior
        to the FFT
    workers : int
        The number of CPUs to use, by default max - 2
    """

    win_fnc: Union[str, Tuple[str, float]] = ("kaiser", 14)
    detrend: Union[str, None] = "linear"
    workers: int = -2

    def run(self, win_data: WindowedData) -> SpectraData:
        """
        Perform the FFT

        Data is padded to the next fast length before performing the FFT to
        speed up processing. Therefore, the output length may not be as
        expected.

        Parameters
        ----------
        win_data : WindowedData
            The input windowed data

        Returns
        -------
        SpectraData
            The Fourier transformed output
        """
        from scipy.fft import next_fast_len, rfftfreq

        metadata_dict = win_data.metadata.dict()
        data = {}
        spectra_levels_metadata = []
        messages = []
        logger.info("Performing fourier transforms of windowed decimated data")
        for ilevel in range(win_data.metadata.n_levels):
            logger.info(f"Transforming level {ilevel}")
            level_metadata = win_data.metadata.levels_metadata[ilevel]
            win_size = level_metadata.win_size
            n_transform = next_fast_len(win_size, real=True)
            logger.debug(f"Padding size {win_size} to next fast len {n_transform}")
            freqs = rfftfreq(n=n_transform, d=1.0 / level_metadata.fs).tolist()
            data[ilevel] = self._get_level_data(
                level_metadata, win_data.get_level(ilevel), n_transform
            )
            spectra_levels_metadata.append(
                self._get_level_metadata(level_metadata, freqs)
            )
            messages.append(f"Calculated spectra for level {ilevel}")
        metadata = self._get_metadata(metadata_dict, spectra_levels_metadata)
        metadata.history.add_record(self._get_record(messages))
        logger.info("Fourier transforms completed")
        return SpectraData(metadata, data)

    def _get_level_data(
        self, metadata: WindowedLevelMetadata, data: np.ndarray, n_transform: int
    ) -> np.ndarray:
        """Run the spectra calculation for a single decimation level"""
        from scipy import signal
        from scipy.fft import rfft

        # detrend and apply window
        if self.detrend is not None:
            data = signal.detrend(data, axis=-1, type=self.detrend)
        win_coeffs = self._get_window(metadata.win_size).astype(data.dtype)
        data = data * win_coeffs
        # perform the fft on the last axis
        return rfft(data, n=n_transform, axis=-1, norm="ortho", workers=self.workers)

    def _get_window(self, win_size: int):
        """Get the window to apply to the data"""
        from scipy.signal import get_window
        from scipy.signal.windows import dpss

        if self.win_fnc == "dpss":
            return dpss(win_size, 5)
        return get_window(self.win_fnc, win_size)

    def _get_level_metadata(
        self, level_metadata: WindowedLevelMetadata, freqs: List[float]
    ) -> SpectraLevelMetadata:
        """Get the spectra metadata for a decimation level"""
        metadata_dict = level_metadata.dict()
        metadata_dict["n_freqs"] = len(freqs)
        metadata_dict["freqs"] = freqs
        return SpectraLevelMetadata(**metadata_dict)

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
    """
    Calculate the spectra values at the evaluation frequencies
    """

    def run(
        self, dec_params: DecimationParameters, spec_data: SpectraData
    ) -> SpectraData:
        """
        Interpolate spectra data to the evaluation frequencies

        This is a simple linear interpolation.

        Parameters
        ----------
        dec_params : DecimationParameters
            The decimation parameters which have the evaluation frequencies for
            each decimation level
        spec_data : SpectraData
            The spectra data

        Returns
        -------
        SpectraData
            The spectra data at the evaluation frequencies
        """
        metadata_dict = spec_data.metadata.dict()
        data = {}
        spectra_levels_metadata = []
        messages = []
        for ilevel in range(spec_data.metadata.n_levels):
            logger.info(f"Reducing freqs to evaluation freqs for level {ilevel}")
            level_metadata = spec_data.metadata.levels_metadata[ilevel]
            freqs = np.array(level_metadata.freqs)
            eval_freqs = np.array(dec_params.get_eval_freqs(ilevel))
            data[ilevel] = self._get_level_data(
                freqs, spec_data.get_level(ilevel), eval_freqs
            )
            spectra_levels_metadata.append(
                self._get_level_metadata(level_metadata, eval_freqs)
            )
        messages.append("Spectra reduced to evaluation frequencies")
        metadata = self._get_metadata(metadata_dict, spectra_levels_metadata)
        metadata.history.add_record(self._get_record(messages))
        logger.info("Fourier coefficients calculated at evaluation frequencies")
        return SpectraData(metadata, data)

    def _get_level_data(
        self, freqs: np.ndarray, data: np.ndarray, eval_freqs: np.ndarray
    ) -> np.ndarray:
        """
        Interpolate the spectra data to the evaluation frequencies

        The input data for a level has shape:

        n_wins x n_chans x n_freqs

        The new output data will have size:

        n_wins x n_chans x n_eval_freqs

        This process is doing a linear interpolation. As this is complex data
        and numpy does not have an interpolation along axis option,
        interpolation is done manually.

        First the evaluation frequencies are interpolated to their indices given
        the current frequencies and indices.

        Then these float indices are used to do the interpolation.

        Parameters
        ----------
        freqs : np.ndarray
            The input data frequencies
        data : np.ndarray
            The input spectra data
        eval_freqs : List[float]
            The evaluation frequencies

        Returns
        -------
        np.ndarray
            Output level data
        """
        index = np.arange(len(freqs))
        eval_indices = np.interp(eval_freqs, freqs, index)
        floors = np.floor(eval_indices).astype(int)
        ceils = np.floor(eval_indices).astype(int)
        portions = eval_indices - floors
        diffs = data[..., ceils] - data[..., floors]
        return data[..., floors] + np.squeeze(diffs[..., np.newaxis, :] * portions)

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


class SpectraDataWriter(ResisticsWriter):
    """Writer of resistics spectra data"""

    def run(self, dir_path: Path, spec_data: SpectraData) -> None:
        """
        Write out SpectraData

        Parameters
        ----------
        dir_path : Path
            The directory path to write to
        spec_data : SpectraData
            Spectra data to write out

        Raises
        ------
        WriteError
            If unable to write to the directory
        """
        from resistics.errors import WriteError

        if not self._check_dir(dir_path):
            WriteError(dir_path, "Unable to write to directory, check logs")
        logger.info(f"Writing spectra data to {dir_path}")
        metadata_path = dir_path / "metadata.json"
        data_path = dir_path / "data"
        np.savez_compressed(data_path, **{str(x): y for x, y in spec_data.data.items()})
        metadata = spec_data.metadata.copy()
        metadata.history.add_record(self._get_record(dir_path, type(spec_data)))
        metadata.write(metadata_path)


class SpectraDataReader(ResisticsProcess):
    """Reader of resistics spectra data"""

    def run(
        self, dir_path: Path, metadata_only: bool = False
    ) -> Union[SpectraMetadata, SpectraData]:
        """
        Read SpectraData

        Parameters
        ----------
        dir_path : Path
            The directory path to read from
        metadata_only : bool, optional
            Flag for getting metadata only, by default False

        Returns
        -------
        Union[SpectraMetadata, SpectraData]
            The SpectraData or SpectraMetadata if metadata_only is True

        Raises
        ------
        ReadError
            If the directory does not exist
        """
        from resistics.errors import ReadError

        if not dir_path.exists():
            raise ReadError(dir_path, "Directory does not exist")
        logger.info(f"Reading spectra data from {dir_path}")
        metadata_path = dir_path / "metadata.json"
        metadata = SpectraMetadata.parse_file(metadata_path)
        if metadata_only:
            return metadata
        data_path = dir_path / "data.npz"
        npz_file = np.load(data_path)
        data = {int(level): npz_file[level] for level in npz_file.files}
        messages = [f"Spectra data read from {dir_path}"]
        metadata.history.add_record(self._get_record(messages))
        return SpectraData(metadata, data)
