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
import pandas as pd
import plotly.graph_objects as go

from resistics.common import ResisticsData, ResisticsProcess, History
from resistics.common import ResisticsWriter, Metadata, WriteableMetadata
from resistics.sampling import HighResDateTime
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
    first_time: HighResDateTime
    last_time: HighResDateTime
    system: str = ""
    serial: str = ""
    wgs84_latitude: float = -999.0
    wgs84_longitude: float = -999.0
    easting: float = -999.0
    northing: float = -999.0
    elevation: float = -999.0
    chans_metadata: Dict[str, ChanMetadata]
    levels_metadata: List[SpectraLevelMetadata]
    ref_time: HighResDateTime
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

    def get_timestamps(self, level: int) -> pd.DatetimeIndex:
        """
        Get the start time of each window

        Note that this does not use high resolution timestamps

        Parameters
        ----------
        level : int
            The decimation level

        Returns
        -------
        pd.DatetimeIndex
            The starts of each window

        Raises
        ------
        ValueError
            If the level is out of range
        """
        from resistics.window import get_win_starts

        if level >= self.metadata.n_levels:
            raise ValueError(f"Level {level} not <= max {self.metadata.n_levels - 1}")
        level_metadata = self.metadata.levels_metadata[level]
        return get_win_starts(
            self.metadata.ref_time,
            level_metadata.win_size,
            level_metadata.olap_size,
            level_metadata.fs,
            level_metadata.n_wins,
            level_metadata.index_offset,
        )

    def plot_stack(
        self,
        level: int,
        max_pts: int = 10_000,
        grouping: str = "12h",
        offset: str = "0h",
    ) -> go.Figure:
        """
        Stack the spectra with optional time grouping

        Parameters
        ----------
        level : int
            The decimation level
        max_pts : int, optional
            The maximum number of points in any individual plot before applying
            lttbc downsampling, by default 10_000
        grouping : str, optional
            A grouping interval as a pandas freq string, by default "6h"
        offset : str, optional
            A time offset to add to the grouping, by default "0h". For instance,
            to plot night time and day time spectra, set grouping to "12h" and
            offset to "6h"

        Returns
        -------
        go.Figure
            The plotly figure
        """
        from resistics.plot import PlotData1D, figure_columns_as_lines, plot_columns_1d

        if grouping is None:
            first_date = pd.Timestamp(self.metadata.first_time.isoformat()).floor("D")
            last_date = pd.Timestamp(self.metadata.last_time.isoformat()).ceil("D")
            grouping = last_date - first_date
        level_metadata = self.metadata.levels_metadata[level]
        df = pd.DataFrame(
            data=np.arange(level_metadata.n_wins),
            index=self.get_timestamps(level),
            columns=["local"],
        )
        # prepare the plot
        subplots = self.metadata.chans
        subplot_columns = {x: [x] for x in subplots}
        y_labels = {x: "Magnitude" for x in subplots}
        fig = figure_columns_as_lines(
            subplots=subplots, y_labels=y_labels, x_label="Frequency"
        )
        fig.update_yaxes(type="log")
        # group by the grouping frequency, iterate over the groups and plot
        for idx, group in df.groupby(pd.Grouper(freq=grouping, offset=offset)):
            stack = np.mean(np.absolute(self.data[level][group["local"]]), axis=0)
            plot_data = PlotData1D(
                x=np.array(level_metadata.freqs), data=stack, rows=self.metadata.chans
            )
            plot_columns_1d(
                fig,
                plot_data,
                subplots,
                subplot_columns,
                max_pts=max_pts,
                label_prefix=str(idx),
            )
        return fig

    def plot_section(self, level: int, grouping="30T") -> go.Figure:
        """
        Plot a spectra section

        Parameters
        ----------
        level : int
            The decimation level to plot
        grouping : str, optional
            The time domain resolution, by default "30T"

        Returns
        -------
        go.Figure
            A plotly figure
        """
        from resistics.plot import figure_columns_as_lines

        level_metadata = self.metadata.levels_metadata[level]
        df = pd.DataFrame(
            data=np.arange(level_metadata.n_wins),
            index=self.get_timestamps(level),
            columns=["local"],
        )
        # create the figure
        subplots = self.metadata.chans
        y_labels = {x: "Frequency Hz" for x in subplots}
        fig = figure_columns_as_lines(
            subplots=subplots, y_labels=y_labels, x_label="Date"
        )
        colorbar_len = 0.90 / self.metadata.n_chans
        colorbar_inc = 0.83 / (self.metadata.n_chans - 1)
        # group by the grouping frequency, iterate over the groups and plot
        data = {}
        for idx, group in df.groupby(pd.Grouper(freq=grouping)):
            data[idx] = np.mean(np.absolute(self.data[level][group["local"]]), axis=0)
        for idx, chan in enumerate(self.metadata.chans):
            df_data = pd.DataFrame(
                data={k: v[idx] for k, v in data.items()}, index=level_metadata.freqs
            )
            z = np.log10(df_data.values)
            z_min = np.ceil(z.min())
            z_max = np.floor(z.max())
            z_range = np.arange(z_min, z_max + 1)
            colorbar = dict(
                title=f"Amplitude {chan}",
                tickvals=z_range,
                ticktext=[f"10^{int(x)}" for x in z_range],
                y=0.92 - idx * colorbar_inc,
                len=colorbar_len,
            )
            heatmap = go.Heatmap(
                z=z,
                x=pd.to_datetime(df_data.columns) + pd.Timedelta(grouping) / 2,
                y=df_data.index,
                zmin=z_min,
                zmax=z_max,
                colorscale="viridis",
                colorbar=colorbar,
            )
            fig.append_trace(heatmap, row=idx + 1, col=1)
        return fig


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

    Examples
    --------
    This example will get periodic decimated data, perfrom windowing and run the
    Fourier transform on the windowed data.

    .. plot::
        :width: 90%

        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from resistics.testing import decimated_data_periodic
        >>> from resistics.window import WindowSetup, Windower
        >>> from resistics.spectra import FourierTransform
        >>> frequencies = {"chan1": [870, 590, 110, 32, 12], "chan2": [480, 375, 210, 60, 45]}
        >>> dec_data = decimated_data_periodic(frequencies, fs=128)
        >>> dec_data.metadata.chans
        ['chan1', 'chan2']
        >>> print(dec_data.to_string())
        <class 'resistics.decimate.DecimatedData'>
                   fs        dt  n_samples           first_time                        last_time
        level
        0      2048.0  0.000488      16384  2021-01-01 00:00:00  2021-01-01 00:00:07.99951171875
        1       512.0  0.001953       4096  2021-01-01 00:00:00    2021-01-01 00:00:07.998046875
        2       128.0  0.007812       1024  2021-01-01 00:00:00      2021-01-01 00:00:07.9921875

        Perform the windowing

        >>> win_params = WindowSetup().run(dec_data.metadata.n_levels, dec_data.metadata.fs)
        >>> win_data = Windower().run(dec_data.metadata.first_time, win_params, dec_data)

        And then the Fourier transform. By default, the data will be (linearly)
        detrended and mutliplied by a Kaiser window prior to the Fourier
        transform

        >>> spec_data = FourierTransform().run(win_data)

        For plotting of magnitude, let's stack the spectra

        >>> freqs_0 = spec_data.metadata.levels_metadata[0].freqs
        >>> data_0 = np.absolute(spec_data.data[0]).mean(axis=0)
        >>> freqs_1 = spec_data.metadata.levels_metadata[1].freqs
        >>> data_1 = np.absolute(spec_data.data[1]).mean(axis=0)
        >>> freqs_2 = spec_data.metadata.levels_metadata[2].freqs
        >>> data_2 = np.absolute(spec_data.data[2]).mean(axis=0)

        Now plot

        >>> plt.subplot(3,1,1) # doctest: +SKIP
        >>> plt.plot(freqs_0, data_0[0], label="chan1") # doctest: +SKIP
        >>> plt.plot(freqs_0, data_0[1], label="chan2") # doctest: +SKIP
        >>> plt.grid()
        >>> plt.title("Decimation level 0") # doctest: +SKIP
        >>> plt.legend() # doctest: +SKIP
        >>> plt.subplot(3,1,2) # doctest: +SKIP
        >>> plt.plot(freqs_1, data_1[0], label="chan1") # doctest: +SKIP
        >>> plt.plot(freqs_1, data_1[1], label="chan2") # doctest: +SKIP
        >>> plt.grid()
        >>> plt.title("Decimation level 1") # doctest: +SKIP
        >>> plt.legend() # doctest: +SKIP
        >>> plt.subplot(3,1,3) # doctest: +SKIP
        >>> plt.plot(freqs_2, data_2[0], label="chan1") # doctest: +SKIP
        >>> plt.plot(freqs_2, data_2[1], label="chan2") # doctest: +SKIP
        >>> plt.grid()
        >>> plt.title("Decimation level 2") # doctest: +SKIP
        >>> plt.legend() # doctest: +SKIP
        >>> plt.xlabel("Frequency") # doctest: +SKIP
        >>> plt.tight_layout() # doctest: +SKIP
        >>> plt.show() # doctest: +SKIP
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
        """
        Run the spectra calculation for a single decimation level

        The input is an array with shape:

        n_wins x n_chans x win_size

        And the output has shape

        n_wins x n_chans x n_transform

        Parameters
        ----------
        metadata : WindowedLevelMetadata
            Level metadata
        data : np.ndarray
            Data to transform
        n_transform : int
            Size of the transform

        Returns
        -------
        np.ndarray
            Transformed data for all windows
        """
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


class SpectraSmootherUniform(ResisticsProcess):
    """
    Smooth a spectra with a uniform filter

    For more information, please refer to:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.uniform_filter1d.html

    Examples
    --------
    Smooth a simple spectra data instance

    >>> from resistics.spectra import SpectraSmootherUniform
    >>> from resistics.testing import spectra_data_basic
    >>> spec_data = spectra_data_basic()
    >>> smooth_data = SpectraSmootherUniform(length_proportion=0.5).run(spec_data)

    Look at the results for the two windows

    >>> spec_data.data[0][0,0]
    array([0.+0.j, 1.+1.j, 2.+2.j, 3.+3.j, 4.+4.j, 5.+5.j, 6.+6.j, 7.+7.j,
           8.+8.j, 9.+9.j])
    >>> smooth_data.data[0][0,0]
    array([0.8+0.8j, 1.2+1.2j, 2. +2.j , 3. +3.j , 4. +4.j , 5. +5.j ,
           6. +6.j , 7. +7.j , 7.8+7.8j, 8.2+8.2j])
    """

    length_proportion: float = 0.1

    def run(self, spec_data: SpectraData) -> SpectraData:
        """
        Smooth spectra data with a uniform smoother

        Parameters
        ----------
        spec_data : SpectraData
            The input spectra data

        Returns
        -------
        SpectraData
            The output spectra data
        """
        import scipy.ndimage as ndimage

        data = {}
        logger.info("Smoothing frequencies with uniform filter")
        messages = ["Smoothing frequencies with uniform filter"]
        for ilevel in range(spec_data.metadata.n_levels):
            n_freqs = spec_data.metadata.levels_metadata[ilevel].n_freqs
            smooth_length = self._get_smooth_length(n_freqs)
            logger.debug(f"Smoothing level {ilevel} with num points {smooth_length}")
            data[ilevel] = ndimage.uniform_filter1d(
                spec_data.get_level(ilevel), smooth_length, axis=-1
            )
            messages.append(f"Smoothed level {ilevel} with num points {smooth_length}")
        metadata = SpectraMetadata(**spec_data.metadata.dict())
        metadata.history.add_record(self._get_record(messages))
        logger.info("Fourier coefficients calculated at evaluation frequencies")
        return SpectraData(metadata, data)

    def _get_smooth_length(self, data_size: int) -> int:
        """Get the smoothing length given the size of the data"""
        length = int(self.length_proportion * data_size)
        if length % 2 == 0:
            length += 1
        if length < 1:
            return 1
        return length


class SpectraSmootherGaussian(ResisticsProcess):
    """
    Smooth a spectra with a gaussian filter

    For more information, please refer to:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter1d.html

    Examples
    --------
    Smooth a simple spectra data instance

    >>> from resistics.spectra import SpectraSmootherGaussian
    >>> from resistics.testing import spectra_data_basic
    >>> spec_data = spectra_data_basic()
    >>> smooth_data = SpectraSmootherGaussian().run(spec_data)

    Look at the results for the two windows

    >>> spec_data.data[0][0,0]
    array([0.+0.j, 1.+1.j, 2.+2.j, 3.+3.j, 4.+4.j, 5.+5.j, 6.+6.j, 7.+7.j,
           8.+8.j, 9.+9.j])
    >>> smooth_data.data[0][0,0]
    array([0.42704095+0.42704095j, 1.06795587+1.06795587j,
           2.00483335+2.00483335j, 3.00013383+3.00013383j,
           4.        +4.j        , 5.        +5.j        ,
           5.99986617+5.99986617j, 6.99516665+6.99516665j,
           7.93204413+7.93204413j, 8.57295905+8.57295905j])
    """

    sigma: float = 3

    def run(self, spec_data: SpectraData) -> SpectraData:
        """
        Run Gaussian filtering of spectra data

        Parameters
        ----------
        spec_data : SpectraData
            Input spectra data

        Returns
        -------
        SpectraData
            Output spectra data
        """
        import scipy.ndimage as ndimage

        data = {}
        logger.info(f"Smoothing frequencies with gaussian filter, sigma {self.sigma}")
        messages = [f"Smoothing frequencies with gaussian filter, sigma {self.sigma}"]
        for ilevel in range(spec_data.metadata.n_levels):
            data[ilevel] = ndimage.gaussian_filter1d(
                spec_data.get_level(ilevel), 1, axis=-1
            )
            messages.append(f"Smoothed level {ilevel} with gaussian filter")
        metadata = SpectraMetadata(**spec_data.metadata.dict())
        metadata.history.add_record(self._get_record(messages))
        logger.info("Fourier coefficients calculated at evaluation frequencies")
        return SpectraData(metadata, data)


class EvaluationFreqs(ResisticsProcess):
    """
    Calculate the spectra values at the evaluation frequencies

    This is done using linear interpolation in the complex domain

    Example
    -------
    The example will show interpolation to evaluation frequencies on a very
    simple example. Begin by generating some example spectra data.

    >>> from resistics.decimate import DecimationSetup
    >>> from resistics.spectra import EvaluationFreqs
    >>> from resistics.testing import spectra_data_basic
    >>> spec_data = spectra_data_basic()
    >>> spec_data.metadata.n_levels
    1
    >>> spec_data.metadata.chans
    ['chan1']
    >>> spec_data.metadata.levels_metadata[0].summary()
    {
        'fs': 180.0,
        'n_wins': 2,
        'win_size': 20,
        'olap_size': 5,
        'index_offset': 0,
        'n_freqs': 10,
        'freqs': [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0]
    }

    The spectra data has only a single channel and a single level which has 2
    windows. Now define our evaluation frequencies.

    >>> eval_freqs = [1, 12, 23, 34, 45, 56, 67, 78, 89]
    >>> dec_setup = DecimationSetup(n_levels=1, per_level=9, eval_freqs=eval_freqs)
    >>> dec_params = dec_setup.run(spec_data.metadata.fs[0])
    >>> dec_params.summary()
    {
        'fs': 180.0,
        'n_levels': 1,
        'per_level': 9,
        'min_samples': 256,
        'eval_freqs': [1.0, 12.0, 23.0, 34.0, 45.0, 56.0, 67.0, 78.0, 89.0],
        'dec_factors': [1],
        'dec_increments': [1],
        'dec_fs': [180.0]
    }

    Now calculate the spectra at the evaluation frequencies

    >>> eval_data = EvaluationFreqs().run(dec_params, spec_data)
    >>> eval_data.metadata.levels_metadata[0].summary()
    {
        'fs': 180.0,
        'n_wins': 2,
        'win_size': 20,
        'olap_size': 5,
        'index_offset': 0,
        'n_freqs': 9,
        'freqs': [1.0, 12.0, 23.0, 34.0, 45.0, 56.0, 67.0, 78.0, 89.0]
    }

    To double check everything is as expected, let's compare the data. Comparing
    window 1 gives

    >>> print(spec_data.data[0][0, 0])
    [0.+0.j 1.+1.j 2.+2.j 3.+3.j 4.+4.j 5.+5.j 6.+6.j 7.+7.j 8.+8.j 9.+9.j]
    >>> print(eval_data.data[0][0, 0])
    [0.1+0.1j 1.2+1.2j 2.3+2.3j 3.4+3.4j 4.5+4.5j 5.6+5.6j 6.7+6.7j 7.8+7.8j
     8.9+8.9j]

    And window 2

    >>> print(spec_data.data[0][1, 0])
    [-1. +1.j  0. +2.j  1. +3.j  2. +4.j  3. +5.j  4. +6.j  5. +7.j  6. +8.j
      7. +9.j  8.+10.j]
    >>> print(eval_data.data[0][1, 0])
    [-0.9+1.1j  0.2+2.2j  1.3+3.3j  2.4+4.4j  3.5+5.5j  4.6+6.6j  5.7+7.7j
      6.8+8.8j  7.9+9.9j]
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
        ceils = np.ceil(eval_indices).astype(int)
        # cast portions to preserve original data type
        # otherwise, can expand complex64 to complex128
        portions = (eval_indices - floors).astype(data.dtype)
        diffs = data[..., ceils] - data[..., floors]
        add = np.squeeze(diffs[..., np.newaxis, :] * portions, axis=-2)
        return data[..., floors] + add

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
