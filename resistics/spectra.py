"""Module containing functions and classes related to Spectra calculation and
manipulation

Spectra are calculated from the windowed, decimated time data. The inbuilt
Fourier transform implementation is inspired by the implementation of the
scipy stft function.
"""

from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from loguru import logger
from pydantic import ConfigDict, PositiveInt

from resistics.common import (
    History,
    Metadata,
    ResisticsData,
    ResisticsModel,
    ResisticsProcess,
    ResisticsWriter,
    WriteableMetadata,
    save_compressed_arrays,
    validate_output_label,
)
from resistics.decimate import DecimationParameters
from resistics.sampling import HighResDateTime
from resistics.time import ChanMetadata
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
    freqs: list[float]
    """List of frequencies"""

    @property
    def nyquist(self) -> float:
        """Get the nyquist frequency"""
        return self.fs / 2


class SpectraMetadata(WriteableMetadata):
    """Metadata for spectra data"""

    model_config = ConfigDict(extra="ignore")

    fs: list[float]
    chans: list[str]
    n_chans: int = 0
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
    chans_metadata: dict[str, ChanMetadata]
    levels_metadata: list[SpectraLevelMetadata]
    ref_time: HighResDateTime
    history: History = History()


class SpectraData(ResisticsData):
    """Class for holding spectra data

    The spectra data is stored in the class as a dictionary mapping decimation
    level to numpy array. The shape of the array for each decimation level is:

    n_wins x n_chans x n_freqs

    :param metadata: Metadata for the spectra data.
    :param data: Per-level complex arrays shaped as windows, channels, and frequencies.
    """

    def __init__(self, metadata: SpectraMetadata, data: dict[int, np.ndarray]) -> None:
        logger.debug(f"Creating SpectraData with data type {data[0].dtype}")
        self.metadata = metadata
        self.data = data

    def get_level(self, level: int) -> np.ndarray:
        """Get spectra data for a decimation level.

        :param level: Decimation level index.

        :return: Complex data shaped as windows, channels, and frequencies.

        :raises ValueError: If the level is outside the available range.
        """
        if level >= self.metadata.n_levels:
            raise ValueError(f"Level {level} not <= max {self.metadata.n_levels - 1}")
        return self.data[level]

    def get_chan(self, level: int, chan: str) -> np.ndarray:
        """Get one channel's spectra for a decimation level.

        :param level: Decimation level index.
        :param chan: Channel component name.

        :return: Complex data shaped as windows by frequencies.

        :raises ChannelNotFoundError: If the channel is unavailable.
        """
        from resistics.errors import ChannelNotFoundError

        if chan not in self.metadata.chans:
            raise ChannelNotFoundError(chan, self.metadata.chans)
        idx = self.metadata.chans.index(chan)
        return self.data[level][..., idx, :]

    def get_chans(self, level: int, chans: list[str]) -> np.ndarray:
        """Get selected channel spectra for a decimation level.

        :param level: Decimation level index.
        :param chans: Channel component names in the requested output order.

        :return: Complex data shaped as windows, selected channels, and frequencies.

        :raises ChannelNotFoundError: If any requested channel is unavailable.
        """
        from resistics.errors import ChannelNotFoundError

        for chan in chans:
            if chan not in self.metadata.chans:
                raise ChannelNotFoundError(chan, self.metadata.chans)
        indices = [self.metadata.chans.index(chan) for chan in chans]
        return self.data[level][..., indices, :]

    def get_freq(self, level: int, idx: int) -> np.ndarray:
        """Get spectra at one frequency index for a decimation level.

        :param level: Decimation level index.
        :param idx: Frequency-bin index.

        :return: Complex data for every window and channel at the frequency.

        :raises ValueError: If the frequency index is outside the level range.
        """
        n_freqs = self.metadata.levels_metadata[level].n_freqs
        if idx < 0 or idx >= n_freqs:
            raise ValueError(f"Freq. index {idx} not 0 <= idx < {n_freqs}")
        return np.squeeze(self.data[level][..., idx])

    def get_mag_phs(
        self, level: int, unwrap: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get magnitude and phase for a decimation level.

        :param level: Decimation level index.
        :param unwrap: Whether to unwrap phase along the final axis.

        :return: Magnitude and phase arrays matching the level data shape.
        """
        spec = self.data[level]
        if unwrap:
            return np.absolute(spec), np.unwrap(np.angle(spec))
        return np.absolute(spec), np.angle(spec)

    def get_timestamps(self, level: int) -> pd.DatetimeIndex:
        """Get the start time of each window

        Note that this does not use high resolution timestamps

        :param level: The decimation level

        :return: The starts of each window

        :raises ValueError: If the level is out of range
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

    def plot(self, max_pts: int | None = 10_000) -> go.Figure:
        """Stack spectra data for all decimation levels

        :param max_pts: The maximum number of points in any individual plot before applying
            LTTB downsampling, by default 10_000. If set to None, no
            downsampling will be applied.

        :return: The plotly figure
        """
        from resistics.plot import get_spectra_stack_fig

        y_labels = dict.fromkeys(self.metadata.chans, "Magnitude")
        fig = get_spectra_stack_fig(self.metadata.chans, y_labels)
        colors = iter(px.colors.qualitative.Plotly)
        for ilevel in range(self.metadata.n_levels):
            level_metadata = self.metadata.levels_metadata[ilevel]
            freqs = np.array(level_metadata.freqs)
            stack = np.mean(np.absolute(self.data[ilevel]), axis=0)
            legend = f"{ilevel} - {level_metadata.fs:.4f} Hz"
            fig = self._add_stack_data(
                fig, freqs, stack, legend, color=next(colors), max_pts=max_pts
            )
        return fig

    def plot_level_stack(
        self,
        level: int,
        max_pts: int = 10_000,
        grouping: str | None = None,
        offset: str = "0h",
    ) -> go.Figure:
        """Stack the spectra for a decimation level with optional time grouping

        :param level: The decimation level
        :param max_pts: The maximum number of points in any individual plot before applying
            LTTB downsampling, by default 10_000
        :param grouping: A grouping interval as a pandas freq string, by default None
        :param offset: A time offset to add to the grouping, by default "0h". For instance,
            to plot night time and day time spectra, set grouping to "12h" and
            offset to "6h"

        :return: The plotly figure
        """
        from resistics.plot import get_spectra_stack_fig

        group_frequency = grouping
        if group_frequency is None:
            first_date = pd.Timestamp(self.metadata.first_time.isoformat()).floor("D")
            last_date = pd.Timestamp(self.metadata.last_time.isoformat()).ceil("D")
            duration_seconds = max(1, int((last_date - first_date).total_seconds()))
            group_frequency = f"{duration_seconds}s"
        level_metadata = self.metadata.levels_metadata[level]
        df = pd.DataFrame(
            data=np.arange(level_metadata.n_wins),
            index=self.get_timestamps(level),
            columns=["local"],
        )
        # group by the grouping frequency, iterate over the groups and plot
        freqs = np.array(level_metadata.freqs)
        y_labels = dict.fromkeys(self.metadata.chans, "Magnitude")
        fig = get_spectra_stack_fig(self.metadata.chans, y_labels)
        colors = iter(px.colors.qualitative.Plotly)
        for idx, group in df.groupby(pd.Grouper(freq=group_frequency, offset=offset)):
            stack = np.mean(np.absolute(self.data[level][group["local"]]), axis=0)
            fig = self._add_stack_data(
                fig, freqs, stack, str(idx), color=next(colors), max_pts=max_pts
            )
        return fig

    def _add_stack_data(
        self,
        fig: go.Figure,
        freqs: np.ndarray,
        data: np.ndarray,
        legend: str,
        color: str = "blue",
        max_pts: int | None = 10_000,
    ) -> go.Figure:
        """Add stacked spectra data to a plot

        :param fig: The figure to add to
        :param freqs: Frequencies
        :param data: The magnitude data
        :param legend: The legend string for the data
        :param color: The color to plot the line, by default "blue"
        :param max_pts: Maximum number of points to plot, by default 10_000. If the number
            of samples in the data is above this, it will be downsampled

        :return: Plotly figure
        """
        from resistics.plot import apply_lttb

        n_chans = data.shape[0]
        for idx in range(n_chans):
            indices, chan_data = apply_lttb(data[idx, :], max_pts)
            chan_freqs = freqs[indices]
            scatter = go.Scattergl(
                x=chan_freqs,
                y=chan_data,
                line={"color": color},
                name=legend,
                legendgroup=legend,
                showlegend=(idx == 0),
            )
            fig.add_trace(scatter, row=idx + 1, col=1)
        return fig

    def plot_level_section(self, level: int, grouping: str = "30T") -> go.Figure:
        """Plot a spectra section

        :param level: The decimation level to plot
        :param grouping: The time domain resolution, by default "30T"

        :return: A plotly figure
        """
        from resistics.plot import get_spectra_section_fig

        level_metadata = self.metadata.levels_metadata[level]
        df = pd.DataFrame(
            data=np.arange(level_metadata.n_wins),
            index=self.get_timestamps(level),
            columns=["local"],
        )

        fig = get_spectra_section_fig(self.metadata.chans)
        colorbar_len = 0.90 / self.metadata.n_chans
        colorbar_inc = (
            0.0 if self.metadata.n_chans == 1 else 0.84 / (self.metadata.n_chans - 1)
        )
        # group by the grouping frequency, iterate over the groups and plot
        data = {}
        for idx, group in df.groupby(pd.Grouper(freq=grouping)):
            data[idx] = np.mean(np.absolute(self.data[level][group["local"]]), axis=0)
        for idx, _chan in enumerate(self.metadata.chans):
            df_data = pd.DataFrame(
                data={k: v[idx] for k, v in data.items()}, index=level_metadata.freqs
            )
            z = np.log10(df_data.values)
            z_min = np.ceil(z.min())
            z_max = np.floor(z.max())
            z_range = np.arange(z_min, z_max + 1)
            colorbar = {
                "tickvals": z_range,
                "ticktext": [f"10^{int(x)}" for x in z_range],
                "y": 0.92 - idx * colorbar_inc,
                "len": colorbar_len,
            }
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
    """Perform a Fourier transform of the windowed data

    The processor is inspired by the scipy.signal.stft function which performs
    a similar process and involves a Fourier transform along the last axis of
    the windowed data.

    :param win_fnc: The window to use before performing the FFT, by default ("kaiser", 14)
    :param detrend: Type of detrending to apply before performing FFT, by default linear
        detrend. Setting to None will not apply any detrending to the data prior
        to the FFT
    :param workers: The number of CPUs to use, by default max - 2

    **Examples**

    This example will get periodic decimated data, perfrom windowing and run the
    Fourier transform on the windowed data.

    ```{plot}
    :width: 90%
    :filename-prefix: fourier-transform

    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> from resistics.testing import decimated_data_periodic
    >>> from resistics.window import WindowSetup, Windower
    >>> from resistics.spectra import FourierTransform
    >>> frequencies = {"chan1": [870, 590, 110, 32, 12], "chan2": [480, 375, 210, 60, 45]}
    >>> dec_data = decimated_data_periodic(frequencies, fs=128)
    >>> dec_data.metadata.chans
    ['chan1', 'chan2']
    >>> print(dec_data.to_string()) # doctest: +NORMALIZE_WHITESPACE
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

    ```
    """

    input_types: ClassVar[dict[str, str]] = {"win_data": "windowed_data"}
    output_type: ClassVar[str] = "spectra_data"
    include_in_default_parameters: ClassVar[bool] = True

    win_fnc: str | tuple[str, float] = ("kaiser", 14)
    detrend: Literal["linear", "constant"] | None = "linear"
    workers: int = -2

    def run(self, win_data: WindowedData) -> SpectraData:
        """Perform the FFT

        Data is padded to the next fast length before performing the FFT to
        speed up processing. Therefore, the output length may not be as
        expected.

        :param win_data: The input windowed data

        :return: The Fourier transformed output
        """
        from scipy.fft import next_fast_len, rfftfreq

        metadata_dict = win_data.metadata.model_dump()
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
        """Run the spectra calculation for a single decimation level

        The input is an array with shape:

        n_wins x n_chans x win_size

        And the output has shape

        n_wins x n_chans x n_transform

        :param metadata: Level metadata
        :param data: Data to transform
        :param n_transform: Size of the transform

        :return: Transformed data for all windows
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

    def _get_window(self, win_size: int) -> np.ndarray:
        """Get coefficients for the configured Fourier window.

        :param win_size: Window size in samples.

        :return: One coefficient per input sample.
        """
        from scipy.signal import get_window
        from scipy.signal.windows import dpss

        if self.win_fnc == "dpss":
            return dpss(win_size, 5)
        return get_window(self.win_fnc, win_size)

    def _get_level_metadata(
        self, level_metadata: WindowedLevelMetadata, freqs: list[float]
    ) -> SpectraLevelMetadata:
        """Get spectra metadata for a decimation level.

        :param level_metadata: Source windowed-level metadata.
        :param freqs: Fourier frequencies in hertz.

        :return: Spectra-level metadata with frequency coordinates.
        """
        metadata_dict = level_metadata.model_dump()
        metadata_dict["n_freqs"] = len(freqs)
        metadata_dict["freqs"] = freqs
        return SpectraLevelMetadata(**metadata_dict)

    def _get_metadata(
        self,
        metadata_dict: dict[str, Any],
        levels_metadata: list[SpectraLevelMetadata],
    ) -> SpectraMetadata:
        """Get aggregate metadata for spectra data.

        :param metadata_dict: Source windowed-data metadata.
        :param levels_metadata: Metadata for each transformed level.

        :return: Aggregate spectra metadata.
        """
        metadata_dict.pop("file_info")
        metadata_dict["levels_metadata"] = levels_metadata
        return SpectraMetadata(**metadata_dict)


class EvaluationFreqs(ResisticsProcess):
    """Calculate the spectra values at the evaluation frequencies

    This is done using linear interpolation in the complex domain

    **Examples**

    The example will show interpolation to evaluation frequencies on a very
    simple example. Begin by generating some example spectra data.

    ```{doctest}
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

    ```

    The spectra data has only a single channel and a single level which has 2
    windows. Now define our evaluation frequencies.

    ```{doctest}
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

    ```

    Now calculate the spectra at the evaluation frequencies

    ```{doctest}
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

    ```

    To double check everything is as expected, let's compare the data. Comparing
    window 1 gives

    ```{doctest}
    >>> print(spec_data.data[0][0, 0])
    [0.+0.j 1.+1.j 2.+2.j 3.+3.j 4.+4.j 5.+5.j 6.+6.j 7.+7.j 8.+8.j 9.+9.j]
    >>> print(eval_data.data[0][0, 0])
    [0.1+0.1j 1.2+1.2j 2.3+2.3j 3.4+3.4j 4.5+4.5j 5.6+5.6j 6.7+6.7j 7.8+7.8j
     8.9+8.9j]

    ```

    And window 2

    ```{doctest}
    >>> print(spec_data.data[0][1, 0])
    [-1. +1.j  0. +2.j  1. +3.j  2. +4.j  3. +5.j  4. +6.j  5. +7.j  6. +8.j
      7. +9.j  8.+10.j]
    >>> print(eval_data.data[0][1, 0])
    [-0.9+1.1j  0.2+2.2j  1.3+3.3j  2.4+4.4j  3.5+5.5j  4.6+6.6j  5.7+7.7j
      6.8+8.8j  7.9+9.9j]

    ```
    """

    input_types: ClassVar[dict[str, str]] = {
        "dec_params": "decimation_parameters",
        "spec_data": "spectra_data",
    }
    output_type: ClassVar[str] = "eval_data"
    include_in_default_parameters: ClassVar[bool] = True

    def execute(
        self, inputs: dict[str, Any], context: Any
    ) -> "EvaluationFrequencyData":
        """Keep the decimation setup with spectra for persistence.

        :param inputs: Flow inputs containing decimation parameters and spectra data.
        :param context: Runtime context; unused by this process.

        :return: Evaluation-frequency spectra and their decimation setup.
        """
        del context
        dec_params = inputs["dec_params"]
        return EvaluationFrequencyData(
            spectra_data=self.run(dec_params, inputs["spec_data"]),
            decimation_parameters=dec_params,
        )

    def run(
        self, dec_params: DecimationParameters, spec_data: SpectraData
    ) -> SpectraData:
        """Interpolate spectra data to the evaluation frequencies

        This is a simple linear interpolation.

        :param dec_params: The decimation parameters which have the evaluation frequencies for
            each decimation level
        :param spec_data: The spectra data

        :return: The spectra data at the evaluation frequencies
        """
        metadata_dict = spec_data.metadata.model_dump()
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
        """Interpolate the spectra data to the evaluation frequencies

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

        :param freqs: The input data frequencies
        :param data: The input spectra data
        :param eval_freqs: The evaluation frequencies

        :return: Output level data
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
        """Get metadata for an evaluation-frequency level.

        :param level_metadata: Source spectra-level metadata.
        :param eval_freqs: Selected evaluation frequencies in hertz.

        :return: Spectra metadata carrying the selected frequency coordinates.
        """
        metadata_dict = level_metadata.model_dump()
        metadata_dict["n_freqs"] = len(eval_freqs)
        metadata_dict["freqs"] = eval_freqs.tolist()
        return SpectraLevelMetadata(**metadata_dict)

    def _get_metadata(
        self, metadata_dict: dict[str, Any], levels_metadata: list[SpectraLevelMetadata]
    ) -> SpectraMetadata:
        """Get aggregate metadata for evaluation-frequency spectra.

        :param metadata_dict: Source spectra metadata.
        :param levels_metadata: Metadata for each interpolated level.

        :return: Aggregate spectra metadata.
        """
        metadata_dict.pop("file_info")
        metadata_dict["levels_metadata"] = levels_metadata
        return SpectraMetadata(**metadata_dict)


class SpectraDataWriter(ResisticsWriter):
    """Writer of resistics spectra data"""

    def run(self, dir_path: Path, data: ResisticsData) -> None:
        """Write out SpectraData

        :param dir_path: The directory path to write to
        :param data: Spectra data to write out

        :raises TypeError: If ``data`` is not spectra data.
        :raises WriteError: If unable to write to the directory
        """
        from resistics.errors import WriteError

        if not isinstance(data, SpectraData):
            raise TypeError("SpectraDataWriter requires SpectraData")
        spec_data = data
        if not self._check_dir(dir_path):
            raise WriteError(dir_path, "Unable to write to directory, check logs")
        logger.info(f"Writing spectra data to {dir_path}")
        metadata_path = dir_path / "metadata.json"
        data_path = dir_path / "data"
        save_compressed_arrays(
            data_path, {str(level): values for level, values in spec_data.data.items()}
        )
        metadata = spec_data.metadata.model_copy()
        metadata.history.add_record(self._get_writer_record(dir_path, type(spec_data)))
        metadata.write(metadata_path)


class SpectraDataReader(ResisticsProcess):
    """Reader of resistics spectra data"""

    def run(
        self, dir_path: Path, metadata_only: bool = False
    ) -> SpectraMetadata | SpectraData:
        """Read SpectraData

        :param dir_path: The directory path to read from
        :param metadata_only: Flag for getting metadata only, by default False

        :return: The SpectraData or SpectraMetadata if metadata_only is True

        :raises ReadError: If the directory does not exist
        """
        from resistics.errors import ReadError

        if not dir_path.exists():
            raise ReadError(dir_path, "Directory does not exist")
        logger.info(f"Reading spectra data from {dir_path}")
        metadata_path = dir_path / "metadata.json"
        metadata = SpectraMetadata.model_validate_json(metadata_path.read_bytes())
        if metadata_only:
            return metadata
        data_path = dir_path / "data.npz"
        npz_file = np.load(data_path)
        data = {int(level): npz_file[level] for level in npz_file.files}
        messages = [f"Spectra data read from {dir_path}"]
        metadata.history.add_record(self._get_record(messages))
        return SpectraData(metadata, data)


class EvaluationFrequencyData(ResisticsModel):
    """Persisted evaluation-frequency spectra and their decimation setup."""

    spectra_data: SpectraData
    decimation_parameters: DecimationParameters


class EvaluationFrequencyReader(ResisticsProcess):
    """Read evaluation-frequency data stored for an MTH5 project run."""

    output_type: ClassVar[str] = "eval_data"
    runtime_requirements: ClassVar[list[str]] = ["project_path", "run_batch"]
    label: str = "default"

    def execute(
        self, inputs: dict[str, Any], context: dict[str, Any]
    ) -> EvaluationFrequencyData:
        """Read spectra and persisted decimation parameters.

        :param inputs: Flow inputs; this reader consumes no upstream artifact.
        :param context: Project path, run batch, and optional output label.

        :return: Persisted evaluation-frequency artifact.
        """
        del inputs
        batch = context["run_batch"]
        label = validate_output_label(context.get("output_label", self.label))
        path = (
            Path(context["project_path"])
            / "data"
            / batch["survey"]
            / batch["station"]
            / batch["run"]
            / "evals"
            / label
        )
        return EvaluationFrequencyData(
            spectra_data=SpectraDataReader().run(path),
            decimation_parameters=DecimationParameters.model_validate_json(
                (path / "decimation_parameters.json").read_bytes()
            ),
        )


class EvaluationFrequencyWriter(ResisticsProcess):
    """Write evaluation-frequency data for later processing."""

    input_types: ClassVar[dict[str, str]] = {"eval_data": "eval_data"}
    output_type: ClassVar[str] = "job_result"
    runtime_requirements: ClassVar[list[str]] = ["project_path", "run_batch"]
    label: str = "default"

    def execute(
        self, inputs: dict[str, Any], context: dict[str, Any]
    ) -> dict[str, str]:
        """Persist spectra and their decimation parameters.

        :param inputs: Flow inputs containing ``eval_data``.
        :param context: Project path, run batch, and optional output label.

        :return: Result containing the persisted evaluation directory.

        :raises ValueError: If the input is not evaluation-frequency data.
        """
        value = inputs["eval_data"]
        if isinstance(value, EvaluationFrequencyData):
            artifact = value
        else:
            raise ValueError(
                "EvaluationFrequencyWriter requires EvaluationFrequencyData"
            )
        batch = context["run_batch"]
        label = validate_output_label(context.get("output_label", self.label))
        path = (
            Path(context["project_path"])
            / "data"
            / batch["survey"]
            / batch["station"]
            / batch["run"]
            / "evals"
            / label
        )
        SpectraDataWriter().run(path, artifact.spectra_data)
        (path / "decimation_parameters.json").write_text(
            artifact.decimation_parameters.model_dump_json(indent=2), encoding="utf-8"
        )
        return {"evaluation_path": str(path)}


class EvaluationFrequencyParameters(ResisticsProcess):
    """Return decimation parameters carried by evaluation-frequency data."""

    input_types: ClassVar[dict[str, str]] = {"eval_data": "eval_data"}
    output_type: ClassVar[str] = "decimation_parameters"

    def execute(
        self, inputs: dict[str, Any], context: dict[str, Any]
    ) -> DecimationParameters:
        """Extract the persisted decimation setup.

        :param inputs: Flow inputs containing ``eval_data``.
        :param context: Runtime context; unused by this process.

        :return: Decimation parameters carried by the artifact.

        :raises ValueError: If the input is not evaluation-frequency data.
        """
        del context
        artifact = inputs["eval_data"]
        if not isinstance(artifact, EvaluationFrequencyData):
            raise ValueError("EvaluationFrequencyParameters requires evaluation data")
        return artifact.decimation_parameters


class SpectraProcess(ResisticsProcess):
    """Parent class for spectra processes"""

    def run(self, spec_data: SpectraData) -> SpectraData:
        """Run a spectra processor.

        :param spec_data: Spectra data to process.

        :return: Processed spectra data.

        :raises NotImplementedError: Always; subclasses must implement this method.
        """
        raise NotImplementedError("Run is not implemented in the parent SpectraProcess")


class SpectraSmootherUniform(SpectraProcess):
    """Smooth a spectra with a uniform filter

    For more information, please refer to:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.uniform_filter1d.html

    **Examples**

    Smooth a simple spectra data instance

    ```{doctest}
    >>> from resistics.spectra import SpectraSmootherUniform
    >>> from resistics.testing import spectra_data_basic
    >>> spec_data = spectra_data_basic()
    >>> smooth_data = SpectraSmootherUniform(length_proportion=0.5).run(spec_data)

    ```

    Look at the results for the two windows

    ```{doctest}
    >>> spec_data.data[0][0,0]
    array([0.+0.j, 1.+1.j, 2.+2.j, 3.+3.j, 4.+4.j, 5.+5.j, 6.+6.j, 7.+7.j,
           8.+8.j, 9.+9.j])
    >>> smooth_data.data[0][0,0]
    array([0.8+0.8j, 1.2+1.2j, 2. +2.j , 3. +3.j , 4. +4.j , 5. +5.j ,
           6. +6.j , 7. +7.j , 7.8+7.8j, 8.2+8.2j])

    ```
    """

    length_proportion: float = 0.1

    def run(self, spec_data: SpectraData) -> SpectraData:
        """Smooth spectra data with a uniform smoother

        :param spec_data: The input spectra data

        :return: The output spectra data
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
        metadata = SpectraMetadata(**spec_data.metadata.model_dump())
        metadata.history.add_record(self._get_record(messages))
        logger.info("Fourier coefficients calculated at evaluation frequencies")
        return SpectraData(metadata, data)

    def _get_smooth_length(self, data_size: int) -> int:
        """Get an odd smoothing length for the frequency-axis size.

        :param data_size: Number of frequency samples.

        :return: Odd smoothing length of at least one sample.
        """
        length = int(self.length_proportion * data_size)
        if length % 2 == 0:
            length += 1
        if length < 1:
            return 1
        return length


class SpectraSmootherGaussian(SpectraProcess):
    """Smooth a spectra with a gaussian filter

    For more information, please refer to:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter1d.html

    **Examples**

    Smooth a simple spectra data instance

    ```{doctest}
    >>> from resistics.spectra import SpectraSmootherGaussian
    >>> from resistics.testing import spectra_data_basic
    >>> spec_data = spectra_data_basic()
    >>> smooth_data = SpectraSmootherGaussian().run(spec_data)

    ```

    Look at the results for the two windows

    ```{doctest}
    >>> spec_data.data[0][0,0]
    array([0.+0.j, 1.+1.j, 2.+2.j, 3.+3.j, 4.+4.j, 5.+5.j, 6.+6.j, 7.+7.j,
           8.+8.j, 9.+9.j])
    >>> smooth_data.data[0][0,0]
    array([1.93603671+1.93603671j, 2.1921536 +2.1921536j ,
           2.67507336+2.67507336j, 3.33255376+3.33255376j,
           4.09862656+4.09862656j, 4.90137344+4.90137344j,
           5.66744624+5.66744624j, 6.32492664+6.32492664j,
           6.8078464 +6.8078464j , 7.06396329+7.06396329j])

    ```
    """

    sigma: float = 3

    def run(self, spec_data: SpectraData) -> SpectraData:
        """Run Gaussian filtering of spectra data

        :param spec_data: Input spectra data

        :return: Output spectra data
        """
        import scipy.ndimage as ndimage

        data = {}
        logger.info(f"Smoothing frequencies with gaussian filter, sigma {self.sigma}")
        messages = [f"Smoothing frequencies with gaussian filter, sigma {self.sigma}"]
        for ilevel in range(spec_data.metadata.n_levels):
            data[ilevel] = ndimage.gaussian_filter1d(
                spec_data.get_level(ilevel), self.sigma, axis=-1
            )
            messages.append(f"Smoothed level {ilevel} with gaussian filter")
        metadata = SpectraMetadata(**spec_data.metadata.model_dump())
        metadata.history.add_record(self._get_record(messages))
        logger.info("Fourier coefficients calculated at evaluation frequencies")
        return SpectraData(metadata, data)
