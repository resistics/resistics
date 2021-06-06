"""
Functions and classes for instrument and sensor calibration of data

Calibration data should be given in the frequency domain and has a magnitude
and phase component (in radians). Calibration data is the impulse response for
an instrument or sensor and is usually deconvolved (division in frequency
domain) from the time data.

Notes
-----
Calibration data for induction coils is given in mV/nT. Because this is
deconvolved from magnetic time data, which is in mV, the resultant magnetic
time data is in nT.
"""
from resistics.spectra import SpectraData
from loguru import logger
from typing import Optional, Any, List, Tuple, Union, Dict
from pathlib import Path
from pydantic import validator
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from resistics.common import ResisticsProcess, WriteableMetadata
from resistics.spectra import SpectraMetadata
from resistics.errors import CalibrationFileNotFound


class CalibrationData(WriteableMetadata):
    """
    Class for holding calibration data

    Calibration is usually the transfer function of the instrument or sensor
    to be removed from the data. It is expected to be in the frequency domain.

    Regarding units:

    - Magnitude units are dependent on use case
    - Phase is in radians
    """

    file_path: Optional[Path]
    """Path to the calibration file"""
    sensor: str = ""
    """Sensor type"""
    serial: Union[int, str]
    """Serial number of the sensor"""
    static_gain: float = 1
    """Static gain to apply"""
    magnitude_unit: str = "mV/nT"
    """Units of the magnitude"""
    frequency: List[float]
    """Frequencies in Hz"""
    magnitude: List[float]
    """Magnitude"""
    phase: List[float]
    """Phase"""
    n_samples: Optional[int] = None
    """Number of data samples"""

    @validator("n_samples", always=True)
    def validate_n_samples(cls, value: Union[int, None], values: Dict[str, Any]) -> int:
        """Validate number of samples"""
        if value is None:
            return len(values["frequency"])
        return value

    def __getitem__(self, arg: str) -> np.ndarray:
        """Get data mainly for the purposes of plotting"""
        if arg == "frequency":
            return np.array(self.frequency)
        if arg == "magnitude":
            return np.array(self.magnitude)
        if arg == "phase":
            return np.array(self.phase)
        raise ValueError(
            f"Unknown arg {arg}, must be: 'frequency', 'magnitude' or 'phase'"
        )

    def plot(
        self, fig: Optional[go.Figure] = None, label_prefix: str = ""
    ) -> go.Figure:
        """
        Plot calibration data

        Parameters
        ----------
        fig : Optional[go.Figure], optional
            Plotly figure, by default None. If no figure is provided, a new one
            will be created.
        label_prefix : str, optional
            Prefix to add to the plot labels, by default ""

        Returns
        -------
        go.Figure
            Plotly figure
        """
        from resistics.plot import figure_columns_as_lines, plot_columns_1d

        subplots = ["Magnitude", "Phase"]
        subplot_columns = {"Magnitude": ["magnitude"], "Phase": ["phase"]}
        y_labels = {"Magnitude": self.magnitude_unit, "Phase": "radians"}
        if fig is None:
            fig = figure_columns_as_lines(
                subplots=subplots, y_labels=y_labels, x_label="Frequency Hz"
            )
            fig.update_xaxes(type="log")
        plot_columns_1d(
            fig,
            self,
            subplots,
            subplot_columns,
            max_pts=None,
            label_prefix=label_prefix,
        )
        return fig

    def x_size(self) -> int:
        """Get x size for plotting"""
        return self.n_samples

    def get_x(self, samples: Optional[np.ndarray] = None) -> np.ndarray:
        """Get x axis for plotting 1-D"""
        freqs = np.array(self.frequency)
        if samples is not None:
            return freqs[samples]
        return freqs

    def to_dataframe(self):
        """Convert to pandas DataFrame"""
        data = {
            "frequency": self.frequency,
            "magnitude": self.magnitude,
            "phase": self.phase,
        }
        df = pd.DataFrame(data=data)
        return df.set_index("frequency")


class CalibrationReader(ResisticsProcess):
    """Parent class for reading calibration data"""

    extension: Optional[str] = None


class InstrumentCalibrationReader(CalibrationReader):
    """Parent class for reading instrument calibration files"""

    def run(self, metadata: SpectraMetadata) -> CalibrationData:
        raise NotImplementedError("To be implemented in child classes")


class SensorCalibrationReader(CalibrationReader):
    """
    Parent class for reading sensor calibration files

    Use this reader for induction coil calibration file readers

    Examples
    --------
    A short example to show how naming substitution works

    >>> from pathlib import Path
    >>> from resistics.testing import time_metadata_1chan
    >>> from resistics.calibrate import SensorCalibrationReader
    >>> calibration_path = Path("test")
    >>> metadata = time_metadata_1chan()
    >>> metadata.chans_metadata["chan1"].sensor = "example"
    >>> metadata.chans_metadata["chan1"].serial = "254"
    >>> calibrator = SensorCalibrationReader(extension=".json")
    >>> calibrator.file_str
    'IC_$sensor$extension'
    >>> file_path = calibrator._get_path(calibration_path, metadata, "chan1")
    >>> file_path.name
    'IC_example.json'

    If the file name has a different pattern, the file_str can be changed as
    required.

    >>> calibrator = SensorCalibrationReader(file_str="$sensor_$serial$extension", extension=".json")
    >>> file_path = calibrator._get_path(calibration_path, metadata, "chan1")
    >>> file_path.name
    'example_254.json'
    """

    file_str: str = "IC_$sensor$extension"

    def run(
        self, dir_path: Path, metadata: SpectraMetadata, chan: str
    ) -> CalibrationData:
        """
        Run the calibration file reader

        Parameters
        ----------
        dir_path : Path
            The directory with calibration files
        metadata : SpectraMetadata
            TimeData metadata
        chan : str
            The channel for which to search for a calibration file

        Raises
        ------
        NotImplementedError
            To be implemented in child readers
        """
        raise NotImplementedError("To be implemented in child classes")

    def _get_path(self, dir_path: Path, metadata: SpectraMetadata, chan: str) -> Path:
        """Get the expected Path to the calibration file"""
        chan_metadata = metadata.chans_metadata[chan]
        name = self.file_str.replace("$sensor", chan_metadata.sensor)
        name = name.replace("$serial", chan_metadata.serial)
        if self.extension is not None:
            name = name.replace("$extension", self.extension)
        return dir_path / name


class SensorCalibrationJSON(SensorCalibrationReader):
    """
    Read in JSON formatted calibration data

    Examples
    --------

    """

    extension: str = ".json"

    def run(
        self, dir_path: Path, metadata: SpectraMetadata, chan: str
    ) -> CalibrationData:
        """
        Get sensor calibration data from JSON file

        Parameters
        ----------
        dir_path : Path
            The directory to search for a calibration file
        metadata : SpectraMetadata
            The time data metadata
        chan : str
            The channel to get calibration data for

        Returns
        -------
        CalibrationData
            The calibration data

        Raises
        ------
        CalibrationFileNotFound
            If no matching file is found in dir_path
        """
        file_path = self._get_path(dir_path, metadata, chan)
        logger.info(f"Searching file {file_path.name} in {dir_path}")
        if not file_path.exists():
            raise CalibrationFileNotFound(dir_path, file_path)
        logger.info(f"Reading file {file_path.name}")

        cal_data = CalibrationData.parse_file(file_path)
        cal_data.file_path = file_path
        return cal_data


class SensorCalibrationTXT(SensorCalibrationReader):
    """
    Read in calibration data from a TXT file

    Use of this calibration reader is discouraged. Instead, where a choice is
    available, users should try and format their calibration data in the JSON
    format as this is immediately much more portable

    See Also
    --------
    CalibrationReaderJSON : Reader for JSON calibration files
    """

    extension = ".TXT"

    def run(
        self, dir_path: Path, metadata: SpectraMetadata, chan: str
    ) -> CalibrationData:
        """
        Get sensor calibration data

        Parameters
        ----------
        dir_path : Path
            The directory to search for a calibration file
        metadata : SpectraMetadata
            The time data metadata
        chan : str
            The channel to get calibration data for

        Returns
        -------
        CalibrationData
            The calibration data

        Raises
        ------
        CalibrationFileNotFound
            If no matching file is found in dir_path
        """
        file_path = self._get_path(dir_path, metadata, chan)
        logger.info(f"Searching file {file_path.name} in {dir_path}")
        if not file_path.exists():
            raise CalibrationFileNotFound(dir_path, file_path)

        logger.info(f"Reading file {file_path.name}")
        with file_path.open("r") as f:
            lines = f.readlines()
        lines = [x.strip() for x in lines]
        data_dict = self._read_metadata(lines)
        df = self._read_data(lines, data_dict)
        data_dict["frequency"] = df.index.values.tolist()
        data_dict["magnitude"] = df["magnitude"].values.tolist()
        data_dict["phase"] = df["phase"].values.tolist()
        data_dict["file_path"] = file_path
        return CalibrationData(**data_dict)

    def _read_metadata(self, lines: List[str]) -> Dict[str, Any]:
        """Read data from the calibration file"""
        serial, sensor = self._get_sensor_details(lines)
        static_gain = self._get_static_gain(lines)
        chopper = self._get_chopper(lines)
        magnitude_unit, phase_unit = self._get_units(lines)
        return {
            "serial": serial,
            "sensor": sensor,
            "static_gain": static_gain,
            "chopper": chopper,
            "magnitude_unit": magnitude_unit,
            "phase_unit": phase_unit,
        }

    def _get_sensor_details(self, lines: List[str]) -> Tuple[str, str]:
        """Get sensor details"""
        serial: str = "1"
        sensor: str = ""
        for line in lines:
            line = line.lower()
            if "serial" in line:
                serial = line.split("=")[1].strip()
            if "sensor" in line:
                sensor = line.split("=")[1].strip()
        return serial, sensor

    def _get_static_gain(self, lines: List[str]) -> float:
        """Get static gain"""
        static_gain = 1.0
        for line in lines:
            if "static gain" in line.lower():
                static_gain = float(line.split("=")[1].strip())
                return static_gain
        return static_gain

    def _get_chopper(self, lines: List[str]) -> bool:
        """Get chopper"""
        for line in lines:
            if "chopper" in line.lower():
                chopper_str = line.split("=")[1].strip()
                return chopper_str == "True"
        return False

    def _get_units(self, lines: List[str]) -> Tuple[str, str]:
        """Get units for the magnitude and phase"""
        magnitude_unit: str = "mV/nT"
        phase_unit: str = "radians"
        for line in lines:
            line = line.lower()
            if "magnitude unit" in line:
                magnitude_unit = line.split("=")[1].strip()
            if "phase unit" in line:
                phase_unit = line.split("=")[1].strip()
        return magnitude_unit, phase_unit

    def _read_data(self, lines: List[str], data_dict: Dict[str, Any]) -> pd.DataFrame:
        """Read the calibration data lines"""
        read_from = self._get_read_from(lines)
        data_lines = self._get_data_lines(lines, read_from)
        # convert lines to data frame
        data = np.array([x.split() for x in data_lines], dtype=float)
        df = pd.DataFrame(data=data, columns=["frequency", "magnitude", "phase"])
        # unit manipulation - change phase to radians
        if data_dict["phase_unit"] == "degrees":
            df["phase"] = df["phase"] * (np.pi / 180)
        df = df.set_index("frequency").sort_index()
        return df

    def _get_read_from(self, lines: List[str]) -> int:
        """Get the line number to read from"""
        for idx, line in enumerate(lines):
            if "CALIBRATION DATA" in line:
                return idx + 1
        raise ValueError("Unable to determine location of data in file")

    def _get_data_lines(self, lines: List[str], idx: int) -> List[str]:
        """Get the data lines out of the file"""
        data_lines: List[str] = []
        while idx < len(lines) and lines[idx] != "":
            data_lines.append(lines[idx])
            idx += 1
        return data_lines


class Calibrator(ResisticsProcess):
    """Parent class for a calibrator"""

    chans: Optional[List[str]] = None
    """List of channels to calibrate"""

    def _get_chans(self, chans: List[str]) -> List[str]:
        """Get the channels to calibrate"""
        if self.chans is None:
            return chans
        return [x for x in self.chans if x in chans]

    def _calibrate(
        self, freqs: List[float], chan_data: np.ndarray, cal_data: CalibrationData
    ) -> np.ndarray:
        """
        Calibrate a channel

        This is essentially a deconvolution, which means a division in frequency
        domain.

        Parameters
        ----------
        freqs : List[float]
            List of frequencies to interpolate calibration data to
        chan_data : np.ndarray
            Channel data
        cal_data : CalibrationData
            CalibrationData instance with calibration information

        Returns
        -------
        np.ndarray
            Calibrated data
        """
        transfunc = self._interpolate(np.array(freqs), cal_data)
        chan_data = chan_data[:, np.newaxis, :] / transfunc[np.newaxis, :]
        return np.squeeze(chan_data)

    def _interpolate(
        self, freqs: np.ndarray, cal_data: CalibrationData
    ) -> pd.DataFrame:
        """
        Interpolate the calibration data to the same frequencies as the time
        data

        Static gain is assumed to already be applied in the magnitude and is not
        applied separately.

        Parameters
        ----------
        freqs : np.ndarray
            The frequencies in the time data
        cal_data : CalibrationData
            The calibration data

        Returns
        -------
        pd.DataFrame
            The data interpolated to the frequencies and with an additional
            column, complex, which is the complex values for the magnitude and
            phase combinations.
        """
        mag = np.interp(freqs, cal_data.frequency, cal_data.magnitude)
        phs = np.interp(freqs, cal_data.frequency, cal_data.phase)
        return mag * np.exp(1j * phs)


class InstrumentCalibrator(Calibrator):

    readers: List[InstrumentCalibrationReader]
    """List of readers for reading in instrument calibration files"""

    def run(self, dir_path, spec_data: SpectraData) -> SpectraData:
        """Run the instrument calibration"""
        raise NotImplementedError("To be implemented")


class SensorCalibrator(Calibrator):

    readers: List[SensorCalibrationReader]
    """List of readers for reading in sensor calibration files"""

    def run(self, dir_path: Path, spec_data: SpectraData) -> SpectraData:
        """Calibrate Spectra data"""
        chans = self._get_chans(spec_data.metadata.chans)
        logger.info(f"Calibrating channels {chans}")
        messages = [f"Calibrating channels {chans}"]
        data = {x: np.array(y) for x, y in spec_data.data.items()}
        for chan in chans:
            logger.info(f"Looking for sensor calibration data for channel {chan}")
            cal_data = self._get_cal_data(dir_path, spec_data.metadata, chan)
            if cal_data is None:
                logger.info(f"No calibration data for channel {chan}")
                messages.append(f"No calibration data for channel {chan}")
                continue
            logger.info(f"Calibrating {chan} with data from {cal_data.file_path}")
            idx = spec_data.metadata.chans.index(chan)
            for ilevel in range(spec_data.metadata.n_levels):
                level_metadata = spec_data.metadata.levels_metadata[ilevel]
                data[ilevel][:, idx] = self._calibrate(
                    level_metadata.freqs, data[ilevel][:, idx], cal_data
                )
            messages.append(f"Calibrated {chan} with data from {cal_data.file_path}")
        metadata = SpectraMetadata(**spec_data.metadata.dict())
        metadata.history.add_record(self._get_record(messages))
        return SpectraData(metadata, data)

    def _get_cal_data(
        self, dir_path: Path, metadata: SpectraMetadata, chan: str
    ) -> Union[CalibrationData, None]:
        """Get the calibration data"""
        cal_data = None
        for reader in self.readers:
            try:
                cal_data = reader.run(dir_path, metadata, chan)
                break
            except CalibrationFileNotFound:
                logger.debug(f"Calibration reader {reader.name} did not find file")
            except Exception:
                logger.debug(f"Calibration reader {reader.name} failed reading file")
        return cal_data
