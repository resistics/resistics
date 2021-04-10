"""
Functions and classes for instrument and sensor calibration of data
"""
from logging import getLogger
from typing import Dict, Any, Union, List, Tuple, Any
from pathlib import Path
import numpy as np
import pandas as pd

from resistics.common import ResisticsData, ResisticsProcess
from resistics.common import ProcessRecord, ProcessHistory, Headers

logger = getLogger(__name__)


calibration_header_specs = {
    "data_file": {"type": str, "default": None},
    "n_samples": {"type": int, "default": None},
    "serial": {"type": int, "default": 1},
    "sensor": {"type": str, "default": ""},
    "static_gain": {"type": float, "default": 1},
    "chopper": {"type": bool, "default": False},
    "mag_unit": {"type": str, "default": "mV/nT"},
    "phs_unit": {"type": str, "default": "radians"},
}


def get_calibration_headers(
    calibration_headers: Dict[str, Any],
) -> Headers:
    """
    Calibration headers

    Parameters
    ----------
    calibration_headers : Dict[str, Any]
        The calibration headers

    Returns
    -------
    Headers
        The Headers instance with any missing values filled in with defaults
    """
    return Headers(calibration_headers, calibration_header_specs)


class CalibrationData(ResisticsData):
    """Class for holding calibration data

    Calibration data should be given in the frequency domain and has a magnitude and phase component (in radians). Calibration data is the impulse response for an instrument or sensor and is usually deconvolved (division in frequency domain) from the time data.

    Notes
    -----
    Calibration data for magnetic channels is given in mV/nT. Because this is deconvolved from magnetic time data, which is in mV, the resultant magnetic time data is in nT.

    Examples
    --------
    .. doctest::

        >>> from resistics.testing import calibration_data_linear
        >>> cal_data = calibration_data_linear(n_samples=10)
        >>> cal_data.headers.summary()
    """

    def __init__(
        self,
        headers: Headers,
        df: pd.DataFrame,
        history: ProcessHistory,
    ) -> None:
        """
        Initialise

        Parameters
        ----------
        headers : Headers
            Calibration file headers
        df : pd.DataFrame
            Calibration file data
        history : ProcessHistory
            The history for the calibration data
        """
        self.headers = headers
        self.df = df
        self.history = history

    def __getitem__(self, column: str) -> np.ndarray:
        """
        Get a column as an numpy array

        Parameters
        ----------
        column : str
            The column to get, should be magnitude or phase

        Returns
        -------
        np.ndarray
            The data
        """
        if column not in ["magnitude", "phase"]:
            raise KeyError("Column must be one of magnitude or phase")
        return self.df[column].values

    def x_size(self) -> int:
        """
        Get the x size for plotting, same as number of samples

        Returns
        -------
        int
            The x size
        """
        return self.headers["n_samples"]

    def get_x(self) -> np.ndarray:
        """
        Return the frequencies

        Returns
        -------
        np.ndarray
            An array of the frequencies
        """
        return self.df.index.values

    def to_string(self) -> str:
        """
        Get class information as a string

        Returns
        -------
        str
            Class information as string
        """
        outstr = f"{self.headers.to_string()}\n"
        outstr += f"{self.history.to_string()}\n"
        outstr += "Data\n"
        outstr += f"{self.df.to_string()}"
        return outstr

    def plot(
        self,
        fig=None,
        label_prefix="",
    ) -> Any:
        """
        Plot calibration data

        Parameters
        ----------
        fig : [type], optional
            Plotly figure, by default None. If no figure is provided, a new one will be created.

        Returns
        -------
        fig
            Plotly figure
        """
        from resistics.plot import figure_columns_as_lines, plot_columns_1d

        subplots = ["Magnitude", "Phase"]
        subplot_columns = {"Magnitude": ["magnitude"], "Phase": ["phase"]}
        y_labels = {
            "Magnitude": self.headers["mag_unit"],
            "Phase": self.headers["phs_unit"],
        }
        if fig is None:
            fig = figure_columns_as_lines(
                subplots=subplots, y_labels=y_labels, x_label="Frequency Hz"
            )
        plot_columns_1d(
            fig,
            self,
            subplots,
            subplot_columns,
            max_pts=None,
            label_prefix=label_prefix,
        )
        return fig


class CalibrationReader(ResisticsProcess):
    def __init__(self, file_path: Path):
        self.file_path = file_path


class InstrumentCalibrationReader(CalibrationReader):
    pass


class SensorCalibrationReader(CalibrationReader):
    def check(self) -> bool:
        """Can the file be read"""
        try:
            headers = self.read_headers()
        except:
            return False
        self.headers = headers
        return True

    def read_headers(self):
        raise NotImplementedError("Should be implemented in child classes")

    def run(self) -> CalibrationData:
        return self.read_data()

    def read_data(self):
        raise NotImplementedError("This should be implemented in child reader classes")


class CalibrationTXT(SensorCalibrationReader):
    def get_names(self, metadata: Dict[str, Any]) -> List[str]:
        """Get the expected name of the calibrate file for the metadata"""
        return [f"IC_{metadata['serial']}.{self.extension}"]

    def read(self, file_path: Path, metadata: Dict[str, Any]) -> CalibrationData:
        from resistics.common import lines_to_array
        import math

        with file_path.open("r") as f:
            lines = f.readlines()
        lines = [x.strip() for x in lines]

        serial, sensor = self._get_sensor_details(lines)
        static_gain = self._get_static_gain(lines)
        chopper = self._get_chopper(lines)
        mag_unit, phs_unit = self._get_units(lines)
        read_from = self._get_read_from(lines)
        data_lines = self._get_data_lines(lines, read_from)
        # convert lines to data frame
        data = lines_to_array(data_lines)
        df = pd.DataFrame(data=data, columns=["frequency", "magnitude", "phase"])
        # unit manipulation - change phase to radians
        if phs_unit == "degrees":
            df["phase"] = df["phase"] * (math.pi / 180)
        df = df.set_index("frequency").sort_index()

        return CalibrationData(
            file_path,
            df,
            ProcessHistory(),
            chopper=chopper,
            serial=serial,
            sensor=sensor,
            static_gain=static_gain,
        )

    def _get_sensor_details(self, lines: List[str]) -> Tuple[int, str]:
        serial: int = 1
        sensor: str = ""
        for line in lines:
            line = line.lower()
            if "serial" in line:
                serial = int(line.split("=")[1].strip())
            if "sensor" in line:
                sensor = line.split("=")[1].strip()
        return serial, sensor

    def _get_static_gain(self, lines: List[str]) -> float:
        static_gain: float = 1
        for line in lines:
            line = line.lower()
            if "static gain" in line:
                static_gain = float(line.split("=")[1].strip())
                return static_gain
        return static_gain

    def _get_chopper(self, lines: List[str]) -> bool:
        for line in lines:
            line = line.lower()
            if "chopper" in line:
                chopper_str = line.split("=")[1].strip()
                if chopper_str == "True":
                    return True
                else:
                    return False
        return False

    def _get_units(self, lines: List[str]) -> Tuple[str, str]:
        mag_unit: str = "mV/nT"
        phs_unit: str = "radians"
        for line in lines:
            line = line.lower()
            if "mag unit" in line:
                mag_unit = line.split("=")[1].strip()
            if "phs unit" in line:
                phs_unit = line.split("=")[1].strip()
        return mag_unit, phs_unit

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


class CalibrationJSON(SensorCalibrationReader):
    def __init__(self, dir_path: Path):
        """Initialise"""
        super().__init__(dir_path, ["serial"], ".json")

    def get_names(self, metadata: Dict[str, Any]) -> List[str]:
        """Get the expected name of the calibrate file for the metadata"""
        return [f"IC_{metadata['serial']}.{self.extension}"]

    def read(self, file_path: Path, metadata: Dict[str, Any]) -> CalibrationData:
        """Read data from a text based calibration file for any type of instrument

        Notes
        -----
        Text based calibration files should be used when reading of other formats appears to be failing or are not supported. Nothing is assumed or promised about the data units.

        Returns
        -------
        CalibrationData
            A calibration data object
        """
        import math
        import json

        with open(file_path, "r") as f:
            json_data = json.load(f)
        # read data
        df = pd.DataFrame(
            {
                "frequency": json_data["data"]["freqs"],
                "magntiude": json_data["data"]["mag"],
                "phase": json_data["data"]["phs"],
            }
        )
        df = df.sort_index()
        # apply corrections
        df["magnitude"] = df["magnitude"] * json_data["static_gain"]
        if json_data["phs_unit"] == "degrees":
            df["phase"] = df["phase"] * (math.pi / 180)

        return CalibrationData(
            file_path,
            df,
            ProcessHistory(),
            chopper=json_data["chopper"],
            serial=json_data["serial"],
            sensor=json_data["sensor"],
            static_gain=json_data["static_gain"],
        )


def write_to_json(self, file_path: Path, cal_data: CalibrationData) -> None:
    """Write out calibration data to a json file

    Parameters
    ----------
    calibrationData : CalibrationData
        Calibration data to write out
    filepath : str
        The file to write out to
    """
    import json

    with file_path.open("w") as json_file:
        json.dump(cal_data.to_dict(), json_file)


def write_to_txt(
    file_path: Path,
    cal_data: Union[CalibrationData, None],
    serial: int = 1,
    sensor: str = "",
    static_gain: float = 1.0,
    chopper: bool = False,
    mag_unit: str = "mV/nT",
    phs_unit: str = "degrees",
    df: Union[pd.DataFrame, None] = None,
) -> None:
    """Write out a calibration data to text file

    Can also be used to write a template calibration file
    """
    if cal_data is not None:
        serial = cal_data.serial
        sensor = cal_data.sensor
        static_gain = cal_data.static_gain
        chopper = cal_data.chopper
        mag_unit = cal_data.mag_unit
        phs_unit = cal_data.phs_unit
        df = cal_data.df

    with file_path.open("w") as f:
        f.write("serial = {}\n".format(serial))
        f.write("sensor = {}\n".format(sensor))
        f.write("static gain = {}\n".format(static_gain))
        f.write("mag unit = {}\n".format(mag_unit))
        f.write("phs unit = {}\n".format(phs_unit))
        f.write("chopper = {}\n".format(chopper))
        f.write("\n")
        f.write("CALIBRATION DATA\n")
        if df is not None:
            for idx, row in df.iterrows():
                mag = row["magnitude"]
                phs = row["phase"]
                f.write(f"{idx:.8e}\t{mag:.8e}\t{phs:.8e}\n")


