from pathlib import Path

import numpy as np

from resistics.calibrate import (
    CalibrationData,
    Calibrator,
    SensorCalibrationJSON,
    SensorCalibrationTXT,
)
from resistics.time import ChanMetadata


def get_cal_data() -> CalibrationData:
    """Get some testing calibrationd data"""
    return CalibrationData(
        sensor="testsens",
        serial=20,
        static_gain=5,
        frequency=[1, 2, 3, 4, 5],
        magnitude=[10, 11, 12, 13, 14],
        phase=[0.1, 0.2, 0.3, 0.4, 0.5],
    )


def test_calibration_interpolation_returns_complex_array():
    """Calibration interpolation should preserve its numerical array contract."""
    frequencies = np.array([1.0, 2.5, 5.0])

    interpolated = Calibrator()._interpolate(frequencies, get_cal_data())

    expected_magnitude = np.array([10.0, 11.5, 14.0])
    expected_phase = np.array([0.1, 0.25, 0.5])
    assert isinstance(interpolated, np.ndarray)
    assert np.iscomplexobj(interpolated)
    np.testing.assert_allclose(
        interpolated, expected_magnitude * np.exp(1j * expected_phase)
    )


def test_sensor_calibration_json(monkeypatch):
    """Test reading of sensor calibration files"""

    def mock_read_bytes(*args):
        """Mock the read_bytes used by pydantic"""
        return get_cal_data().json().encode()

    monkeypatch.setattr(Path, "read_bytes", mock_read_bytes)
    reader = SensorCalibrationJSON()
    cal_data = reader.read_calibration_data(
        Path("test.json"), ChanMetadata(name="Ex", data_files=[])
    )
    cal_data_reference = get_cal_data()
    cal_data_reference.file_path = Path("test.json")
    assert cal_data == cal_data_reference


class MockFileObject:
    """Mock for a file object"""

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        return

    def readlines(self):
        """Readlines"""
        cal_data = get_cal_data()
        lines = []
        lines.append(f"Serial = {cal_data.serial}")
        lines.append(f"Sensor = {cal_data.sensor}")
        lines.append(f"Static gain = {cal_data.static_gain}")
        lines.append("Magnitude unit = mV/nT")
        lines.append("Phase unit = radians")
        lines.append("Chopper = False")
        lines.append("")
        lines.append("CALIBRATION DATA")
        for f, m, p in zip(
            cal_data.frequency, cal_data.magnitude, cal_data.phase, strict=False
        ):
            lines.append(f"{f} {m} {p}")
        return lines


def test_sensor_calibration_txt(monkeypatch):
    """Test reading of text based calibration files"""

    def mock_open(*args, **kwargs):
        """Return mock file object"""
        return MockFileObject()

    monkeypatch.setattr(Path, "open", mock_open)
    reader = SensorCalibrationTXT()
    cal_data = reader.read_calibration_data(
        Path("test.txt"), ChanMetadata(name="Ex", data_files=[])
    )
    cal_data_reference = get_cal_data()
    cal_data_reference.file_path = Path("test.txt")
    assert cal_data == cal_data_reference
