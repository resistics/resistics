"""
Module containing the resistics configuration

The configuration is an essential part of a resistics environment. It defines
many dependencies, such as which data readers to use for time series data or
calibration data and processing options.

Configuration allows users to insert their own dependencies and processors to
work with data.

Configurations can be saved to and loaded from JSON files.
"""
from typing import List

from resistics.common import ResisticsModel, ResisticsProcess
from resistics.time import TimeReader, TimeReaderNumpy, TimeReaderAscii
from resistics.time import InterpolateNans, RemoveMean
from resistics.calibrate import SensorCalibrationJSON, SensorCalibrator
from resistics.decimate import DecimationSetup
from resistics.decimate import Decimator
from resistics.window import WindowSetup, Windower
from resistics.spectra import FourierTransform, EvaluationFreqs
from resistics.transfunc import TransferFunction, ImpedanceTensor
from resistics.regression import RegressionPreparerGathered, SolverScikitTheilSen


class Configuration(ResisticsModel):
    """
    The resistics configuration

    Configuration can be customised by users who wish to use their own custom
    processes for certain steps. In most cases, customisation will be for:

    - Implementing new time data readers
    - Implementing readers for specific calibration formats
    - Adding new features to extract from the data

    Examples
    --------
    Frequently, configuration will be used to change data readers.

    >>> from resistics.letsgo import get_default_configuration
    >>> config = get_default_configuration()
    >>> config.name
    'default'
    >>> for tr in config.time_readers:
    ...     tr.summary()
    {
        'name': 'TimeReaderAscii',
        'apply_scalings': True,
        'extension': '.ascii'
    }
    {
        'name': 'TimeReaderNumpy',
        'apply_scalings': True,
        'extension': '.npy'
    }
    >>> config.sensor_calibrator.summary()
    {
        'name': 'SensorCalibrator',
        'chans': None,
        'readers': [
            {
                'name': 'SensorCalibrationJSON',
                'extension': '.json',
                'file_str': 'IC_$sensor$extension'
            }
        ]
    }

    To change these, it's best to make a new configuration with a different name

    >>> from resistics.letsgo import Configuration
    >>> from resistics.time import TimeReaderNumpy
    >>> config = Configuration(name="myconfig", time_readers=[TimeReaderNumpy(apply_scalings=False)])
    >>> for tr in config.time_readers:
    ...     tr.summary()
    {
        'name': 'TimeReaderNumpy',
        'apply_scalings': False,
        'extension': '.npy'
    }

    Or for the sensor calibration

    >>> from resistics.calibrate import SensorCalibrator, SensorCalibrationTXT
    >>> calibration_reader = SensorCalibrationTXT(file_str="lemi120_IC_$serial$extension")
    >>> calibrator = SensorCalibrator(chans=["Hx", "Hy", "Hz"], readers=[calibration_reader])
    >>> config = Configuration(name="myconfig", sensor_calibrator=calibrator)
    >>> config.sensor_calibrator.summary()
    {
        'name': 'SensorCalibrator',
        'chans': ['Hx', 'Hy', 'Hz'],
        'readers': [
            {
                'name': 'SensorCalibrationTXT',
                'extension': '.TXT',
                'file_str': 'lemi120_IC_$serial$extension'
            }
        ]
    }

    As a final example, create a configuration which used targetted windowing
    instead of specified window sizes

    >>> from resistics.letsgo import Configuration
    >>> from resistics.window import WindowerTarget
    >>> config = Configuration(name="window_target", windower=WindowerTarget(target=500))
    >>> config.name
    'window_target'
    >>> config.windower.summary()
    {
        'name': 'WindowerTarget',
        'target': 500,
        'min_size': 64,
        'olap_proportion': 0.25
    }
    """

    name: str
    """The name of the configuration"""
    time_readers: List[TimeReader] = [TimeReaderAscii(), TimeReaderNumpy()]
    """Time readers in the configuration"""
    time_processors: List[ResisticsProcess] = [InterpolateNans(), RemoveMean()]
    """List of time processors to run"""
    dec_setup: ResisticsProcess = DecimationSetup()
    """Process to calculate decimation parameters"""
    decimator: ResisticsProcess = Decimator()
    """Process to decimate time data"""
    win_setup: ResisticsProcess = WindowSetup()
    """Process to calculate windowing parameters"""
    windower: ResisticsProcess = Windower()
    """Process to window the decimated data"""
    fourier: ResisticsProcess = FourierTransform()
    """Process to perform the fourier transform"""
    spectra_processors: List[ResisticsProcess] = []
    """List of processors to run on spectra data"""
    evals: ResisticsProcess = EvaluationFreqs()
    """Process to get the spectra data at the evaluation frequencies"""
    sensor_calibrator: ResisticsProcess = SensorCalibrator(
        readers=[SensorCalibrationJSON()]
    )
    """The sensor calibrator and associated calibration file readers"""
    tf: TransferFunction = ImpedanceTensor()
    """The transfer function to solve"""
    regression_preparer: ResisticsProcess = RegressionPreparerGathered()
    """Process to prepare linear equations"""
    solver: ResisticsProcess = SolverScikitTheilSen()
    """The solver to use to estimate the regression parameters"""


def get_default_configuration() -> Configuration:
    """Get the default configuration"""
    return Configuration(name="default")
