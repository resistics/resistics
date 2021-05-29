"""
This module is the main interface to resistics and includes:

- Classes and functions for making, loading and using resistics projects
- Functions for processing data
"""
from loguru import logger
from typing import Iterator, Union, Optional, List, Dict, Any
from pathlib import Path
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from resistics.errors import MetadataReadError, TimeDataReadError
from resistics.common import ResisticsModel, WriteableMetadata, ResisticsProcess
from resistics.sampling import HighResDateTime
from resistics.time import TimeMetadata, TimeData
from resistics.time import TimeReader, TimeReaderNumpy, TimeReaderAscii
from resistics.time import InterpolateNans, RemoveMean
from resistics.calibrate import SensorCalibrationJSON, SensorCalibrator
from resistics.decimate import DecimationParameters, DecimationSetup
from resistics.decimate import Decimator, DecimatedData
from resistics.window import WindowSetup, Windower, WindowedData
from resistics.spectra import FourierTransform, EvaluationFreqs, SpectraData
from resistics.regression import ImpedanceTensor, RegressionPreparer
from resistics.regression import RegressionInputData


PROJ_FILE = "resistics.json"
PROJ_DIRS = {
    "time": "time",
    "calibration": "calibrate",
    "spectra": "spectra",
    "features": "features",
    "masks": "masks",
    "results": "results",
    "images": "images",
}


def get_calibration_path(proj_dir: Path) -> Path:
    """Get the path to the calibration data"""
    return proj_dir / PROJ_DIRS["calibration"]


def get_meas_time_path(proj_dir: Path, site_name: str, meas_name: str) -> Path:
    """Get path to measurement time data"""
    return proj_dir / PROJ_DIRS["time"] / site_name / meas_name


def get_meas_spectra_path(
    proj_dir: Path, site_name: str, meas_name: str, config_name: str
) -> Path:
    """Get path to measurement spectra data"""
    return proj_dir / PROJ_DIRS["spectra"] / site_name / meas_name / config_name


def get_meas_features_path(
    proj_dir: Path, site_name: str, meas_name: str, config_name: str
) -> Path:
    """Get path to measurement features data"""
    return proj_dir / PROJ_DIRS["features"] / site_name / meas_name / config_name


def get_fs_mask_path(
    proj_dir: Path, site_name: str, config_name: str, fs: float
) -> Path:
    """Get path to sampling frequency mask data"""
    from resistics.common import fs_to_string

    mask_name = f"{fs_to_string(fs)}.pkl"
    return proj_dir / PROJ_DIRS["masks"] / site_name / config_name / mask_name


def get_fs_results_path(
    proj_dir: Path, site_name: str, config_name: str, fs: float
) -> Path:
    """Get path to sampling frequency results"""
    from resistics.common import fs_to_string

    results_name = f"{fs_to_string(fs)}.json"
    return proj_dir / PROJ_DIRS["results"] / site_name / config_name / results_name


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
    evals: ResisticsProcess = EvaluationFreqs()
    """Process to get the spectra data at the evaluation frequencies"""
    sensor_calibrator: ResisticsProcess = SensorCalibrator(
        readers=[SensorCalibrationJSON()]
    )
    """The sensor calibrator and associated calibration file readers"""
    regression_preparer: ResisticsProcess = RegressionPreparer(tf=ImpedanceTensor())
    """Process to prepare linear equations"""


def get_default_configuration():
    """Get the default configuration"""
    return Configuration(name="default")


class Measurement(ResisticsModel):
    """
    Class for interfacing with a measurement

    The class holds the original time series metadata and can provide
    information about other types of data
    """

    site_name: str
    dir_path: Path
    metadata: TimeMetadata
    reader: TimeReader
    config: Configuration

    @property
    def name(self) -> str:
        """Get the name of the measurement"""
        return self.dir_path.name


class Site(ResisticsModel):
    """
    Class for describing Sites

    .. note::

        This should essentially describe a single instrument setup. If the same
        site is re-occupied later with a different instrument setup, it is
        suggested to split this into a different site.
    """

    dir_path: Path
    measurements: Dict[str, Measurement]
    begin_time: HighResDateTime
    end_time: HighResDateTime
    config: Configuration

    def __iter__(self) -> Iterator:
        """Iterator over measurements"""
        return self.measurements.values().__iter__()

    def __getitem__(self, meas_name: str) -> Measurement:
        """Get a measurement"""
        return self.get_measurement(meas_name)

    @property
    def name(self) -> str:
        """The Site name"""
        return self.dir_path.name

    @property
    def n_meas(self) -> int:
        """Get the number of measurements"""
        return len(self.measurements)

    def fs(self) -> List[float]:
        """Get the sampling frequencies in the Site"""
        fs = [x.metadata.fs for x in self.measurements.values()]
        return sorted(list(set(fs)))

    def get_measurement(self, meas_name: str) -> Measurement:
        """Get a measurement"""
        from resistics.errors import MeasurementNotFoundError

        if meas_name not in self.measurements:
            raise MeasurementNotFoundError(self.name, meas_name)
        return self.measurements[meas_name]

    def get_measurements(self, fs: Optional[float] = None) -> Dict[str, Measurement]:
        """Get dictionary of measurements with optional filter by sampling frequency"""
        if fs is None:
            return self.measurements
        return {
            name: meas
            for name, meas in self.measurements.items()
            if meas.metadata.fs == fs
        }

    def plot(self) -> go.Figure:
        """Plot the site timeline"""
        df = self.to_dataframe()
        if len(df.index) == 0:
            logger.error("No measurements found to plot")
            return
        fig = px.timeline(
            df,
            x_start="first_time",
            x_end="last_time",
            y="name",
            color="fs",
            title=self.name,
        )
        return fig

    def to_dataframe(self) -> pd.DataFrame:
        """
        Get measurements list in a pandas DataFrame

        .. note::

            Measurement first and last times are converted to pandas Timestamps
            as these are more universally useful in a pandas DataFrame. However,
            this may result in a loss of precision, especially at high sampling
            frequencies.

        Returns
        -------
        pd.DataFrame
            Site measurement DataFrame
        """
        data = [
            [
                x.name,
                x.metadata.fs,
                x.metadata.first_time.isoformat(),
                x.metadata.last_time.isoformat(),
            ]
            for x in self.measurements.values()
        ]
        df = pd.DataFrame(data=data, columns=["name", "fs", "first_time", "last_time"])
        df["first_time"] = pd.to_datetime(df["first_time"])
        df["last_time"] = pd.to_datetime(df["last_time"])
        df["site"] = self.name
        return df


class ProjectMetadata(WriteableMetadata):
    """Project metadata"""

    ref_time: HighResDateTime
    location: str = ""
    country: str = ""
    year: int = -999
    description: str = ""
    contributors: List[str] = []


class Project(ResisticsModel):
    """
    Class to describe a resistics project

    The resistics Project Class connects all resistics data. It is an essential
    part of processing data with resistics.

    Resistics projects are in directory with several sub-directories. Project
    metadata is saved in the resistics.json file at the top level directory.
    """

    dir_path: Path
    begin_time: HighResDateTime
    end_time: HighResDateTime
    metadata: ProjectMetadata
    sites: Dict[str, Site] = {}
    config: Configuration

    def __iter__(self) -> Iterator:
        """Iterator over sites"""
        return self.sites.values().__iter__()

    def __getitem__(self, site_name: str) -> Site:
        """Get a Site instance given the name of the Site"""
        return self.get_site(site_name)

    @property
    def n_sites(self) -> int:
        """The number of sites"""
        return len(self.sites)

    def fs(self) -> List[float]:
        """Get sampling frequencies in the Project"""
        fs = set()
        for site in self.sites.values():
            fs = fs.union(set(site.fs()))
        return sorted(list(fs))

    def get_site(self, site_name: str) -> Site:
        """Get a Site object given the Site name"""
        from resistics.errors import SiteNotFoundError

        if site_name not in self.sites:
            raise SiteNotFoundError(site_name)
        return self.sites[site_name]

    def get_sites(self, fs: Optional[float] = None) -> Dict[str, Site]:
        """
        Get sites

        Parameters
        ----------
        fs : Optional[float], optional
            Filter by sites which have at least a single recording at a
            specified sampling frequency, by default None

        Returns
        -------
        Dict[str, Site]
            Dictionary of site name to Site
        """
        if fs is None:
            return self.sites
        return {name: site for name, site in self.sites.items() if fs in site.fs()}

    def get_concurrent(self, site_name: str) -> List[str]:
        """
        Find sites that recorded conscurrently to a specified site

        Parameters
        ----------
        site_name : str
            Search for sites recording concurrently to this site

        Returns
        -------
        List[str]
            List of site names which were recording concurrently
        """
        site_start = self.sites[site_name].begin_time
        site_end = self.sites[site_name].end_time
        concurrent = []
        for site in self.sites.values():
            if site.name == site_name:
                continue
            if site.end_time < site_start:
                continue
            if site.begin_time > site_end:
                continue
            concurrent.append(site)
        return concurrent

    def plot(self) -> go.Figure:
        """Plot a timeline of the project"""
        df = self.to_dataframe()
        df["Site"] = df["site"]
        df["Sampling frequency, Hz"] = df["fs"].values.astype(str)
        if len(df.index) == 0:
            logger.error("No measurements found to plot")
            return
        fig = px.timeline(
            df,
            x_start="first_time",
            x_end="last_time",
            y="Site",
            color="Sampling frequency, Hz",
            title=str(self.dir_path),
        )
        return fig

    def to_dataframe(self) -> pd.DataFrame:
        """Detail Project recordings in a DataFrame"""
        df = pd.DataFrame(columns=["name", "first_time", "last_time", "fs", "site"])
        for site in self.sites.values():
            df = df.append(site.to_dataframe())
        return df


class ProjectCreator(ResisticsProcess):
    """Process to create a project"""

    dir_path: Path
    metadata: ProjectMetadata

    def run(self):
        """
        Create the project

        Raises
        ------
        ProjectCreateError
            If an existing project found
        """
        from resistics.errors import ProjectCreateError
        from resistics.common import is_dir, assert_dir

        metadata_path = self.dir_path / PROJ_FILE
        if self.dir_path.exists() and is_dir(self.dir_path) and metadata_path.exists():
            raise ProjectCreateError(
                self.dir_path, "Existing project found, try loading"
            )
        elif self.dir_path.exists():
            logger.warning("Directory already exists, the project will be saved here")
            assert_dir(self.dir_path)
        else:
            logger.info("Making directory for project")
            self.dir_path.mkdir(parents=True)
        self._make_subdirs()
        self.metadata.write(metadata_path)
        logger.info(f"Project created in {self.dir_path}")

    def _make_subdirs(self):
        """Make project subdirectories"""
        for data_type, subdir in PROJ_DIRS.items():
            subdir_path = self.dir_path / subdir
            if not subdir_path.exists():
                logger.info(f"Making {data_type} data subdirectory: {subdir_path}")
                subdir_path.mkdir()


class ProjectLoader(ResisticsProcess):
    """Project loader"""

    dir_path: Path
    config: Configuration

    def run(self) -> Project:
        """
        Load a project

        Returns
        -------
        Project
            Project instance

        Raises
        ------
        ProjectLoadError
            If the resistcs project metadata is not found
        """
        from resistics.errors import ProjectLoadError
        from resistics.common import assert_dir, dir_subdirs

        assert_dir(self.dir_path)
        metadata_path = self.dir_path / PROJ_FILE
        if not metadata_path.exists():
            raise ProjectLoadError(
                self.dir_path, f"Resistics project file {metadata_path} not found"
            )
        self._check_subdirs()

        metadata = ProjectMetadata.parse_file(metadata_path)
        time_subdirs = dir_subdirs(self.dir_path / PROJ_DIRS["time"])
        sites = {}
        for site_dir in time_subdirs:
            site = self._load_site(site_dir)
            sites[site_dir.name] = site
        if len(sites) > 0:
            begin_time = min([x.begin_time for x in sites.values()])
            end_time = max([x.end_time for x in sites.values()])
        else:
            begin_time = datetime.now()
            end_time = datetime.now()
        return Project(
            dir_path=self.dir_path,
            metadata=metadata,
            begin_time=begin_time,
            end_time=end_time,
            sites=sites,
            config=self.config,
        )

    def _check_subdirs(self) -> bool:
        """Returns True if all require project subdirectories exist otherwise False"""
        from resistics.common import assert_dir

        for subdir in PROJ_DIRS.values():
            subdir_path = self.dir_path / subdir
            assert_dir(subdir_path)
            return False
        return True

    def _load_site(self, site_dir: Path) -> Site:
        """Load a Site"""
        from resistics.common import dir_subdirs

        subdirs = dir_subdirs(site_dir)
        measurements = {}
        for meas_dir in subdirs:
            meas = self._load_measurement(site_dir.name, meas_dir)
            if meas is not None:
                measurements[meas_dir.name] = meas
        if len(measurements) > 0:
            begin_time = min([x.metadata.first_time for x in measurements.values()])
            end_time = max([x.metadata.last_time for x in measurements.values()])
        else:
            begin_time = datetime.now()
            end_time = datetime.now()
        return Site(
            dir_path=site_dir,
            begin_time=begin_time,
            end_time=end_time,
            measurements=measurements,
            config=self.config,
        )

    def _load_measurement(
        self, site_name: str, meas_dir: Path
    ) -> Union[Measurement, None]:
        """
        Load a measurement

        The loader tries to use any TimeReader provided in the configuration to
        load the measurement. If no compatible reader is found, the measurement
        will be ignored.

        Parameters
        ----------
        site_name : str
            The name of the Site
        meas_dir : Path
            The measurement subdirectory in the site time directory

        Returns
        -------
        Union[Measurement, None]
            Measurement if reading was successful, else None
        """
        for reader in self.config.time_readers:
            try:
                metadata = reader.run(dir_path=meas_dir, metadata_only=True)
                logger.info(f"Read measurement {meas_dir} with {reader.name}")
                return Measurement(
                    site_name=site_name,
                    dir_path=meas_dir,
                    metadata=metadata,
                    reader=reader,
                    config=self.config,
                )
            except Exception:
                logger.warning(
                    f"Failed to read measurement {meas_dir} with {reader.name}"
                )
        logger.error(f"No reader found for measurement {meas_dir}")
        return None


def load(dir_path: Union[Path, str], config: Optional[Configuration] = None) -> Project:
    """
    Load an existing project

    Parameters
    ----------
    dir_path : Union[Path, str]
        The project directory
    config : Optional[Configuration], optional
        A configuration of parameters to use

    Returns
    -------
    Project
        A project instance

    Raises
    ------
    ProjectLoadError
        If the loading failed
    """
    if isinstance(dir_path, str):
        dir_path = Path(dir_path)
    if config is None:
        config = get_default_configuration()
    loader = ProjectLoader(dir_path=dir_path, config=config)
    return loader.run()


def reload(proj: Project):
    """
    Reload a project

    Parameters
    ----------
    proj : Project
        The project to reload

    Returns
    -------
    Project
        The reloaded project
    """
    return load(dir_path=proj.dir_path, config=proj.config)


def new(dir_path: Union[Path, str], proj_info: Dict[str, Any]) -> bool:
    """
    Create a new project

    Parameters
    ----------
    dir_path : Union[Path, str]
        The directory to create the project in
    proj_info : Dict[str, Any]
        Any project details

    Returns
    -------
    bool
        True if the creator was successful
    """
    if isinstance(dir_path, str):
        dir_path = Path(dir_path)
    metadata = ProjectMetadata(**proj_info)
    ProjectCreator(dir_path=dir_path, metadata=metadata).run()
    return True


def run_time_processors(config: Configuration, time_data: TimeData) -> TimeData:
    """
    Process time data

    Parameters
    ----------
    config : Configuration
        The configuration
    time_data : TimeData
        Time data to process

    Returns
    -------
    TimeData
        Process time data
    """
    for process in config.time_processors:
        logger.info(f"Running processor {process.name}")
        time_data = process.run(time_data)
    return time_data


def run_decimation(
    config: Configuration,
    time_data: TimeData,
    dec_params: Optional[DecimationParameters] = None,
) -> DecimatedData:
    """
    Decimate TimeData

    Parameters
    ----------
    config : Configuration
        The configuration
    time_data : TimeData
        Time data to decimate
    dec_params : DecimationParameters
        Number of levels, decimation factors etc.

    Returns
    -------
    DecimatedData
        Decimated time data
    """
    logger.info("Decimating time data")
    if dec_params is None:
        dec_params = config.dec_setup.run(time_data.metadata.fs)
    return config.decimator.run(dec_params, time_data)


def run_windowing(
    config: Configuration, ref_time: HighResDateTime, dec_data: DecimatedData
) -> WindowedData:
    """
    Window time data

    Parameters
    ----------
    config : Configuration
        The configuration
    ref_time : HighResDateTime
        The reference time
    dec_data : DecimatedData
        Decimated data to window

    Returns
    -------
    WindowedData
        The windowed data
    """
    logger.info("Windowing time data")
    win_params = config.win_setup.run(dec_data.metadata.n_levels, dec_data.metadata.fs)
    return config.windower.run(ref_time, win_params, dec_data)


def run_fft(config: Configuration, win_data: WindowedData) -> SpectraData:
    """
    Run Fourier transform

    Parameters
    ----------
    config : Configuration
        The configuration
    win_data : WindowedData
        Windowed data

    Returns
    -------
    SpectraData
        Fourier transformed windowed data
    """
    logger.info("Calculating spectra data")
    return config.fourier.run(win_data)


def run_evals(
    config: Configuration, dec_params: DecimationParameters, spec_data: SpectraData
) -> SpectraData:
    """
    Run evaluation frequency data calculator

    Parameters
    ----------
    config : Configuration
        The configuration
    dec_params : DecimationParameters
        Decimation parameters with the evaluation frequencies
    spec_data : SpectraData
        The spectra data

    Returns
    -------
    SpectraData
        Spectra data at evaluation frequencies
    """
    logger.info("Calculating fourier coefficients at evaluation frequencies")
    return config.evals.run(dec_params, spec_data)


def run_sensor_calibration(
    config: Configuration, calibration_path: Path, spec_data: SpectraData
) -> SpectraData:
    """
    Run calibration

    Parameters
    ----------
    config : Configuration
        The configuration
    calibration_path : Path
        Path to calibration data
    spec_data : SpectraData
        Spectra data to calibrate

    Returns
    -------
    SpectraData
        Calibrated spectra data
    """
    logger.info("Calibrating time data")
    return config.sensor_calibrator.run(calibration_path, spec_data)


def run_regression_preparer(
    config: Configuration, spec_data: SpectraData
) -> RegressionInputData:
    """
    Prepare linear regression data

    Parameters
    ----------
    config : Configuration
        The configuration
    spec_data : SpectraData
        Spectra data

    Returns
    -------
    RegressionInputData
        Regression inputs for all evaluation frequencies
    """
    logger.info("Preparing regression input data")
    return config.regression_preparer.run(spec_data)


def quick(
    dir_path: Path,
    config: Optional[Configuration] = None,
    calibration_path: Optional[Path] = None,
) -> RegressionInputData:
    """Quick processing of a single data directory"""
    logger.info(f"Processing data in {dir_path}")
    if config is None:
        config = get_default_configuration()

    logger.info("Reading time data")
    time_data = None
    for reader in config.time_readers:
        try:
            time_data = reader.run(dir_path)
        except MetadataReadError:
            logger.error(f"Unable to read metadata with reader {type(reader)}")
        except TimeDataReadError:
            logger.error("Failed reading time data")
        except Exception:
            logger.error("Unknown problem reading time data")
    if time_data is None:
        raise ValueError("Time data was not read")

    ref_time = time_data.metadata.first_time
    time_data = run_time_processors(config, time_data)
    dec_params = config.dec_setup.run(time_data.metadata.fs)
    dec_data = run_decimation(config, time_data, dec_params=dec_params)
    win_data = run_windowing(config, ref_time, dec_data)
    spec_data = run_fft(config, win_data)
    eval_data = run_evals(config, dec_params, spec_data)
    if calibration_path is not None:
        eval_data = run_sensor_calibration(config, calibration_path, eval_data)
    reg_data = run_regression_preparer(config, eval_data)
    return reg_data


def process_time_to_spectra(proj: Project, site_name: str, meas_name: str) -> None:
    """
    Process from time data to Fourier spectra

    Parameters
    ----------
    proj : Project
        The project
    site_name : str
        The name of the site
    meas_name : str
        The name of the measurement to process
    """
    from resistics.spectra import SpectraDataWriter

    logger.info(f"Processing measurement {site_name}, {meas_name}")
    config = proj.config
    calibration_path = get_calibration_path(proj.dir_path)

    meas = proj[site_name][meas_name]
    time_data = meas.reader.run(meas.dir_path, metadata=meas.metadata)
    time_data = run_time_processors(config, time_data)
    dec_params = config.dec_setup.run(time_data.metadata.fs)
    dec_data = run_decimation(config, time_data, dec_params=dec_params)
    win_data = run_windowing(config, proj.metadata.ref_time, dec_data)
    spec_data = run_fft(config, win_data)
    eval_data = run_evals(config, dec_params, spec_data)
    eval_data = run_sensor_calibration(config, calibration_path, eval_data)
    spectra_path = get_meas_spectra_path(
        proj.dir_path, meas.site_name, meas.name, config.name
    )
    logger.info(f"Saving evaluation frequency data to {spectra_path}")
    SpectraDataWriter().run(spectra_path, eval_data)


def process_spectra_to_tf(
    proj: Project,
    fs: float,
    out_site: str,
    in_site: Optional[str] = None,
    remote_site: Optional[str] = None,
    masks: Optional[Dict[str, str]] = None,
):
    from resistics.gather import Selector

    sites = [proj[out_site]]
    if in_site is not None:
        sites.append(proj[in_site])
    if remote_site is not None:
        sites.append(proj[remote_site])

    config = proj.config
    dec_params = config.dec_setup.run(fs)
    Selector().run(proj, sites, dec_params)
