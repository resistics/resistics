"""
This module is the main interface to resistics and includes:

- Classes and functions for making, loading and using resistics projects
- Functions for processing data
"""
from loguru import logger
from typing import Optional, Dict, Union, Any
from pathlib import Path
from datetime import datetime

from resistics.errors import MetadataReadError, TimeDataReadError
from resistics.common import ResisticsProcess, ResisticsModel
from resistics.config import Configuration, get_default_configuration
from resistics.project import PROJ_FILE, PROJ_DIRS, ProjectMetadata, Project
from resistics.project import Site, Measurement
from resistics.sampling import DateTimeLike, HighResDateTime
from resistics.time import TimeData
from resistics.decimate import DecimationParameters
from resistics.decimate import DecimatedData
from resistics.window import WindowedData
from resistics.spectra import SpectraData
from resistics.gather import GatheredData
from resistics.regression import RegressionInputData, Solution


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


class ProjectLoader(ResisticsProcess):
    """Project loader"""

    dir_path: Path

    def run(self, config: Configuration) -> Project:
        """
        Load a project

        Parameters
        ----------
        config : Configuration
            The configuration for the purposes of getting the time readers

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
            site = self._load_site(site_dir, config)
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
        )

    def _check_subdirs(self) -> bool:
        """Returns True if all require project subdirectories exist otherwise False"""
        from resistics.common import assert_dir

        for subdir in PROJ_DIRS.values():
            subdir_path = self.dir_path / subdir
            assert_dir(subdir_path)
            return False
        return True

    def _load_site(self, site_dir: Path, config: Configuration) -> Site:
        """Load a Site"""
        from resistics.common import dir_subdirs

        subdirs = dir_subdirs(site_dir)
        measurements = {}
        for meas_dir in subdirs:
            meas = self._load_measurement(site_dir.name, meas_dir, config)
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
        )

    def _load_measurement(
        self, site_name: str, meas_dir: Path, config: Configuration
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
        config : Configuration
            Configuration which is used for the time readers

        Returns
        -------
        Union[Measurement, None]
            Measurement if reading was successful, else None
        """
        for reader in config.time_readers:
            try:
                metadata = reader.run(dir_path=meas_dir, metadata_only=True)
                logger.info(f"Read measurement {meas_dir} with {reader.name}")
                return Measurement(
                    site_name=site_name,
                    dir_path=meas_dir,
                    metadata=metadata,
                    reader=reader,
                )
            except Exception:
                logger.debug(
                    f"Failed to read measurement {meas_dir} with {reader.name}"
                )
        logger.error(f"No reader found for measurement {meas_dir}")
        return None


class ResisticsEnvironment(ResisticsModel):
    """
    A Resistics environment which combines a project and a configuration
    """

    proj: Project
    """The project"""
    config: Configuration
    """The configuration for processing"""


def load(
    dir_path: Union[Path, str], config: Optional[Configuration] = None
) -> ResisticsEnvironment:
    """
    Load an existing project into a ResisticsEnvironment

    Parameters
    ----------
    dir_path : Union[Path, str]
        The project directory
    config : Optional[Configuration], optional
        A configuration of parameters to use

    Returns
    -------
    ResisticsEnvironment
        The ResisticsEnvironment combining a project and a configuration

    Raises
    ------
    ProjectLoadError
        If the loading failed
    """
    if config is None:
        config = get_default_configuration()
    if isinstance(dir_path, str):
        dir_path = Path(dir_path)
    proj = ProjectLoader(dir_path=dir_path).run(config)
    return ResisticsEnvironment(proj=proj, config=config)


def reload(resenv: ResisticsEnvironment) -> ResisticsEnvironment:
    """
    Reload the project in the ResisticsEnvironment

    Parameters
    ----------
    resenv : ResisticsEnvironment
        The current resistics environment

    Returns
    -------
    ResisticsEnvironment
        The resistics environment with the project reloaded
    """
    return load(dir_path=resenv.proj.dir_path, config=resenv.config)


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


def run_spectra_processors(
    config: Configuration, spec_data: SpectraData
) -> SpectraData:
    """
    Run any spectra processors

    Parameters
    ----------
    config : Configuration
        The configuration
    spec_data : SpectraData
        Spectra data

    Returns
    -------
    SpectraData
        Processed spectra data
    """
    for process in config.spectra_processors:
        logger.info(f"Running processor {process.name}")
        spec_data = process.run(spec_data)
    return spec_data


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
    config: Configuration, gathered_data: GatheredData
) -> RegressionInputData:
    """
    Prepare linear regression data

    Parameters
    ----------
    config : Configuration
        The configuration
    gathered_data : GatheredData
        Gathered data to input into the regression

    Returns
    -------
    RegressionInputData
        Regression inputs for all evaluation frequencies
    """
    logger.info("Preparing regression input data")
    return config.regression_preparer.run(config.tf, gathered_data)


def run_solver(config: Configuration, reg_data: RegressionInputData) -> Solution:
    """
    Run the regression solver

    Parameters
    ----------
    config : Configuration
        The configuration
    reg_data : RegressionInputData
        The regression input data

    Returns
    -------
    Solution
        Transfer function estimate
    """
    logger.info(f"Running solver {config.solver.name}")
    return config.solver.run(reg_data)


def quick_read(config: Configuration, dir_path: Path) -> TimeData:
    """
    Read time data folder

    Parameters
    ----------
    config : Configuration
        Configuration with appropriate readers, otherwise returns None
    dir_path : Path
        The directory path to read

    Returns
    -------
    TimeData
        The read TimeData

    Raises
    ------
    TimeDataReadError
        If unable to read data
    """
    for reader in config.time_readers:
        logger.info(f"Attempting to read data with reader {reader.name}")
        try:
            return reader.run(dir_path)
        except MetadataReadError:
            logger.debug(f"Unable to read metadata with reader {reader.name}")
        except TimeDataReadError:
            logger.debug(f"Failed reading time data with reader {reader.name}")
        except Exception:
            logger.debug("Unknown problem reading time data")
    reader_names = [reader.name for reader in config.time_readers]
    raise TimeDataReadError(
        dir_path, f"Unable to read time data with readers {reader_names}"
    )


def quick_view(
    dir_path: Path,
    config: Optional[Configuration] = None,
    decimate: bool = False,
    max_pts: int = 10_000,
):
    """
    Quick plotting of time data

    Parameters
    ----------
    dir_path : Path
        The directory path
    config : Optional[Configuration], optional
        The configuration with the required time readers, by default None
    decimate : bool, optional
        Boolean flag for decimating, by default False
    max_pts : Optional[int], optional
        Max points in lttb decimation, by default 10_000

    Raises
    ------
    ValueError
        If time data fails reading
    """
    logger.info(f"Plotting data in {dir_path}")
    if config is None:
        config = get_default_configuration()

    time_data = quick_read(config, dir_path)
    time_data = run_time_processors(config, time_data)
    if not decimate:
        time_data.plot(max_pts=max_pts).show()
        return
    dec_params = config.dec_setup.run(time_data.metadata.fs)
    dec_data = run_decimation(config, time_data, dec_params=dec_params)
    dec_data.plot(max_pts=max_pts).show()


def quick_tf(
    dir_path: Path,
    config: Optional[Configuration] = None,
    calibration_path: Optional[Path] = None,
) -> Solution:
    """
    Quickly calculate out a transfer function for time data in its own directory

    Parameters
    ----------
    dir_path : Path
        The directory path
    config : Optional[Configuration], optional
        A configuration instance, by default None
    calibration_path : Optional[Path], optional
        The path to the calibration data, by default None

    Returns
    -------
    Solution
        Transfer function estimate
    """
    from resistics.gather import QuickGather

    logger.info(f"Processing data in {dir_path}")
    if config is None:
        config = get_default_configuration()

    time_data = quick_read(config, dir_path)
    ref_time = time_data.metadata.first_time
    time_data = run_time_processors(config, time_data)
    dec_params = config.dec_setup.run(time_data.metadata.fs)
    dec_data = run_decimation(config, time_data, dec_params=dec_params)
    win_data = run_windowing(config, ref_time, dec_data)
    spec_data = run_fft(config, win_data)
    spec_data = run_spectra_processors(config, spec_data)
    eval_data = run_evals(config, dec_params, spec_data)
    if calibration_path is not None:
        eval_data = run_sensor_calibration(config, calibration_path, eval_data)
    gathered_data = QuickGather().run(dir_path, dec_params, config.tf, eval_data)
    reg_data = run_regression_preparer(config, gathered_data)
    return run_solver(config, reg_data)


def process_time(
    resenv: ResisticsEnvironment,
    site_name: str,
    meas_name: str,
    out_site: str,
    out_meas: str,
    from_time: Optional[DateTimeLike] = None,
    to_time: Optional[DateTimeLike] = None,
) -> None:
    """
    Process time data and save as a new measurement

    This is useful when resampling data to use with other measurements

    Parameters
    ----------
    resenv : ResisticsEnvironment
        The resistics environment
    site_name : str
        The name of the site with the data to process
    meas_name : str
        The name of the measurement to process
    out_site : str
        The site to output the data to
    out_meas : str
        The name of the measurement to output the data to
    from_time : Optional[DateTimeLike], optional
        Time to get the time data from, by default None
    to_time : Optional[DateTimeLike], optional
        Time to get the time data up to, by default None
    """
    from resistics.time import TimeWriterNumpy
    from resistics.project import get_meas_time_path

    logger.info(f"Running time processors on meas {meas_name} from site {site_name}")
    meas = resenv.proj[site_name][meas_name]
    time_data = meas.reader.run(
        meas.dir_path, metadata=meas.metadata, from_time=from_time, to_time=to_time
    )
    time_data = run_time_processors(resenv.config, time_data)
    out_path = get_meas_time_path(resenv.proj.dir_path, out_site, out_meas)
    writer = TimeWriterNumpy()
    writer.run(out_path, time_data)


def process_time_to_evals(
    resenv: ResisticsEnvironment, site_name: str, meas_name: str
) -> None:
    """
    Process from time data to Fourier spectra

    Parameters
    ----------
    resenv : ResisticsEnvironment
        The resistics environment containing the project and configuration
    site_name : str
        The name of the site
    meas_name : str
        The name of the measurement to process
    """
    from resistics.project import get_calibration_path, get_meas_evals_path
    from resistics.spectra import SpectraDataWriter

    proj = resenv.proj
    config = resenv.config
    calibration_path = get_calibration_path(proj.dir_path)

    logger.info(f"Processing measurement {site_name}, {meas_name}")
    meas = proj[site_name][meas_name]
    time_data = meas.reader.run(meas.dir_path, metadata=meas.metadata)
    time_data = run_time_processors(config, time_data)
    dec_params = config.dec_setup.run(time_data.metadata.fs)
    dec_data = run_decimation(config, time_data, dec_params=dec_params)
    win_data = run_windowing(config, proj.metadata.ref_time, dec_data)
    spec_data = run_fft(config, win_data)
    spec_data = run_spectra_processors(config, spec_data)
    eval_data = run_evals(config, dec_params, spec_data)
    eval_data = run_sensor_calibration(config, calibration_path, eval_data)
    evals_path = get_meas_evals_path(
        proj.dir_path, meas.site_name, meas.name, config.name
    )
    logger.info(f"Saving evaluation frequency data to {evals_path}")
    SpectraDataWriter().run(evals_path, eval_data)


def process_evals_to_tf(
    resenv: ResisticsEnvironment,
    fs: float,
    out_site: str,
    in_site: Optional[str] = None,
    cross_site: Optional[str] = None,
    masks: Optional[Dict[str, str]] = None,
    postfix: Optional[str] = None,
) -> Solution:
    """
    Process spectra to transfer functions

    Parameters
    ----------
    resenv : ResisticsEnvironment
        The resistics environment
    fs : float
        The sampling frequency to process
    out_site : str
        The name of the output site
    in_site : Optional[str], optional
        The name of the input site, by default None. This should be used for
        intersite processing
    cross_site : Optional[str], optional
        The name of the cross site, by default None. This is usually the site to
        use as the remote reference.
    masks : Optional[Dict[str, str]], optional
        Any masks to apply, by default None
    postfix : Optional[str]
        String to add to the end of solution, by default None

    Returns
    -------
    Solution
        Transfer function estimate
    """
    from resistics.gather import Selector, ProjectGather
    from resistics.project import get_results_path, get_solution_name

    proj = resenv.proj
    config = resenv.config

    sites = [out_site]
    if in_site is not None:
        sites.append(in_site)
    if cross_site is not None:
        sites.append(cross_site)

    dec_params = config.dec_setup.run(fs)
    selection = Selector().run(config.name, proj, sites, dec_params)
    gathered_data = ProjectGather().run(
        config.name,
        proj,
        selection,
        config.tf,
        out_name=out_site,
        in_name=in_site,
        cross_name=cross_site,
    )
    reg_data = run_regression_preparer(config, gathered_data)
    solution = run_solver(config, reg_data)
    solution_path = get_results_path(proj.dir_path, out_site, config.name)
    if not solution_path.exists():
        solution_path.mkdir(parents=True)
    solution_name = get_solution_name(
        fs, solution.tf.name, solution.tf.variation, postfix
    )
    solution.write(solution_path / solution_name)
    return solution


def get_solution(
    resenv: ResisticsEnvironment,
    site_name: str,
    config_name: str,
    fs: float,
    tf_name: str,
    tf_var: str,
    postfix: Optional[str] = None,
) -> Solution:
    """
    Get a solution

    Parameters
    ----------
    resenv : ResisticsEnvironment
        The resistics environment
    site_name : str
        The site for which to get the solution
    config_name : str
        The configuration that was used
    fs : float
        The sampling frequency
    tf_name : str
        The transfer function name
    tf_var : str
        The transfer function variation
    postfix : Optional[str], optional
        Any postfix on the solution, by default None

    Returns
    -------
    Solution
        The solution
    """
    from resistics.project import get_results_path, get_solution_name

    proj = resenv.proj
    solution_path = get_results_path(proj.dir_path, site_name, config_name)
    solution_name = get_solution_name(fs, tf_name, tf_var, postfix)
    return Solution.parse_file(solution_path / solution_name)
