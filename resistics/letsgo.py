"""
This module is the main interface to resistics and includes:

- Classes and functions for making, loading and using resistics projects
- Functions for processing data
"""
from loguru import logger
from typing import Optional, Dict, Union, List
from pathlib import Path
import pandas as pd

from resistics.common import ResisticsModel
from resistics.config import Configuration, get_default_configuration
from resistics.project import Project
from resistics.project import init as init_project
from resistics.project import load as load_project
from resistics.sampling import DateTimeLike, HighResDateTime
from resistics.time import TimeData
from resistics.decimate import DecimationParameters
from resistics.decimate import DecimatedData
from resistics.window import WindowedData, get_win_table
from resistics.spectra import SpectraData
from resistics.gather import GatheredData
from resistics.regression import RegressionInputData, Solution


def new(
    project_path: Union[Path, str],
    mth5_path: Union[Path, str],
    ref_time: DateTimeLike,
    overwrite: bool = False,
    plugin_paths: Optional[List[Union[Path, str]]] = None,
) -> bool:
    """
    Create a new MTH5-backed project.

    Parameters
    ----------
    project_path : Union[Path, str]
        Directory to create the project in.
    mth5_path : Union[Path, str]
        Path to an existing MTH5 file.
    ref_time : DateTimeLike
        Project reference time.
    overwrite : bool
        Overwrite an existing project metadata file if present.
    plugin_paths : Optional[List[Union[Path, str]]]
        Optional project plugin paths.

    Returns
    -------
    bool
        True if the project was created.
    """
    return init_project(
        project_path=project_path,
        mth5_path=mth5_path,
        ref_time=ref_time,
        overwrite=overwrite,
        plugin_paths=plugin_paths,
    )


class ResisticsEnvironment(ResisticsModel):
    """
    A Resistics environment which combines a project and a configuration
    """

    proj: Project
    """The project"""
    config: Configuration
    """The configuration for processing"""


def load(
    project_path: Union[Path, str], config: Optional[Configuration] = None
) -> ResisticsEnvironment:
    """
    Load an existing project into a ResisticsEnvironment

    Parameters
    ----------
    project_path : Union[Path, str]
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
    proj = load_project(project_path)
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
    return load(project_path=resenv.proj.project_path, config=resenv.config)


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


def _get_project(project: Union[ResisticsEnvironment, Project, Path, str]) -> Project:
    """Resolve a project-like object to a loaded MTH5 project."""
    if isinstance(project, ResisticsEnvironment):
        return project.proj
    if isinstance(project, Project):
        return project
    return load_project(project)


def quick_read(
    project: Union[ResisticsEnvironment, Project, Path, str],
    survey: str,
    station: str,
    run: str,
    config: Optional[Configuration] = None,
    chans: Optional[List[str]] = None,
    from_time: Optional[DateTimeLike] = None,
    to_time: Optional[DateTimeLike] = None,
    from_sample: Optional[int] = None,
    to_sample: Optional[int] = None,
) -> TimeData:
    """
    Read an MTH5 run.

    Parameters
    ----------
    project : Union[ResisticsEnvironment, Project, Path, str]
        Loaded project, resistics environment, or project path.
    survey : str
        MTH5 survey name.
    station : str
        MTH5 station name.
    run : str
        MTH5 run name.
    config : Optional[Configuration], optional
        Configuration with appropriate readers, by default None.
    from_time : Union[DateTimeLike, None], optional
        Timestamp to read from, by default None
    to_time : Union[DateTimeLike, None], optional
        Timestamp to read to, by default None
    from_sample : Union[int, None], optional
        Sample to read from, by default None
    to_sample : Union[int, None], optional
        Sample to read to, by default None

    Returns
    -------
    TimeData
        The read TimeData

    Raises
    ------
    TimeDataReadError
        If unable to read data
    """
    logger.info(f"Reading MTH5 run {survey}/{station}/{run}")
    proj = _get_project(project)
    return proj.read_run(
        survey=survey,
        station=station,
        run=run,
        chans=chans,
        from_time=from_time,
        to_time=to_time,
        from_sample=from_sample,
        to_sample=to_sample,
    )


def quick_view(
    project: Union[ResisticsEnvironment, Project, Path, str],
    survey: str,
    station: str,
    run: str,
    config: Optional[Configuration] = None,
    chans: Optional[List[str]] = None,
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

    Returns
    -------
    go.Figure
        Plotly figure

    Raises
    ------
    ValueError
        If time data fails reading
    """
    logger.info(f"Plotting MTH5 run {survey}/{station}/{run}")
    if config is None:
        config = get_default_configuration()

    time_data = quick_read(project, survey, station, run, config, chans=chans)
    time_data = run_time_processors(config, time_data)
    if not decimate:
        return time_data.plot(max_pts=max_pts)
    dec_params = config.dec_setup.run(time_data.metadata.fs)
    dec_data = run_decimation(config, time_data, dec_params=dec_params)
    return dec_data.plot(max_pts=max_pts)


def quick_spectra(
    project: Union[ResisticsEnvironment, Project, Path, str],
    survey: str,
    station: str,
    run: str,
    config: Optional[Configuration] = None,
    chans: Optional[List[str]] = None,
) -> SpectraData:
    """
    Quick plotting of time data

    Parameters
    ----------
    dir_path : Path
        The directory path
    config : Optional[Configuration], optional
        The configuration with the required time readers, by default None

    Returns
    -------
    SpectraData
        The spectra data

    Raises
    ------
    ValueError
        If time data fails reading
    """
    logger.info(f"Getting spectra for MTH5 run {survey}/{station}/{run}")
    if config is None:
        config = get_default_configuration()

    time_data = quick_read(project, survey, station, run, config, chans=chans)
    ref_time = time_data.metadata.first_time
    time_data = run_time_processors(config, time_data)
    dec_params = config.dec_setup.run(time_data.metadata.fs)
    dec_data = run_decimation(config, time_data, dec_params=dec_params)
    win_data = run_windowing(config, ref_time, dec_data)
    return run_fft(config, win_data)


def quick_tf(
    project: Union[ResisticsEnvironment, Project, Path, str],
    survey: str,
    station: str,
    run: str,
    config: Optional[Configuration] = None,
    chans: Optional[List[str]] = None,
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

    logger.info(f"Processing MTH5 run {survey}/{station}/{run}")
    if config is None:
        config = get_default_configuration()

    time_data = quick_read(project, survey, station, run, config, chans=chans)
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
    gathered_data = QuickGather().run(
        Path(survey) / station / run, dec_params, config.tf, eval_data
    )
    reg_data = run_regression_preparer(config, gathered_data)
    return run_solver(config, reg_data)


def profile_windowing(
    project: Union[ResisticsEnvironment, Project, Path, str],
    survey: str,
    station: str,
    run: str,
    config: Optional[Configuration] = None,
    chans: Optional[List[str]] = None,
    ref_time: Optional[DateTimeLike] = None,
) -> Dict[int, pd.DataFrame]:
    """
    Profile windowing for a measurement

    This function will return window tables for each decimation level. Note that
    any time processses are run first in case these changes the start or end
    time of the data.

    Parameters
    ----------
    dir_path : Path
        Directory path of the time data
    config : Optional[Configuration], optional
        Configuration to use, by default None. If not provided, the default
        configuration will be used.
    ref_time : Optional[DateTimeLike], optional
        A reference time to perform windowing against, by default None. If not
        provided, the start time of the recording will be used as the reference
        time.

    Returns
    -------
    Dict[int, pd.DataFrame]
        Mapping from decimation level to a pandas DataFrame of the windows for
        the decimation level
    """
    from resistics.sampling import to_datetime

    logger.info(f"Profiling windowing for MTH5 run {survey}/{station}/{run}")
    if config is None:
        config = get_default_configuration()

    time_data = quick_read(project, survey, station, run, config, chans=chans)
    if ref_time is None:
        ref_time = time_data.metadata.first_time
    else:
        ref_time = to_datetime(ref_time)
    time_data = run_time_processors(config, time_data)
    dec_params = config.dec_setup.run(time_data.metadata.fs)
    dec_data = run_decimation(config, time_data, dec_params=dec_params)
    win_params = config.win_setup.run(dec_data.metadata.n_levels, dec_data.metadata.fs)
    profiles = {}
    for ilevel in range(0, dec_data.metadata.n_levels):
        logger.info(f"Profiling windowing for level {ilevel}")
        win_size = win_params.get_win_size(ilevel)
        olap_size = win_params.get_olap_size(ilevel)
        level_metadata = dec_data.metadata.levels_metadata[ilevel]
        profiles[ilevel] = get_win_table(ref_time, level_metadata, win_size, olap_size)
    return profiles


def process_time(
    resenv: ResisticsEnvironment,
    site_name: str,
    meas_name: str,
    out_site: str,
    out_meas: str,
    input_from_time: Optional[DateTimeLike] = None,
    input_to_time: Optional[DateTimeLike] = None,
    output_from_time: Optional[DateTimeLike] = None,
    output_to_time: Optional[DateTimeLike] = None,
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
    input_from_time : Optional[DateTimeLike], optional
        Time to read data from for the input data, by default None. If None, the
        first time of the time series data is used.
    input_to_time : Optional[DateTimeLike], optional
        Time to read data to for the input data, by default None. If None, the
        last time of the input time series data is used.
    output_from_time : Optional[DateTimeLike], optional
        Time to output data from, by default None. If None, the first time of
        input data is used.
    output_to_time : Optional[DateTimeLike], optional
        Time to output data to, by default None. If None, the last time of the
        input data is used.
    """
    raise NotImplementedError(
        "process_time writes directory-based NumPy measurements and is no longer "
        "a public workflow. Public input is MTH5-only; use quick_read or "
        "process_run_to_evals with survey, station, and run selections."
    )


def process_run_to_evals(
    resenv: ResisticsEnvironment,
    survey: str,
    station: str,
    run: str,
    chans: Optional[List[str]] = None,
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
    from resistics.project import get_run_data_path
    from resistics.spectra import SpectraDataWriter

    proj = resenv.proj
    config = resenv.config
    calibration_path = proj.project_path / "calibrate"

    logger.info(f"Processing MTH5 run {survey}/{station}/{run}")
    time_data = proj.read_run(survey=survey, station=station, run=run, chans=chans)
    time_data = run_time_processors(config, time_data)
    dec_params = config.dec_setup.run(time_data.metadata.fs)
    dec_data = run_decimation(config, time_data, dec_params=dec_params)
    win_data = run_windowing(config, proj.ref_time, dec_data)
    spec_data = run_fft(config, win_data)
    spec_data = run_spectra_processors(config, spec_data)
    eval_data = run_evals(config, dec_params, spec_data)
    eval_data = run_sensor_calibration(config, calibration_path, eval_data)
    evals_path = get_run_data_path(proj.project_path, survey, station, run) / config.name
    logger.info(f"Saving evaluation frequency data to {evals_path}")
    SpectraDataWriter().run(evals_path, eval_data)


def process_time_to_evals(*args, **kwargs) -> None:
    """Deprecated alias kept to fail with an actionable MTH5-only message."""
    raise NotImplementedError(
        "process_time_to_evals used legacy site/measurement inputs. Use "
        "process_run_to_evals(resenv, survey, station, run) instead."
    )


def process_evals_to_tf(
    resenv: ResisticsEnvironment,
    fs: float,
    station_path: str,
    in_station_path: Optional[str] = None,
    remote_station_path: Optional[str] = None,
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
    from resistics.project import get_results_path, get_solution_name

    raise NotImplementedError(
        "process_evals_to_tf still depends on legacy site/measurement gather "
        "objects. The MTH5-only project API is in place; gathering needs the "
        "next migration step to read derived run artifacts by station_path."
    )


def get_solution(
    resenv: ResisticsEnvironment,
    station_path: str,
    output_label: str,
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
    output_label : str
        The output label used by the processing job
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
    survey, station = station_path.split("/", 1)
    solution_path = get_results_path(proj.project_path, survey, station, output_label)
    solution_name = get_solution_name(fs, tf_name, tf_var, postfix)
    return Solution.model_validate_json((solution_path / solution_name).read_bytes())
