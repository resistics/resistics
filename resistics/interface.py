from pathlib import Path
from typing import Optional, Union, Dict, Any
from logging import getLogger

from resistics.project import Project, ProjectLoader, ProjectCreator
from resistics.time import TimeReaderNumpy, TimeReaderAscii
from resistics.decimate import DecimationSetup, Decimator
from resistics.window import WindowSetup, Windower

logger = getLogger(__name__)


default_setup = {
    "project_creator": ProjectCreator,
    "project_loader": ProjectLoader,
    "time_readers": [TimeReaderAscii, TimeReaderNumpy],
    "time_process": [],
    "decimation_setup": DecimationSetup,
    "decimator": Decimator,
    "window_setup": WindowSetup,
    "windower": Windower,
}


def get_resistics_setup(user_setup: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Get the resistics setup

    Parameters
    ----------
    user_setup : Optional[Dict[str, Any]], optional
        Any user specific options, by default None

    Returns
    -------
    Dict[str, Any]
        The resistics setup
    """
    from copy import deepcopy

    if user_setup is None:
        return deepcopy(default_setup)
    for key, value in default_setup.items():
        if key not in user_setup:
            user_setup[key] = deepcopy(value)
    return user_setup


def load(proj_dir: Union[Path, str], setup: Optional[Dict[str, Any]] = None) -> Project:
    """
    Load an existing project

    Parameters
    ----------
    proj_dir : Union[Path, str]
        The project directory
    setup : Optional[Dict[str, Any]], optional
        A setup to use with the project, by default None

    Returns
    -------
    Project
        A project instance

    Raises
    ------
    ProjectLoadError
        If the loading failed
    """
    from resistics.errors import ProjectLoadError

    if isinstance(proj_dir, str):
        proj_dir = Path(proj_dir)
    if setup is None:
        setup = get_resistics_setup()

    loader = ProjectLoader(proj_dir)
    if not loader.check():
        raise ProjectLoadError(proj_dir, "Error loading project")
    return loader.run(setup)


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
    return load(proj.proj_dir, proj.setup)


def new(
    proj_dir: Union[Path, str],
    proj_info: Dict[str, Any],
    setup: Optional[Dict[str, Any]] = None,
) -> Project:
    """
    Create a new resistics project

    Parameters
    ----------
    proj_dir : Union[Path, str]
        The project directory
    proj_info : Dict[str, Any]
        Project information in a dictionary. 'ref_time' is a required key and
        the value should be the reference time for the project.
    setup : Optional[Dict[str, Any]], optional
        Optional setup defining classes to use for various data objects and
        processes, by default None

    Returns
    -------
    Project
        A project

    Raises
    ------
    ProjectCreateError
        If info is not provided in the correct format, primarily reference time
    ProjectCreateError
        If checks fail, likely caused because of an existing project at the same
        location
    """
    from resistics.errors import ProjectCreateError
    from resistics.common import format_dict
    from resistics.project import info_metadata_specs

    if setup is None:
        setup = default_setup
    if isinstance(proj_dir, str):
        proj_dir = Path(proj_dir)
    # try formatting the metadata
    try:
        proj_info = format_dict(proj_info, info_metadata_specs)
    except (KeyError, TypeError):
        raise ProjectCreateError(proj_dir, "Failed to format info. See docs")

    obj = setup["project_creator"]
    creator = obj(proj_dir, proj_info)
    if not creator.check():
        raise ProjectCreateError(proj_dir, "Error creating project, see logs")
    creator.run()
    return load(proj_dir, setup)
