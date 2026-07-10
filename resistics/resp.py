"""Compatibility shim for the MTH5-backed project API.

The canonical public project API now lives in :mod:`resistics.project`. This
module remains temporarily so older MTH5 prototype examples that import
``resistics.resp`` keep working during the migration.
"""
from resistics.project import (
    PROJ_DIRS,
    PROJ_FILE,
    Project,
    ProjectMetadata,
    check_project,
    get_flow_path,
    get_job_path,
    get_log_path,
    get_parameters_path,
    get_results_path,
    get_run_data_path,
    get_solution_name,
    init,
    load,
)

__all__ = [
    "PROJ_DIRS",
    "PROJ_FILE",
    "Project",
    "ProjectMetadata",
    "check_project",
    "get_flow_path",
    "get_job_path",
    "get_log_path",
    "get_parameters_path",
    "get_results_path",
    "get_run_data_path",
    "get_solution_name",
    "init",
    "load",
]
