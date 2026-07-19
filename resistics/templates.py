"""Built-in, editable processing templates for new resistics projects."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from resistics.flow import (
    FlowDefinition,
    ParameterSet,
    default_parameter_set,
    mask_calculation_flow,
    model_from_yaml_file,
    model_to_yaml_file,
    remote_reference_mt_flow,
    single_site_mt_flow,
    single_site_mt_target_flow,
)

if TYPE_CHECKING:
    from resistics.gather import GatherCriteria

SINGLE_SITE_FLOW_FILENAME = "single_site_mt_standard.yaml"
SINGLE_SITE_TARGET_FLOW_FILENAME = "single_site_mt_target.yaml"
REMOTE_REFERENCE_FLOW_FILENAME = "remote_reference_mt.yaml"
MASK_CALCULATION_FLOW_FILENAME = "mask_calculation.yaml"

SINGLE_SITE_CRITERIA_FILENAME = "single_site.yaml"
REMOTE_REFERENCE_CRITERIA_FILENAME = "remote_reference.yaml"

DEFAULT_FLOW_FILENAME = SINGLE_SITE_FLOW_FILENAME
DEFAULT_PARAMETERS_FILENAME = "default.yaml"


def builtin_processing_templates(
    project_path: Path | None = None,
) -> dict[str, dict[str, FlowDefinition | ParameterSet | GatherCriteria]]:
    """Return fresh models for the default flows and parameter sets."""
    from resistics.gather import (
        GatherCriteria,
        RateGatherCriteria,
        StationGatherCriteria,
    )

    flows: dict[str, FlowDefinition] = {
        SINGLE_SITE_FLOW_FILENAME: single_site_mt_flow(),
        SINGLE_SITE_TARGET_FLOW_FILENAME: single_site_mt_target_flow(),
        REMOTE_REFERENCE_FLOW_FILENAME: remote_reference_mt_flow(),
        MASK_CALCULATION_FLOW_FILENAME: mask_calculation_flow(),
    }
    parameters: dict[str, ParameterSet] = {
        DEFAULT_PARAMETERS_FILENAME: default_parameter_set(project_path),
    }
    criteria: dict[str, GatherCriteria] = {
        SINGLE_SITE_CRITERIA_FILENAME: GatherCriteria(),
        REMOTE_REFERENCE_CRITERIA_FILENAME: GatherCriteria(
            stations={
                "survey/target": StationGatherCriteria(
                    sampling_frequencies={
                        128.0: RateGatherCriteria(remote_references=["survey/remote"])
                    }
                )
            }
        ),
    }
    return {"flows": flows, "parameters": parameters, "criteria": criteria}


def _install_builtin_templates(project_path: Path, resource_type: str) -> list[Path]:
    """Write missing templates of one type without replacing user files."""
    templates = builtin_processing_templates(project_path)[resource_type]
    destination = project_path / "processing" / resource_type
    destination.mkdir(parents=True, exist_ok=True)
    installed = []
    for filename, model in templates.items():
        path = destination / filename
        if path.exists():
            continue
        model_to_yaml_file(model, path)
        installed.append(path)
    return installed


def install_builtin_flow_templates(project_path: Path) -> list[Path]:
    """Write all missing built-in flow templates without modifying user flows."""
    return _install_builtin_templates(project_path, "flows")


def install_builtin_parameter_templates(project_path: Path) -> list[Path]:
    """Install or extend the shared defaults without replacing user values."""
    directory = project_path / "processing" / "parameters"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / DEFAULT_PARAMETERS_FILENAME
    defaults = default_parameter_set(project_path)
    installed = []
    if not path.exists():
        model_to_yaml_file(defaults, path)
        installed.append(path)
    else:
        current = model_from_yaml_file(ParameterSet, path)
        missing = {
            process: values
            for process, values in defaults.processes.items()
            if process not in current.processes
        }
        if missing:
            current.processes.update(missing)
            model_to_yaml_file(current, path)
            installed.append(path)

    return installed


def install_builtin_criteria_templates(project_path: Path) -> list[Path]:
    """Write missing criteria examples without modifying user criteria."""
    return _install_builtin_templates(project_path, "criteria")


def install_builtin_processing_templates(project_path: Path) -> list[Path]:
    """Write all missing built-in templates without replacing user files."""
    return (
        install_builtin_flow_templates(project_path)
        + install_builtin_parameter_templates(project_path)
        + install_builtin_criteria_templates(project_path)
    )
