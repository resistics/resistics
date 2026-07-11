"""Built-in, editable processing templates for new resistics projects."""

from __future__ import annotations

from pathlib import Path

from resistics.flow import (
    FlowDefinition,
    ParameterSet,
    default_parameter_set,
    model_to_yaml_file,
    standard_mt_flow,
)

DEFAULT_FLOW_FILENAME = "standard_mt.yaml"
DEFAULT_PARAMETERS_FILENAME = "default_mt.yaml"
QUICK_PARAMETERS_FILENAME = "quick_mt.yaml"


def builtin_processing_templates() -> dict[str, FlowDefinition | ParameterSet]:
    """Return fresh models for the processing templates installed in projects."""
    flow = standard_mt_flow()
    default_parameters = default_parameter_set(flow)
    default_parameters.name = "Default MT"
    default_parameters.description = "Conservative single-run OLS impedance processing."

    quick_parameters = default_parameters.model_copy(deep=True)
    quick_parameters.name = "Quick MT"
    quick_parameters.description = "Faster exploratory MT processing for a single run."
    quick_parameters.values["decimate"].update({"n_levels": 4, "per_level": 3})

    return {
        DEFAULT_FLOW_FILENAME: flow,
        DEFAULT_PARAMETERS_FILENAME: default_parameters,
        QUICK_PARAMETERS_FILENAME: quick_parameters,
    }


def _install_builtin_templates(project_path: Path, filenames: set[str]) -> list[Path]:
    """Write selected missing built-in templates without replacing user files."""
    destinations = {
        DEFAULT_FLOW_FILENAME: project_path / "processing" / "flows",
        DEFAULT_PARAMETERS_FILENAME: project_path / "processing" / "parameters",
        QUICK_PARAMETERS_FILENAME: project_path / "processing" / "parameters",
    }
    installed = []
    for filename, model in builtin_processing_templates().items():
        if filename not in filenames:
            continue
        path = destinations[filename] / filename
        if path.exists():
            continue
        model_to_yaml_file(model, path)
        installed.append(path)
    return installed


def install_builtin_flow_templates(project_path: Path) -> list[Path]:
    """Write missing built-in flow templates without modifying parameter sets."""
    return _install_builtin_templates(project_path, {DEFAULT_FLOW_FILENAME})


def install_builtin_parameter_templates(project_path: Path) -> list[Path]:
    """Write missing built-in parameter templates without modifying flows."""
    return _install_builtin_templates(
        project_path, {DEFAULT_PARAMETERS_FILENAME, QUICK_PARAMETERS_FILENAME}
    )


def install_builtin_processing_templates(project_path: Path) -> list[Path]:
    """Write all missing built-in templates without replacing user files."""
    return _install_builtin_templates(
        project_path,
        {
            DEFAULT_FLOW_FILENAME,
            DEFAULT_PARAMETERS_FILENAME,
            QUICK_PARAMETERS_FILENAME,
        },
    )
