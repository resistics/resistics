"""
Processing flow definitions for app-authored resistics workflows.

The classes in this module deliberately model a constrained MT processing DAG
rather than a general workflow platform.  They are safe to use from the desktop
app and from standalone scripts because they do not import the legacy processing
stack at module import time.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Type, TypeVar

from pydantic import BaseModel, Field, field_validator


ProgressCallback = Callable[[Dict[str, Any]], None]
StepHandler = Callable[[Dict[str, Any], Dict[str, Any], Dict[str, Any]], Any]
ModelT = TypeVar("ModelT", bound=BaseModel)


class ParameterDefinition(BaseModel):
    """A user-editable parameter exposed by a processing step."""

    name: str
    kind: str = "str"
    description: str = ""
    default: Any = None
    required: bool = False
    choices: Optional[List[Any]] = None

    def validate_value(self, value: Any) -> Any:
        """Validate and lightly coerce a parameter value."""
        if value is None:
            if self.required and self.default is None:
                raise ValueError(f"Parameter '{self.name}' is required")
            return self.default
        if self.choices is not None and value not in self.choices:
            raise ValueError(
                f"Parameter '{self.name}' must be one of {self.choices}, got {value!r}"
            )
        if self.kind == "int":
            return int(value)
        if self.kind == "float":
            return float(value)
        if self.kind == "bool":
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.lower() in {"1", "true", "yes", "on"}
            return bool(value)
        if self.kind == "list" and not isinstance(value, list):
            raise ValueError(f"Parameter '{self.name}' must be a list")
        return value


class StepDefinition(BaseModel):
    """Registry metadata for a processing step type."""

    type_id: str
    display_name: str
    description: str = ""
    input_types: List[str] = Field(default_factory=list)
    output_type: str
    parameters: List[ParameterDefinition] = Field(default_factory=list)
    runtime_requirements: List[str] = Field(default_factory=list)
    optional: bool = False

    def default_parameters(self) -> Dict[str, Any]:
        """Return default parameter values keyed by parameter name."""
        return {param.name: param.default for param in self.parameters}

    def validate_parameters(self, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate values for this step and fill omitted defaults."""
        values = dict(values)
        valid_names = {param.name for param in self.parameters}
        unknown = sorted(set(values) - valid_names)
        if unknown:
            raise ValueError(f"Unknown parameter(s) for {self.type_id}: {unknown}")
        return {
            param.name: param.validate_value(values.get(param.name, param.default))
            for param in self.parameters
        }


class FlowNode(BaseModel):
    """A node instance in a processing flow."""

    id: str
    type: str
    inputs: List[str] = Field(default_factory=list)
    enabled: bool = True
    position: Dict[str, float] = Field(default_factory=dict)


class FlowDefinition(BaseModel):
    """Serializable processing DAG definition."""

    name: str
    description: str = ""
    version: str = "1"
    nodes: List[FlowNode]

    @field_validator("nodes")
    def validate_nodes_not_empty(cls, value: List[FlowNode]) -> List[FlowNode]:
        if not value:
            raise ValueError("A flow must contain at least one node")
        return value

    def node_map(self) -> Dict[str, FlowNode]:
        """Return nodes keyed by id."""
        return {node.id: node for node in self.nodes}


class ParameterSet(BaseModel):
    """Parameter values for nodes in a flow."""

    name: str
    description: str = ""
    values: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    def for_node(self, node_id: str) -> Dict[str, Any]:
        """Return parameter values for a node."""
        return dict(self.values.get(node_id, {}))


class ProcessingJob(BaseModel):
    """A runnable binding of flow, parameters, runtime data, and output name."""

    name: str
    flow: FlowDefinition
    parameters: ParameterSet
    runtime: Dict[str, Any] = Field(default_factory=dict)
    output_label: str = "result"


class StepRegistry:
    """Registry of known processing steps."""

    def __init__(self, steps: Optional[Iterable[StepDefinition]] = None):
        self._steps: Dict[str, StepDefinition] = {}
        for step in steps or []:
            self.register(step)

    def register(self, step: StepDefinition) -> None:
        """Register a step definition."""
        if step.type_id in self._steps:
            raise ValueError(f"Step type already registered: {step.type_id}")
        self._steps[step.type_id] = step

    def get(self, type_id: str) -> StepDefinition:
        """Get a step definition by type id."""
        try:
            return self._steps[type_id]
        except KeyError as exc:
            raise ValueError(f"Unknown step type: {type_id}") from exc

    def all(self) -> List[StepDefinition]:
        """Return all registered step definitions."""
        return list(self._steps.values())


class FlowValidationResult(BaseModel):
    """Result of validating a processing job."""

    ok: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class FlowValidator:
    """Validate flow graph shape, parameters, and runtime bindings."""

    def __init__(self, registry: StepRegistry):
        self.registry = registry

    def validate(self, processing_job: ProcessingJob) -> FlowValidationResult:
        """Validate a processing job."""
        errors: List[str] = []
        warnings: List[str] = []
        flow = processing_job.flow
        nodes = flow.node_map()
        if len(nodes) != len(flow.nodes):
            errors.append("Flow contains duplicate node ids")

        for node in flow.nodes:
            try:
                step = self.registry.get(node.type)
            except ValueError as exc:
                errors.append(str(exc))
                continue

            for input_id in node.inputs:
                if input_id not in nodes:
                    errors.append(f"Node '{node.id}' references missing input '{input_id}'")
                    continue
                input_step = self.registry.get(nodes[input_id].type)
                if step.input_types and input_step.output_type not in step.input_types:
                    errors.append(
                        f"Node '{node.id}' cannot accept output '{input_step.output_type}' "
                        f"from '{input_id}'"
                    )

            try:
                step.validate_parameters(processing_job.parameters.for_node(node.id))
            except ValueError as exc:
                errors.append(f"Node '{node.id}': {exc}")

            for key in step.runtime_requirements:
                if processing_job.runtime.get(key) in (None, ""):
                    errors.append(f"Runtime value '{key}' is required by node '{node.id}'")

        try:
            topological_order(flow)
        except ValueError as exc:
            errors.append(str(exc))

        if not processing_job.output_label.strip():
            errors.append("output_label is required")

        return FlowValidationResult(ok=not errors, errors=errors, warnings=warnings)


def topological_order(flow: FlowDefinition) -> List[FlowNode]:
    """Return enabled nodes in topological order or raise for cycles."""
    nodes = {node.id: node for node in flow.nodes if node.enabled}
    incoming = {node_id: 0 for node_id in nodes}
    outgoing: Dict[str, List[str]] = {node_id: [] for node_id in nodes}

    for node in nodes.values():
        for input_id in node.inputs:
            if input_id not in nodes:
                continue
            incoming[node.id] += 1
            outgoing[input_id].append(node.id)

    queue = deque([node_id for node_id, count in incoming.items() if count == 0])
    ordered: List[FlowNode] = []
    while queue:
        node_id = queue.popleft()
        ordered.append(nodes[node_id])
        for child_id in outgoing[node_id]:
            incoming[child_id] -= 1
            if incoming[child_id] == 0:
                queue.append(child_id)

    if len(ordered) != len(nodes):
        raise ValueError("Flow contains a cycle")
    return ordered


class FlowExecutor:
    """Execute a validated flow locally in topological order."""

    def __init__(
        self,
        registry: StepRegistry,
        handlers: Optional[Dict[str, StepHandler]] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ):
        self.registry = registry
        self.handlers = handlers or {}
        self.progress_callback = progress_callback

    def run(self, processing_job: ProcessingJob) -> Dict[str, Any]:
        """Run a flow and return node results keyed by node id."""
        validation = FlowValidator(self.registry).validate(processing_job)
        if not validation.ok:
            raise ValueError("; ".join(validation.errors))

        results: Dict[str, Any] = {}
        for node in topological_order(processing_job.flow):
            step = self.registry.get(node.type)
            params = step.validate_parameters(processing_job.parameters.for_node(node.id))
            inputs = {input_id: results[input_id] for input_id in node.inputs}
            self._emit({"event": "started", "node_id": node.id, "step_type": node.type})
            try:
                handler = self.handlers.get(node.type, _default_handler)
                results[node.id] = handler(inputs, params, processing_job.runtime)
            except Exception as exc:
                self._emit(
                    {
                        "event": "failed",
                        "node_id": node.id,
                        "step_type": node.type,
                        "error": str(exc),
                    }
                )
                raise
            self._emit({"event": "completed", "node_id": node.id, "step_type": node.type})
        return results

    def _emit(self, event: Dict[str, Any]) -> None:
        if self.progress_callback is not None:
            self.progress_callback(event)


def _default_handler(
    inputs: Dict[str, Any], params: Dict[str, Any], runtime: Dict[str, Any]
) -> Dict[str, Any]:
    """Default dry-run handler used before real processing adapters are wired."""
    return {"inputs": sorted(inputs), "params": params, "runtime": dict(runtime)}


def builtin_step_registry() -> StepRegistry:
    """Return the built-in MT processing step registry."""
    return StepRegistry(
        [
            StepDefinition(
                type_id="mth5_read",
                display_name="Read MTH5",
                description="Read selected survey, station, run, and channels from MTH5.",
                output_type="time_data",
                runtime_requirements=["survey", "station", "run"],
                parameters=[
                    ParameterDefinition(
                        name="channels",
                        kind="list",
                        default=["Ex", "Ey", "Hx", "Hy"],
                    ),
                    ParameterDefinition(name="from_time", kind="str", default=""),
                    ParameterDefinition(name="to_time", kind="str", default=""),
                ],
            ),
            StepDefinition(
                type_id="time_processors",
                display_name="Time Processors",
                description="Apply time-domain cleaning before decimation.",
                input_types=["time_data"],
                output_type="time_data",
                parameters=[
                    ParameterDefinition(name="remove_mean", kind="bool", default=True),
                    ParameterDefinition(
                        name="interpolate_nans", kind="bool", default=True
                    ),
                ],
            ),
            StepDefinition(
                type_id="decimate",
                display_name="Decimate",
                description="Create decimation levels for spectral processing.",
                input_types=["time_data"],
                output_type="decimated_data",
                parameters=[
                    ParameterDefinition(
                        name="n_levels", kind="int", default=8, required=True
                    ),
                    ParameterDefinition(
                        name="per_level", kind="int", default=5, required=True
                    ),
                    ParameterDefinition(
                        name="div_factor", kind="int", default=2, required=True
                    ),
                ],
            ),
            StepDefinition(
                type_id="window",
                display_name="Window",
                description="Window decimated data.",
                input_types=["decimated_data"],
                output_type="windowed_data",
                parameters=[
                    ParameterDefinition(
                        name="min_size", kind="int", default=256, required=True
                    ),
                    ParameterDefinition(
                        name="overlap", kind="float", default=0.25, required=True
                    ),
                ],
            ),
            StepDefinition(
                type_id="fft",
                display_name="FFT",
                description="Transform windowed data to spectra.",
                input_types=["windowed_data"],
                output_type="spectra_data",
                parameters=[
                    ParameterDefinition(name="window_type", kind="str", default="parzen"),
                ],
            ),
            StepDefinition(
                type_id="evals",
                display_name="Evaluation Frequencies",
                description="Select spectra at evaluation frequencies.",
                input_types=["spectra_data"],
                output_type="eval_data",
                parameters=[
                    ParameterDefinition(name="f_min", kind="float", default=0.0),
                    ParameterDefinition(name="n_freqs", kind="int", default=0),
                ],
            ),
            StepDefinition(
                type_id="calibrate",
                display_name="Calibrate",
                description="Apply sensor calibration when calibration data is available.",
                input_types=["eval_data"],
                output_type="eval_data",
                optional=True,
                parameters=[
                    ParameterDefinition(name="enabled", kind="bool", default=True),
                    ParameterDefinition(
                        name="calibration_path", kind="str", default=""
                    ),
                ],
            ),
            StepDefinition(
                type_id="gather",
                display_name="Gather",
                description="Gather output, input, and optional remote reference data.",
                input_types=["eval_data"],
                output_type="regression_input",
                runtime_requirements=["station"],
                parameters=[
                    ParameterDefinition(
                        name="remote_reference", kind="str", default=""
                    ),
                ],
            ),
            StepDefinition(
                type_id="solve_tf",
                display_name="Solve Transfer Function",
                description="Estimate transfer-function components.",
                input_types=["regression_input"],
                output_type="transfer_function",
                parameters=[
                    ParameterDefinition(
                        name="solver",
                        kind="str",
                        default="ols",
                        choices=["ols", "robust"],
                    ),
                    ParameterDefinition(name="tf", kind="str", default="impedance"),
                ],
            ),
            StepDefinition(
                type_id="write_results",
                display_name="Write Results",
                description="Write job metadata and transfer-function result files.",
                input_types=["transfer_function"],
                output_type="job_result",
                runtime_requirements=["project_path"],
                parameters=[
                    ParameterDefinition(name="overwrite", kind="bool", default=False),
                ],
            ),
        ]
    )


def standard_mt_flow() -> FlowDefinition:
    """Return the built-in Standard MT visual flow."""
    return FlowDefinition(
        name="Standard MT",
        description="Single-station MT processing with optional remote reference.",
        nodes=[
            FlowNode(id="read", type="mth5_read", position={"x": 40, "y": 120}),
            FlowNode(
                id="time_processors",
                type="time_processors",
                inputs=["read"],
                position={"x": 260, "y": 120},
            ),
            FlowNode(
                id="decimate",
                type="decimate",
                inputs=["time_processors"],
                position={"x": 480, "y": 120},
            ),
            FlowNode(
                id="window",
                type="window",
                inputs=["decimate"],
                position={"x": 700, "y": 120},
            ),
            FlowNode(id="fft", type="fft", inputs=["window"], position={"x": 920, "y": 120}),
            FlowNode(
                id="evals",
                type="evals",
                inputs=["fft"],
                position={"x": 1140, "y": 120},
            ),
            FlowNode(
                id="calibrate",
                type="calibrate",
                inputs=["evals"],
                position={"x": 1360, "y": 120},
            ),
            FlowNode(
                id="gather",
                type="gather",
                inputs=["calibrate"],
                position={"x": 1580, "y": 120},
            ),
            FlowNode(
                id="solve_tf",
                type="solve_tf",
                inputs=["gather"],
                position={"x": 1800, "y": 120},
            ),
            FlowNode(
                id="write_results",
                type="write_results",
                inputs=["solve_tf"],
                position={"x": 2020, "y": 120},
            ),
        ],
    )


def default_parameter_set(
    flow: FlowDefinition, registry: Optional[StepRegistry] = None
) -> ParameterSet:
    """Return defaults for all nodes in a flow."""
    registry = registry or builtin_step_registry()
    return ParameterSet(
        name=f"{flow.name} defaults",
        values={
            node.id: registry.get(node.type).default_parameters()
            for node in flow.nodes
        },
    )


def model_to_dict(model: BaseModel) -> Dict[str, Any]:
    """Return a dict for a pydantic v2 model."""
    return model.model_dump()


def model_to_yaml(model: BaseModel) -> str:
    """Serialize a model to YAML."""
    import yaml

    return yaml.safe_dump(model_to_dict(model), sort_keys=False)


def model_from_yaml(model_type: Type[ModelT], yaml_text: str) -> ModelT:
    """Parse a model from YAML text."""
    import yaml

    data = yaml.safe_load(yaml_text) or {}
    return model_type.model_validate(data)


def model_to_yaml_file(model: BaseModel, path: Path) -> None:
    """Write a model to a YAML file."""
    path.write_text(model_to_yaml(model))


def model_from_yaml_file(model_type: Type[ModelT], path: Path) -> ModelT:
    """Read a model from a YAML file."""
    return model_from_yaml(model_type, path.read_text())
