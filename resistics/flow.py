"""Portable, library-first processing flow definitions.

Flows name concrete Python process classes and their data dependencies.  They
deliberately contain no UI layout or process parameter values.
"""

from __future__ import annotations

import inspect
import sys
from collections import deque
from collections.abc import Iterable
from importlib import import_module
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from resistics.common import (
    CancellationCallback,
    ProcessingCancelled,
    ProcessingProgressCallback,
    ProcessingProgressEvent,
    ProcessingProgressState,
    ResisticsProcess,
    validate_output_label,
)

ProgressCallback = ProcessingProgressCallback


class FlowNode(BaseModel):
    """One concrete process invocation in a processing DAG."""

    model_config = ConfigDict(extra="forbid")

    id: str
    process: str
    inputs: dict[str, str] = Field(default_factory=dict)
    configuration_source: Literal["parameters", "criteria"] = "parameters"


class FlowStage(BaseModel):
    """A DAG executed once for each run or station/rate batch."""

    model_config = ConfigDict(extra="forbid")

    stage_id: str
    scope: Literal["run", "station_rate"]
    nodes: list[FlowNode]

    @field_validator("nodes")
    @classmethod
    def validate_nodes_not_empty(cls, value: list[FlowNode]) -> list[FlowNode]:
        if not value:
            raise ValueError("A flow stage must contain at least one node")
        return value

    def node_map(self) -> dict[str, FlowNode]:
        """Return this stage's nodes keyed by their unique identifiers.

        :return: This stage's nodes keyed by their unique identifiers.
        """
        return {node.id: node for node in self.nodes}


class FlowDefinition(BaseModel):
    """Serializable staged processing definition.

    **Examples**

    Build a one-stage flow from a concrete process node.

    ```{doctest}
    >>> from resistics.flow import FlowDefinition, FlowNode, FlowStage
    >>> node = FlowNode(id="remove_mean", process="resistics.time.RemoveMean")
    >>> stage = FlowStage(stage_id="runs", scope="run", nodes=[node])
    >>> flow = FlowDefinition(id="example", name="Example", stages=[stage])
    >>> flow.flow_stages()[0].nodes[0].id
    'remove_mean'

    ```
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    description: str = ""
    version: str = "2"
    stages: list[FlowStage]

    @model_validator(mode="after")
    def validate_stages(self) -> FlowDefinition:
        if not self.stages:
            raise ValueError("A flow must contain at least one stage")
        return self

    def flow_stages(self) -> list[FlowStage]:
        """Return the explicitly declared flow stages.

        :return: The explicitly declared flow stages.
        """
        return list(self.stages)


class ParameterSet(BaseModel):
    """Process-class configuration shared by one or more flows.

    **Examples**

    Parameters are keyed by the same qualified class paths used by flow nodes.

    ```{doctest}
    >>> from resistics.flow import ParameterSet
    >>> path = "resistics.time.RemoveMean"
    >>> parameters = ParameterSet(name="example", processes={path: {}})
    >>> parameters.for_process(path)
    {}

    ```
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    processes: dict[str, dict[str, Any]] = Field(default_factory=dict)

    def for_process(self, process: str) -> dict[str, Any]:
        """Return an independent parameter mapping for one process path.

        :param process: Qualified process path.
        :return: An independent parameter mapping for one process path.
        """
        return dict(self.processes.get(process, {}))


class ProcessingJob(BaseModel):
    """A runnable binding of flow, process parameters, runtime, and output."""

    name: str
    flow: FlowDefinition
    parameters: ParameterSet
    runtime: dict[str, Any] = Field(default_factory=dict)
    output_label: str = "default"

    @field_validator("output_label")
    @classmethod
    def validate_output_label_value(cls, value: str) -> str:
        return validate_output_label(value)


class ProcessDescriptor(BaseModel):
    """App-safe description of one flow-exposed process."""

    path: str
    display_name: str
    description: str
    input_types: dict[str, str]
    output_type: str
    runtime_requirements: list[str]
    parameter_schema: dict[str, Any]


BUILTIN_PROCESS_MODULES = (
    "resistics.time",
    "resistics.decimate",
    "resistics.window",
    "resistics.spectra",
    "resistics.mask",
    "resistics.gather",
    "resistics.regression",
)


def process_path(process_class: type[ResisticsProcess]) -> str:
    """Return the stable qualified path used in flow and parameter YAML.

    :param process_class: Concrete process class.
    :return: The stable qualified path used in flow and parameter YAML.
    """
    return f"{process_class.__module__}.{process_class.__name__}"


def resolve_process_class(
    path: str, project_path: Path | None = None
) -> type[ResisticsProcess]:
    """Resolve and validate a process class named directly by a flow node.

    :param path: Path or routed coordinates to process.
    :param project_path: Project root used to locate configuration and artifacts.
    :return: The value produced when this operation completes.
    :raises ValueError: If the requested operation cannot satisfy its contract.
    """
    if project_path is not None:
        project_import_path = str(Path(project_path))
        if project_import_path not in sys.path:
            sys.path.insert(0, project_import_path)
    module_name, separator, class_name = path.rpartition(".")
    if not separator:
        raise ValueError(f"Process must be a qualified class path: {path!r}")
    try:
        process_class = getattr(import_module(module_name), class_name)
    except Exception as exc:
        raise ValueError(f"Unable to import process {path!r}: {exc}") from exc
    if not isinstance(process_class, type) or not issubclass(
        process_class, ResisticsProcess
    ):
        raise ValueError(f"Process {path!r} must subclass ResisticsProcess")
    if process_class is ResisticsProcess or process_class.output_type is None:
        raise ValueError(f"Process {path!r} does not declare a flow output_type")
    return process_class


def process_descriptor(
    path: str, project_path: Path | None = None
) -> ProcessDescriptor:
    """Build a UI-safe descriptor from a directly resolved process path.

    :param path: Import path of the process class.
    :param project_path: Optional project whose trusted plugins may supply the process.

    :return: Serializable process metadata for discovery and editing.

    :raises ValueError: If the path does not identify a concrete flow process.
    """
    process_class = resolve_process_class(path, project_path)
    output_type = process_class.output_type
    if output_type is None:
        # ``resolve_process_class`` enforces this invariant. Keep the check local
        # too so the descriptor contract remains explicit to static analyzers.
        raise ValueError(f"Process {path!r} does not declare a flow output_type")
    try:
        parameter_schema = process_class.model_json_schema()
    except Exception:
        # A legacy process can contain a callable or another value that has no
        # JSON-schema representation. It remains executable; the app simply
        # cannot render a strongly typed editor for that parameter set yet.
        parameter_schema = {"type": "object", "additionalProperties": True}
    return ProcessDescriptor(
        path=path,
        display_name=process_class.__name__,
        description=inspect.getdoc(process_class) or "",
        input_types=dict(process_class.input_types),
        output_type=output_type,
        runtime_requirements=list(process_class.runtime_requirements),
        parameter_schema=parameter_schema,
    )


class ProcessCatalog:
    """Discover flow processes for a project without governing execution.

    :param project_path: Optional project root whose trusted plugins are discovered.
    """

    def __init__(self, project_path: Path | None = None):
        self.project_path = None if project_path is None else Path(project_path)

    def discover(self) -> list[ProcessDescriptor]:
        """Return all built-in and trusted-plugin process descriptors.

        :return: All built-in and trusted-plugin process descriptors.
        """
        classes = {}
        for module in self._modules():
            for _, value in inspect.getmembers(module, inspect.isclass):
                if (
                    value is ResisticsProcess
                    or not issubclass(value, ResisticsProcess)
                    or value.output_type is None
                ):
                    continue
                classes[process_path(value)] = value
        return [process_descriptor(path) for path in sorted(classes)]

    def _modules(self) -> Iterable[Any]:
        for module_name in BUILTIN_PROCESS_MODULES:
            yield import_module(module_name)
        if self.project_path is None:
            return
        plugins_path = self.project_path / "plugins"
        if not plugins_path.is_dir():
            return
        parent = str(self.project_path)
        if parent not in sys.path:
            sys.path.insert(0, parent)
        for path in sorted(plugins_path.rglob("*.py")):
            if path.name == "__init__.py":
                continue
            module_name = "plugins." + ".".join(
                path.relative_to(plugins_path).with_suffix("").parts
            )
            yield import_module(module_name)


class FlowValidationResult(BaseModel):
    """Validation outcome containing all discovered errors and warnings."""

    ok: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class FlowValidator:
    """Validate graph dependencies, concrete processes, and configurations.

    :param available_runtime: Runtime keys supplied by the surrounding job runner.
    """

    def __init__(self, available_runtime: Iterable[str] | None = None):
        self.available_runtime = set(available_runtime or [])

    @staticmethod
    def _process(path: str) -> type[ResisticsProcess]:
        return resolve_process_class(path)

    def validate(  # noqa: C901 - validation decomposition is owned by Phase 5.7
        self,
        processing_job: ProcessingJob,
        stages: Iterable[FlowStage] | None = None,
    ) -> FlowValidationResult:
        """Validate a job's selected stages without executing any process.

        :param processing_job: The fully bound flow and parameter set to validate.
        :param stages: A stage subset, or all flow stages when omitted.

        :return: All validation errors and warnings found in a single pass.
        """
        errors: list[str] = []
        flow = processing_job.flow
        selected_stages = list(stages) if stages is not None else flow.flow_stages()
        for process_path in processing_job.parameters.processes:
            try:
                self._process(process_path)
            except ValueError as exc:
                errors.append(str(exc))
        for stage in selected_stages:
            nodes = stage.node_map()
            mask_names: dict[str, str] = {}
            if len(nodes) != len(stage.nodes):
                errors.append(f"Stage '{stage.stage_id}' contains duplicate node ids")
            for node in stage.nodes:
                try:
                    process = self._process(node.process)
                except ValueError as exc:
                    errors.append(str(exc))
                    continue
                if set(node.inputs) != set(process.input_types):
                    errors.append(
                        f"Node '{node.id}' inputs must be {sorted(process.input_types)}, "
                        f"got {sorted(node.inputs)}"
                    )
                for port, upstream_id in node.inputs.items():
                    if port not in process.input_types:
                        continue
                    upstream = nodes.get(upstream_id)
                    if upstream is None:
                        errors.append(
                            f"Node '{node.id}' references missing input '{upstream_id}'"
                        )
                        continue
                    try:
                        upstream_process = self._process(upstream.process)
                    except ValueError as exc:
                        errors.append(str(exc))
                        continue
                    expected = process.input_types[port]
                    if upstream_process.output_type != expected:
                        errors.append(
                            f"Node '{node.id}' port '{port}' requires '{expected}', got "
                            f"'{upstream_process.output_type}' from '{upstream_id}'"
                        )
                instance = None
                try:
                    instance = process(
                        **processing_job.parameters.for_process(node.process)
                    )
                except ValidationError as exc:
                    errors.append(f"Process '{node.process}': {exc}")
                if instance is not None:
                    from resistics.mask import WindowMaskProcess

                    if isinstance(instance, WindowMaskProcess):
                        previous = mask_names.get(instance.name)
                        if previous is not None:
                            errors.append(
                                f"Mask name {instance.name!r} is produced by both "
                                f"nodes '{previous}' and '{node.id}'"
                            )
                        else:
                            mask_names[instance.name] = node.id
                if (
                    node.configuration_source == "criteria"
                    and "criteria" not in self.available_runtime
                    and processing_job.runtime.get("criteria") is None
                ):
                    errors.append(
                        f"Criteria configuration is required by node '{node.id}'"
                    )
                errors.extend(
                    f"Runtime value '{key}' is required by node '{node.id}'"
                    for key in process.runtime_requirements
                    if (
                        processing_job.runtime.get(key) in (None, "")
                        and key not in self.available_runtime
                    )
                )
            try:
                topological_order(stage)
            except ValueError as exc:
                errors.append(str(exc))
        try:
            validate_output_label(processing_job.output_label)
        except ValueError as exc:
            errors.append(str(exc))
        return FlowValidationResult(ok=not errors, errors=errors)


def topological_order(flow: FlowStage) -> list[FlowNode]:
    """Return nodes in dependency order or raise for a cycle.

    :param flow: Flow definition to inspect or execute.
    :return: Nodes in dependency order or raise for a cycle.
    :raises ValueError: If the requested operation cannot satisfy its contract.
    """
    nodes = flow.node_map()
    incoming = dict.fromkeys(nodes, 0)
    outgoing: dict[str, list[str]] = {node_id: [] for node_id in nodes}
    for node in flow.nodes:
        for upstream_id in node.inputs.values():
            if upstream_id in nodes:
                incoming[node.id] += 1
                outgoing[upstream_id].append(node.id)
    ready = deque(node_id for node_id, count in incoming.items() if count == 0)
    ordered = []
    while ready:
        node_id = ready.popleft()
        ordered.append(nodes[node_id])
        for downstream_id in outgoing[node_id]:
            incoming[downstream_id] -= 1
            if incoming[downstream_id] == 0:
                ready.append(downstream_id)
    if len(ordered) != len(nodes):
        raise ValueError("Flow contains a cycle")
    return ordered


class FlowExecutor:
    """Execute directly resolved processes in a flow stage.

    A flow contains only importable process paths.  The executor deliberately
    has no registry or built-in dispatch table: processes are instantiated from
    those paths and receive the batch context supplied by the job runner.

    :param progress_callback: Optional callback that receives processing progress events.
    :param cancellation_callback: Optional callback that reports whether execution should stop.
    """

    def __init__(
        self,
        progress_callback: ProgressCallback | None = None,
        cancellation_callback: CancellationCallback | None = None,
    ):
        self.progress_callback = progress_callback
        self.cancellation_callback = cancellation_callback

    def run(
        self,
        processing_job: ProcessingJob,
        contexts: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Run every stage once with optional stage-specific runtime context.

        :param processing_job: Validated flow, parameters, runtime, and output binding.
        :param contexts: Optional runtime values keyed by stage identifier.
        :return: Outputs produced by the completed operation.
        :raises ValueError: If the requested operation cannot satisfy its contract.
        """
        contexts = contexts or {}
        available = set(processing_job.runtime)
        available.add("output_label")
        for context in contexts.values():
            available.update(context)
        validation = FlowValidator(available).validate(processing_job)
        if not validation.ok:
            raise ValueError("; ".join(validation.errors))
        stage_results: dict[str, Any] = {}
        for stage in processing_job.flow.flow_stages():
            stage_results[stage.stage_id] = self.run_stage(
                processing_job, stage, contexts.get(stage.stage_id, {})
            )
        return stage_results

    def run_stage(
        self,
        processing_job: ProcessingJob,
        stage: FlowStage,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run one stage for one concrete run or station/rate batch.

        :param processing_job: Validated flow, parameters, runtime, and output binding.
        :param stage: Flow stage to execute.
        :param context: Runtime values supplied by the flow executor.
        :return: Outputs produced by the completed operation.
        :raises ProcessingCancelled: If cancellation is requested before or during a node.
        :raises Exception: If a process node fails.
        """
        runtime = dict(processing_job.runtime)
        runtime.update(context or {})
        # A job's output label is its artifact namespace.  Do not allow an
        # individual stage context to redirect data into another namespace.
        runtime["output_label"] = processing_job.output_label
        results: dict[str, Any] = {}
        for node in topological_order(stage):
            if self.cancellation_callback is not None and self.cancellation_callback():
                self._emit(
                    ProcessingProgressEvent(
                        state=ProcessingProgressState.cancelled,
                        task=node.id,
                        current=0,
                        total=1,
                        message=f"Cancelled: {node.id}",
                        stage_id=stage.stage_id,
                        node_id=node.id,
                        process=node.process,
                    )
                )
                raise ProcessingCancelled("Processing job cancelled")
            process_class = resolve_process_class(node.process)
            params = processing_job.parameters.for_process(node.process)
            self._emit(
                ProcessingProgressEvent(
                    state=ProcessingProgressState.started,
                    task=node.id,
                    current=0,
                    total=1,
                    message=f"Started: {node.id}",
                    stage_id=stage.stage_id,
                    node_id=node.id,
                    process=node.process,
                )
            )
            try:
                inputs = {
                    port: results[node_id] for port, node_id in node.inputs.items()
                }
                instance = (
                    runtime["criteria"]
                    if node.configuration_source == "criteria"
                    else process_class(**params)
                )
                process_context = dict(runtime)
                process_context["_resistics_progress_callback"] = (
                    lambda event, stage=stage, node=node: self._emit(
                        event.model_copy(
                            update={
                                "stage_id": event.stage_id or stage.stage_id,
                                "node_id": event.node_id or node.id,
                                "process": event.process or node.process,
                            }
                        )
                    )
                )
                process_context["_resistics_cancellation_callback"] = (
                    self.cancellation_callback
                )
                results[node.id] = instance.execute(inputs, process_context)
            except ProcessingCancelled:
                raise
            except Exception as exc:
                self._emit(
                    ProcessingProgressEvent(
                        state=ProcessingProgressState.failed,
                        task=node.id,
                        current=0,
                        total=1,
                        message=f"Failed: {node.id}",
                        stage_id=stage.stage_id,
                        node_id=node.id,
                        process=node.process,
                        error=str(exc),
                    )
                )
                raise
            self._emit(
                ProcessingProgressEvent(
                    state=ProcessingProgressState.completed,
                    task=node.id,
                    current=1,
                    total=1,
                    message=f"Completed: {node.id}",
                    stage_id=stage.stage_id,
                    node_id=node.id,
                    process=node.process,
                )
            )
        return results

    def _emit(self, event: ProcessingProgressEvent) -> None:
        if self.progress_callback is not None:
            self.progress_callback(event)


def _node(node_id: str, process: str, **inputs: str) -> FlowNode:
    return FlowNode(id=node_id, process=process, inputs=inputs)


def _time_to_evals_nodes(windower: str) -> list[FlowNode]:
    nodes = [
        _node("read", "resistics.time.MTH5TimeReader"),
        _node("interpolate_nans", "resistics.time.InterpolateNans", time_data="read"),
        _node("remove_mean", "resistics.time.RemoveMean", time_data="interpolate_nans"),
        _node(
            "decimation_setup",
            "resistics.decimate.DecimationSetup",
            time_data="remove_mean",
        ),
        _node(
            "decimator",
            "resistics.decimate.Decimator",
            dec_params="decimation_setup",
            time_data="remove_mean",
        ),
        _node("window_setup", "resistics.window.WindowSetup", dec_data="decimator"),
        _node(
            "windower",
            windower,
            win_params="window_setup",
            dec_data="decimator",
        ),
        _node(
            "fourier_transform",
            "resistics.spectra.FourierTransform",
            win_data="windower",
        ),
        _node(
            "evaluation_frequencies",
            "resistics.spectra.EvaluationFreqs",
            dec_params="decimation_setup",
            spec_data="fourier_transform",
        ),
    ]
    nodes.append(
        _node(
            "write_evaluation_frequencies",
            "resistics.spectra.EvaluationFrequencyWriter",
            eval_data="evaluation_frequencies",
        )
    )
    return nodes


def _evals_to_tf_nodes() -> list[FlowNode]:
    """Nodes that gather persisted run artifacts for one station/rate batch.

    :return: Nodes that gather persisted run artifacts for one station/rate batch.
    """
    return [
        FlowNode(
            id="criteria",
            process="resistics.gather.GatherCriteria",
            configuration_source="criteria",
        ),
        _node("transfer_function", "resistics.regression.ImpedanceTensorSetup"),
        _node(
            "gather",
            "resistics.gather.Gather",
            selection="criteria",
            tf="transfer_function",
        ),
        _node(
            "regression_preparer",
            "resistics.regression.RegressionPreparerGathered",
            tf="transfer_function",
            gathered_data="gather",
        ),
        _node(
            "solver",
            "resistics.regression.SolverOLS",
            regression_input="regression_preparer",
        ),
        _node(
            "write_solution",
            "resistics.regression.SolutionWriter",
            solution="solver",
        ),
    ]


def mask_calculation_flow() -> FlowDefinition:
    """Full per-run pipeline with independent time and amplitude mask branches.

    :return: Full per-run pipeline with independent time and amplitude mask branches.
    """
    nodes = _time_to_evals_nodes("resistics.window.Windower")
    writer = nodes.pop()
    nodes.extend(
        [
            _node(
                "time_mask",
                "resistics.mask.TimeMask",
                win_data="windower",
                dec_params="decimation_setup",
            ),
            _node(
                "amplitude_mask",
                "resistics.mask.AbsoluteAmplitudeMask",
                win_data="windower",
                dec_params="decimation_setup",
            ),
            writer,
        ]
    )
    return FlowDefinition(
        id="mask_calculation",
        name="Mask Calculation",
        description=(
            "Persist evaluation frequencies and named time/amplitude masks for "
            "each run; gather them in a subsequent evaluations-to-TF job."
        ),
        stages=[
            FlowStage(stage_id="time_to_evals_and_masks", scope="run", nodes=nodes)
        ],
    )


def mask_calculation_parameter_set() -> ParameterSet:
    """Editable defaults paired with {py:func}`mask_calculation_flow`.

    :return: Editable defaults paired with {py:func}`mask_calculation_flow`.
    """
    from resistics.mask import AbsoluteAmplitudeMask, DailyTimeRange, TimeMask

    time_mask = TimeMask(
        daily_include=[DailyTimeRange(from_time="20:00", to_time="06:00")],
    )
    amplitude_mask = AbsoluteAmplitudeMask(
        limits={
            "Ex": {"maximum": 100_000},
            "Ey": {"maximum": 100_000},
            "Hx": {"maximum": 100_000},
            "Hy": {"maximum": 100_000},
        },
    )
    return ParameterSet(
        name="Mask Calculation",
        description="Editable UTC night and absolute peak-amplitude masks.",
        processes={
            "resistics.mask.TimeMask": time_mask.model_dump(mode="json"),
            "resistics.mask.AbsoluteAmplitudeMask": amplitude_mask.model_dump(
                mode="json"
            ),
        },
    )


def single_site_mt_flow() -> FlowDefinition:
    """Run all time-to-evaluations work before station/rate regression.

    :return: Outputs produced by the completed operation.
    """
    return FlowDefinition(
        id="single_site_mt_standard",
        name="Single-Site MT (Standard Windowing)",
        description="Persist all run evaluation artifacts, then gather them for one station/rate regression.",
        stages=[
            FlowStage(
                stage_id="time_to_evals",
                scope="run",
                nodes=_time_to_evals_nodes("resistics.window.Windower"),
            ),
            FlowStage(
                stage_id="evals_to_tf",
                scope="station_rate",
                nodes=_evals_to_tf_nodes(),
            ),
        ],
    )


def single_site_mt_target_flow() -> FlowDefinition:
    """Single-site MT with target-count windows and durable stage boundary.

    :return: Single-site MT with target-count windows and durable stage boundary.
    """
    return FlowDefinition(
        id="single_site_mt_target",
        name="Single-Site MT (Target Windowing)",
        description="Persist all target-window evaluation artifacts, then gather them by station/rate.",
        stages=[
            FlowStage(
                stage_id="time_to_evals",
                scope="run",
                nodes=_time_to_evals_nodes("resistics.window.WindowerTarget"),
            ),
            FlowStage(
                stage_id="evals_to_tf",
                scope="station_rate",
                nodes=_evals_to_tf_nodes(),
            ),
        ],
    )


def remote_reference_mt_flow() -> FlowDefinition:
    """Remote-reference MT using criteria to select each reference station.

        The durable evaluation-frequency stage is deliberately the same as the
        standard single-site flow.  The second stage receives a criteria file,
        whose ``remote_references`` mapping selects the reference station for each
        target station/rate batch.

    :return: Remote-reference MT using criteria to select each reference station.
    """
    return FlowDefinition(
        id="remote_reference_mt",
        name="Remote-Reference MT",
        description="Persist evaluation frequencies for every selected run, then use criteria to gather each target with its remote reference.",
        stages=[
            FlowStage(
                stage_id="time_to_evals",
                scope="run",
                nodes=_time_to_evals_nodes("resistics.window.Windower"),
            ),
            FlowStage(
                stage_id="evals_to_tf",
                scope="station_rate",
                nodes=_evals_to_tf_nodes(),
            ),
        ],
    )


def default_parameter_set(project_path: Path | None = None) -> ParameterSet:
    """Return defaults for all discovered opted-in process classes.

    :param project_path: Project root used to locate configuration and artifacts.
    :return: Defaults for all discovered opted-in process classes.
    :raises ValueError: If the requested operation cannot satisfy its contract.
    """
    processes = {}
    for descriptor in ProcessCatalog(project_path).discover():
        process_class = resolve_process_class(descriptor.path)
        if not process_class.include_in_default_parameters:
            continue
        try:
            processes[descriptor.path] = process_class().model_dump(
                mode="json", exclude={"name"}
            )
        except Exception as exc:
            raise ValueError(
                f"Default parameters unavailable for {descriptor.path}: {exc}"
            ) from exc
    if "resistics.window.WindowerTarget" in processes:
        processes["resistics.window.WindowerTarget"]["target"] = 500
    return ParameterSet(
        name="Default processing parameters",
        description="Shared defaults for built-in concrete processes.",
        processes=processes,
    )


def parameter_set_for_flow(
    flow: FlowDefinition,
    name: str,
    description: str,
    project_path: Path | None = None,
) -> ParameterSet:
    """Return default parameters for only the configurable processes in ``flow``.

    :param flow: Flow definition to inspect or execute.
    :param name: Stable name used for the persisted resource.
    :param description: Human-readable description for the parameter set.
    :param project_path: Project root used to locate configuration and artifacts.
    :return: Default parameters for only the configurable processes in ``flow``.
    """
    defaults = default_parameter_set(project_path)
    processes = {
        process: values
        for process, values in defaults.processes.items()
        if process
        in {
            node.process
            for stage in flow.flow_stages()
            for node in stage.nodes
            if node.configuration_source == "parameters"
        }
    }
    return ParameterSet(name=name, description=description, processes=processes)


def single_site_mt_parameter_set(project_path: Path | None = None) -> ParameterSet:
    """Default parameters for the standard single-site MT flow.

    :param project_path: Project root used to locate configuration and artifacts.
    :return: Default parameters for the standard single-site MT flow.
    """
    return parameter_set_for_flow(
        single_site_mt_flow(),
        "Single-Site MT (Standard Windowing)",
        "Defaults for the standard single-site MT processing flow.",
        project_path,
    )


def single_site_mt_target_parameter_set(
    project_path: Path | None = None,
) -> ParameterSet:
    """Default parameters for the target-window single-site MT flow.

    :param project_path: Project root used to locate configuration and artifacts.
    :return: Default parameters for the target-window single-site MT flow.
    """
    return parameter_set_for_flow(
        single_site_mt_target_flow(),
        "Single-Site MT (Target Windowing)",
        "Defaults for the target-window single-site MT processing flow.",
        project_path,
    )


def remote_reference_mt_parameter_set(
    project_path: Path | None = None,
) -> ParameterSet:
    """Default parameters for the remote-reference MT flow.

    :param project_path: Project root used to locate configuration and artifacts.
    :return: Default parameters for the remote-reference MT flow.
    """
    return parameter_set_for_flow(
        remote_reference_mt_flow(),
        "Remote-Reference MT",
        "Defaults for remote-reference MT processing; choose references in criteria.",
        project_path,
    )


def model_to_dict(model: BaseModel) -> dict[str, Any]:
    """Convert a Pydantic flow model to plain Python values.

    :param model: Pydantic model to serialize.
    :return: The value produced when this operation completes.
    """
    return model.model_dump()


def model_to_yaml(model: BaseModel) -> str:
    """Serialize a Pydantic flow model as readable YAML.

    :param model: Pydantic model to serialize.
    :return: The value produced when this operation completes.
    """
    import yaml

    return yaml.safe_dump(model_to_dict(model), sort_keys=False)


def model_from_yaml[ModelT: BaseModel](
    model_type: type[ModelT], yaml_text: str
) -> ModelT:
    """Validate YAML text as the requested Pydantic flow model.

    :param model_type: Concrete Pydantic model class used for validation.
    :param yaml_text: YAML document to parse and validate.
    :return: The value produced when this operation completes.
    :raises ValueError: If the requested operation cannot satisfy its contract.
    """
    import yaml

    data = yaml.safe_load(yaml_text) or {}
    if model_type is ParameterSet and {"values", "step_values"}.intersection(data):
        raise ValueError(
            "Legacy parameter YAML is not supported. Replace the parameter file "
            "and restore the current defaults."
        )
    return model_type.model_validate(data)


def model_to_yaml_file(model: BaseModel, path: Path) -> None:
    """Serialize a Pydantic flow model to a YAML file.

    :param model: Pydantic model to serialize.
    :param path: Path or routed coordinates to process.
    """
    path.write_text(model_to_yaml(model))


def model_from_yaml_file[ModelT: BaseModel](
    model_type: type[ModelT], path: Path
) -> ModelT:
    """Read and validate a Pydantic flow model from a YAML file.

    :param model_type: Concrete Pydantic model class used for validation.
    :param path: Path or routed coordinates to process.
    :return: The value produced when this operation completes.
    """
    return model_from_yaml(model_type, path.read_text())
