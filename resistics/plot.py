"""Module to help plotting various data"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from html import escape
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tsdownsample import LTTBDownsampler, NaNMinMaxLTTBDownsampler

from resistics.flow_graph import (
    FLOW_CARD_HEIGHT,
    FLOW_CARD_TEXT_SIZE,
    FLOW_CARD_WIDTH,
    FLOW_LAYOUT_PADDING,
    FLOW_LAYOUT_VERTEX_SPACING,
    _flow_node_key,
    _FlowGraphEdge,
    _FlowGraphNode,
    _FlowGraphRenderSpec,
    _render_flow_graph,
)

if TYPE_CHECKING:
    from resistics.flow import FlowDefinition, FlowNode, FlowStage, ProcessDescriptor
    from resistics.job import JobScope, ResolvedJob

PLOTLY_TEMPLATE = "seaborn"
PLOTLY_MARGIN = {"l": 0, "r": 0, "b": 0, "t": 50}
FLOW_CARD_LABEL_LINE_LENGTH = 18
FLOW_CARD_LABEL_MAX_LINES = 2
JOB_CARD_WIDTH = 260
JOB_CARD_BASE_HEIGHT = 118
JOB_CARD_LINE_HEIGHT = 17
JOB_CARD_TEXT_SIZE = 12
JOB_CARD_VALUE_MAX_LENGTH = 38
_LTTB_DOWNSAMPLER = LTTBDownsampler()
_NAN_LTTB_DOWNSAMPLER = NaNMinMaxLTTBDownsampler()


def _flow_descriptors(
    flow: FlowDefinition, project_path: Path | None
) -> dict[str, ProcessDescriptor]:
    """Resolve the process contracts used by a flow.

    :param flow: Flow definition to inspect or execute.
    :param project_path: Project root used to locate configuration and artifacts.
    :return: Process descriptors keyed by qualified process path.
    """
    from resistics.flow import process_descriptor

    process_paths = {
        node.process for stage in flow.flow_stages() for node in stage.nodes
    }
    return {path: process_descriptor(path, project_path) for path in process_paths}


def _validate_flow_node(
    node: FlowNode,
    nodes: Mapping[str, FlowNode],
    descriptors: Mapping[str, ProcessDescriptor],
) -> None:
    """Validate the dependency contracts needed to render one node.

    :param node: Flow node within the stage.
    :param nodes: Nodes used by this operation.
    :param descriptors: Descriptors used by this operation.
    :raises ValueError: If the requested operation cannot satisfy its contract.
    """
    descriptor = descriptors[node.process]
    if set(node.inputs) != set(descriptor.input_types):
        raise ValueError(
            f"Node '{node.id}' inputs must be {sorted(descriptor.input_types)}, "
            f"got {sorted(node.inputs)}"
        )
    for port, upstream_id in node.inputs.items():
        upstream = nodes.get(upstream_id)
        if upstream is None:
            raise ValueError(
                f"Node '{node.id}' references missing input '{upstream_id}'"
            )
        expected_type = descriptor.input_types[port]
        actual_type = descriptors[upstream.process].output_type
        if actual_type != expected_type:
            raise ValueError(
                f"Node '{node.id}' port '{port}' requires '{expected_type}', "
                f"got '{actual_type}' from '{upstream_id}'"
            )


def _validate_flow_stage(
    stage: FlowStage, descriptors: Mapping[str, ProcessDescriptor]
) -> None:
    """Validate one stage before constructing its dependency graph.

    :param stage: Flow stage to execute.
    :param descriptors: Descriptors used by this operation.
    :raises ValueError: If the requested operation cannot satisfy its contract.
    """
    from resistics.flow import topological_order

    nodes = stage.node_map()
    if len(nodes) != len(stage.nodes):
        raise ValueError(f"Stage '{stage.stage_id}' contains duplicate node ids")
    for node in topological_order(stage):
        _validate_flow_node(node, nodes, descriptors)


def _flow_node_hover(
    stage: FlowStage, node: FlowNode, descriptor: ProcessDescriptor
) -> str:
    """Return escaped HTML details for a Plotly node tooltip.

    :param stage: Flow stage to execute.
    :param node: Flow node within the stage.
    :param descriptor: Discovered process metadata for the node.
    :return: Escaped HTML details for a Plotly node tooltip.
    """
    inputs = (
        "<br>".join(
            f"{escape(name)}: {escape(value_type)}"
            for name, value_type in descriptor.input_types.items()
        )
        or "None"
    )
    runtime = ", ".join(map(escape, descriptor.runtime_requirements)) or "None"
    return (
        f"<b>{escape(node.id)}</b><br>{escape(descriptor.path)}<br><br>"
        f"<b>Stage</b><br>{escape(stage.stage_id)} · {escape(stage.scope)}<br><br>"
        f"<b>Inputs</b><br>{inputs}<br><br>"
        f"<b>Output</b><br>{escape(descriptor.output_type)}<br><br>"
        f"<b>Configuration</b><br>{escape(node.configuration_source)}<br><br>"
        f"<b>Runtime requirements</b><br>{runtime}"
    )


def _flow_label_lines(value: str) -> list[str]:
    """Wrap a node label at readable word boundaries for a compact card.

    :param value: Value to validate or normalize.
    :return: Wrap a node label at readable word boundaries for a compact card.
    """
    words = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", value)
    words = re.sub(r"[_-]+", " ", words).split()
    words = [
        word[index : index + FLOW_CARD_LABEL_LINE_LENGTH]
        for word in words
        for index in range(0, len(word), FLOW_CARD_LABEL_LINE_LENGTH)
    ]
    lines = []
    for word in words:
        if not lines or len(lines[-1]) + len(word) + 1 > FLOW_CARD_LABEL_LINE_LENGTH:
            lines.append(word)
        else:
            lines[-1] = f"{lines[-1]} {word}"
    if len(lines) > FLOW_CARD_LABEL_MAX_LINES:
        lines = lines[:FLOW_CARD_LABEL_MAX_LINES]
        lines[-1] = f"{lines[-1].rstrip('…')}…"
    return lines or [value]


def _flow_node_label(node: FlowNode, descriptor: ProcessDescriptor) -> str:
    """Return a compact, wrapped node label while preserving hover detail.

    :param node: Flow node within the stage.
    :param descriptor: Discovered process metadata for the node.
    :return: A compact, wrapped node label while preserving hover detail.
    """
    node_id = "<br>".join(escape(line) for line in _flow_label_lines(node.id))
    process_name = "<br>".join(
        escape(line) for line in _flow_label_lines(descriptor.display_name)
    )
    return f"<b>{node_id}</b><br>{process_name}"


def _flow_graph(
    flow: FlowDefinition, descriptors: Mapping[str, ProcessDescriptor]
) -> tuple[list[_FlowGraphNode], list[_FlowGraphEdge]]:
    """Return flow nodes and input edges after validating each stage.

    :param flow: Flow definition to inspect or execute.
    :param descriptors: Descriptors used by this operation.
    :return: Validated staged nodes and typed dependency edges.
    """
    nodes = []
    edges = []
    for stage_index, stage in enumerate(flow.flow_stages()):
        _validate_flow_stage(stage, descriptors)
        node_map = stage.node_map()
        nodes.extend(
            (stage_index, stage, node, descriptors[node.process])
            for node in stage.nodes
        )
        for node in stage.nodes:
            descriptor = descriptors[node.process]
            for port, upstream_id in node.inputs.items():
                edges.append(
                    (
                        stage_index,
                        _flow_node_key(stage, node_map[upstream_id]),
                        _flow_node_key(stage, node),
                        port,
                        descriptor.input_types[port],
                    )
                )

    return nodes, edges


def _flow_render_spec(
    flow: FlowDefinition, nodes: list[_FlowGraphNode]
) -> _FlowGraphRenderSpec:
    """Adapt a flow definition to the shared graph presentation contract.

    :param flow: Flow whose title is displayed.
    :param nodes: Validated nodes used to prepare card labels and hover details.

    :return: Flow-specific presentation consumed by the shared renderer.
    """
    labels = {}
    hovers = {}
    for _, stage, node, descriptor in nodes:
        key = _flow_node_key(stage, node)
        labels[key] = _flow_node_label(node, descriptor)
        hovers[key] = _flow_node_hover(stage, node, descriptor)
    return _FlowGraphRenderSpec(
        title=f"Flow: {escape(flow.name)}",
        labels=labels,
        hovers=hovers,
        card_width=FLOW_CARD_WIDTH,
        card_height=FLOW_CARD_HEIGHT,
        card_text_size=FLOW_CARD_TEXT_SIZE,
        marker_size=FLOW_CARD_HEIGHT,
        vertex_spacing=FLOW_LAYOUT_VERTEX_SPACING,
        margin_top=80,
        minimum_height=500,
        height_base=420,
        legend_y=0.96,
        x_padding=FLOW_LAYOUT_PADDING,
        y_padding=FLOW_LAYOUT_PADDING,
        metadata={},
    )


def plot_flow(flow: FlowDefinition, project_path: Path | None = None) -> go.Figure:
    """Build a ranked, interactive Plotly figure for a staged processing flow.

    :param flow: Processing flow to render.
    :param project_path: Project used to resolve local process plugins.

    :return: Interactive staged flow graph.

    **Examples**

    Build a figure directly from a portable flow definition.

    ```{doctest}
    >>> from resistics.flow import FlowDefinition, FlowNode, FlowStage
    >>> from resistics.plot import plot_flow
    >>> node = FlowNode(id="read", process="resistics.time.MTH5TimeReader")
    >>> stage = FlowStage(stage_id="runs", scope="run", nodes=[node])
    >>> flow = FlowDefinition(id="read", name="Read", stages=[stage])
    >>> figure = plot_flow(flow)
    >>> len(figure.data) > 0
    True

    ```
    """
    descriptors = _flow_descriptors(flow, project_path)
    nodes, edges = _flow_graph(flow, descriptors)
    return _render_flow_graph(flow, nodes, edges, _flow_render_spec(flow, nodes))


def _job_compact_value(value: Any) -> str:
    """Return one readable card-sized representation of a configuration value.

    :param value: Value to validate or normalize.
    :return: One readable card-sized representation of a configuration value.
    """
    if isinstance(value, dict):
        rendered = f"{{{len(value)} keys}}"
    elif isinstance(value, (list, tuple, set)):
        if len(value) <= 3 and all(
            isinstance(item, (str, int, float, bool, type(None))) for item in value
        ):
            rendered = ", ".join(str(item) for item in value)
        else:
            rendered = f"[{len(value)} items]"
    elif isinstance(value, str):
        rendered = value
    else:
        rendered = str(value)
    if len(rendered) > JOB_CARD_VALUE_MAX_LENGTH:
        return f"{rendered[: JOB_CARD_VALUE_MAX_LENGTH - 1]}…"
    return rendered


def _job_configuration_lines(node: FlowNode, resolved_job: ResolvedJob) -> list[str]:
    """Return compact, card-ready configuration details for one job node.

    :param node: Flow node within the stage.
    :param resolved_job: Resolved job used by this operation.
    :return: Compact, card-ready configuration details for one job node.
    """
    if node.configuration_source == "criteria":
        criteria_name = (
            resolved_job.criteria_path.name
            if resolved_job.criteria_path is not None
            else "default criteria"
        )
        policy_count = (
            0 if resolved_job.criteria is None else len(resolved_job.criteria.stations)
        )
        return [
            f"Criteria: {escape(criteria_name)}",
            f"Policies: {policy_count} station(s)",
        ]

    values = resolved_job.processing_job.parameters.for_process(node.process)
    if not values:
        return ["No parameters"]
    return [
        f"{escape(str(name))} = {escape(_job_compact_value(value))}"
        for name, value in values.items()
    ]


def _job_card_label(
    node: FlowNode,
    descriptor: ProcessDescriptor,
    configuration_lines: list[str],
) -> tuple[str, int]:
    """Return an expanded card label and its rendered line count.

    :param node: Flow node within the stage.
    :param descriptor: Discovered process metadata for the node.
    :param configuration_lines: Configuration lines used by this operation.
    :return: An expanded card label and its rendered line count.
    """
    node_lines = _flow_label_lines(node.id)
    process_lines = _flow_label_lines(descriptor.display_name)
    title = "<br>".join(escape(line) for line in node_lines)
    process = "<br>".join(escape(line) for line in process_lines)
    details = "<br>".join(configuration_lines)
    return (
        f"<b>{title}</b><br>{process}<br><br>{details}",
        len(node_lines) + len(process_lines) + 1 + len(configuration_lines),
    )


def _job_preformatted(value: Any) -> str:
    """Format structured configuration for an indented Plotly hover section.

    :param value: Value to validate or normalize.
    :return: Format structured configuration for an indented Plotly hover section.
    """
    rendered = json.dumps(value, indent=2, sort_keys=True, default=str)
    return rendered.replace(" ", "&nbsp;").replace("\n", "<br>")


def _job_node_hover(
    stage: FlowStage,
    node: FlowNode,
    descriptor: ProcessDescriptor,
    resolved_job: ResolvedJob,
) -> str:
    """Return the full effective job configuration for a node hover tooltip.

    :param stage: Flow stage to execute.
    :param node: Flow node within the stage.
    :param descriptor: Discovered process metadata for the node.
    :param resolved_job: Resolved job used by this operation.
    :return: The full effective job configuration for a node hover tooltip.
    """
    base = _flow_node_hover(stage, node, descriptor)
    if node.configuration_source == "criteria":
        criteria_name = (
            resolved_job.criteria_path.name
            if resolved_job.criteria_path is not None
            else "default empty criteria"
        )
        criteria = (
            {}
            if resolved_job.criteria is None
            else resolved_job.criteria.model_dump(mode="json")
        )
        return (
            f"{base}<br><br><b>Criteria configuration ({escape(criteria_name)})</b>"
            f"<br>{_job_preformatted(criteria)}"
        )

    values = resolved_job.processing_job.parameters.for_process(node.process)
    if not values:
        return f"{base}<br><br><b>Parameter-file values</b><br>None supplied; process defaults apply"
    return f"{base}<br><br><b>Parameter-file values</b><br>{_job_preformatted(values)}"


def _job_scope_summary(scope: JobScope) -> str:
    """Return a compact description of the filters applied to a job.

    :param scope: Scope used by this operation.
    :return: A compact description of the filters applied to a job.
    """
    parts = []
    if scope.surveys:
        parts.append(f"surveys: {', '.join(scope.surveys)}")
    if scope.stations:
        parts.append(f"stations: {', '.join(scope.stations)}")
    if scope.sampling_frequencies:
        rates = ", ".join(f"{rate:g} Hz" for rate in scope.sampling_frequencies)
        parts.append(f"rates: {rates}")
    if scope.stages:
        parts.append(f"stages: {', '.join(scope.stages)}")
    return "; ".join(parts) if parts else "all surveys, stations, and rates"


def _job_render_spec(
    resolved_job: ResolvedJob,
    flow: FlowDefinition,
    nodes: list[_FlowGraphNode],
) -> _FlowGraphRenderSpec:
    """Adapt a resolved job to the shared graph presentation contract.

    :param resolved_job: Job providing effective configuration and execution scope.
    :param flow: Selected job stages represented as a flow.
    :param nodes: Validated nodes used to prepare expanded job cards.

    :return: Job-specific presentation consumed by the shared renderer.
    """
    labels = {}
    hovers = {}
    line_counts = []
    for _, stage, node, descriptor in nodes:
        key = _flow_node_key(stage, node)
        label, line_count = _job_card_label(
            node, descriptor, _job_configuration_lines(node, resolved_job)
        )
        labels[key] = label
        hovers[key] = _job_node_hover(stage, node, descriptor, resolved_job)
        line_counts.append(line_count)

    card_height = max(
        JOB_CARD_BASE_HEIGHT,
        28 + max(line_counts, default=0) * JOB_CARD_LINE_HEIGHT,
    )
    vertex_spacing = max(
        FLOW_LAYOUT_VERTEX_SPACING,
        JOB_CARD_WIDTH + 60,
        card_height + 80,
    )
    definition = resolved_job.definition
    batches = resolved_job.batches
    target_runs = sum(len(batch.run_paths) for batch in batches)
    criteria_name = (
        resolved_job.criteria_path.name
        if resolved_job.criteria_path is not None
        else "default criteria"
    )
    header = (
        f"Flow: {escape(flow.name)} ({escape(resolved_job.flow_path.name)})"
        f" · Parameters: {escape(resolved_job.processing_job.parameters.name)} "
        f"({escape(resolved_job.parameters_path.name)})"
        f" · Criteria: {escape(criteria_name)}"
    )
    work_plan = (
        f"Output: {escape(definition.output_label)}"
        f" · Overwrite: {'yes' if definition.overwrite else 'no'}"
        f" · Scope: {escape(_job_scope_summary(definition.scope))}"
        f" · Work: {len(flow.stages)} stage(s), {len(batches)} target batch(es), "
        f"{target_runs} target run(s)"
    )
    return _FlowGraphRenderSpec(
        title=(
            f"Job: {escape(definition.name)}<br><br><sup>{header}</sup>"
            f"<br><sup>{work_plan}</sup>"
        ),
        labels=labels,
        hovers=hovers,
        card_width=JOB_CARD_WIDTH,
        card_height=card_height,
        card_text_size=JOB_CARD_TEXT_SIZE,
        marker_size=max(card_height, JOB_CARD_WIDTH),
        vertex_spacing=vertex_spacing,
        margin_top=160,
        minimum_height=620,
        height_base=460,
        legend_y=0.88,
        x_padding=JOB_CARD_WIDTH / 2 + 40,
        y_padding=card_height / 2 + 50,
        metadata={"job": True},
    )


def plot_job(resolved_job: ResolvedJob, project_path: Path | None = None) -> go.Figure:
    """Build an execution-plan plot from one fully resolved processing job.

    :param resolved_job: Validated job, selected stages, and concrete work batches.
    :param project_path: Project used to resolve local process plugins. When omitted, the job's
        runtime project path is used when present.

    :return: Interactive job graph with effective configuration details.
    """
    if project_path is None:
        runtime_path = resolved_job.processing_job.runtime.get("project_path")
        project_path = None if runtime_path is None else Path(runtime_path)

    flow = resolved_job.processing_job.flow.model_copy(
        update={"stages": list(resolved_job.stages)}
    )
    descriptors = _flow_descriptors(flow, project_path)
    nodes, edges = _flow_graph(flow, descriptors)
    return _render_flow_graph(
        flow,
        nodes,
        edges,
        _job_render_spec(resolved_job, flow, nodes),
    )


def lttb_downsample(
    x: np.ndarray, y: np.ndarray, max_pts: int = 5_000
) -> tuple[np.ndarray, np.ndarray]:
    """Downsample x, y for visualisation

    :param x: x array
    :param y: y array
    :param max_pts: Maximum number of points after downsampling, by default 5000

    :return: (new_x, new_y), the downsampled x and y arrays

    :raises ValueError: If the size of x does not match the size of y

    **Notes**

    Selection indices are applied to the original arrays so their dtypes are
    preserved. Floating-point data containing NaNs uses the NaN-aware
    MinMaxLTTB implementation to retain markers for visible Plotly gaps.

    **Examples**

    Downsampling preserves the first and final coordinates.

    ```{doctest}
    >>> import numpy as np
    >>> from resistics.plot import lttb_downsample
    >>> x, y = lttb_downsample(np.arange(10), np.arange(10), max_pts=4)
    >>> len(x), x[[0, -1]].tolist(), y[[0, -1]].tolist()
    (4, [0, 9], [0, 9])

    ```
    """
    if x.size != y.size:
        raise ValueError(f"x size {x.size} must equal y size {y.size}")
    if max_pts >= x.size:
        return x, y

    contiguous_x = np.ascontiguousarray(x)
    contiguous_y = np.ascontiguousarray(y)
    downsampler = (
        _NAN_LTTB_DOWNSAMPLER
        if np.issubdtype(y.dtype, np.inexact) and np.isnan(y).any()
        else _LTTB_DOWNSAMPLER
    )
    indices = downsampler.downsample(contiguous_x, contiguous_y, n_out=max_pts)
    return x[indices], y[indices]


def apply_lttb(data: np.ndarray, max_pts: int | None) -> tuple[np.ndarray, np.ndarray]:
    """A helper function for applying lttb downsampling if max_pts is not None

    :param data: The data to downsample
    :param max_pts: The maximum number of points or None. If None, no downsamping is
        performed

    :return: Indices and data selected for plotting

    **Examples**

    ``None`` retains every sample and its positional index.

    ```{doctest}
    >>> import numpy as np
    >>> from resistics.plot import apply_lttb
    >>> indices, values = apply_lttb(np.array([2.0, 3.0]), None)
    >>> indices.tolist(), values.tolist()
    ([0, 1], [2.0, 3.0])

    ```
    """
    indices = np.arange(data.size)
    if max_pts is None:
        return indices, data

    indices, data = lttb_downsample(indices, data, max_pts)
    return indices, data


def plot_timeline(
    df: pd.DataFrame,
    y_col: str,
    title: str = "Timeline",
    ref_time: pd.Timestamp | None = None,
) -> go.Figure:
    """Plot a timeline

    The function converts pd.Timestamps to Python datetime objects which are
    supported more fully in serialization.

    :param df: DataFrame with the first and last times of the horizontal bars
    :param y_col: The column to use for the y axis
    :param title: The title for the plot, by default "Timeline"
    :param ref_time: The reference time, by default None

    :return: Plotly figure
    """

    # get range for x axis
    min_time = pd.Timestamp(df["start"].min())
    max_time = pd.Timestamp(df["end"].max())
    if ref_time is not None and ref_time < min_time:
        min_time = ref_time
    # get axis range and covert to datetime
    pad = 0.1 * (max_time - min_time)
    range_start = pd.Timestamp(min_time - pad).to_pydatetime()
    range_end = pd.Timestamp(max_time + pad).to_pydatetime()

    # sort for ordering
    df = df.sort_values([y_col, "start"])

    fig = px.timeline(
        df,
        x_start="start",
        x_end="end",
        y=y_col,
        color="sample_rate",
        hover_data=["survey", "station", "run"],
        title=title,
    )
    if ref_time is not None:
        fig.add_vline(
            x=ref_time.to_pydatetime(),
            line_width=3,
            line_dash="dash",
            line_color="red",
        )
    fig.update_layout(template=PLOTLY_TEMPLATE, margin=dict(PLOTLY_MARGIN))
    fig.update_xaxes(range=[range_start, range_end])
    fig.update_yaxes(title=None, autorange="reversed")
    fig.update_layout(legend={"itemclick": False, "itemdoubleclick": False})
    return fig


def get_calibration_fig() -> go.Figure:
    """Get a figure for plotting calibration data

    :return: Plotly figure
    """
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        subplot_titles=["Magnitude", "Phase"],
        vertical_spacing=0.05,
    )
    fig.update_xaxes(type="log", row=1, col=1)
    fig.update_yaxes(title_text="Magnitude, nT/mV", type="log", row=1, col=1)
    fig.update_xaxes(title_text="Frequency, Hz", type="log", row=2, col=1)
    fig.update_yaxes(title_text="Phase, radians", row=2, col=1)
    fig.layout.update(template=PLOTLY_TEMPLATE, margin=dict(PLOTLY_MARGIN))
    return fig


def get_time_fig(chans: list[str], y_axis_label: dict[str, str]) -> go.Figure:
    """Get a figure for plotting time data

    :param chans: The channels to plot
    :param y_axis_label: The labels to use for the y axis

    :return: Plotly figure
    """
    fig = make_subplots(
        rows=len(chans),
        cols=1,
        shared_xaxes=True,
        subplot_titles=[f"Channel {chan}" for chan in chans],
        vertical_spacing=0.05,
    )
    for idx, chan in enumerate(chans):
        fig.update_yaxes(title_text=y_axis_label[chan], row=idx + 1, col=1)
    fig.layout.update(template=PLOTLY_TEMPLATE, margin=dict(PLOTLY_MARGIN))
    return fig


def get_spectra_stack_fig(chans: list[str], y_axis_label: dict[str, str]) -> go.Figure:
    """Get a figure for plotting spectra stack data

    :param chans: The channels to plot
    :param y_axis_label: The y axis labels

    :return: Plotly figure
    """
    fig = make_subplots(
        rows=len(chans),
        cols=1,
        shared_xaxes=True,
        subplot_titles=[f"Channel {chan}" for chan in chans],
        vertical_spacing=0.05,
    )
    for idx, chan in enumerate(chans):
        fig.update_xaxes(type="log")
        fig.update_yaxes(title_text=y_axis_label[chan], type="log", row=idx + 1, col=1)
    fig.update_xaxes(title_text="Frequency, Hz", row=len(chans), col=1)
    fig.layout.update(template=PLOTLY_TEMPLATE, margin=dict(PLOTLY_MARGIN))
    return fig


def get_spectra_section_fig(chans: list[str]) -> go.Figure:
    """Get figure for plotting spectra sections

    :param chans: The channels to plot

    :return: Plotly figure
    """
    fig = make_subplots(
        rows=len(chans),
        cols=1,
        subplot_titles=[f"Channel {chan}" for chan in chans],
        vertical_spacing=0.05,
        x_title="Date",
        y_title="Frequency, Hz",
    )
    fig.layout.update(template=PLOTLY_TEMPLATE, margin=dict(PLOTLY_MARGIN))
    return fig
