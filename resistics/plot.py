"""
Module to help plotting various data
"""

from __future__ import annotations

import json
import re
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
    _FlowGraphNode,
    _FlowGraphRenderSpec,
    _render_flow_graph,
)

if TYPE_CHECKING:
    from resistics.flow import FlowDefinition
    from resistics.job import ResolvedJob

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


def _flow_descriptors(flow, project_path: Path | None):
    """Resolve the process contracts used by a flow."""
    from resistics.flow import process_descriptor

    process_paths = {
        node.process for stage in flow.flow_stages() for node in stage.nodes
    }
    return {path: process_descriptor(path, project_path) for path in process_paths}


def _validate_flow_node(node, nodes, descriptors) -> None:
    """Validate the dependency contracts needed to render one node."""
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


def _validate_flow_stage(stage, descriptors) -> None:
    """Validate one stage before constructing its dependency graph."""
    from resistics.flow import topological_order

    nodes = stage.node_map()
    if len(nodes) != len(stage.nodes):
        raise ValueError(f"Stage '{stage.stage_id}' contains duplicate node ids")
    for node in topological_order(stage):
        _validate_flow_node(node, nodes, descriptors)


def _flow_node_hover(stage, node, descriptor) -> str:
    """Return escaped HTML details for a Plotly node tooltip."""
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
    """Wrap a node label at readable word boundaries for a compact card."""
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


def _flow_node_label(node, descriptor) -> str:
    """Return a compact, wrapped node label while preserving hover detail."""
    node_id = "<br>".join(escape(line) for line in _flow_label_lines(node.id))
    process_name = "<br>".join(
        escape(line) for line in _flow_label_lines(descriptor.display_name)
    )
    return f"<b>{node_id}</b><br>{process_name}"


def _flow_graph(flow, descriptors):
    """Return flow nodes and input edges after validating each stage."""
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

    Parameters
    ----------
    flow : FlowDefinition
        Flow whose title is displayed.
    nodes : list[_FlowGraphNode]
        Validated nodes used to prepare card labels and hover details.

    Returns
    -------
    _FlowGraphRenderSpec
        Flow-specific presentation consumed by the shared renderer.
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

    Parameters
    ----------
    flow : FlowDefinition
        Processing flow to render.
    project_path : Path | None
        Project used to resolve local process plugins.

    Returns
    -------
    go.Figure
        Interactive staged flow graph.
    """
    descriptors = _flow_descriptors(flow, project_path)
    nodes, edges = _flow_graph(flow, descriptors)
    return _render_flow_graph(flow, nodes, edges, _flow_render_spec(flow, nodes))


def _job_compact_value(value: Any) -> str:
    """Return one readable card-sized representation of a configuration value."""
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


def _job_configuration_lines(node, resolved_job) -> list[str]:
    """Return compact, card-ready configuration details for one job node."""
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
    node, descriptor, configuration_lines: list[str]
) -> tuple[str, int]:
    """Return an expanded card label and its rendered line count."""
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
    """Format structured configuration for an indented Plotly hover section."""
    rendered = json.dumps(value, indent=2, sort_keys=True, default=str)
    return rendered.replace(" ", "&nbsp;").replace("\n", "<br>")


def _job_node_hover(stage, node, descriptor, resolved_job) -> str:
    """Return the full effective job configuration for a node hover tooltip."""
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


def _job_scope_summary(scope) -> str:
    """Return a compact description of the filters applied to a job."""
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

    Parameters
    ----------
    resolved_job : ResolvedJob
        Job providing effective configuration and execution scope.
    flow : FlowDefinition
        Selected job stages represented as a flow.
    nodes : list[_FlowGraphNode]
        Validated nodes used to prepare expanded job cards.

    Returns
    -------
    _FlowGraphRenderSpec
        Job-specific presentation consumed by the shared renderer.
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

    Parameters
    ----------
    resolved_job : ResolvedJob
        Validated job, selected stages, and concrete work batches.
    project_path : Path | None
        Project used to resolve local process plugins. When omitted, the job's
        runtime project path is used when present.

    Returns
    -------
    go.Figure
        Interactive job graph with effective configuration details.
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
    """
    Downsample x, y for visualisation

    Parameters
    ----------
    x : np.ndarray
        x array
    y : np.ndarray
        y array
    max_pts : int, optional
        Maximum number of points after downsampling, by default 5000

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (new_x, new_y), the downsampled x and y arrays

    Raises
    ------
    ValueError
        If the size of x does not match the size of y

    Notes
    -----
    Selection indices are applied to the original arrays so their dtypes are
    preserved. Floating-point data containing NaNs uses the NaN-aware
    MinMaxLTTB implementation to retain markers for visible Plotly gaps.
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
    """
    A helper function for applying lttb downsampling if max_pts is not None

    Parameters
    ----------
    data : np.ndarray
        The data to downsample
    max_pts : Union[int, None]
        The maximum number of points or None. If None, no downsamping is
        performed

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Indices and data selected for plotting
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
    """
    Plot a timeline

    The function converts pd.Timestamps to Python datetime objects which are
    supported more fully in serialization.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with the first and last times of the horizontal bars
    y_col : str
        The column to use for the y axis
    title : str, optional
        The title for the plot, by default "Timeline"
    ref_time : Optional[pd.Timestamp], optional
        The reference time, by default None

    Returns
    -------
    go.Figure
        Plotly figure
    """

    # get range for x axis
    min_time = pd.Timestamp(df["start"].min())
    max_time = pd.Timestamp(df["end"].max())
    if ref_time is not None and ref_time < min_time:
        min_time = ref_time
    # get axis range and covert to datetime
    pad = 0.1 * (max_time - min_time)
    range_start = (min_time - pad).to_pydatetime()
    range_end = (max_time + pad).to_pydatetime()

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
    """
    Get a figure for plotting calibration data

    Returns
    -------
    go.Figure
        Plotly figure
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
    """
    Get a figure for plotting time data

    Parameters
    ----------
    chans : List[str]
        The channels to plot
    y_axis_label : Dict[str, str]
        The labels to use for the y axis

    Returns
    -------
    go.Figure
        Plotly figure
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
    """
    Get a figure for plotting spectra stack data

    Parameters
    ----------
    chans : List[str]
        The channels to plot
    y_axis_label : Dict[str, str]
        The y axis labels

    Returns
    -------
    go.Figure
        Plotly figure
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
    """
    Get figure for plotting spectra sections

    Parameters
    ----------
    chans : List[str]
        The channels to plot

    Returns
    -------
    go.Figure
        Plotly figure
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
