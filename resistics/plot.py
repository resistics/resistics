"""
Module to help plotting various data
"""

from html import escape
import json
from pathlib import Path
import re
from typing import Any, List, Dict, Tuple, Optional, Union
from fast_sugiyama import from_edges
import numpy as np
import pandas as pd
import lttbc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PLOTLY_TEMPLATE = "seaborn"
PLOTLY_MARGIN = dict(l=0, r=0, b=0, t=50)
FLOW_STAGE_COLOURS = ("#1D4ED8", "#047857", "#6D28D9", "#9A3412")
FLOW_CARD_BORDER_COLOUR = "#172033"
FLOW_CARD_TEXT_COLOUR = "#FFFFFF"
FLOW_PAPER_COLOUR = "#FFFFFF"
FLOW_PLOT_COLOUR = "#F8FAFC"
FLOW_LAYOUT_PADDING = 100
FLOW_LAYOUT_VERTEX_SPACING = 180
FLOW_CARD_WIDTH = 160
FLOW_CARD_HEIGHT = 96
FLOW_CARD_TEXT_SIZE = 13
FLOW_CARD_LABEL_LINE_LENGTH = 18
FLOW_CARD_LABEL_MAX_LINES = 2
FLOW_ARROW_CLEARANCE = 16
JOB_CARD_WIDTH = 260
JOB_CARD_BASE_HEIGHT = 118
JOB_CARD_LINE_HEIGHT = 17
JOB_CARD_TEXT_SIZE = 12
JOB_CARD_VALUE_MAX_LENGTH = 38


def _flow_descriptors(flow, project_path: Optional[Path]):
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


def _flow_node_key(stage, node) -> str:
    """Return a globally unique key for a staged flow node."""
    return f"{stage.stage_id}:{node.id}"


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


def _flow_edge_hover(port: str, value_type: str) -> str:
    """Return escaped HTML details for a Plotly input-edge tooltip."""
    return f"<b>{escape(port)}</b><br>Expected type: {escape(value_type)}"


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
        for node in stage.nodes:
            nodes.append((stage_index, stage, node, descriptors[node.process]))
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


def _flow_layout(flow, descriptors, vertex_spacing: int = FLOW_LAYOUT_VERTEX_SPACING):
    """Return one shared ranked layout for all stages in a flow."""
    nodes, edges = _flow_graph(flow, descriptors)
    keys = [_flow_node_key(stage, node) for _, stage, node, _ in nodes]
    node_numbers = {key: index for index, key in enumerate(keys)}
    original_numbers = set(node_numbers.values())
    layout_edges = [
        (node_numbers[source], node_numbers[target])
        for _, source, target, _, _ in edges
    ]
    if not original_numbers:
        return nodes, edges, node_numbers, {}, []

    synthetic_root = len(node_numbers)
    targets = {target for _, target in layout_edges}
    root_numbers = sorted(original_numbers.difference(targets))
    layout_edges.extend((synthetic_root, root) for root in root_numbers)
    layouts = from_edges(
        layout_edges,
        dummy_vertices=True,
        vertex_spacing=vertex_spacing,
        check_layout=True,
    ).dot_layout()
    positions = layouts.to_dict()
    positions.pop(synthetic_root)
    routed_edges = [
        edge for _, _, _, solver_edges in layouts for edge in solver_edges or []
    ]

    return nodes, edges, node_numbers, positions, routed_edges


def _routed_adjacency(routed_edges):
    """Group solver edge segments by their source vertex."""
    adjacency = {}
    for start, end in routed_edges:
        adjacency.setdefault(start, []).append(end)
    return adjacency


def _routed_path(source, target, adjacency, original_numbers):
    """Find a path that contains no original nodes between its endpoints."""
    pending = [(source, [source])]
    visited = set()
    while pending:
        current, path = pending.pop()
        if current == target:
            return path
        if current in visited:
            continue
        visited.add(current)
        candidates = [
            candidate
            for candidate in adjacency.get(current, [])
            if candidate not in original_numbers or candidate == target
        ]
        pending.extend((candidate, [*path, candidate]) for candidate in candidates)
    return None


def _flow_edge_path(source, target, routed_edges, original_numbers, positions):
    """Trace one original edge through the solver's dummy routing vertices."""
    path = _routed_path(
        source, target, _routed_adjacency(routed_edges), original_numbers
    )
    if path is None:
        path = [source, target]
    return [positions[number] for number in path]


def _flow_arrow_position(
    path,
    card_width: float = FLOW_CARD_WIDTH,
    card_height: float = FLOW_CARD_HEIGHT,
) -> tuple[float, float]:
    """Return the terminal arrowhead position just outside its target card."""
    start_x, start_y = path[-2]
    end_x, end_y = path[-1]
    dx = end_x - start_x
    dy = end_y - start_y
    length = max((dx**2 + dy**2) ** 0.5, 1.0)
    unit_x = dx / length
    unit_y = dy / length
    distances = []
    if unit_x:
        distances.append(card_width / 2 / abs(unit_x))
    if unit_y:
        distances.append(card_height / 2 / abs(unit_y))
    arrow_offset = min(distances) + FLOW_ARROW_CLEARANCE
    return (
        end_x - arrow_offset * unit_x,
        end_y - arrow_offset * unit_y,
    )


def _flow_arrow_annotation(
    path,
    colour: str,
    card_width: float = FLOW_CARD_WIDTH,
    card_height: float = FLOW_CARD_HEIGHT,
) -> dict:
    """Return a native Plotly arrowhead aligned to the final routed segment."""
    start_x, start_y = path[-2]
    end_x, end_y = _flow_arrow_position(path, card_width, card_height)
    return dict(
        x=end_x,
        y=end_y,
        ax=start_x,
        ay=start_y,
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        showarrow=True,
        arrowhead=3,
        arrowsize=1.35,
        arrowwidth=2.25,
        arrowcolor=colour,
    )


def _flow_arrow_trace(
    path,
    colour: str,
    card_width: float = FLOW_CARD_WIDTH,
    card_height: float = FLOW_CARD_HEIGHT,
    meta: Optional[dict] = None,
    legendgroup: Optional[str] = None,
) -> go.Scatter:
    """Return a trace-based arrowhead that can remain below card labels."""
    start_x, start_y = path[-2]
    end_x, end_y = _flow_arrow_position(path, card_width, card_height)
    angle = float(np.degrees(np.arctan2(end_x - start_x, end_y - start_y)))
    return go.Scatter(
        x=[end_x],
        y=[end_y],
        mode="markers",
        marker=dict(
            color=colour,
            size=15,
            symbol="arrow",
            angle=angle,
            angleref="up",
        ),
        hoverinfo="skip",
        showlegend=False,
        legendgroup=legendgroup,
        meta=meta,
    )


def _flow_card_polygon(
    x: float,
    y: float,
    card_width: float = FLOW_CARD_WIDTH,
    card_height: float = FLOW_CARD_HEIGHT,
) -> tuple[list[float | None], list[float | None]]:
    """Return a rectangular card path that can be toggled with its legend group."""
    half_width = card_width / 2
    half_height = card_height / 2
    return (
        [
            x - half_width,
            x + half_width,
            x + half_width,
            x - half_width,
            x - half_width,
            None,
        ],
        [
            y - half_height,
            y - half_height,
            y + half_height,
            y + half_height,
            y - half_height,
            None,
        ],
    )


def plot_flow(flow, project_path: Optional[Path] = None) -> go.Figure:
    """Build a ranked, interactive Plotly figure for a staged processing flow."""
    descriptors = _flow_descriptors(flow, project_path)
    nodes, edges, node_numbers, positions, routed_edges = _flow_layout(
        flow, descriptors
    )
    figure = go.Figure()
    original_numbers = set(node_numbers.values())
    for stage_index, source, target, port, value_type in edges:
        colour = FLOW_STAGE_COLOURS[stage_index % len(FLOW_STAGE_COLOURS)]
        path = _flow_edge_path(
            node_numbers[source],
            node_numbers[target],
            routed_edges,
            original_numbers,
            positions,
        )
        x, y = zip(*path)
        figure.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                line=dict(color=colour, width=2),
                text=[_flow_edge_hover(port, value_type)] * len(path),
                hovertemplate="%{text}<extra></extra>",
                name="Flow input",
                showlegend=False,
                legendgroup=f"stage-{stage_index}",
                meta={
                    "kind": "edge",
                    "stage": stage_index,
                    "source": source,
                    "target": target,
                    "port": port,
                },
            )
        )
        figure.add_trace(
            _flow_arrow_trace(
                path,
                colour,
                legendgroup=f"stage-{stage_index}",
                meta={
                    "kind": "arrow",
                    "stage": stage_index,
                    "source": source,
                    "target": target,
                    "port": port,
                },
            )
        )

    for stage_index, stage in enumerate(flow.flow_stages()):
        stage_nodes = [
            (node, descriptor)
            for node_stage_index, node_stage, node, descriptor in nodes
            if node_stage_index == stage_index and node_stage is stage
        ]
        if not stage_nodes:
            continue
        colour = FLOW_STAGE_COLOURS[stage_index % len(FLOW_STAGE_COLOURS)]
        x = [
            positions[node_numbers[_flow_node_key(stage, node)]][0]
            for node, _ in stage_nodes
        ]
        y = [
            positions[node_numbers[_flow_node_key(stage, node)]][1]
            for node, _ in stage_nodes
        ]
        card_x = []
        card_y = []
        for node_x, node_y in zip(x, y, strict=True):
            x_path, y_path = _flow_card_polygon(node_x, node_y)
            card_x.extend(x_path)
            card_y.extend(y_path)
        figure.add_trace(
            go.Scatter(
                x=card_x,
                y=card_y,
                mode="lines",
                fill="toself",
                fillcolor=colour,
                line=dict(color=FLOW_CARD_BORDER_COLOUR, width=1.5),
                hoverinfo="skip",
                showlegend=False,
                legendgroup=f"stage-{stage_index}",
                meta={"kind": "cards", "stage": stage_index},
            )
        )
        figure.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="text",
                text=[
                    _flow_node_label(node, descriptor)
                    for node, descriptor in stage_nodes
                ],
                textposition="middle center",
                textfont=dict(color=FLOW_CARD_TEXT_COLOUR, size=FLOW_CARD_TEXT_SIZE),
                hoverinfo="skip",
                showlegend=False,
                legendgroup=f"stage-{stage_index}",
                meta={"kind": "labels", "stage": stage_index},
            )
        )
        figure.add_trace(
            go.Scatter(
                x=x,
                y=y,
                marker=dict(
                    color="rgba(0, 0, 0, 0.01)",
                    size=FLOW_CARD_HEIGHT,
                    symbol="square",
                ),
                mode="markers",
                ids=[_flow_node_key(stage, node) for node, _ in stage_nodes],
                customdata=[
                    _flow_node_hover(stage, node, descriptor)
                    for node, descriptor in stage_nodes
                ],
                hovertemplate="%{customdata}<extra></extra>",
                name=f"{stage.stage_id} · {stage.scope}",
                showlegend=False,
                legendgroup=f"stage-{stage_index}",
                meta={"kind": "nodes", "stage": stage_index},
            )
        )
        figure.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines+markers",
                line=dict(color=colour, width=2),
                marker=dict(
                    color=colour,
                    line=dict(color=FLOW_CARD_BORDER_COLOUR, width=1.5),
                    size=13,
                    symbol="square",
                ),
                name=f"{stage.stage_id} · {stage.scope}",
                legendgroup=f"stage-{stage_index}",
                hoverinfo="skip",
                meta={"kind": "legend", "stage": stage_index},
            )
        )

    x_values = [position[0] for position in positions.values()]
    y_values = [position[1] for position in positions.values()]
    x_min, x_max = min(x_values, default=0.0), max(x_values, default=0.0)
    y_min, y_max = min(y_values, default=0.0), max(y_values, default=0.0)
    figure.update_layout(
        title=dict(
            text=f"Flow: {escape(flow.name)}",
            x=0.5,
            xanchor="center",
            y=0.99,
            yanchor="top",
            font=dict(size=20),
        ),
        paper_bgcolor=FLOW_PAPER_COLOUR,
        plot_bgcolor=FLOW_PLOT_COLOUR,
        font=dict(color=FLOW_CARD_BORDER_COLOUR),
        margin=dict(l=0, r=0, b=0, t=80),
        height=max(500, 420 + int((y_max - y_min) * 1.25)),
        hovermode="closest",
        legend=dict(
            title="Stages",
            orientation="v",
            x=0.01,
            xanchor="left",
            y=0.96,
            yanchor="top",
            bgcolor="rgba(255, 255, 255, 0.92)",
            bordercolor="#CBD5E1",
            borderwidth=1,
            groupclick="togglegroup",
        ),
    )
    figure.update_xaxes(
        visible=False,
        range=[x_min - FLOW_LAYOUT_PADDING, x_max + FLOW_LAYOUT_PADDING],
    )
    figure.update_yaxes(
        visible=False,
        range=[y_min - FLOW_LAYOUT_PADDING, y_max + FLOW_LAYOUT_PADDING],
    )
    return figure


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
        policy_count = 0 if resolved_job.criteria is None else len(
            resolved_job.criteria.stations
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


def _job_card_label(node, descriptor, configuration_lines: list[str]) -> tuple[str, int]:
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
    """Escape structured configuration for an indented Plotly hover section."""
    rendered = json.dumps(value, indent=2, sort_keys=True, default=str)
    return escape(rendered).replace(" ", "&nbsp;").replace("\n", "<br>")


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
            {} if resolved_job.criteria is None else resolved_job.criteria.model_dump(mode="json")
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


def plot_job(resolved_job, project_path: Optional[Path] = None) -> go.Figure:
    """Build an execution-plan plot from one fully resolved processing job."""
    if project_path is None:
        runtime_path = resolved_job.processing_job.runtime.get("project_path")
        project_path = None if runtime_path is None else Path(runtime_path)

    flow = resolved_job.processing_job.flow.model_copy(
        update={"stages": list(resolved_job.stages)}
    )
    descriptors = _flow_descriptors(flow, project_path)
    nodes, edges = _flow_graph(flow, descriptors)
    labels = {}
    line_counts = []
    for _, stage, node, descriptor in nodes:
        key = _flow_node_key(stage, node)
        label, line_count = _job_card_label(
            node, descriptor, _job_configuration_lines(node, resolved_job)
        )
        labels[key] = label
        line_counts.append(line_count)
    card_height = max(
        JOB_CARD_BASE_HEIGHT,
        28 + max(line_counts, default=0) * JOB_CARD_LINE_HEIGHT,
    )
    vertex_spacing = max(
        FLOW_LAYOUT_VERTEX_SPACING,
        int(JOB_CARD_WIDTH + 60),
        int(card_height + 80),
    )
    nodes, edges, node_numbers, positions, routed_edges = _flow_layout(
        flow, descriptors, vertex_spacing=vertex_spacing
    )

    figure = go.Figure()
    original_numbers = set(node_numbers.values())
    for stage_index, source, target, port, value_type in edges:
        colour = FLOW_STAGE_COLOURS[stage_index % len(FLOW_STAGE_COLOURS)]
        path = _flow_edge_path(
            node_numbers[source],
            node_numbers[target],
            routed_edges,
            original_numbers,
            positions,
        )
        x, y = zip(*path)
        figure.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                line=dict(color=colour, width=2),
                text=[_flow_edge_hover(port, value_type)] * len(path),
                hovertemplate="%{text}<extra></extra>",
                name="Flow input",
                showlegend=False,
                legendgroup=f"stage-{stage_index}",
                meta={
                    "kind": "edge",
                    "stage": stage_index,
                    "source": source,
                    "target": target,
                    "port": port,
                    "job": True,
                },
            )
        )
        figure.add_trace(
            _flow_arrow_trace(
                path,
                colour,
                JOB_CARD_WIDTH,
                card_height,
                meta={
                    "kind": "arrow",
                    "stage": stage_index,
                    "source": source,
                    "target": target,
                    "port": port,
                    "job": True,
                },
                legendgroup=f"stage-{stage_index}",
            )
        )

    for stage_index, stage in enumerate(flow.flow_stages()):
        stage_nodes = [
            (node, descriptor)
            for node_stage_index, node_stage, node, descriptor in nodes
            if node_stage_index == stage_index and node_stage is stage
        ]
        if not stage_nodes:
            continue
        colour = FLOW_STAGE_COLOURS[stage_index % len(FLOW_STAGE_COLOURS)]
        x = [
            positions[node_numbers[_flow_node_key(stage, node)]][0]
            for node, _ in stage_nodes
        ]
        y = [
            positions[node_numbers[_flow_node_key(stage, node)]][1]
            for node, _ in stage_nodes
        ]
        card_x = []
        card_y = []
        for node_x, node_y in zip(x, y, strict=True):
            x_path, y_path = _flow_card_polygon(
                node_x, node_y, JOB_CARD_WIDTH, card_height
            )
            card_x.extend(x_path)
            card_y.extend(y_path)
        figure.add_trace(
            go.Scatter(
                x=card_x,
                y=card_y,
                mode="lines",
                fill="toself",
                fillcolor=colour,
                line=dict(color=FLOW_CARD_BORDER_COLOUR, width=1.5),
                hoverinfo="skip",
                showlegend=False,
                legendgroup=f"stage-{stage_index}",
                meta={"kind": "cards", "stage": stage_index, "job": True},
            )
        )
        figure.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="text",
                text=[labels[_flow_node_key(stage, node)] for node, _ in stage_nodes],
                textposition="middle center",
                textfont=dict(color=FLOW_CARD_TEXT_COLOUR, size=JOB_CARD_TEXT_SIZE),
                hoverinfo="skip",
                showlegend=False,
                legendgroup=f"stage-{stage_index}",
                meta={"kind": "labels", "stage": stage_index, "job": True},
            )
        )
        figure.add_trace(
            go.Scatter(
                x=x,
                y=y,
                marker=dict(
                    color="rgba(0, 0, 0, 0.01)",
                    size=max(card_height, JOB_CARD_WIDTH),
                    symbol="square",
                ),
                mode="markers",
                ids=[_flow_node_key(stage, node) for node, _ in stage_nodes],
                customdata=[
                    _job_node_hover(stage, node, descriptor, resolved_job)
                    for node, descriptor in stage_nodes
                ],
                hovertemplate="%{customdata}<extra></extra>",
                name=f"{stage.stage_id} · {stage.scope}",
                showlegend=False,
                legendgroup=f"stage-{stage_index}",
                meta={"kind": "nodes", "stage": stage_index, "job": True},
            )
        )
        figure.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines+markers",
                line=dict(color=colour, width=2),
                marker=dict(
                    color=colour,
                    line=dict(color=FLOW_CARD_BORDER_COLOUR, width=1.5),
                    size=13,
                    symbol="square",
                ),
                name=f"{stage.stage_id} · {stage.scope}",
                legendgroup=f"stage-{stage_index}",
                hoverinfo="skip",
                meta={"kind": "legend", "stage": stage_index, "job": True},
            )
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
    x_values = [position[0] for position in positions.values()]
    y_values = [position[1] for position in positions.values()]
    x_min, x_max = min(x_values, default=0.0), max(x_values, default=0.0)
    y_min, y_max = min(y_values, default=0.0), max(y_values, default=0.0)
    figure.update_layout(
        title=dict(
            text=(
                f"Job: {escape(definition.name)}<br><br><sup>{header}</sup>"
                f"<br><sup>{work_plan}</sup>"
            ),
            x=0.5,
            xanchor="center",
            y=0.99,
            yanchor="top",
            font=dict(size=20),
        ),
        paper_bgcolor=FLOW_PAPER_COLOUR,
        plot_bgcolor=FLOW_PLOT_COLOUR,
        font=dict(color=FLOW_CARD_BORDER_COLOUR),
        margin=dict(l=0, r=0, b=0, t=160),
        height=max(620, 460 + int((y_max - y_min) * 1.25)),
        hovermode="closest",
        legend=dict(
            title="Stages",
            orientation="v",
            x=0.01,
            xanchor="left",
            y=0.88,
            yanchor="top",
            bgcolor="rgba(255, 255, 255, 0.92)",
            bordercolor="#CBD5E1",
            borderwidth=1,
            groupclick="togglegroup",
        ),
    )
    figure.update_xaxes(
        visible=False,
        range=[x_min - JOB_CARD_WIDTH / 2 - 40, x_max + JOB_CARD_WIDTH / 2 + 40],
    )
    figure.update_yaxes(
        visible=False,
        range=[y_min - card_height / 2 - 50, y_max + card_height / 2 + 50],
    )
    return figure


def lttb_downsample(
    x: np.ndarray, y: np.ndarray, max_pts: int = 5_000
) -> Tuple[np.ndarray, np.ndarray]:
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
    """
    if x.size != y.size:
        raise ValueError(f"x size {x.size} must equal y size {y.size}")
    if max_pts >= x.size:
        return x, y

    x_dtype = x.dtype
    y_dtype = y.dtype
    nx, ny = lttbc.downsample(
        x.astype(np.float32),
        y.astype(np.float32),
        max_pts,
    )
    return nx.astype(x_dtype), ny.astype(y_dtype)


def apply_lttb(
    data: np.ndarray, max_pts: Union[int, None]
) -> Tuple[np.ndarray, np.ndarray]:
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
    ref_time: Optional[pd.Timestamp] = None,
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

    def check_datetime(x):
        if isinstance(x, pd.Timestamp):
            return x.to_pydatetime()
        return x

    # get range for x axis
    min_time = df["start"].min()
    max_time = df["end"].max()
    if ref_time is not None and ref_time < min_time:
        ref_time = check_datetime(ref_time)
        min_time = ref_time
    # get axis range and covert to datetime
    pad = 0.1 * (max_time - min_time)
    min_time = check_datetime(min_time - pad)
    max_time = check_datetime(max_time + pad)

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
        fig.add_vline(x=ref_time, line_width=3, line_dash="dash", line_color="red")
    fig.update_layout(template=PLOTLY_TEMPLATE, margin=dict(PLOTLY_MARGIN))
    fig.update_xaxes(range=[min_time, max_time])
    fig.update_yaxes(title=None, autorange="reversed")
    fig.update_layout(legend=dict(itemclick=False, itemdoubleclick=False))
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


def get_time_fig(chans: List[str], y_axis_label: Dict[str, str]) -> go.Figure:
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


def get_spectra_stack_fig(chans: List[str], y_axis_label: Dict[str, str]) -> go.Figure:
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


def get_spectra_section_fig(chans: List[str]) -> go.Figure:
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
