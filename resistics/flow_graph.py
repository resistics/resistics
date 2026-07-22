"""Shared Plotly rendering for adapted flow and job graphs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from html import escape
from itertools import pairwise
from typing import TYPE_CHECKING, Any

import numpy as np
import plotly.graph_objects as go
from fast_sugiyama import from_edges

if TYPE_CHECKING:
    from resistics.flow import FlowDefinition, FlowNode, FlowStage, ProcessDescriptor

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
FLOW_ARROW_CLEARANCE = 16
type _FlowGraphNode = tuple[
    int,
    "FlowStage",
    "FlowNode",
    "ProcessDescriptor",
]
type _FlowGraphEdge = tuple[int, str, str, str, str]


@dataclass(frozen=True)
class _FlowGraphRenderSpec:
    """Immutable presentation contract for the private graph renderer.

    Attributes
    ----------
    title : str
        Escaped Plotly title markup.
    labels : Mapping[str, str]
        Card label markup keyed by staged node id.
    hovers : Mapping[str, str]
        Node hover markup keyed by staged node id.
    card_width : float
        Width of every node card.
    card_height : float
        Height of every node card.
    card_text_size : int
        Font size for node labels.
    marker_size : float
        Size of the transparent node hover target.
    vertex_spacing : int
        Minimum spacing requested from the graph layout.
    margin_top : int
        Figure margin reserved above the graph.
    minimum_height : int
        Minimum rendered figure height.
    height_base : int
        Base height before graph span is added.
    legend_y : float
        Vertical legend position in paper coordinates.
    x_padding : float
        Horizontal axis padding around laid-out nodes.
    y_padding : float
        Vertical axis padding around laid-out nodes.
    metadata : Mapping[str, Any]
        Adapter-specific values added to every trace's metadata.
    """

    title: str
    labels: Mapping[str, str]
    hovers: Mapping[str, str]
    card_width: float
    card_height: float
    card_text_size: int
    marker_size: float
    vertex_spacing: int
    margin_top: int
    minimum_height: int
    height_base: int
    legend_y: float
    x_padding: float
    y_padding: float
    metadata: Mapping[str, Any]


def _flow_node_key(stage, node) -> str:
    """Return a globally unique key for a staged flow node."""
    return f"{stage.stage_id}:{node.id}"


def _flow_edge_hover(port: str, value_type: str) -> str:
    """Return escaped HTML details for a Plotly input-edge tooltip."""
    return f"<b>{escape(port)}</b><br>Expected type: {escape(value_type)}"


def _flow_layout(
    nodes: list[_FlowGraphNode],
    edges: list[_FlowGraphEdge],
    vertex_spacing: int = FLOW_LAYOUT_VERTEX_SPACING,
) -> tuple[
    dict[str, int],
    dict[int | str, tuple[float, float] | tuple[int, int]],
    list[tuple[int | str, int | str]],
]:
    """Return one shared ranked layout for validated graph nodes and edges.

    Parameters
    ----------
    nodes : list[_FlowGraphNode]
        Adapted staged nodes.
    edges : list[_FlowGraphEdge]
        Adapted dependency edges.
    vertex_spacing : int
        Minimum spacing passed to the layout solver.

    Returns
    -------
    tuple[dict[str, int], dict[int | str, tuple[float, float] | tuple[int, int]], list[tuple[int | str, int | str]]]
        Node numbers, coordinates, and routed edge segments.
    """
    keys = [_flow_node_key(stage, node) for _, stage, node, _ in nodes]
    node_numbers = {key: index for index, key in enumerate(keys)}
    original_numbers = set(node_numbers.values())
    layout_edges = [
        (node_numbers[source], node_numbers[target])
        for _, source, target, _, _ in edges
    ]
    if not original_numbers:
        return node_numbers, {}, []

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

    return node_numbers, positions, routed_edges


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
    return {
        "x": end_x,
        "y": end_y,
        "ax": start_x,
        "ay": start_y,
        "xref": "x",
        "yref": "y",
        "axref": "x",
        "ayref": "y",
        "showarrow": True,
        "arrowhead": 3,
        "arrowsize": 1.35,
        "arrowwidth": 2.25,
        "arrowcolor": colour,
    }


def _flow_arrow_trace(
    path,
    colour: str,
    card_width: float = FLOW_CARD_WIDTH,
    card_height: float = FLOW_CARD_HEIGHT,
    meta: dict | None = None,
    legendgroup: str | None = None,
) -> go.Scatter:
    """Return a trace-based arrowhead that can remain below card labels."""
    start_x, start_y = path[-2]
    end_x, end_y = _flow_arrow_position(path, card_width, card_height)
    angle = float(np.degrees(np.arctan2(end_x - start_x, end_y - start_y)))
    return go.Scatter(
        x=[end_x],
        y=[end_y],
        mode="markers",
        marker={
            "color": colour,
            "size": 15,
            "symbol": "arrow",
            "angle": angle,
            "angleref": "up",
        },
        hoverinfo="skip",
        showlegend=False,
        legendgroup=legendgroup,
        meta=meta,
    )


def _flow_edge_label_position(path) -> tuple[float, float]:
    """Return the distance-weighted midpoint of a routed edge path."""
    segments = [
        ((end_x - start_x) ** 2 + (end_y - start_y) ** 2) ** 0.5
        for (start_x, start_y), (end_x, end_y) in pairwise(path)
    ]
    total_length = sum(segments)
    if total_length == 0:
        return path[0]
    midpoint = total_length / 2
    traversed = 0.0
    for (start_x, start_y), (end_x, end_y), length in zip(
        path, path[1:], segments, strict=False
    ):
        if traversed + length >= midpoint:
            fraction = (midpoint - traversed) / length
            return (
                start_x + fraction * (end_x - start_x),
                start_y + fraction * (end_y - start_y),
            )
        traversed += length
    return path[-1]


def _flow_edge_label_trace(
    path,
    value_type: str,
    legendgroup: str | None = None,
    meta: dict | None = None,
) -> go.Scatter:
    """Return a visible data-type label positioned on a routed edge."""
    x, y = _flow_edge_label_position(path)
    return go.Scatter(
        x=[x],
        y=[y],
        mode="text",
        text=[escape(value_type)],
        textposition="middle center",
        textfont={"color": FLOW_CARD_BORDER_COLOUR, "size": 13},
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


def _graph_trace_metadata(
    spec: _FlowGraphRenderSpec,
    kind: str,
    stage_index: int,
    **details: Any,
) -> dict[str, Any]:
    """Build consistent trace metadata for one rendered graph element.

    Parameters
    ----------
    spec : _FlowGraphRenderSpec
        Adapter-owned graph presentation.
    kind : str
        Stable element kind used by interaction and tests.
    stage_index : int
        Zero-based stage owning the element.
    **details : Any
        Edge or node identity fields for this element.

    Returns
    -------
    dict[str, Any]
        Complete metadata shared by flow and job traces.
    """
    return {
        "kind": kind,
        "stage": stage_index,
        **details,
        **spec.metadata,
    }


def _render_flow_graph(
    flow: FlowDefinition,
    nodes: list[_FlowGraphNode],
    edges: list[_FlowGraphEdge],
    spec: _FlowGraphRenderSpec,
) -> go.Figure:
    """Render an adapted flow or job graph without inspecting job state.

    Parameters
    ----------
    flow : FlowDefinition
        Flow providing ordered stage membership.
    nodes : list[_FlowGraphNode]
        Validated nodes adapted from the source model.
    edges : list[_FlowGraphEdge]
        Validated typed dependencies between the nodes.
    spec : _FlowGraphRenderSpec
        Adapter-owned labels, hover text, title, metadata, and dimensions.

    Returns
    -------
    go.Figure
        Ranked interactive graph with shared rendering behaviour.
    """
    node_numbers, positions, routed_edges = _flow_layout(
        nodes, edges, vertex_spacing=spec.vertex_spacing
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
        x, y = zip(*path, strict=False)
        figure.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                line={"color": colour, "width": 2},
                text=[_flow_edge_hover(port, value_type)] * len(path),
                hovertemplate="%{text}<extra></extra>",
                name="Flow input",
                showlegend=False,
                legendgroup=f"stage-{stage_index}",
                meta=_graph_trace_metadata(
                    spec,
                    "edge",
                    stage_index,
                    source=source,
                    target=target,
                    port=port,
                ),
            )
        )
        figure.add_trace(
            _flow_arrow_trace(
                path,
                colour,
                spec.card_width,
                spec.card_height,
                legendgroup=f"stage-{stage_index}",
                meta=_graph_trace_metadata(
                    spec,
                    "arrow",
                    stage_index,
                    source=source,
                    target=target,
                    port=port,
                ),
            )
        )
        figure.add_trace(
            _flow_edge_label_trace(
                path,
                value_type,
                legendgroup=f"stage-{stage_index}",
                meta=_graph_trace_metadata(
                    spec,
                    "edge_label",
                    stage_index,
                    source=source,
                    target=target,
                    port=port,
                ),
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
                node_x, node_y, spec.card_width, spec.card_height
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
                line={"color": FLOW_CARD_BORDER_COLOUR, "width": 1.5},
                hoverinfo="skip",
                showlegend=False,
                legendgroup=f"stage-{stage_index}",
                meta=_graph_trace_metadata(spec, "cards", stage_index),
            )
        )
        figure.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="text",
                text=[
                    spec.labels[_flow_node_key(stage, node)] for node, _ in stage_nodes
                ],
                textposition="middle center",
                textfont={
                    "color": FLOW_CARD_TEXT_COLOUR,
                    "size": spec.card_text_size,
                },
                hoverinfo="skip",
                showlegend=False,
                legendgroup=f"stage-{stage_index}",
                meta=_graph_trace_metadata(spec, "labels", stage_index),
            )
        )
        figure.add_trace(
            go.Scatter(
                x=x,
                y=y,
                marker={
                    "color": "rgba(0, 0, 0, 0.01)",
                    "size": spec.marker_size,
                    "symbol": "square",
                },
                mode="markers",
                ids=[_flow_node_key(stage, node) for node, _ in stage_nodes],
                customdata=[
                    spec.hovers[_flow_node_key(stage, node)] for node, _ in stage_nodes
                ],
                hovertemplate="%{customdata}<extra></extra>",
                name=f"{stage.stage_id} · {stage.scope}",
                showlegend=False,
                legendgroup=f"stage-{stage_index}",
                meta=_graph_trace_metadata(spec, "nodes", stage_index),
            )
        )
        figure.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines+markers",
                line={"color": colour, "width": 2},
                marker={
                    "color": colour,
                    "line": {"color": FLOW_CARD_BORDER_COLOUR, "width": 1.5},
                    "size": 13,
                    "symbol": "square",
                },
                name=f"{stage.stage_id} · {stage.scope}",
                legendgroup=f"stage-{stage_index}",
                hoverinfo="skip",
                meta=_graph_trace_metadata(spec, "legend", stage_index),
            )
        )

    x_values = [position[0] for position in positions.values()]
    y_values = [position[1] for position in positions.values()]
    x_min, x_max = min(x_values, default=0.0), max(x_values, default=0.0)
    y_min, y_max = min(y_values, default=0.0), max(y_values, default=0.0)
    figure.update_layout(
        title={
            "text": spec.title,
            "x": 0.5,
            "xanchor": "center",
            "y": 0.99,
            "yanchor": "top",
            "font": {"size": 20},
        },
        paper_bgcolor=FLOW_PAPER_COLOUR,
        plot_bgcolor=FLOW_PLOT_COLOUR,
        font={"color": FLOW_CARD_BORDER_COLOUR},
        margin={"l": 0, "r": 0, "b": 0, "t": spec.margin_top},
        height=max(
            spec.minimum_height,
            spec.height_base + int((y_max - y_min) * 1.25),
        ),
        hovermode="closest",
        legend={
            "title": "Stages",
            "orientation": "v",
            "x": 0.01,
            "xanchor": "left",
            "y": spec.legend_y,
            "yanchor": "top",
            "bgcolor": "rgba(255, 255, 255, 0.92)",
            "bordercolor": "#CBD5E1",
            "borderwidth": 1,
            "groupclick": "togglegroup",
        },
    )
    figure.update_xaxes(
        visible=False,
        range=[x_min - spec.x_padding, x_max + spec.x_padding],
    )
    figure.update_yaxes(
        visible=False,
        range=[y_min - spec.y_padding, y_max + spec.y_padding],
    )
    return figure
