"""
Module to help plotting various data
"""
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
import pandas as pd
import lttbc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PLOTLY_TEMPLATE = "seaborn"
PLOTLY_MARGIN = dict(l=0, r=0, b=0, t=50)


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
    Apply lttb downsampling if max_pts is not None

    There is a helper function

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
    min_time = df["first_time"].min()
    if ref_time is not None and ref_time < min_time:
        min_time = ref_time
    max_time = df["last_time"].max()
    pad = 0.1 * (max_time - min_time)
    min_time = min_time - pad
    max_time = max_time + pad

    # sort for ordering
    df = df.sort_values([y_col, "first_time"])

    fig = px.timeline(
        df, x_start="first_time", x_end="last_time", y=y_col, color="fs", title=title
    )
    if ref_time is not None:
        fig.add_vline(x=ref_time, line_width=3, line_dash="dash", line_color="red")
    fig.update_layout(template=PLOTLY_TEMPLATE, margin=dict(PLOTLY_MARGIN))
    fig.update_xaxes(range=[min_time, max_time])
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
