"""
Module to help plotting various data
"""
from typing import List, Dict, Union, Tuple, Optional
import numpy as np
import plotly.graph_objects as go

from resistics.common import ResisticsData


def lttb_downsample(
    x: np.ndarray, y: np.ndarray, max_pts: int = 5000
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
    import lttbc

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


class PlotData1D(ResisticsData):
    """
    Class to help plot various 1-D data

    As this takes a full instance of x, it is not recommended for long time
    series as this will have a memory impact
    """

    def __init__(self, x: np.ndarray, data: np.ndarray, rows: List[str]):
        """
        Initialise

        Parameters
        ----------
        x : np.ndarray
            The x array
        data : np.ndarray
            The data, which is n_rows x n_x
        rows : List[str]
            The name of each row
        """
        self.x = x
        self.data = data
        self.rows = rows

    def __getitem__(self, row: str) -> np.ndarray:
        """Get data for a row by name"""
        index = self.rows.index(row)
        return self.data[index, :]

    def x_size(self) -> int:
        """Get the x size"""
        return len(self.x)

    def get_x(self, samples: Optional[np.ndarray] = None) -> np.ndarray:
        """Get x values"""
        if samples is None:
            return self.x
        self.x[samples]


def figure_columns_as_lines(
    subplots: List[str],
    y_labels: Dict[str, str],
    x_label: str,
    title: Union[str, None] = None,
) -> go.Figure:
    """
    Get a figure for columnar data with specified subplots

    Parameters
    ----------
    subplots : List[str]
        The subplot titles
    y_labels : Dict[str, str]
        y labels for each subplot, with subplot as key and label as value
    x_label : str
        The x label, assumed to be the same for all subplots as this is for
        columnar data
    title : Union[str, None], optional
        Title of the figure, by default None

    Returns
    -------
    go.Figure
        A plotly figure

    Raises
    ------
    ValueError
        If the ylabels specification does not match the subplot specification
    """
    from plotly.subplots import make_subplots

    if set(subplots) != set(y_labels.keys()):
        raise ValueError(f"Mismatch between ylabels {y_labels} and subplots {subplots}")
    if title is None:
        title = "Data plot"

    n_subplots = len(subplots)
    fig = make_subplots(
        rows=n_subplots,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=subplots,
    )
    fig.update_layout(title_text=title)
    fig.update_xaxes(title_text=x_label, row=n_subplots, col=1)
    for idx, subplot in enumerate(subplots):
        fig.update_yaxes(title_text=y_labels[subplot], row=idx + 1, col=1)
    return fig


def plot_columns_1d(
    fig,
    data: ResisticsData,
    subplots: List[str],
    subplot_columns: Dict[str, List[str]],
    max_pts: Union[int, None] = 5000,
    label_prefix: str = "",
) -> None:
    """View timeseries data as a line plot"""
    if label_prefix != "":
        label_prefix = f"{label_prefix} : "
    for idx, subplot in enumerate(subplots):
        for column in subplot_columns[subplot]:
            if max_pts is not None:
                nx, ny = lttb_downsample(
                    np.arange(data.x_size()), data[column], max_pts
                )
                nx = data.get_x(samples=nx)
            else:
                nx = data.get_x()
                ny = data[column]
            lineplot = go.Scattergl(x=nx, y=ny, name=f"{label_prefix}{column}")
            fig.add_trace(lineplot, row=idx + 1, col=1)
    return fig
