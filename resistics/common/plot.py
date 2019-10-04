import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib.dates import (
    DateFormatter,
    DayLocator,
    AutoDateLocator,
    AutoDateFormatter,
)
from datetime import datetime, timedelta
from typing import Dict, Tuple

from resistics.common.checks import parseKeywords


def savePlot(path: str, filename: str, fig: plt.figure) -> str:
    """Save plot whilst checking for overwriting
    
    Parameters
    ----------
    path : str
        The path to save the file
    filename : str
        Filename to save to
    fig : plt.figure   
        Matplotlig figure to save
    
    Returns
    -------
    str
        The filename written out to
    """
    import os

    filenameForWriting = filename
    fileExists = True
    counter = 1
    while fileExists:
        filepath = os.path.join(path, "{}.png".format(filenameForWriting))
        if os.path.exists(filepath):
            filenameForWriting = "{}.{}".format(filename, counter)
            counter += 1
        else:
            fileExists = False
            break
    filepath = os.path.join(path, "{}.png".format(filenameForWriting))
    fig.savefig(filepath)
    return filenameForWriting


def getPlotRowsAndCols(maxCols: int, numPlots: int = 0) -> Tuple[int, int]:
    """Get the numbers of rows and columns for plots

    Parameters
    ----------
    maxCols : int
        The maximum number of columns
    numPlots : int
        The of subplots
    """
    if numPlots <= maxCols:
        ncols = numPlots
        nrows = 1
    else:
        ncols = maxCols
        nrows = int(np.ceil(1.0 * numPlots / maxCols))
    return nrows, ncols


def plotOptionsStandard(**kwargs) -> Dict:
    """Get a set of standard plot options

    Parameters
    ----------
    figsize : Tuple, optional
        Set the figure size
    plotfonts : Dict, optional
        Font sizes to use for plotting fonts
    block : bool, optional
        Boolean flag for blocking execution when plot is shown

    Returns
    -------
    Dict
        Dictionary of standard plot options
    """
    default: Dict = {}
    default["figsize"] = (20, 12)
    default["plotfonts"] = getViewFonts()
    default["block"] = True
    default = parseKeywords(default, kwargs)
    return default


def plotOptionsTime(**kwargs) -> Dict:
    """Get default plot options for plotting time data 

    Parameters
    ----------
    figsize : Tuple, optional
        Set the figure size
    plotfonts : Dict, optional
        Font sizes to use for plotting fonts
    block : bool, optional
        Boolean flag for blocking execution when plot is shown
    Eylim : List[float], optional
        y limits for electric data
    Hylim : List[float], optional
        y limits for magnetic data

    Returns
    -------
    out : Dict
        Dictionary of default plot options for plotting time data
    """
    default = plotOptionsStandard()
    default["Eylim"] = []
    default["Hylim"] = []
    default = parseKeywords(default, kwargs)
    return default


def plotOptionsSpec(**kwargs) -> Dict:
    """Get default plot options for plotting spectra data 

    Parameters
    ----------
    figsize : Tuple, optional
        Set the figure size
    plotfonts : Dict, optional
        Font sizes to use for plotting fonts
    block : bool, optional
        Boolean flag for blocking execution when plot is shown
    amplim : List[float], optional
        Amplitude limits for plotting spectra

    Returns
    -------
    out : Dict
        Dictionary of default plot options for plotting spectra data
    """

    default = plotOptionsStandard()
    default["amplim"] = []
    default = parseKeywords(default, kwargs)
    return default


def plotOptionsTransferFunction(**kwargs) -> Dict:
    """Get default plot options for plotting transfer function data 

    Parameters
    ----------
    figsize : Tuple, optional
        Set the figure size
    plotfonts : Dict, optional
        Font sizes to use for plotting fonts
    block : bool, optional
        Boolean flag for blocking execution when plot is shown
    res_ylim : List[float], optional
        y limits for resistivity data
    phase_ylim : List[float], optional
        y limits for phase data
    xlim : List[float], optional
        x limits for transfer function data

    Returns
    -------
    out : Dict
        Dictionary of default plot options for plotting spectra data
    """
    default = plotOptionsStandard()
    default["figsize"] = None
    default["res_ylim"] = [0.01, 10000]
    default["phase_ylim"] = [-20, 90]
    default["xlim"] = [0.0001, 10000]
    default = parseKeywords(default, kwargs)
    return default


def getTransferFunctionFigSize(oneplot: bool, npolarisations: int) -> Tuple[int, int]:
    """Get the plot size for a transfer function plot

    Notes
    -----
    This is awkward because of maintaining aspect ratio for the apparent resistivity. A better solution might be required at some point.

    Parameters
    ----------
    oneplot : bool
        All polarisations in one plot
    npolarisations : int
        Number of polarisations to plot
    
    Returns
    -------
    figsize : Tuple
        The figure size
    """
    if oneplot:
        return (6, 10)
    else:
        return (5 * npolarisations, 9)


def plotOptionsTipper(**kwargs) -> Dict:
    """Get default plot options for plotting transfer function data 

    Parameters
    ----------
    figsize : Tuple, optional
        Set the figure size
    plotfonts : Dict, optional
        Font sizes to use for plotting fonts
    block : bool, optional
        Boolean flag for blocking execution when plot is shown
    length_ylim : List[float], optional
        y limits for length data
    angle_ylim : List[float], optional
        y limits for angle data
    xlim : List[float], optional
        x limits for transfer function data

    Returns
    -------
    out : Dict
        Dictionary of default plot options for plotting spectra data
    """
    default = plotOptionsStandard()
    default["figsize"] = (16, 5)
    default["length_ylim"] = [0.001, 1000]
    default["angle_ylim"] = [-30, 30]
    default["xlim"] = [0.0001, 10000]
    default = parseKeywords(default, kwargs)
    return default


def getViewFonts() -> Dict:
    """Get default plot font options for viewing plots   

    Returns
    -------
    out : Dict
        Dictionary of default plot font options
    """
    plotFonts: Dict = {}
    plotFonts["suptitle"] = 14
    plotFonts["title"] = 12
    plotFonts["axisLabel"] = 12
    plotFonts["axisTicks"] = 12
    plotFonts["legend"] = 12
    return plotFonts


def getPlotFonts() -> Dict:
    """Get default plot font options for saving plots   

    Returns
    -------
    out : Dict
        Dictionary of default plot font options
    """
    plotFonts: Dict = {}
    plotFonts["suptitle"] = 18
    plotFonts["title"] = 16
    plotFonts["axisLabel"] = 16
    plotFonts["axisTicks"] = 14
    plotFonts["legend"] = 14
    return plotFonts


def getPresentationFonts() -> Dict:
    """Get default plot font options for presentation plots   

    Returns
    -------
    out : Dict
        Dictionary of default plot font options
    """
    plotFonts: Dict = {}
    plotFonts["suptitle"] = 22
    plotFonts["title"] = 20
    plotFonts["axisLabel"] = 20
    plotFonts["axisTicks"] = 20
    plotFonts["legend"] = 20
    return plotFonts


def getPaperFonts() -> Dict:
    """Get default plot font options for paper plots   

    Returns
    -------
    out : Dict
        Dictionary of default plot font options
    """
    plotFonts: Dict = {}
    plotFonts["suptitle"] = 18
    plotFonts["title"] = 17
    plotFonts["axisLabel"] = 16
    plotFonts["axisTicks"] = 15
    plotFonts["legend"] = 15
    return plotFonts


def transferFunctionColours() -> Dict[str, str]:
    """Get colours from the different components of standard magnetotelluric transfer functions
    
    Returns
    -------
    Dict[str, str]
        Dictionary mapping magnetotelluric transfer funciton components to colours
    """
    colours = {
        "ExHx": "orange",
        "EyHy": "green",
        "ExHy": "red",
        "EyHx": "blue",
        "HzHx": "gray",
        "HzHy": "magenta",
    }
    return colours


def colorbar2dTime():
    """Get default colormap for 2d plots with time   

    Returns
    -------
    out : plt.cm
        Colormap
    """
    return plt.cm.viridis


def colorbar2dSpectra():
    """Return colorbar for plotting spectra sections

    Returns
    -------
    out : plt.cm
        Colormap
    """
    return plt.cm.magma


def colorbar2dOther():
    """Alternative colormap for 2d data 

    Returns
    -------
    out : plt.cm
        Colormap
    """
    return plt.cm.plasma


def colorbarMultiline():
    """Alternative colormap for plotting multiple line plots 

    Returns
    -------
    out : plt.cm
        Colormap
    """
    return plt.cm.cividis


def addLegends(fig: plt.figure, pos: int = 1) -> None:
    """Add legends to all subplots in figure
    
    Parameters
    ----------
    fig : magplotlib.pyplot.figure
        A matplotlib figure
    pos : int
        Location of legend  
    """
    axList = fig.axes
    for ax in axList:
        ax.legend(loc=pos)
