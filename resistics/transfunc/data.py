import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import List, Dict, Tuple

from resistics.common.base import ResisticsBase
from resistics.common.print import listToString, arrayToString
from resistics.common.plot import (
    getViewFonts,
    getTransferFunctionFigSize,
    transferFunctionColours,
)


class TransferFunctionData(ResisticsBase):
    """Class for holding information time data

    Attributes
    ----------
    freq : np.ndarray
        The evaluation frequencies for which transfer functions were estimated
    period : np.ndarray
        The evaluation periods for which the transfer functions were estimated (1/freq)
    data : Dict
        The transfer function data keyed by polarisations. Data is in the units E = mV, H = nT
    polarisations : List[str]
        List of polarisations in the data
    variances : Dict
        The variances of the estimates to give an idea of uncertainty

    Methods
    -------
    __init__(freq, data, variances)
        Initialise the transfer function data
    getComponent(component)
        Set data with parameters
    getVariance(component)
        Add a comment to the dataset
    getResAndPhase(component) 
        A datetime array of the sample times
    getResAndPhaseErrors(component)
        View the spectra data 
    view()
        View the impedance tensor components
    viewTipper()
        View the tipper
    printList()
        Class status returned as list of strings        

    Notes
    -----
    Information about data units
    Magnetic permeability in nT . m / A
    Electric (E) data is in mV/m 
    Magnetic (H) data is in nT
    Z = E/H is in mV / m . nT
    Units of resistance = Ohm = V / A        
    T = H/H is dimensionless              
    """

    def __init__(self, freq: np.ndarray, data: Dict, variances: Dict):
        """Initialise and set object parameters

        Parameters
        ----------
        freq : np.ndarray
            Evaluation frequency array
        data : Dict
            Dictionary of data keyed by polarisation and with array values. Data units expected to be E = mV, H = nT
        variances : Dict
            Dictionary of uncertainties keyed by polarisation and with array values 
        """
        self.freq: np.ndarray = freq
        self.period: np.ndarray = np.reciprocal(self.freq)
        self.data: Dict = data
        self.polarisations: List[str] = sorted(self.data.keys())
        self.variances: Dict = variances

    def __getitem__(self, component: str) -> np.ndarray:
        """Get data for a polarisation

        Parameters
        ----------
        component : str
            The polarisation to return the data for
        
        Returns
        -------
        np.ndarray
            The data for the polarisation component
        """
        return self.getComponent(component)

    def getComponent(self, component: str) -> np.ndarray:
        """Get data for a polarisation

        Parameters
        ----------
        component : str
            The polarisation to return the data for
        
        Returns
        -------
        np.ndarray
            The data for the polarisation component
        """
        return self.data[component]

    def __iter__(self):
        """Return the component iterator
        
        Returns
        -------
        list_iterator
            An iterator for the components
        """
        return iter(self.components)

    def getVariance(self, component: str) -> np.ndarray:
        """Get the variance for a component

        Parameters
        ----------
        component : str
            The polarisation to return the data for
        
        Returns
        -------
        np.ndarray
            The data for the polarisation component
        """
        return self.variances[component]

    def getResAndPhase(self, component: str) -> Tuple[np.ndarray, np.ndarray]:
        """Return the resistivity and phase for a component

        The raw data is expected to be units E = mV, H = nT
        Resisitivity is calculated as 0.2 * period * (abs(data))^2
        Phase is calculated as the complex angle, unwrapped and then shifted for components ExHx and ExHy to put them in 0-90 degree quadrant

        Parameters
        ----------
        component : str
            The polarisation to return the data for
        
        Returns
        -------
        res : np.ndarray
            The resistivity for the evaluation frequencies
        phase : np.ndarray
            The phase for the evaluation frequencies
        """
        data = self.getComponent(component)
        res = 0.2 * self.period * np.power(np.absolute(data), 2)
        phase = np.angle(data)
        # check, can we unwrap into specific quadrant
        phase = np.unwrap(phase)
        # convert to degrees
        phase = phase * 180 / np.pi
        if component == "ExHx" or component == "ExHy":
            # do modulo 360, see if this helps
            phase = np.mod(phase, 360)
            phase = phase - 180
        return res, phase

    def getResAndPhaseErrors(self, component: str) -> Tuple[np.ndarray, np.ndarray]:
        """Return the resistivity and phase errors for a component

        The raw data is expected to be units E = mV, H = nT
        Errors are calculated parameterically (for confidence intervals)
        Error in resistivity is given by 1.96 * sqrt(2 * period * res * var / 5.0)
        Error in phase is given by 1.96 * (180 / pi) * (sqrt(var / 2.0) / abs(data))

        .. todo:: 
        
            Is there are better way to calculate errors

        Parameters
        ----------
        component : str
            The polarisation to return the data for
        
        Returns
        -------
        resError : np.ndarray
            The errors in resistivity data for the evaluation frequencies
        phaseError : np.ndarray
            The errors in phase data for the evaluation frequencies
        """
        data = self.getComponent(component)
        res, phase = self.getResAndPhase(component)
        var = self.getVariance(component)
        resError = 1.96 * np.sqrt(2 * self.period * res * var / 5.0)
        phaseError = 1.96 * (180 / np.pi) * (np.sqrt(var / 2.0) / np.absolute(data))
        # calculate uncertainties on apparent resistivity and phase
        return resError, phaseError

    def getTipper(self):
        """Return tipper length and angle

        The tipper are the Tx = HzHx and Ty = HzHy components
        The tipper length is sqrt(Re(Tx)^2 + Re(Ty)^2)
        The tipper angle is arctan (Re(Ty)/Re(Tx))

        This needs HzHx and HzHy to be components of the transfer function
        
        Returns
        -------
        tipperLength : np.ndarray
            The tipper length
        tipperAngle : np.ndarray
            The tipper angle
        """
        if "HzHy" not in self.polarisations or "HzHx" not in self.polarisations:
            return False, False
        txRe = np.real(self.getComponent("HzHx"))
        txIm = np.imag(self.getComponent("HzHx"))
        tyRe = np.real(self.getComponent("HzHy"))
        tyIm = np.imag(self.getComponent("HzHy"))
        tipperLength = np.sqrt(np.power(txRe, 2) + np.power(tyRe, 2))
        tipperAngleRe = np.arctan(tyRe / txRe)
        tipperAngleIm = np.arctan(tyIm / txIm)
        return tipperLength, tipperAngleRe, tipperAngleIm

    def viewImpedance(self, **kwargs) -> Figure:
        """Plots the transfer function data

        For resistivity data, both axes are log scale (period and resistivity). For phase data, period is in log scale and phase is linear scale.
        Units, x axis is seconds, resistivity is Ohm m and phase is degrees.

        Parameters
        ----------
        polarisations : List[str], optional
            Polarisations to plot
        fig : matplotlib.pyplot.figure, optional
            A figure object
        oneplot : bool, optional   
            Boolean flag for plotting all polarisations on one plot rather than separate plots               
        colours : Dict[str, str], optional
            Colours dictionary for plotting impedance components 
        mk : str, optional
            Plot marker type
        ls : str, optional
            Line style  
        plotfonts : Dict, optional
            A dictionary of plot fonts
        label : str, optional
            Label for the plots
        xlim : List, optional
            Limits for the x axis
        res_ylim : List, optional
            Limits for the resistivity y axis
        phase_ylim : List, optional
            Limits for the phase y axis 

        Returns
        -------
        plt.figure
            Matplotlib figure object            
        """
        polarisations = (
            kwargs["polarisations"] if "polarisations" in kwargs else self.polarisations
        )

        # limits
        xlim = kwargs["xlim"] if "xlim" in kwargs else [1e-3, 1e4]
        res_ylim = kwargs["res_ylim"] if "res_ylim" in kwargs else [1e-2, 1e3]
        phase_ylim = kwargs["phase_ylim"] if "phase_ylim" in kwargs else [-30, 120]

        # markers
        colours = (
            kwargs["colours"] if "colours" in kwargs else transferFunctionColours()
        )
        mk = kwargs["mk"] if "mk" in kwargs else "o"
        ls = kwargs["ls"] if "ls" in kwargs else "none"
        plotfonts = kwargs["plotfonts"] if "plotfonts" in kwargs else getViewFonts()

        # calculate number of rows and columns
        oneplot = False
        if "oneplot" in kwargs and kwargs["oneplot"]:
            oneplot = True
        nrows = 2
        ncols = 1 if oneplot else len(polarisations)
        # a multiplier to make sure all the components end up on the right plot
        plotNumMult = 0 if ncols > 1 else 1

        # plot
        if "fig" in kwargs:
            fig = plt.figure(kwargs["fig"].number)
        else:
            figsize = getTransferFunctionFigSize(oneplot, len(polarisations))
            fig = plt.figure(figsize=figsize)

        st = fig.suptitle(
            "Impedance tensor apparent resistivity and phase",
            fontsize=plotfonts["suptitle"],
        )
        st.set_y(0.98)

        for idx, pol in enumerate(polarisations):
            res, phase = self.getResAndPhase(pol)
            resError, phaseError = self.getResAndPhaseErrors(pol)
            label = kwargs["label"] + " - {}".format(pol) if "label" in kwargs else pol
            # plot resistivity
            ax1 = plt.subplot(nrows, ncols, idx + 1 - plotNumMult * idx)
            # the title
            if not oneplot:
                plt.title("Polarisation {}".format(pol), fontsize=plotfonts["title"])
            else:
                plt.title(
                    "Polarisations {}".format(listToString(polarisations)),
                    fontsize=plotfonts["title"],
                )
            # plot the data
            ax1.errorbar(
                self.period,
                res,
                yerr=resError,
                ls=ls,
                marker=mk,
                markersize=7,
                markerfacecolor="white",
                markeredgecolor=colours[pol],
                mew=1.1,
                color=colours[pol],
                ecolor=colours[pol],
                elinewidth=1.0,
                capsize=4,
                barsabove=False,
                label=label,
            )
            ax1.set_xscale("log")
            ax1.set_yscale("log")
            ax1.set_aspect("equal", adjustable="box")
            # axis options
            plt.ylabel("Apparent Res. [Ohm m]", fontsize=plotfonts["axisLabel"])
            plt.xlim(xlim)
            plt.ylim(res_ylim)
            # set tick sizes
            for lab in ax1.get_xticklabels() + ax1.get_yticklabels():
                lab.set_fontsize(plotfonts["axisTicks"])

            # plot phase
            ax2 = plt.subplot(nrows, ncols, ncols + idx + 1 - plotNumMult * idx)
            # plot the data
            ax2.errorbar(
                self.period,
                phase,
                yerr=phaseError,
                ls="none",
                marker=mk,
                markersize=7,
                markerfacecolor="white",
                markeredgecolor=colours[pol],
                mew=1.1,
                color=colours[pol],
                ecolor=colours[pol],
                elinewidth=1.0,
                capsize=4,
                barsabove=False,
                label=label,
            )
            ax2.set_xscale("log")
            # axis options
            plt.xlabel("Period [s]", fontsize=plotfonts["axisLabel"])
            plt.ylabel("Phase [degrees]", fontsize=plotfonts["axisLabel"])
            plt.xlim(xlim)
            plt.ylim(phase_ylim)
            # set tick sizes
            for lab in ax2.get_xticklabels() + ax2.get_yticklabels():
                lab.set_fontsize(plotfonts["axisTicks"])

        # add the legend
        for idx, pol in enumerate(polarisations):
            ax1 = plt.subplot(nrows, ncols, idx + 1 - plotNumMult * idx)
            leg = plt.legend(loc="lower left", fontsize=plotfonts["legend"])
            leg.get_frame().set_linewidth(0.0)
            leg.get_frame().set_facecolor("w")
            plt.grid(True, ls="--")
            ax2 = plt.subplot(nrows, ncols, ncols + idx + 1 - plotNumMult * idx)
            leg = plt.legend(loc="lower left", fontsize=plotfonts["legend"])
            leg.get_frame().set_linewidth(0.0)
            leg.get_frame().set_facecolor("w")
            plt.grid(True, ls="--")

        # show if the figure is not in keywords
        if "fig" not in kwargs:
            # layout options
            plt.tight_layout()
            fig.subplots_adjust(top=0.92)
            plt.show()

        return fig

    def viewTipper(self, **kwargs) -> Figure:
        """Plots tipper data where available

        For length data, both axes are log scale (period and length). For angle data, period is in log scale and phase is linear scale.
        Units, x axis is seconds, length is dimensionless and angle is degrees.

        Parameters
        ----------
        fig : matplotlib.pyplot.figure, optional
            A figure object
        cols : bool, optional
            There are three tipper plots: tipper length, tipper real angle, tipper imaginary angle. These can either be plotted in rows or columns. Set to True to plot in with one row and three columns.
        colours : str, optional
            Colour of plot
        mk : str, optional
            Plot marker type
        ls : str, optional
            Line style  
        plotfonts : Dict, optional
            A dictionary of plot fonts
        label : str, optional
            Label for the plots
        xlim : List, optional
            Limits for the x axis
        length_ylim : List, optional
            Limits for the length y axis
        angle_ylim : List, optional
            Limits for the angle y axis            
        """
        if "HzHx" not in self.polarisations or "HzHy" not in self.polarisations:
            return False

        # rows or columns
        cols = kwargs["cols"] if "cols" in kwargs else True
        if cols:
            nrows = 1
            ncols = 3
        else:
            nrows = 3
            ncols = 1
        # plot options
        mk = kwargs["mk"] if "mk" in kwargs else "o"
        ls = kwargs["ls"] if "ls" in kwargs else "none"
        plotfonts = kwargs["plotfonts"] if "plotfonts" in kwargs else getViewFonts()

        # limits
        xlim = kwargs["xlim"] if "xlim" in kwargs else [1e-3, 1e4]
        length_ylim = kwargs["length_ylim"] if "length_ylim" in kwargs else [1e-2, 1e3]
        angle_ylim = kwargs["angle_ylim"] if "angle_ylim" in kwargs else [-30, 30]

        fig = (
            plt.figure(kwargs["fig"].number)
            if "fig" in kwargs
            else plt.figure(figsize=(16, 7))
        )
        st = fig.suptitle("Tipper")
        st.set_y(0.98)

        # plot
        tipperLength, tipperAngleRe, tipperAngleIm = self.getTipper()
        # lengthError, angleError = self.getTipperErrors()
        label = kwargs["label"] if "label" in kwargs else "tipper"
        # plot resistivity
        ax1 = plt.subplot(nrows, ncols, 1)
        plt.title("Tipper Length")
        ax1.errorbar(
            self.period,
            tipperLength,
            yerr=tipperLength / 50,
            ls=ls,
            marker=mk,
            markerfacecolor="white",
            markersize=9,
            color="red",
            ecolor="red",
            mew=1.1,
            elinewidth=1.0,
            capsize=4,
            barsabove=False,
            label=label,
        )
        ax1.set_xscale("log")
        ax1.set_yscale("log")
        # axis options
        plt.xlabel("Period [s]", fontsize=plotfonts["axisLabel"])
        plt.ylabel("Length [dimensionless]", fontsize=plotfonts["axisLabel"])
        plt.xlim(xlim)
        plt.ylim(length_ylim)
        # set tick sizes and legend
        for lab in ax1.get_xticklabels() + ax1.get_yticklabels():
            lab.set_fontsize(plotfonts["axisTicks"])
        leg = plt.legend(loc="upper right", fontsize=plotfonts["legend"])
        leg.get_frame().set_linewidth(0.0)
        leg.get_frame().set_facecolor("w")
        plt.grid(True, ls="--")

        # plot real tipper angles
        ax2 = plt.subplot(nrows, ncols, 2)
        plt.title("Tipper Angle Real")
        ax2.errorbar(
            self.period,
            tipperAngleRe,
            yerr=tipperAngleRe / 50,
            ls="none",
            marker=mk,
            markerfacecolor="white",
            markersize=9,
            color="red",
            ecolor="red",
            mew=1.1,
            elinewidth=1.0,
            capsize=4,
            barsabove=False,
            label=label,
        )
        ax2.set_xscale("log")
        # axis options
        plt.xlabel("Period [s]", fontsize=plotfonts["axisLabel"])
        plt.ylabel("Angle [degrees]", fontsize=plotfonts["axisLabel"])
        plt.xlim(xlim)
        plt.ylim(angle_ylim)
        # set tick sizes
        for lab in ax2.get_xticklabels() + ax2.get_yticklabels():
            lab.set_fontsize(plotfonts["axisTicks"])
        leg = plt.legend(loc="upper right", fontsize=plotfonts["legend"])
        leg.get_frame().set_linewidth(0.0)
        leg.get_frame().set_facecolor("w")
        plt.grid(True, ls="--")

        # plot imaginary tipper angles
        ax3 = plt.subplot(nrows, ncols, 3)
        plt.title("Tipper Angle Imaginary")
        ax3.errorbar(
            self.period,
            tipperAngleIm,
            yerr=tipperAngleIm / 50,
            ls="none",
            marker=mk,
            markerfacecolor="white",
            markersize=9,
            color="red",
            ecolor="red",
            mew=1.1,
            elinewidth=1.0,
            capsize=4,
            barsabove=False,
            label=label,
        )
        ax3.set_xscale("log")
        # axis options
        plt.xlabel("Period [s]", fontsize=plotfonts["axisLabel"])
        plt.ylabel("Angle [degrees]", fontsize=plotfonts["axisLabel"])
        plt.xlim(xlim)
        plt.ylim(angle_ylim)
        # set tick sizes
        for lab in ax3.get_xticklabels() + ax3.get_yticklabels():
            lab.set_fontsize(plotfonts["axisTicks"])
        leg = plt.legend(loc="upper right", fontsize=plotfonts["legend"])
        leg.get_frame().set_linewidth(0.0)
        leg.get_frame().set_facecolor("w")
        plt.grid(True, ls="--")

        # show if the figure is not in keywords
        if "fig" not in kwargs:
            # layout options
            plt.tight_layout()
            fig.subplots_adjust(top=0.90)
            plt.show()

        return fig

    def printList(self) -> List[str]:
        """Class information as a list of strings

        Returns
        -------
        out : List[str]
            List of strings with information
        """
        textLst = []
        textLst.append("Frequencies [Hz] = {}".format(arrayToString(self.freq)))
        textLst.append("Periods [s] = {}".format(arrayToString(self.period)))
        textLst.append("Polarisations = {}".format(self.polarisations))
        return textLst
