import os
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Union

# import from package
from resistics.dataObjects.dataObject import DataObject
from resistics.utilities.utilsPlotter import getViewFonts


class CalibrationData(DataObject):
    """Class for holding calibration data

    Calibration data should be given in the frequency domain and has a magnitude and phase component (in radians). Calibration data is the impulse response for an instrument or sensor and is usually deconvolved (division in frequency domain) from the time data.
    
    Notes
    -----
    Calibration data for magnetic channels is given in mV/nT. Because this is deconvolved from magnetic time data, which is in mV, the resultant magnetic time data is in nT. 

    Attributes
    ----------
    filename : str
        The filename the calibration data was read from
    numSamples : int
        The number of samples in the calibration data
    freqs : np.ndarray
        The frequency points where calibration data is defined
    magnitude : np.ndarray
        The magnitude data
    magnitudeUnit : str
        The magnitude unit, defaulted to mV/nT
    phase : np.ndarray
        The phase data in radians
    phaseUnit : str
        The phase unit, defaulted to radians
    chopper : bool
        Boolean flag to note whether chopper is on or not

    Methods
    -------
    __init__(kwargs)
        Initialise the time data
    view(kwargs)
        View the spectra data 
    printList()
        Class status returned as list of strings          
    """

    def __init__(
        self,
        filename: str,
        freqs: np.ndarray,
        magnitude: np.ndarray,
        phase: np.ndarray,
        staticGain: float = 1,
        chopper: bool = False
    ) -> None:
        """Initialise and set object parameters

        Parameters
        ----------
        freqs : np.ndarray
            Array of frequencies for which the impulse response is defined
        magnitude : np.ndarray
            Magnitude of impulse response
        phase : np.ndarray
            Phase of impulse reponse in radians 
        """

        self.filename: str = filename
        self.freqs: np.ndarray = freqs
        self.magnitude: np.ndarray = magnitude
        self.magnitudeUnit = "mV/nT"
        self.phase: np.ndarray = phase
        self.phaseUnit = "radians"
        self.numSamples: int = len(freqs)
        self.staticGain: float = staticGain
        self.chopper: bool = chopper
        self.serial = 1

    def view(self, **kwargs) -> plt.figure:
        """Plot of the calibration function

        Parameters
        ----------                
        fig : matplotlib.pyplot.figure, optional
            A figure object
        plotFonts : Dict, optional
            A dictionary of plot fonts
        label : str, optional
            Label for the plots
        xlim : List, optional
            Limits for the x axis
        ylim_mag : List, optional
            Limits for the magnitude y axis
        ylim_phase : List, optional
            Limits for the phase y axis
        legened : bool
            Boolean flag for adding a legend
        
        Returns
        -------
        plt.figure
            Matplotlib figure object            
        """

        if "fig" in kwargs:
            fig = plt.figure(kwargs["fig"].number)
        else :
            fig = plt.figure(figsize=(8, 8))
        plotFonts = kwargs["plotFonts"] if "plotFonts" in kwargs else getViewFonts()

        # plot magnitude
        plt.subplot(2, 1, 1)
        plt.title("Impulse response magnitude", fontsize=plotFonts["title"])
        lab = kwargs["label"] if "label" in kwargs else self.filename
        plt.loglog(self.freqs, self.magnitude, label=lab)
        if "xlim" in kwargs:
            plt.xlim(kwargs["xlim"])
        if "ylim_mag" in kwargs:
            plt.ylim(kwargs["ylim_mag"])   
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Magnitude [{}]".format(self.magnitudeUnit))     
        plt.grid(True)  
        # legend
        if "legend" in kwargs and kwargs["legend"]:
            plt.legend(loc=2)  

        # plot phase
        plt.subplot(2, 1, 2)
        plt.title("Impulse response phase", fontsize=plotFonts["title"])
        lab = kwargs["label"] if "label" in kwargs else self.filename
        plt.semilogx(self.freqs, self.phase, label=lab)
        if "xlim" in kwargs:
            plt.xlim(kwargs["xlim"])
        if "ylim_phase" in kwargs:
            plt.ylim(kwargs["ylim_phase"])   
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Phase [{}]".format(self.phaseUnit))   
        plt.grid(True)   
        # legend
        if "legend" in kwargs and kwargs["legend"]:
            plt.legend(loc=3)   

        # show if the figure is not in keywords
        if "fig" not in kwargs:
            plt.tight_layout(rect=[0, 0.02, 1, 0.96])
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
        textLst.append("Filename = {}".format(self.filename))
        textLst.append("Serial = {}".format(self.serial))
        textLst.append("Static gain = {:.2f}".format(self.staticGain))
        textLst.append("Chopper = {}".format(self.chopper))
        textLst.append("Number of frequency points = {:d}".format(self.numSamples))
        textLst.append("Calibration data:")
        for ii in range(0, self.numSamples):
            textLst.append(
                "\t{:.2f}\t{:.2f}\t{:.2f}".format(
                    self.freqs[ii], self.magnitude[ii], self.phase[ii]
                )
            )
        return textLst
