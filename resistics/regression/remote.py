import os
import numpy as np
from typing import List, Tuple, Dict, Union

from resistics.window.selector import WindowSelector
from resistics.spectra.data import PowerData
from resistics.regression.local import LocalRegressor
from resistics.regression.robust import (
    olsModel,
    chatterjeeMachler,
    mmestimateModel,
)


class RemoteRegressor(LocalRegressor):
    """Perform remote reference processing of magnetotelluric data

    Remote reference processing uses a different equation for spectral power than the single site processor. The equation that the processor uses is:

    Attributes
    ----------
    winSelector : WindowSelector
        A window selector object which defines which windows to use in the linear model
    decParams : DecimationParameters
        DecimationParameters object with information about the decimation scheme
    winParams : WindowParameters
        WindowParameters object with information about the windowing
    outpath : str 
        Location to put the calculated transfer functions (Edi files)
    inSite : str 
        The site to use for the input channels 
    inChannels: List[str] (["Hx", "Hy"])
        List of hannels to use as input channels for the linear system
    inSize : int 
        Number of input channels
    outSite : str 
        The site to use for the output channels
    outChannels : List[str] (["Ex", "Ey"])
        List of channels to use as output channels for the linear system
    outSize : int
        Number of output channels
    remoteSite : str
        The site to use as a remote reference
    remoteCross : List[str]
        The channels to use as remote reference channels
    remoteSize : int
        Number of remote reference channels
    intercept : bool (default False)
        Flag for including an intercept (static) term in the linear system
    method : str (options, "ols", "cm") 
        String for describing what solution method to use
    win : str (default hann)
        Window function to use in robust solution
    winSmooth : int (default -1)
        The size of the window smoother. If -1, this will be autocalculated based on data size
    postpend : str (default "")
        String to postpend to the output filename to help file management
    evalFreq : List[float] or np.ndarray
        The evaluation frequencies
    impedances : List

    variances : List

    Methods
    -------
    __init__(winSelector, outpath)
        Initialise the remote reference processor
    setRemote(remoteSite, remoteCross)
        Set the remote site and channels
    process()
        Perform the remote reference processing
    prepareLinearEqn(data)
        Prepare regressors and observations for regression from cross-power data
    robustProcess(numWindows, obs, reg)      
        Robust regression processing   
    olsProcess(numWindows, obs, reg)      
        Ordinary least squares processing
    stackedProcess(data)
        Stacked processing                  
    printList()
        Class status returned as list of strings
    """

    def __init__(self, winSelector, outpath: str):
        self.winSelector: WindowSelector = winSelector
        self.decParams = winSelector.decParams
        self.winParams = winSelector.winParams
        self.outpath: str = outpath
        # configuration parameters
        self.ncores = 0
        self.method: str = "cm"
        # input site options
        self.inSite: str = "dummy"
        self.inChannels: List[str] = ["Hx", "Hy"]
        self.inSize: int = len(self.inChannels)
        self.inCross: List[str] = self.defaultInCross(self.inChannels)
        # output site options
        self.outSite: str = "dummy"
        self.outChannels: List[str] = ["Ex", "Ey"]
        self.outSize: int = len(self.outChannels)
        self.outCross: List[str] = self.defaultOutCross(self.outChannels)
        # remote site options
        self.remoteSite: str = ""
        self.remoteCross: List[str] = ["Hx", "Hy"]
        self.remoteSize = len(self.remoteCross)
        # solution options
        self.intercept: bool = False
        self.stack: bool = False
        self.mmOptions: Union[None, Dict] = None
        self.cmOptions: Union[None, Dict] = None
        self.olsOptions: Union[None, Dict] = None
        # smoothing options
        self.smoothFunc: str = "hann"
        self.smoothLen: int = None
        # attributes to store transfer function data
        self.evalFreq = []
        self.parameters = []
        self.variances = []
        # output filename
        self.postpend: str = ""

    def defaultInCross(self, inChannels: List[str]) -> List[str]:
        """Returning the default cross channels for the input site

        For remote reference processing, the default is to only use cross channels from the remote site.
        
        Parameters
        ----------
        inChannels : List[str]
            The input channels
        
        Returns
        -------
        inCross : List[str]
            The cross channels for the input site
        """
        return []

    def defaultOutCross(self, outChannels: List[str]) -> List[str]:
        """Returning the default cross channels for the output site

        For remote reference processing, the default is to only use cross channels from the remote site.
        
        Parameters
        ----------
        outChannels : List[str]
            The output channels
        
        Returns
        -------
        outCross : List[str]
            The cross channels for the output site
        """
        return []

    def defaultRemoteCross(self) -> List[str]:
        """Returning the default cross channels for the remote site

        For remote reference processing, the default is to only use cross channels from the remote site. These are usually the magnetic channels for standard impedance tensor calculations.
        
        Returns
        -------
        remoteCross : List[str]
            The cross channels for the remote site
        """
        return ["Hx", "Hy"]

    def setRemote(self, remoteSite: str, remoteCross: Union[List[str], None],) -> None:
        """Set information about remote site channels

        For remote reference processing, it is usual to use the magnetic channels as the cross channels.
    
        Parameters
        ----------
        remoteSite : str
            Site to use for input channel data
        remoteCross : List[str]
            Channels to use from the remote refence site as crosspower channels
        """
        self.remoteSite = remoteSite
        self.remoteCross = remoteCross
        if self.remoteCross is None:
            self.remoteCross = self.defaultRemoteCross()
        self.remoteSize = len(remoteCross)
        self.printText(
            "Remote reference set with site {} and cross channels {}".format(
                self.remoteSite, self.remoteCross
            )
        )

    def processBatch(
        self, batch: Dict, batchedWindows: set, evalFreq: np.ndarray, smoothLen: int
    ) -> Tuple[List[PowerData], np.ndarray]:
        """Process a single batch
        
        Parameters
        ----------
        batch : Dict
            The batch information (about the spectra readers, the global window range etc)
        batchedWindws : set
            The windows in the batch. This is a set, so unsorted. A sorted version with correct order gets returned by the function
        evalFreq : np.ndarray
            The evaluation frequencies
        smoothLen : int
            The smoothing length for the crosspower data

        Returns
        -------
        crosspowers : List[PowerData]
            List of PowerData with the crosspowers
        batchedWindows : np.ndarray
            The correctly ordered windows for the batch
        """
        from resistics.regression.crosspowers import remoteCrosspowers

        # collect spectrum data and set batchedWindows to ensure proper order
        inReader = batch[self.inSite]
        inData, batchedWindows = inReader.readBinaryBatchGlobal(
            globalIndices=batchedWindows
        )
        if self.outSite != self.inSite:
            outReader = batch[self.outSite]
            outData, _gIndicesOut = outReader.readBinaryBatchGlobal(
                globalIndices=batchedWindows
            )
        else:
            outData = inData
        # remote data
        remoteReader = batch[self.remoteSite]
        remoteData, _rgIndicesOut = remoteReader.readBinaryBatchGlobal(
            globalIndices=batchedWindows
        )

        self.printText("Calculating batch crosspowers...")
        crosspowers = remoteCrosspowers(
            self.ncores,
            inData,
            self.inChannels,
            self.inCross,
            outData,
            self.outChannels,
            self.outCross,
            remoteData,
            self.remoteCross,
            smoothLen,
            self.smoothFunc,
            evalFreq,
        )
        self.printText("Calculation of crosspowers for batch complete")
        return crosspowers, batchedWindows

    def getCrossSize(self):
        """This essentially returns the number of equations per window

        The total number of cross channels that have been used in calculating crosspowers

        Returns
        -------
        crossSize : int
            The total number of cross channels used for crosspowers
        """
        return len(self.inCross) + len(self.outCross) + len(self.remoteCross)

    def prepareLinearEqn(self, crosspowerData: np.ndarray, eIdx: int):
        r"""Prepare data as a linear equation for the robust regression

        This prepares the data for the following type of solution,

        .. math::
            y = Ax,

        where :math:`y` is the observations, :math:`A` is the regressors and :math:`x` is the unknown. 

        The number of observations is number of windows * number of cross-power channels
        The shapes of the arrays are as follows:
        
            - y is (number of output channels, number of observations)
            - A is (number of output channels, number of observations, number of input channels)
            - x is (number of output channels, number of input channels)

        Consider the impedance tensor,

        .. math::
            :nowrap:

            \begin{eqnarray}
            E_x & = & Z_{xx} H_x + Z_{xy} H_y \\
            E_y & = & Z_{yx} H_x + Z_{yy} H_y 
            \end{eqnarray}  

        Here, there are two input channels, :math:`H_x`, :math:`H_y` and two output channels :math:`E_x` and :math:`E_y`. In total, there are four components of the unknown impedance tensor, :math:`Z_{xx}`, :math:`Z_{xy}`, :math:`Z_{yx}`, :math:`Z_{yy}` (number of input channels * number of output channels). The number of observations is the number of windows multiplied by the number of channels used for cross-power spectra.     

        Parameters
        ----------
        crosspowerData : np.ndarray
            Cross-power spectral data at evaluation frequencies
        eIdex : int
            The evaluation frequency index (for the decimation level)

        Returns
        -------
        numWindows : int
            The number of windows included in the regression (after bad value removal)
        obs : np.ndarray
            Observations array
        reg : np.ndarray 
            Regressors array
        """
        numWindows = crosspowerData.size
        # remove windows with bad values
        numWindows, crosspowerData = self.checkForBadValues(crosspowerData)
        # calculate out the number of equations per window per output variable
        numInCross = len(self.inCross)
        numOutCross = len(self.outCross)
        crossSize = self.getCrossSize()
        # add In and Out to the required channels
        inChans = [inChan + "In" for inChan in self.inChannels]
        inCross = [inChan + "In" for inChan in self.inCross]
        outChans = [outChan + "Out" for outChan in self.outChannels]
        outCross = [outChan + "Out" for outChan in self.outCross]
        remoteCross = [remoteChan + "RR" for remoteChan in self.remoteCross]
        # construct our arrays
        obs = np.empty(shape=(self.outSize, crossSize * numWindows), dtype="complex")
        reg = np.empty(
            shape=(self.outSize, crossSize * numWindows, self.inSize), dtype="complex"
        )
        for iWin, winPower in enumerate(crosspowerData):
            iOffset = iWin * crossSize
            for iOut, outChan in enumerate(outChans):
                # add in all the observations and regressors for this window
                # the cross channels from the input data
                for iC, xC in enumerate(inCross):
                    obs[iOut, iOffset + iC] = winPower.getPower(outChan, xC, eIdx)
                    for iIn, inChan in enumerate(inChans):
                        reg[iOut, iOffset + iC, iIn] = winPower.getPower(
                            inChan, xC, eIdx
                        )
                # the cross channels from the output data
                for iC, xC in enumerate(outCross):
                    iC += numInCross
                    obs[iOut, iOffset + iC] = winPower.getPower(outChan, xC, eIdx)
                    for iIn, inChan in enumerate(inChans):
                        reg[iOut, iOffset + iC, iIn] = winPower.getPower(
                            inChan, xC, eIdx
                        )
                # the cross channels from the remote data
                for iC, xC in enumerate(remoteCross):
                    iC += numInCross + numOutCross
                    obs[iOut, iOffset + iC] = winPower.getPower(outChan, xC, eIdx)
                    for iIn, inChan in enumerate(inChans):
                        reg[iOut, iOffset + iC, iIn] = winPower.getPower(
                            inChan, xC, eIdx
                        )
        return numWindows, obs, reg

    def printList(self) -> List[str]:
        """Class information as a list of strings

        Returns
        -------
        out : list
            List of strings with information
        """
        textLst: List[str] = []
        textLst.append("Input Site = {}".format(self.inSite))
        textLst.append("Input Site Channels = {}".format(self.inChannels))
        textLst.append("Input Site Cross Channels = {}".format(self.inCross))
        textLst.append("Output Site = {}".format(self.outSite))
        textLst.append("Output Site Channels = {}".format(self.outChannels))
        textLst.append("Output Site Cross Channels = {}".format(self.outCross))
        textLst.append("Remote Site = {}".format(self.remoteSite))
        textLst.append("Remote Site Cross Channels = {}".format(self.remoteCross))
        textLst.append("Sample frequency = {:.3f}".format(self.decParams.sampleFreq))
        return textLst
