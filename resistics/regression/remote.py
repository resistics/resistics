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

    Remote reference processing uses by default channels from the remote reference site as cross spectra channels. By default, no channels from the input site or output site are used to calculate cross spectra.

    Note that methods shown here are only for the RemoteRegressor child class. See the parent LocalRegressor for more information.

    Attributes
    ----------
    winSelector : WindowSelector
        A WindowSelector instance which will be used to help read in the correct data
    decParams : DecimationParameters
        A DecimationParameters instance defining the decimation scheme
    winParams : WindowParameters
        A WindowParameters instance with the windowing parameters
    outpath : str
        The path to write the transfer function data out to
    ncores : int
        The number of cores to use
    method : str
        The solution method
    inSite : str
        The input site
    inChannels : List[str]
        The input channels
    inSize : int 
        The number of input channels
    inCross : List[str] 
        The channels to use from the input site to calculate crosspowers
    outSite: str = "dummy"
        The output site
    outChannels : List[str] 
        The output channels
    outSize : int
        The number of output channels
    outCross : List[str]
        The channels to use from the output site to calculate crosspowers 
    remoteSite : str
        The remote reference site
    remoteCross : List[str]
        The channels to use from the remote reference site to calculate crosspowers 
    remoteSize : int
        The number of channels from the remote reference site which will be used to calculate crosspowers
    intercept : bool
        Flag for adding an intercept term
    interceptChannel : str
        The name of the pseudo channel that will act as intercept
    mmOptions : Union[Dict, None]
        Options from MM estimates
    cmOptions : Union[Dict, None]
        Options for Chaterjee Machler solutions
    olsOptions : Union[Dict, None]
        Options for ordinary least squares
    smoothFunc : str
        The window function to use
    smoothLen : Union[int, None]
        The smooth length to use. If None, will calculate automatically.
    evalFreq : List
        The calculated evaluation frequencies
    parameters : List
        The parameters calculated for each evaluation frequency
    variances : List
        The variances calculated for each evaluation frequency
    postpend : str
        The postpend string to add to the transfer function output file

    Methods
    -------
    __init__(proj, winSelector, outpath)
        Initialise with a window selector and the path to write the result out to
    defaultInCross()
        The default cross channels to use from the input site
    defaultOutCross()
        The default cross channels to use from the output site
    defaultRemoteCross()
        The default cross channels to use from the remote site
    setRemote(remoteSite, remoteCross)
        Set the remote site and the channels from the remote site for which to calculate cross spectra
    processBatch(batch, batchedWindows, evalFreq, smoothLen)
        Process a single batch of spectra data
    getCrossSize()
        Get the number of channels to calculate cross spectra for   
    prepareLinearEqn(crosspowerData, eIdx)
        Prepare regressors and observations for an evaluation frequency from crosspowers data
    printList()
        Class status returned as list of strings
    """

    def __init__(self, winSelector, outpath: str) -> None:
        """Initialise the RemoteRegressor

        Parameters
        ----------
        winSelector : WindowSelector
            A window selector instance
        outpath : str
            The path to write the transfer function data to
        """
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
        self.remoteSize: int = len(self.remoteCross)
        # solution options
        self.intercept: bool = False
        self.interceptChannel: str = "intercept"
        self.stack: bool = False
        self.mmOptions: Union[None, Dict] = None
        self.cmOptions: Union[None, Dict] = None
        self.olsOptions: Union[None, Dict] = None
        # smoothing options
        self.smoothFunc: str = "hann"
        self.smoothLen: int = None
        # attributes to store transfer function data
        self.evalFreq: List[float] = []
        self.parameters: List[np.ndarray] = []
        self.variances: List[np.ndarray] = []
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

        # set up a unit channel to act as the intercept term in the input
        if self.intercept:
            for data in inData:
                data.addUnitChannel(self.interceptChannel)

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
                    obs[iOut, iOffset + iC] = winPower[outChan, xC, eIdx]
                    for iIn, inChan in enumerate(inChans):
                        reg[iOut, iOffset + iC, iIn] = winPower[inChan, xC, eIdx]
                # the cross channels from the output data
                for iC, xC in enumerate(outCross):
                    iC += numInCross
                    obs[iOut, iOffset + iC] = winPower[outChan, xC, eIdx]
                    for iIn, inChan in enumerate(inChans):
                        reg[iOut, iOffset + iC, iIn] = winPower[inChan, xC, eIdx]
                # the cross channels from the remote data
                for iC, xC in enumerate(remoteCross):
                    iC += numInCross + numOutCross
                    obs[iOut, iOffset + iC] = winPower[outChan, xC, eIdx]
                    for iIn, inChan in enumerate(inChans):
                        reg[iOut, iOffset + iC, iIn] = winPower[inChan, xC, eIdx]
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
