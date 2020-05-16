"""Mocked classes for the purposes of testing
This is somewhat better documented because the testing is a bit more complex
"""
from typing import Tuple, List, Dict
import numpy as np


def add_noise(data: np.ndarray, maxnoise: float = 0.05, seed: int = 0,) -> np.ndarray:
    """Add noise to an array
    
    Parameters
    ----------
    data : np.ndarray
        The array to add noise too
    maxnoise : float, optional
        The maximum noise as a proportion of the value. Default 0.05.
    seed : int
        The seed for the random number generator
    
    Returns
    -------
    np.ndarray
        data with noise added
    """
    # make noise proportional to val
    np.random.seed(seed)
    mult_array = np.random.randint(-100, 100, 5) / 100
    maxnoise_array = data * maxnoise
    return data + mult_array * maxnoise_array


def add_intercept(
    sitedata: List[Dict[str, np.ndarray]], Cx: float, Cy: float, Ct: float
) -> List[Dict[str, np.ndarray]]:
    """Add constant terms to Ex, Ey and Hz, the standard outputs for impedance tensors and tippers

    Parameters
    ----------
    sitedata : List[Dict[str, np.ndarray]]
        An array of dictionaries with the data
    Cx : float
        The constant term for Ex
    Cy : float
        The constant term for Ey
    Ct : float
        The constant term for Hz
    
    Returns
    -------
    List[Dict[str, np.ndarray]]
        Site data with constant terms added
    """
    for win in sitedata:
        win["Ex"] = win["Ex"] + Cx
        win["Ey"] = win["Ey"] + Cy
        win["Hz"] = win["Hz"] + Ct
    return sitedata


def mock_localsite(intercept: bool = False) -> List[Dict[str, np.ndarray]]:
    """Returns mock data for a local site
    
    Parameters
    ----------
    intercept : bool, optional
        Flag for adding an intercept term to some values. Default is False.

    Returns
    -------
    List[Dict[str, np.ndarray]]
        Mock data for a local site for three windows
    """
    Cx = 5
    Cy = -9
    Ct = 4
    localsite = [
        {
            "Ex": np.array([18, 11, 35, 39, 26]),
            "Ey": np.array([23, 11, 38, 37, 32]),
            "Hx": np.array([1, 2, 5, 8, 2]),
            "Hy": np.array([3, 1, 4, 3, 4]),
            "Hz": np.array([26, 12, 42, 40, 36]),
        },
        {
            "Ex": np.array([27, 8, 64, 239, 22]),
            "Ey": np.array([29, 9, 72, 262, 22]),
            "Hx": np.array([4, 1, 8, 33, 4]),
            "Hy": np.array([3, 1, 8, 28, 2]),
            "Hz": np.array([32, 10, 80, 290, 24]),
        },
        {
            "Ex": np.array([143, 478, 131, 56, 19]),
            "Ey": np.array([176, 557, 146, 63, 20]),
            "Hx": np.array([11, 51, 17, 7, 3]),
            "Hy": np.array([22, 65, 16, 7, 2]),
            "Hz": np.array([198, 622, 162, 70, 22]),
        },
    ]
    if intercept:
        return add_intercept(localsite, Cx, Cy, Ct)
    return localsite


def mock_intersite(intercept: bool = False) -> List[Dict[str, np.ndarray]]:
    """Returns mock data for an intersite site
    
    Parameters
    ----------
    intercept : bool, optional
        Flag for adding an intercept term to some values. Default is False.

    Returns
    -------
    List[Dict[str, np.ndarray]]
        Mock data for an intersite site for three windows
    """
    Cx = -3
    Cy = 7
    Ct = 6
    intersite = [
        {
            "Ex": np.array([30, 20, 62, 72, 44]),
            "Ey": np.array([7, 9, 24, 35, 12]),
            "Hx": np.array([1, 2, 5, 8, 2]),
            "Hy": np.array([3, 1, 4, 3, 4]),
            "Hz": np.array([14, 13, 37, 49, 22]),
        },
        {
            "Ex": np.array([48, 14, 112, 422, 40]),
            "Ey": np.array([19, 5, 40, 160, 18,]),
            "Hx": np.array([4, 1, 8, 33, 4]),
            "Hy": np.array([3, 1, 8, 28, 2]),
            "Hz": np.array([29, 8, 64, 249, 26]),
        },
        {
            "Ex": np.array([242, 826, 230, 98, 34]),
            "Ey": np.array([66, 269, 84, 35, 14]),
            "Hx": np.array([11, 51, 17, 7, 3]),
            "Hy": np.array([22, 65, 16, 7, 2]),
            "Hz": np.array([121, 450, 133, 56, 21]),
        },
    ]
    if intercept:
        return add_intercept(intersite, Cx, Cy, Ct)
    return intersite


def mock_remotesite(intercept: bool = False) -> List[Dict[str, np.ndarray]]:
    """Returns mock data for an remote reference site
    
    Parameters
    ----------
    intercept : bool, optional
        Flag for adding an intercept term to some values. Default is False.

    Returns
    -------
    List[Dict[str, np.ndarray]]
        Mock data for an remote reference site for three windows
    """
    Cx = -11
    Cy = 24
    Ct = 10
    remotesite = [
        {
            "Ex": np.array([14, 8, 26, 28, 20]),
            "Ey": np.array([8, 11, 29, 43, 14]),
            "Hx": np.array([1, 2, 5, 8, 2]),
            "Hy": np.array([3, 1, 4, 3, 4]),
            "Hz": np.array([33, 21, 66, 75, 48]),
        },
        {
            "Ex": np.array([20, 6, 48, 178, 16]),
            "Ey": np.array([23, 6, 48, 193, 22]),
            "Hx": np.array([4, 1, 8, 33, 4]),
            "Hy": np.array([3, 1, 8, 28, 2]),
            "Hz": np.array([51, 15, 120, 450, 42]),
        },
        {
            "Ex": np.array([110, 362, 98, 42, 14]),
            "Ey": np.array([77, 320, 101, 42, 17]),
            "Hx": np.array([11, 51, 17, 7, 3]),
            "Hy": np.array([22, 65, 16, 7, 2]),
            "Hz": np.array([264, 891, 246, 105, 36]),
        },
    ]
    if intercept:
        return add_intercept(remotesite, Cx, Cy, Ct)
    return remotesite


def mock_data(
    remote: bool = False,
    localnoise: bool = False,
    internoise: bool = False,
    remotenoise: bool = False,
    intercept: bool = False,
) -> Tuple[List[Dict[str, List[int]]]]:
    """Return some mock data
    
    More details in spreadsheet. However, impedance tensor and tipper vals:
    localsite:
    Zxx = 3, Zxy = 5, Zyx = 2, Zyy = 7, Tx = 2, Tx = 8
    intersite:
    Zxx = 6, Zxy = 8, Zyx = 4, Zyy = 1, Tx = 5, Tx = 3
    remotesite:
    Zxx = 2, Zxy = 4, Zyx = 5, Zyy = 1, Tx = 6, Tx = 9

    All input magnetic fields Hx and Hy are the same to match the assumption that magnetic fields are slowly varying.

    Parameters
    ----------
    remote : bool, optional
        Flag for returning data for remote reference too. Default False.
    localnoise : bool, optional
        Flag for adding some noise to localsite. Default False.
    internoise : bool, optional
        Flag for adding some noise to intersite. Default False.
    remotenoise : bool, optional
        Flag for adding some noise to remotesite. Default False.
    intercept : bool, optional
        Flag for adding constant intercept term to the data. Default is False.
    
    Returns
    -------
    localsite:
        Values for the localsite
    intersite
        Values for the intersite
    remotesite
        If remote=True, returns values for the remotesite too
    """
    localsite = mock_localsite(intercept=intercept)
    intersite = mock_intersite(intercept=intercept)
    remotesite = mock_remotesite(intercept=intercept)

    if localnoise:
        for iw, win in enumerate(localsite):
            for ic, chan in enumerate(win):
                win[chan] = add_noise(win[chan], seed=iw + ic)

    if internoise:
        for iw, win in enumerate(intersite):
            for ic, chan in enumerate(win):
                win[chan] = add_noise(win[chan], seed=iw + ic)

    if remotenoise:
        for iw, win in enumerate(remotesite):
            for ic, chan in enumerate(win):
                win[chan] = add_noise(win[chan], seed=iw + ic)

    if remote:
        return localsite, intersite, remotesite
    return localsite, intersite


class mock_window_selector:
    """Mock window selector"""

    def __init__(
        self,
        localnoise: bool = False,
        internoise: bool = False,
        remotenoise: bool = False,
        intercept: bool = False,
    ):
        """Initialise mock window selector
        
        Parameters
        ----------
        localnoise : bool, optional
            Flag for adding some noise to localsite. Default False.
        internoise : bool, optional
            Flag for adding some noise to intersite. Default False.
        remotenoise : bool, optional
            Flag for adding some noise to remotesite. Default False.
        intercept : bool, optional
            Flag for adding constant terms to equations. Default False.
        """
        self.decParams = mock_decimation_parameters()
        self.winParams = mock_window_parameters()
        self.sites = ["local", "inter", "remote"]
        self.specdir = "test"
        # set noise adding flags
        self.localnoise = localnoise
        self.internoise = internoise
        self.remotenoise = remotenoise
        self.intercept = intercept

    def getDataSize(self, declevel):
        return 5

    def getNumSharedWindows(self, declevel):
        return 3

    def getUnmaskedWindowsLevel(self, declevel):
        return set([1, 2, 3])

    def getWindowsForFreq(self, declevel, eIdx):
        return set([1, 2, 3])

    def getSpecReaderBatches(self, declevel):
        localsite, intersite, remotesite = mock_data(
            remote=True,
            localnoise=self.localnoise,
            internoise=self.internoise,
            remotenoise=self.remotenoise,
            intercept=self.intercept,
        )
        windows = [1, 2, 3]
        local_spec = mock_spectra_io(localsite, windows)
        inter_spec = mock_spectra_io(intersite, windows)
        remote_spec = mock_spectra_io(remotesite, windows)
        return [
            {
                "globalrange": [1, 3],
                "local": local_spec,
                "inter": inter_spec,
                "remote": remote_spec,
            }
        ]


class mock_decimation_parameters:
    """Mock decimation parameters"""

    def __init__(self):
        import numpy as np

        self.numLevels = 1
        self.evalfreq = np.array([24, 40])

    def getEvalFrequenciesForLevel(self, declevel):
        return self.evalfreq


class mock_window_parameters:
    """Mock window parameters"""

    def __init__(self):
        return


class mock_spectra_data:
    """Mock spectra data"""

    def __init__(self, data):
        import numpy as np

        self.data = {}
        for chan in data:
            self.data[chan] = np.array(data[chan])
        self.chans = list(self.data.keys())
        self.windowSize = 8
        self.dataSize = len(self.data[self.chans[0]])
        self.freqArray = np.array([0, 16, 32, 48, 64])
        self.sampleFreq = 128
        self.startTime = "2020-01-01 00:00:00.00"
        self.stopTime = "2020-01-01 01:00:00.00"
    
    def addUnitChannel(self, newchan):
        self.data[newchan] = np.ones(shape=(self.dataSize))
        self.chans.append(newchan)


class mock_spectra_io:
    """Mock spectra io"""

    def __init__(self, data, windows):
        self.data = []
        for d in data:
            self.data.append(mock_spectra_data(d))
        self.windows = windows

    def readBinaryBatchGlobal(self, globalIndices):
        return self.data, self.windows

    def closeFile(self):
        return
