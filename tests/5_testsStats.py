import os
import sys
import numpy as np
import math
from datetime import datetime, timedelta
import copy
import matplotlib.pyplot as plt
# import spectra related stuff
from magpy.calculators.spectrumCalculator import SpectrumCalculator
from magpy.ioHandlers.spectrumWriter import SpectrumWriter
from magpy.ioHandlers.spectrumReader import SpectrumReader
# imports statistics stuff
from magpy.ioHandlers.statisticIO import StatisticIO
from magpy.dataObjects.statisticData import StatisticData
# import utils
from magpy.utilities.utilsProcess import *
from magpy.utilities.utilsIO import *

statPath = os.path.join("testData", "statfiles", "spectra")


def readStat():
    generalPrint("testsWindowDecimate",
                 "Running test function: readStat")
    statIO = StatisticIO(datapath=statPath)
    # read coherence
    cohStatData = statIO.read("cohStat", 0)
    if cohStatData:  #statIO will return False if nothing found, so worth checking
        cohStatData.printInfo()
        cohStatData.view(0)
        cohStatData.histogram(0)
    # now let's read the tfStat
    tfStatData = statIO.read("tfStat", 0)
    if tfStatData:
        tfStatData.printInfo()
        # look at a histogram
        tfStatData.histogram(0)
        # the min and max are too far away from the data, let's add some xlimits
        tfStatData.histogram(0, xlim=[-200, 200])
        # next let's look at the data
        tfStatData.view(0, ylim=[-3000, 3000])
        # but no, we might want to overlay a different colour on the plot
        # let's try and overlay the coherence data
        # but first, we need to set up a map between the winStats in each statistic
        # this defines which stats get overlaid
        colourmap = {
            "ExHxReal": "cohExHx",
            "ExHxImag": "cohExHx",
            "EyHyReal": "cohEyHy",
            "EyHyImag": "cohEyHy",
            "ExHyReal": "cohExHy",
            "ExHyImag": "cohExHy",
            "EyHxReal": "cohEyHx",
            "EyHxImag": "cohEyHx"
        }
        # and let's also define the colour limits
        # coherence is between 0 and 1, so these are our limits
        clim = [0, 1]
        tfStatData.view(
            0,
            ylim=[-200, 200],
            colorstat=cohStatData,
            colormap=colourmap,
            clim=clim,
            colortitle="Coherence")
        # finally, let's crossplot
        tfStatData.crossplot(0, xlim=[-200, 200], ylim=[-200, 200])
        # and again, add coherence colours
        tfStatData.crossplot(
            0,
            xlim=[-200, 200],
            ylim=[-200, 200],
            colorstat=cohStatData,
            colormap=colourmap,
            clim=clim,
            colortitle="Coherence")


readStat()