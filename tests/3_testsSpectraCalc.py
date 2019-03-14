import os
import sys
import numpy as np
import math
from datetime import datetime, timedelta
import copy
import matplotlib.pyplot as plt
# import readers
from magpy.ioHandlers.dataReaderSpam import DataReaderSPAM
from magpy.ioHandlers.dataReaderATS import DataReaderATS
from magpy.ioHandlers.dataReaderInternal import DataReaderInternal
# import writers
from magpy.ioHandlers.dataWriterInternal import DataWriterInternal
# import signal processors and calibrator
from magpy.calculators.signalProcessor import SignalProcessor
from magpy.calculators.calibrator import Calibrator
# import spectra related stuff
from magpy.calculators.spectrumCalculator import SpectrumCalculator
from magpy.ioHandlers.spectrumWriter import SpectrumWriter
from magpy.ioHandlers.spectrumReader import SpectrumReader
# import utils
from magpy.utilities.utilsProcess import *
from magpy.utilities.utilsIO import *
from magpy.utilities.utilsPlotter import addLegends

# read in a test dataset
atsPath = os.path.join("testData", "ats")
specPath = os.path.join("testData", "atsSpec")
calPath = os.path.join("testData", "calibration")


# let's read in an original dataset
# and then another dataset which has been signal processed
def standardDataTest():
    breakPrint()
    generalPrint("testsSpectralCalc",
                 "Running test function: standardDataTest")
    # read in data
    reader = DataReaderATS(atsPath)
    timeData = reader.getPhysicalSamples()
    timeData.printInfo()
    # now let's try the calibrator
    cal = Calibrator(calPath)
    cal.printInfo()
    sensors = reader.getSensors(timeData.chans)
    serials = reader.getSerials(timeData.chans)
    choppers = reader.getChoppers(timeData.chans)
    timeData = cal.calibrate(timeData, sensors, serials, choppers)
    timeData.printInfo()
    specCal = SpectrumCalculator(timeData.sampleFreq, timeData.numSamples)
    specData = specCal.calcFourierCoeff(timeData)
    specData.printInfo()
    # now try writing out the spectra
    refTime = datetime.strptime("2016-02-21 00:00:00", "%Y-%m-%d %H:%M:%S")
    specWriter = SpectrumWriter(specPath, refTime)
    specWriter.openBinaryForWriting("spectra", 0, specData.sampleFreq,
                                    timeData.numSamples, 0, 0, 1,
                                    specData.chans)
    specWriter.writeBinary(specData, 0)
    specWriter.writeCommentsFile(specData.comments)
    specWriter.closeFile()
    # now try and read everything in again and get the spec data
    specReader = SpectrumReader(specPath)
    specReader.openBinaryForReading("spectra", 0)
    specReader.printInfo()
    specData = specReader.readBinaryWindowLocal(0)
    specData.printInfo()


def smallDataTest():
    breakPrint()
    generalPrint("testsSpectralCalc", "Running test function: smallDataTest")
    # read in data
    reader = DataReaderATS(atsPath)
    startTime1 = "2016-02-21 03:00:00"
    stopTime1 = "2016-02-21 03:30:00"
    timeData = reader.getPhysicalData(startTime1, stopTime1)
    timeData.printInfo()
    # now let's try the calibrator
    cal = Calibrator(calPath)
    cal.printInfo()
    sensors = reader.getSensors(timeData.chans)
    serials = reader.getSerials(timeData.chans)
    choppers = reader.getChoppers(timeData.chans)
    timeData = cal.calibrate(timeData, sensors, serials, choppers)
    timeData.printInfo()
    specCal = SpectrumCalculator(timeData.sampleFreq, timeData.numSamples)
    specData = specCal.calcFourierCoeff(timeData)
    specData.printInfo()
    fig = plt.figure(figsize=(10, 10))
    specData.view(fig=fig, label="Raw spectral data")
    # now let's try and process something
    sproc = SignalProcessor()
    timeData2 = reader.getPhysicalData(startTime1, stopTime1)
    timeData2 = cal.calibrate(timeData2, sensors, serials, choppers)
    timeData2 = sproc.notchFilter(timeData2, 50.0)
    timeData2 = sproc.notchFilter(timeData2, 16.6667)
    specData2 = specCal.calcFourierCoeff(timeData2)
    specData2.printInfo()
    specData2.view(fig=fig, label="Notch filtered")
    # set plot properties
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    addLegends(fig)
    plt.show()


standardDataTest()
smallDataTest()