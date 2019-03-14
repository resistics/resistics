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
# import window and decimation classes
from magpy.calculators.decimationParameters import DecimationParams
from magpy.calculators.decimator import Decimator
from magpy.calculators.windowParameters import WindowParams
from magpy.calculators.windower import Windower
# import utils
from magpy.utilities.utilsProcess import *
from magpy.utilities.utilsIO import *
from magpy.utilities.utilsPlotter import addLegends

# read in a test dataset
atsPath = os.path.join("testData", "ats")
specPath = os.path.join("testData", "atsSpec_noNotch")
specPathNotch = os.path.join("testData", "atsSpec_notch")
calPath = os.path.join("testData", "calibration")
refTime = "2016-02-21 00:00:00"
datetimeRef = datetime.strptime(refTime, "%Y-%m-%d %H:%M:%S")


def testWindowDecimate():
    breakPrint()
    generalPrint("testsWindowDecimate",
                 "Running test function: testWindowDecimate")
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
    # calculate decimation parameters
    decParams = DecimationParams(timeData.sampleFreq)
    decParams.setDecimationParams(7, 6)
    decParams.printInfo()
    numLevels = decParams.numLevels
    timeData.addComment(
        "Decimating with {} levels and {} frequencies per level".format(
            numLevels, decParams.freqPerLevel))
    # now do window parameters
    winParams = WindowParams(decParams)
    winParams.printInfo()
    # create the decimator
    dec = Decimator(timeData, decParams)
    dec.printInfo()

    for iDec in range(0, numLevels):
        # get the data for the current level
        check = dec.incrementLevel()
        if not check:
            break  # not enough data
        timeData = dec.timeData
        # create the windower and give it window parameters for current level
        fsDec = dec.sampleFreq
        win = Windower(datetimeRef, timeData, winParams.getWindowSize(iDec),
                       winParams.getOverlap(iDec))
        numWindows = win.numWindows
        if numWindows < 2:
            break  # do no more decimation

        # add some comments
        timeData.addComment(
            "Evaluation frequencies for level {} are {}".format(
                iDec,
                listToString(decParams.getEvalFrequenciesForLevel(iDec))))
        timeData.addComment(
            "Windowing with window size {} samples, overlap {} samples and {} windows"
            .format(
                winParams.getWindowSize(iDec), winParams.getOverlap(iDec),
                numWindows))

        # create the spectrum calculator and statistics calculators
        specCalc = SpectrumCalculator(fsDec, winParams.getWindowSize(iDec))
        # get ready a file to save the spectra
        specWrite = SpectrumWriter(specPath, datetimeRef)
        specWrite.openBinaryForWriting("spectra", iDec, fsDec,
                                       winParams.getWindowSize(iDec),
                                       winParams.getOverlap(iDec),
                                       win.winOffset, numWindows,
                                       timeData.chans)

        # loop though windows, calculate spectra and save
        for iW in range(0, numWindows):
            # get the window data
            winData = win.getData(iW)
            # calculate spectra
            specData = specCalc.calcFourierCoeff(winData)
            # write out spectra
            specWrite.writeBinary(specData, iW)

        # close spectra file
        specWrite.writeCommentsFile(timeData.comments)
        specWrite.closeFile()


def testWindowDecimateNotch():
    breakPrint()
    generalPrint("testsWindowDecimate",
                 "Running test function: testWindowDecimateNotch")    
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
    # let's apply some filtering and notching
    sproc = SignalProcessor()
    timeData = sproc.notchFilter(timeData, 50.0)
    timeData = sproc.notchFilter(timeData, 16.6667)
    timeData = sproc.lowPass(timeData, 50)
    # calculate decimation parameters
    decParams = DecimationParams(timeData.sampleFreq)
    decParams.setDecimationParams(7, 6)
    decParams.printInfo()
    numLevels = decParams.numLevels
    timeData.addComment(
        "Decimating with {} levels and {} frequencies per level".format(
            numLevels, decParams.freqPerLevel))
    # now do window parameters
    winParams = WindowParams(decParams)
    winParams.printInfo()
    # create the decimator
    dec = Decimator(timeData, decParams)
    dec.printInfo()

    for iDec in range(0, numLevels):
        # get the data for the current level
        check = dec.incrementLevel()
        if not check:
            break  # not enough data
        timeData = dec.timeData
        # create the windower and give it window parameters for current level
        fsDec = dec.sampleFreq
        win = Windower(datetimeRef, timeData, winParams.getWindowSize(iDec),
                       winParams.getOverlap(iDec))
        numWindows = win.numWindows
        if numWindows < 2:
            break  # do no more decimation

        # add some comments
        timeData.addComment(
            "Evaluation frequencies for level {} are {}".format(
                iDec,
                listToString(decParams.getEvalFrequenciesForLevel(iDec))))
        timeData.addComment(
            "Windowing with window size {} samples, overlap {} samples and {} windows"
            .format(
                winParams.getWindowSize(iDec), winParams.getOverlap(iDec),
                numWindows))

        # create the spectrum calculator and statistics calculators
        specCalc = SpectrumCalculator(fsDec, winParams.getWindowSize(iDec))
        # get ready a file to save the spectra
        specWrite = SpectrumWriter(specPathNotch, datetimeRef)
        specWrite.openBinaryForWriting("spectra", iDec, fsDec,
                                       winParams.getWindowSize(iDec),
                                       winParams.getOverlap(iDec),
                                       win.winOffset, numWindows,
                                       timeData.chans)

        # loop though windows, calculate spectra and save
        for iW in range(0, numWindows):
            # get the window data
            winData = win.getData(iW)
            # calculate spectra
            specData = specCalc.calcFourierCoeff(winData)
            # write out spectra
            specWrite.writeBinary(specData, iW)

        # close spectra file
        specWrite.writeCommentsFile(timeData.comments)
        specWrite.closeFile()


def testReadSpectra():
    breakPrint()
    generalPrint("testsWindowDecimate",
                 "Running test function: testReadSpectra")    
    specReader = SpectrumReader(specPath)
    specReader.openBinaryForReading("spectra", 0)
    numWindows = specReader.getNumWindows()
    specReader.printInfo()
    inc = int(numWindows / 10)
    win = 0
    fig = plt.figure(figsize=(14, 10))
    while win < numWindows:
        specData = specReader.readBinaryWindowLocal(win)
        specData.printInfo()
        specData.view(fig=fig)
        win += inc
    # plot
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    plt.show()


def testCompareSpectra():
    breakPrint()
    generalPrint("testsWindowDecimate",
                 "Running test function: testCompareSpectra")      
    # read in no notch
    specReader = SpectrumReader(specPath)
    specReader.openBinaryForReading("spectra", 0)
    numWindows = specReader.getNumWindows()
    specReader.printInfo()
    win = int(numWindows / 2)
    fig = plt.figure(figsize=(14, 10))
    specData = specReader.readBinaryWindowLocal(win)
    specData.printInfo()
    specData.view(fig=fig, label="No notch")
    # now the notched version
    specReader = SpectrumReader(specPathNotch)
    specReader.openBinaryForReading("spectra", 0)
    specDataNotch = specReader.readBinaryWindowLocal(win)
    specDataNotch.printInfo()
    specDataNotch.view(fig=fig, label="Notched")
    # plot
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    addLegends(fig)
    plt.show()


testWindowDecimate()
testWindowDecimateNotch()
testReadSpectra()
testCompareSpectra()