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
from magpy.ioHandlers.dataReaderPhoenix import DataReaderPhoenix
# import writers
from magpy.ioHandlers.dataWriterInternal import DataWriterInternal
# import signal processors
from magpy.calculators.signalProcessor import SignalProcessor
# import utils
from magpy.utilities.utilsProcess import *
from magpy.utilities.utilsIO import *
from magpy.utilities.utilsPlotter import addLegends

# read in a test dataset
atsPath = os.path.join("testData", "ats")
spamPath = os.path.join("testData", "spam")


def plotTest():
    breakPrint()
    generalPrint("testsSignalProcess", "Running test function: plotTest")
    # read in data
    atsReader = DataReaderATS(atsPath)
    # now get some data
    startTime1 = "2016-02-21 03:00:00"
    stopTime1 = "2016-02-21 03:05:00"
    timeData1 = atsReader.getPhysicalData(startTime1, stopTime1)
    timeData1.printInfo()
    # get a second set of data
    startTime2 = "2016-02-21 03:03:00"
    stopTime2 = "2016-02-21 03:05:00"
    timeData2 = atsReader.getPhysicalData(startTime2, stopTime2)
    timeData2.printInfo()
    # now plot
    fig = plt.figure(figsize=(20, 10))
    timeData1.view(
        sampleStop=timeData1.numSamples - 1,
        fig=fig,
        xlim=[timeData1.startTime, timeData2.stopTime])
    timeData2.view(
        sampleStop=timeData2.numSamples - 1,
        fig=fig,
        xlim=[timeData1.startTime, timeData2.stopTime])
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    addLegends(fig)
    plt.show()


def lpFilterTest():
    breakPrint()
    generalPrint("testsSignalProcess", "Running test function: lpFilterTest")
    # read in data
    atsReader = DataReaderATS(atsPath)
    # now get some data
    startTime = "2016-02-21 03:00:00"
    stopTime = "2016-02-21 04:00:00"
    timeData = atsReader.getPhysicalData(startTime, stopTime)
    timeDataSave = copy.deepcopy(timeData)
    timeData.printInfo()
    # now do some signal processing
    sproc = SignalProcessor()
    timeData = sproc.lowPass(timeData, 4)
    timeData.printInfo()
    # now let's plot and see what happens
    fig = plt.figure(figsize=(20, 10))
    timeDataSave.view(fig=fig, label="Raw data")
    timeData.view(fig=fig, label="Low pass filtered")
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    addLegends(fig)
    plt.show()


def hpFilterTest():
    breakPrint()
    generalPrint("testsSignalProcess", "Running test function: hpFilterTest")
    # read in data
    atsReader = DataReaderATS(atsPath)
    # now get some data
    startTime = "2016-02-21 03:00:00"
    stopTime = "2016-02-21 04:00:00"
    timeData = atsReader.getPhysicalData(startTime, stopTime)
    timeDataSave = copy.deepcopy(timeData)
    timeData.printInfo()
    # now do some signal processing
    sproc = SignalProcessor()
    timeData = sproc.highPass(timeData, 24)
    timeData.printInfo()
    # now let's plot and see what happens
    fig = plt.figure(figsize=(20, 10))
    timeDataSave.view(fig=fig, label="Raw data")
    timeData.view(fig=fig, label="High pass filtered")
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    addLegends(fig)
    plt.show()


def bpFilterTest():
    breakPrint()
    generalPrint("testsSignalProcess", "Running test function: bpFilterTest")
    # read in data
    atsReader = DataReaderATS(atsPath)
    # now get some data
    startTime = "2016-02-21 03:00:00"
    stopTime = "2016-02-21 04:00:00"
    timeData = atsReader.getPhysicalData(startTime, stopTime, remnans=True)
    timeDataSave = copy.deepcopy(timeData)
    timeData.printInfo()
    # now do some signal processing
    sproc = SignalProcessor()
    timeData = sproc.bandPass(timeData, 8, 24)
    timeData.printInfo()
    # now let's plot and see what happens
    fig = plt.figure(figsize=(20, 10))
    timeDataSave.view(fig=fig, label="Raw data")
    timeData.view(fig=fig, label="Band pass filtered")
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    addLegends(fig)
    plt.show()


def notchFilterTest():
    breakPrint()
    generalPrint("testsSignalProcess",
                 "Running test function: notchFilterTest")
    # read in data
    atsReader = DataReaderATS(atsPath)
    # now get some data
    startTime = "2016-02-21 03:00:00"
    stopTime = "2016-02-21 04:00:00"
    timeData = atsReader.getPhysicalData(startTime, stopTime, remnans=True)
    timeDataSave = copy.deepcopy(timeData)
    timeData.printInfo()
    # now do some signal processing
    sproc = SignalProcessor()
    timeData = sproc.notchFilter(timeData, 50)
    timeData = sproc.notchFilter(timeData, 16.66667)
    timeData.printInfo()
    # now let's plot and see what happens
    fig = plt.figure(figsize=(20, 10))
    timeDataSave.view(fig=fig, label="Raw data")
    timeData.view(fig=fig, label="Notch filtered")
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    addLegends(fig)
    plt.show()


def testInterpolate():
    breakPrint()
    generalPrint("testsSignalProcess",
                 "Running test function: testInterpolate")
    # read in data
    spamReader = DataReaderSPAM(spamPath)
    # now get some data
    startTime = "2016-02-07 02:00:00"
    stopTime = "2016-02-07 03:00:00"
    # let's get some unscaled data
    timeData = spamReader.getUnscaledData(startTime, stopTime)
    timeData.printInfo()
    timeDataSave = copy.deepcopy(timeData)
    timeData.printInfo()
    # now do some signal processing
    sproc = SignalProcessor()
    timeData = sproc.interpolateToSecond(timeData)
    timeData.printInfo()
    # now let's plot and see what happens
    fig = plt.figure(figsize=(20, 10))
    x = timeData.getDateArray()
    xlim = [timeDataSave.startTime, x[100]]
    timeDataSave.view(sampleStop=100, fig=fig, xlim=xlim, label="Raw data")
    timeData.view(
        sampleStop=100, fig=fig, xlim=xlim, label="Interpolated to second")
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    addLegends(fig)
    plt.show()


def testFillGap():
    breakPrint()
    generalPrint("testsSignalProcess", "Running test function: testFillGap")
    # read in data
    atsReader = DataReaderATS(atsPath)
    # now get some data
    startTime1 = "2016-02-21 03:00:00"
    stopTime1 = "2016-02-21 03:08:00"
    timeData1 = atsReader.getPhysicalData(startTime1, stopTime1)
    # more data
    startTime2 = "2016-02-21 03:10:00"
    stopTime2 = "2016-02-21 03:15:00"
    timeData2 = atsReader.getPhysicalData(startTime2, stopTime2)
    # now do some signal processing
    sproc = SignalProcessor()
    timeData = sproc.fillGap(timeData1, timeData2)
    timeData.printInfo()
    # now let's plot and see what happens
    fig = plt.figure(figsize=(20, 10))
    x = timeData.getDateArray()
    xlim = [timeData1.startTime, timeData2.stopTime]
    timeData.view(
        sampleStop=timeData.numSamples - 1,
        fig=fig,
        xlim=xlim,
        label="Gap filled")
    timeData1.view(
        sampleStop=timeData1.numSamples - 1,
        fig=fig,
        xlim=xlim,
        label="Section1")
    timeData2.view(
        sampleStop=timeData2.numSamples - 1,
        fig=fig,
        xlim=xlim,
        label="Section2")
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    addLegends(fig)
    plt.show()

    # now test with spam data
    spamReader = DataReaderSPAM(spamPath)
    # now get some data
    startTime1 = "2016-02-07 02:00:00"
    stopTime1 = "2016-02-07 02:08:00"
    timeData1 = spamReader.getPhysicalData(startTime1, stopTime1)
    # more data
    startTime2 = "2016-02-07 02:10:00"
    stopTime2 = "2016-02-07 02:15:00"
    timeData2 = spamReader.getPhysicalData(startTime2, stopTime2)
    # now do some signal processing
    sproc = SignalProcessor()
    timeData = sproc.fillGap(timeData1, timeData2)
    timeData.printInfo()
    # now let's plot and see what happens
    fig = plt.figure(figsize=(20, 10))
    x = timeData.getDateArray()
    xlim = [timeData1.startTime, timeData2.stopTime]
    timeData.view(
        sampleStop=timeData.numSamples - 1,
        fig=fig,
        xlim=xlim,
        label="Gap filled")
    timeData1.view(
        sampleStop=timeData1.numSamples - 1,
        fig=fig,
        xlim=xlim,
        label="Section1")
    timeData2.view(
        sampleStop=timeData2.numSamples - 1,
        fig=fig,
        xlim=xlim,
        label="Section2")
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    addLegends(fig)
    plt.show()


def testResample():
    breakPrint()
    generalPrint("testsSignalProcess", "Running test function: testResample")
    # read in data
    spamReader = DataReaderSPAM(spamPath)
    # now get some data
    startTime = "2016-02-07 02:00:00"
    stopTime = "2016-02-07 03:00:00"
    # let's get some unscaled data
    timeData = spamReader.getUnscaledData(startTime, stopTime)
    timeData.printInfo()
    timeDataSave = copy.deepcopy(timeData)
    timeData.printInfo()
    # now do some signal processing
    sproc = SignalProcessor()
    timeData = sproc.resample(timeData, 50)
    timeData.printInfo()
    # now let's plot and see what happens
    fig = plt.figure(figsize=(20, 10))
    x = timeData.getDateArray()
    xlim = [timeDataSave.startTime, x[4000]]
    timeDataSave.view(sampleStop=20000, fig=fig, xlim=xlim, label="Raw data")
    timeData.view(sampleStop=4000, fig=fig, xlim=xlim, label="Resampled data")
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    addLegends(fig)
    plt.show()


def testRemoveZeros():
    breakPrint()
    generalPrint("testsSignalProcess",
                 "Running test function: testRemoveZeros")
    # read in data
    spamReader = DataReaderSPAM(spamPath)
    spamReader.printInfo()
    # now get some data
    startTime = "2016-02-07 02:30:00"
    stopTime = "2016-02-07 02:35:00"
    timeData = spamReader.getPhysicalData(startTime, stopTime)
    timeData.printInfo()
    numSamples = timeData.numSamples
    # let's set some samples to zero
    sampleStart = int(0.3 * numSamples)
    sampleStop = int(0.4 * numSamples)
    for c in timeData.chans:
        timeData.data[c][sampleStart:sampleStop] = 0
    # now plot
    timeDataProcessed = copy.deepcopy(timeData)
    timeDataProcessed.data = removeZeros(timeDataProcessed.data)
    fig = plt.figure(figsize=(20, 10))
    timeData.view(sampleStop=timeData.numSamples, fig=fig, label="With zeros")
    timeDataProcessed.view(
        sampleStop=timeData.numSamples, fig=fig, label="Without zeros")
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    addLegends(fig)
    plt.show()


def testRemoveNans():
    breakPrint()
    generalPrint("testsSignalProcess", "Running test function: testRemoveNans")
    # read in data
    spamReader = DataReaderSPAM(spamPath)
    spamReader.printInfo()
    # now get some data
    startTime = "2016-02-07 02:30:00"
    stopTime = "2016-02-07 02:35:00"
    timeData = spamReader.getPhysicalData(startTime, stopTime)
    timeData.printInfo()
    numSamples = timeData.numSamples
    # let's set some samples to zero
    sampleStart = int(0.3 * numSamples)
    sampleStop = int(0.4 * numSamples)
    for c in timeData.chans:
        timeData.data[c][sampleStart:sampleStop] = np.nan
    # now plot
    timeDataProcessed = copy.deepcopy(timeData)
    timeDataProcessed.data = removeNans(timeDataProcessed.data)
    fig = plt.figure(figsize=(20, 10))
    timeData.view(sampleStop=timeData.numSamples, fig=fig, label="With nans")
    timeDataProcessed.view(
        sampleStop=timeData.numSamples, fig=fig, label="Without nans")
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    addLegends(fig)
    plt.show()


plotTest()
lpFilterTest()
hpFilterTest()
bpFilterTest()
notchFilterTest()
testInterpolate()
testFillGap()
testResample()
testRemoveZeros()
testRemoveNans()