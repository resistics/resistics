#!/usr/bin/python
import os
import sys
import numpy as np
import math
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
# import readers
from magpy.ioHandlers.dataReaderSpam import DataReaderSPAM
from magpy.ioHandlers.dataReaderATS import DataReaderATS
from magpy.ioHandlers.dataReaderInternal import DataReaderInternal
from magpy.ioHandlers.dataReaderPhoenix import DataReaderPhoenix
# import writers
from magpy.ioHandlers.dataWriterInternal import DataWriterInternal
from magpy.ioHandlers.dataWriterAscii import DataWriterAscii
# import utils
from magpy.utilities.utilsProcess import *
from magpy.utilities.utilsIO import *

# data paths
atsPath = os.path.join("testData", "ats")
ats_2ascii = os.path.join("testData", "atsAscii")
spamPath = os.path.join("testData", "spam")
spam_2internal = os.path.join("testData", "spamInternal")
spam_2internalSubset = os.path.join("testData", "spamInternalSubset")
phoenixPath = os.path.join("testData", "phoenix")
phoenix_2internal = os.path.join("testData", "phoenixInternal")


def testATS():
    breakPrint()
    generalPrint("testsDataReader", "Running test function: testATS")
    # read in ATS data
    atsReader = DataReaderATS(atsPath)
    atsReader.printInfo()
    # now get some data
    startTime = "2016-02-21 03:00:00"
    stopTime = "2016-02-21 04:00:00"
    # let's get some unscaled data
    unscaledData = atsReader.getUnscaledData(startTime, stopTime)
    unscaledData.printInfo()
    # now let's look at the data
    unscaledData.view(sampleEnd=20000)
    # let's try physical data
    physicalData = atsReader.getPhysicalData(startTime, stopTime)
    physicalData.printInfo()
    # let's view it
    physicalData.view()
    # but all we see is 50Hz and 16Hz noise
    # so we can apply a low pass filter
    physicalData.view(sampleEnd=20000, lpfilt=4)
    # now let's try and write it out
    # as the ascii format
    writer = DataWriterAscii()
    writer.setOutPath(ats_2ascii)
    writer.writeDataset(atsReader)


def testSPAM():
    breakPrint()
    generalPrint("testsDataReader", "Running test function: testSPAM")
    # read in spam data
    spamReader = DataReaderSPAM(spamPath)
    spamReader.printInfo()
    # now get some data
    startTime = "2016-02-07 02:00:00"
    stopTime = "2016-02-07 03:00:00"
    # let's get some unscaled data
    unscaledData = spamReader.getUnscaledData(startTime, stopTime)
    unscaledData.printInfo()
    # now let's look at the data
    unscaledData.view(sampleEnd=20000, lpfilt=4)
    # let's try physical data
    physicalData = spamReader.getPhysicalData(startTime, stopTime)
    physicalData.printInfo()
    # let's view it
    physicalData.view(sampleEnd=20000)
    # but all we see is 50Hz and 16Hz noise
    # so we can apply a low pass filter
    physicalData.view(sampleEnd=20000, lpfilt=4)
    # now let's try and write it out
    # as the internal format
    writer = DataWriterInternal()
    writer.setOutPath(spam_2internal)
    writer.writeDataset(spamReader)
    # let's try reading in again and see what's in the comments
    reader = DataReaderInternal(spam_2internal)
    reader.printInfo()
    reader.printComments()

    # now try and write out a smaller subset of data
    writer.setOutPath(spam_2internalSubset)
    physicalData.printInfo()
    chanHeaders, chanMap = spamReader.getChanHeaders()
    writer.writeData(spamReader.getHeaders(), chanHeaders, physicalData)
    # let's try reading in again
    reader = DataReaderInternal(spam_2internalSubset)
    reader.printInfo()
    reader.printComments()


def testPhoenix():
    breakPrint()
    generalPrint("testsDataReader", "Running test function: testPhoenix")
    # read in spam data
    phoenixReader = DataReaderPhoenix(phoenixPath)
    phoenixReader.printInfo()
    # now get some data
    startTime = "2011-11-14 02:00:00"
    stopTime = "2011-11-14 03:00:00"
    # # let's get some unscaled data
    unscaledData = phoenixReader.getUnscaledData(startTime, stopTime)
    unscaledData.printInfo()
    # now let's look at the data
    unscaledData.view(sampleEnd=20000)
    # let's try physical data
    physicalData = phoenixReader.getPhysicalData(startTime, stopTime)
    physicalData.printInfo()
    # let's view it
    physicalData.view(sampleEnd=20000)
    # apply low pass filter
    physicalData.view(sampleEnd=20000, lpfilt=4)
    # now let's try and reformat it
    # only reformat the continuous for now
    phoenixReader.reformatContinuous(phoenix_2internal)
    # phoenixReader.reformatHigh(phoenix_2internal, ts=[4])


testATS()
testSPAM()
testPhoenix()
