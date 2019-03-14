import os
import sys
sys.path.append(os.path.join("..", "core", "ioHandlers"))
sys.path.append(os.path.join("..", "core", "calculators"))
sys.path.append(os.path.join("..", "core", "utilities"))
sys.path.append(os.path.join("..", "core", "dataObjects"))
import numpy as np
import math
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
# import spectra related stuff
from magpy.calculators.spectrumCalculator import SpectrumCalculator
from magpy.ioHandlers.spectrumWriter import SpectrumWriter
from magpy.ioHandlers.spectrumReader import SpectrumReader
# import window and decimation classes
from magpy.calculators.decimationParameters import DecimationParams
from magpy.calculators.decimator import Decimator
from magpy.calculators.windowParameters import WindowParams
from magpy.calculators.windowSelectorBasic import WindowSelectorBasic
# import processors
from magpy.calculators.processorSingleSite import ProcessorSingleSite
# import transfer function stuff
from magpy.ioHandlers.transferFunctionReader import TransferFunctionReader
from magpy.ioHandlers.transferFunctionWriter import TransferFunctionWriter
from magpy.dataObjects.transferFunctionData import TransferFunctionData
# import utils
from magpy.utilities.utilsProcess import *
from magpy.utilities.utilsIO import *

# When not using the project infrastructure, the basic window selector has to be used
# this is because the standard window selector makes heavy use of project setup
# using the basic window selector precludes the use of remote sites or different input sites
# and as such, is a simple, single site processing window selector

specPath = os.path.join("testData", "atsSpec_noNotch")
tfPath = os.path.join("testData", "transfuncfiles")
filepart1 = "dummy_fs128_spectra"
postpend = "noNotch_noConstraints"
postpendTimeConstraint = "noNotch_timeConstraint"

def process():
    breakPrint()
    generalPrint("testsProcess",
                 "Running test function: process")    
    specReader = SpectrumReader(specPath)
    specReader.openBinaryForReading("spectra", 0)
    specReader.printInfo()
    sampleFreq = specReader.getSampleFreq()
    # calculate decimation parameters
    decParams = DecimationParams(sampleFreq)
    decParams.setDecimationParams(7, 6)
    decParams.printInfo()
    numLevels = decParams.numLevels
    # now do window parameters
    winParams = WindowParams(decParams)			
    # now do the window selector
    winSelector = WindowSelectorBasic(specPath, decParams, winParams)         
    # calculate the shared windows and print info
    winSelector.calcSharedWindows()
    winSelector.printInfo()
    winSelector.printDatetimeConstraints()
    winSelector.printWindowMasks()
    winSelector.printSharedIndices()
    winSelector.printWindowsForFrequency()
    # now process
    processor = ProcessorSingleSite(winSelector, tfPath)
    processor.setPostpend(postpend)	
    processor.printInfo()
    processor.process()	 
    # now let's look at the output
    tfFile = os.path.join(tfPath, "128", "{}_{}".format(filepart1, postpend))
    tfReader = TransferFunctionReader(tfFile)
    tfReader.tfData.view()


def processWithDatetimeConstraints():
    breakPrint()
    generalPrint("testsProcess",
                 "Running test function: processWithDatetimeConstraints")     
    specReader = SpectrumReader(specPath)
    specReader.openBinaryForReading("spectra", 0)
    specReader.printInfo()
    sampleFreq = specReader.getSampleFreq()
    # calculate decimation parameters
    decParams = DecimationParams(sampleFreq)
    decParams.setDecimationParams(7, 6)
    decParams.printInfo()
    numLevels = decParams.numLevels
    # now do window parameters
    winParams = WindowParams(decParams)			
    # now do the window selector
    winSelector = WindowSelectorBasic(specPath, decParams, winParams)
    winSelector.addTimeConstraint("03:00:00", "05:00:00")         
    # calculate the shared windows and print info
    winSelector.calcSharedWindows()
    winSelector.printInfo()
    winSelector.printDatetimeConstraints()
    winSelector.printWindowMasks()
    winSelector.printSharedIndices()
    winSelector.printWindowsForFrequency()  
    # now process
    processor = ProcessorSingleSite(winSelector, tfPath)
    processor.setPostpend(postpendTimeConstraint)	
    processor.printInfo()
    processor.process()	 
    # now let's look at the output
    tfFile = os.path.join(tfPath, "128", "{}_{}".format(filepart1, postpendTimeConstraint))
    tfReader = TransferFunctionReader(tfFile)
    tfReader.tfData.view()


# def processWithMask():
  

process()
processWithDatetimeConstraints()