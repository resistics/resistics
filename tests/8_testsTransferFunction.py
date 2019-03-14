import os
import sys
import numpy as np
import math
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
# import readers
from magpy.ioHandlers.transferFunctionReader import TransferFunctionReader
# import writers
from magpy.ioHandlers.transferFunctionWriter import TransferFunctionWriter
# import dataobjects
from magpy.dataObjects.transferFunctionData import TransferFunctionData
# import utils
from magpy.utilities.utilsProcess import *
from magpy.utilities.utilsIO import *

ediPath = os.path.join("testData", "transfuncfiles", "transFunc.edi")
edi2edi = os.path.join("testData", "transfuncfiles", "edi2edi.edi")
edi2internal = os.path.join("testData", "transfuncfiles", "edi2internal")
internalPath = os.path.join("testData", "transfuncfiles", "transFunc_internal")
internal2edi = os.path.join("testData", "transfuncfiles", "internal2edi.edi")


def readEdi():
    breakPrint()
    generalPrint("testsTransferFunction",
                 "Running test function: readEdi")       
    reader = TransferFunctionReader(ediPath)
    tfData = reader.tfData
    # tfData.view(xlim=[1e-3, 1e4], phase_ylim=[-30, 120], res_ylim=[1e-2, 1e3])
    tfData.view()
    # try the one plot options with fewer polarisations
    tfData.view(oneplot=True, polarisations=["ExHy", "EyHx"])


def writeEdi():
    breakPrint()
    generalPrint("testsTransferFunction",
                 "Running test function: writeEdi")       
    reader = TransferFunctionReader(ediPath)
    tfData = reader.tfData
    writer = TransferFunctionWriter(edi2edi, tfData)
    writer.writeEdi()
    reader.read(edi2edi)
    reader.tfData.view()


def ediToInternal():
    breakPrint()
    generalPrint("testsTransferFunction",
                 "Running test function: ediToInternal")       
    reader = TransferFunctionReader(ediPath)
    tfData = reader.tfData
    writer = TransferFunctionWriter(edi2internal, tfData)
    writer.write()
    reader.read(edi2internal)
    reader.tfData.view()


def internalToEdi():
    breakPrint()
    generalPrint("testsTransferFunction",
                 "Running test function: internalToEdi")       
    reader = TransferFunctionReader(internalPath)
    tfData = reader.tfData
    writer = TransferFunctionWriter(internal2edi, tfData)
    writer.writeEdi()
    reader.read(internal2edi)
    reader.tfData.view()


readEdi()
writeEdi()
ediToInternal()
internalToEdi()