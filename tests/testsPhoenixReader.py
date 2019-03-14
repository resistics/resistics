import os
import sys
import numpy as np
from scipy.signal import detrend
import math
from datetime import datetime, timedelta
import struct
# import readers
from magpy.ioHandlers.dataReaderPhoenix import DataReaderPhoenix
from magpy.ioHandlers.dataReaderInternal import DataReaderInternal
# import writers
from magpy.ioHandlers.dataWriterInternal import DataWriterInternal
# import inbuilt
# from projectDefault import *
# from projectViewTime import *
# import utils
from magpy.utilties.utilsProcess import *
from magpy.utilties.utilsIO import *
# graphing
import matplotlib.pyplot as plt

# def readCoil(coilFile):

# coilPath = os.path.join("..", "..", "Data", "riftVolc", "202", "COIL1547.CLC")
# coilPath = os.path.join("..", "..", "Data", "andes", "calibration", "COIL1424.CLC")
# coilFile = open(coilPath, "rb")
# # utc
# print struct.unpack("8b", coilFile.read(8))
# # snum
# print struct.unpack("h", coilFile.read(2))
# # nChn
# print struct.unpack("b", coilFile.read(1))
# # stat
# print struct.unpack("b", coilFile.read(1))
# # next
# print struct.unpack("8b", coilFile.read(8))
# # hw
# print struct.unpack("12s", coilFile.read(12))
# # version
# print struct.unpack("12s", coilFile.read(12))
# #
# print struct.unpack("i", coilFile.read(4))
# print struct.unpack("i", coilFile.read(4))
# print struct.unpack("i", coilFile.read(4))
# print struct.unpack("i", coilFile.read(4))
# print struct.unpack("i", coilFile.read(4))
# # coil
# print struct.unpack("12s", coilFile.read(12))
# # hatt, hnom, cphc
# print struct.unpack("d", coilFile.read(8))
# print struct.unpack("d", coilFile.read(8))
# print struct.unpack("d", coilFile.read(8))
# #
# coilFile.seek(32, 1)
# #
# i = 0
# while i < 44:
#     print struct.unpack("h", coilFile.read(2))
#     print struct.unpack("f", coilFile.read(4))
#     print struct.unpack("h", coilFile.read(2))
#     print struct.unpack("f", coilFile.read(4))
#     print struct.unpack("f", coilFile.read(4))
#     print struct.unpack("f", coilFile.read(4))
#     print struct.unpack("f", coilFile.read(4))
#     print struct.unpack("f", coilFile.read(4))
#     print struct.unpack("f", coilFile.read(4))
#     i += 1
# #
# coilFile.close()

### test the data reader
dataPath = os.path.join("..", "..", "Data", "riftVolc", "202")
dataReader = DataReaderPhoenix(dataPath)
dataReader.printInfo()
dataReader.printDataFileList()
print(dataReader.getSamplesRatesTS())
print(dataReader.getNumberSamplesTS())
dataReader.printTableFile()
startTime = "2017-04-07 23:00:00"
endTime = "2017-04-08 01:00:00"
timeData = dataReader.getUnscaledData(startTime, endTime)
timeData.printInfo()
timeData.view(sampleStart=0, sampleEnd=20000)

### now try and reformat
# outpath = os.path.join("..", "..", "Data", "riftVolc", "202_reformat")
# dataReader.reformat(outpath)

### create a project
# projectPath = (os.path.join("..", "..", "Data", "riftVolcProject"))
# projectMakeDefault(projectPath, "2017-04-07 06:00:00")
# proj = projectLoad(projectPath, "mtProj.prj")

### let's look at some time
# projectViewTime(proj, "2017-04-08 02:00:00", "2017-04-08 04:30:00", freqs=[15], save=True, chans=["Ex", "Ey", "Hx", "Hy", "Hz"])
# projectViewTime(proj, "2017-04-07 09:16:00", "2017-04-07 09:16:16", freqs=[150], save=True, chans=["Ex", "Ey", "Hx", "Hy", "Hz"])
# projectViewTime(proj, "2017-04-07 09:33:00", "2017-04-07 09:33:01", freqs=[2400], save=True, chans=["Ex", "Ey", "Hx", "Hy", "Hz"])

### Check the andes data
# dataPath = os.path.join("..", "..", "Data", "andes", "TS")
# dataReader = DataReaderPhoenix(dataPath)
# dataReader.printInfo()
# dataReader.printDataFileList()
# print dataReader.getSamplesRatesTS()
# print dataReader.getNumberSamplesTS()
# dataReader.printTableFile()
# startTime = "2011-11-14 02:00:00"
# endTime = "2011-11-14 02:30:00"
# data = dataReader.getUnscaledData(startTime, endTime)
# print data["numSamples"]
# print data["sampleFreq"]
# print data["startTime"]
# print data["endTime"]
# plt.figure()
# for idx, chan in enumerate(data["chans"]):
#     plt.subplot(dataReader.getNumChannels(), 1, idx+1)
#     plt.title(chan)
#     plt.plot(detrend(data[chan], type="linear"))
# plt.show()
### TRY THE ANDES project
# projectPath = (os.path.join("..", "..", "Data", "andesProject"))
# projectMakeDefault(projectPath, "2011-11-10 06:00:00")
# proj = projectLoad(projectPath, "mtProj.prj")
# projectViewTime(proj, "2011-11-13 22:00:00", "2011-11-14 02:30:00", freqs=[15], save=True, chans=["Ex", "Ey", "Hx", "Hy", "Hz"])
# let's try reformatting
