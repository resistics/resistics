import os

# from resistics.ioHandlers.dataReaderLemiB423 import DataReaderLemiB423
from resistics.ioHandlers.dataReaderLemiB423 import DataReaderLemiB423, measB423Headers, folderB423Headers

# lemiPath = os.path.join("timeData", "lemiB423_3")
# folderB423Headers(lemiPath, 2000, hxSensor = 712, hySensor = 710, hzSensor = 714)

# read in lemi data
lemiPath = os.path.join("timeData", "lemiB423_2")
startTime = "2016-10-19 08:35:39"
stopTime = "2016-10-19 08:35:44"
measB423Headers(lemiPath, 2000, hxSensor = 712, hySensor = 710, hzSensor = 714)

# lemiPath = os.path.join("timeData", "lemiB423")
# startTime = "2019-05-27 15:00:00"
# stopTime = "2019-05-27 16:00:00"
# measB423Headers(lemiPath, 500, hxSensor = 712, hySensor = 710, hzSensor = 714, dx=60, dy=60.7)

lemiReader = DataReaderLemiB423(lemiPath)
lemiReader.printInfo()

# get physical data from Lemi
import matplotlib.pyplot as plt

unscaledLemiData = lemiReader.getUnscaledData(startTime, stopTime, scale=False, chans=["Hx", "Hy", "Hz"])
unscaledLemiData.printInfo()
fig = plt.figure(figsize=(16, 3 * unscaledLemiData.numChans))
unscaledLemiData.view(fig=fig, sampleStop=unscaledLemiData.numSamples)
fig.tight_layout(rect=[0, 0.02, 1, 0.96])
plt.show()

unscaledLemiData = lemiReader.getUnscaledData(startTime, stopTime, scale=True, chans=["Hx", "Hy", "Hz"])
unscaledLemiData.printInfo()
fig = plt.figure(figsize=(16, 3 * unscaledLemiData.numChans))
unscaledLemiData.view(fig=fig, sampleStop=unscaledLemiData.numSamples)
fig.tight_layout(rect=[0, 0.02, 1, 0.96])
plt.show()

physicalLemiData = lemiReader.getPhysicalData(startTime, stopTime, remaverage=False, chans=["Hx", "Hy", "Hz"])
physicalLemiData.printInfo()
fig = plt.figure(figsize=(16, 3 * physicalLemiData.numChans))
physicalLemiData.view(fig=fig, sampleStop=physicalLemiData.numSamples)
fig.tight_layout(rect=[0, 0.02, 1, 0.96])
plt.show()

physicalLemiData = lemiReader.getPhysicalData(startTime, stopTime, remaverage=True, chans=["Hx", "Hy", "Hz"])
physicalLemiData.printInfo()
fig = plt.figure(figsize=(16, 3 * physicalLemiData.numChans))
physicalLemiData.view(fig=fig, sampleStop=physicalLemiData.numSamples)
fig.tight_layout(rect=[0, 0.02, 1, 0.96])
plt.show()

