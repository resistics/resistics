import os

# from resistics.ioHandlers.dataReaderLemiB423 import DataReaderLemiB423
from resistics.ioHandlers.dataReaderLemiB423E import DataReaderLemiB423E, measB423EHeaders, folderB423EHeaders

# lemiPath = os.path.join("timeData", "lemiB423E_2")
# folderB423EHeaders(lemiPath, 500, dx=60, dy=60.7)

lemiPath = os.path.join("timeData", "lemiB423E")
startTime = "2019-05-27 14:00:00"
stopTime = "2019-05-27 15:00:00"
measB423EHeaders(lemiPath, 500, dx=60, dy=60.7)

lemiReader = DataReaderLemiB423E(lemiPath)
lemiReader.printInfo()

# get physical data from Lemi
import matplotlib.pyplot as plt

# unscaledLemiData = lemiReader.getUnscaledData(startTime, stopTime, scale=False)
unscaledLemiData = lemiReader.getUnscaledSamples(startSample=0, endSample=62000, scale=False)
unscaledLemiData.printInfo()
fig = plt.figure(figsize=(16, 2 * unscaledLemiData.numChans))
unscaledLemiData.view(fig=fig, sampleStop=unscaledLemiData.numSamples)
fig.tight_layout(rect=[0, 0.02, 1, 0.96])
plt.show()

# unscaledLemiData = lemiReader.getUnscaledData(startTime, stopTime, scale=True)
unscaledLemiData = lemiReader.getUnscaledSamples(startSample=0, endSample=62000, scale=True)
unscaledLemiData.printInfo()
fig = plt.figure(figsize=(16, 2 * unscaledLemiData.numChans))
unscaledLemiData.view(fig=fig, sampleStop=unscaledLemiData.numSamples)
fig.tight_layout(rect=[0, 0.02, 1, 0.96])
plt.show()

physicalLemiData = lemiReader.getPhysicalSamples(startSample=0, endSample=62000, remaverage=False)
physicalLemiData.printInfo()
fig = plt.figure(figsize=(16, 2 * physicalLemiData.numChans))
physicalLemiData.view(fig=fig, sampleStop=physicalLemiData.numSamples)
fig.tight_layout(rect=[0, 0.02, 1, 0.96])
plt.show()

physicalLemiData = lemiReader.getPhysicalSamples(startSample=0, endSample=62000, remaverage=True)
physicalLemiData.printInfo()
fig = plt.figure(figsize=(16, 2 * physicalLemiData.numChans))
physicalLemiData.view(fig=fig, sampleStop=physicalLemiData.numSamples)
fig.tight_layout(rect=[0, 0.02, 1, 0.96])
plt.show()

