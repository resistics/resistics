import os

from resistics.ioHandlers.dataReaderLemiB423 import (
    DataReaderLemiB423,
    measB423Headers,
    folderB423Headers,
)

lemiPath = os.path.join("timeData", "lemiB423")
measB423Headers(
    lemiPath, 500, hxSensor=712, hySensor=710, hzSensor=714, hGain=16, dx=60, dy=60.7
)

lemiReader = DataReaderLemiB423(lemiPath)
lemiReader.printInfo()
lemiReader.printDataFileInfo()

# plot data
import matplotlib.pyplot as plt

unscaledLemiData = lemiReader.getUnscaledSamples()
unscaledLemiData.printInfo()
fig = plt.figure(figsize=(16, 3 * unscaledLemiData.numChans))
unscaledLemiData.view(fig=fig, sampleStop=10000)
fig.tight_layout(rect=[0, 0.02, 1, 0.96])
fig.savefig(os.path.join("images", "lemiB423Unscaled.png"))

startTime = "2019-05-27 15:00:00"
stopTime = "2019-05-27 15:00:15"
unscaledLemiData = lemiReader.getUnscaledData(startTime, stopTime, scale=True)
unscaledLemiData.printInfo()
fig = plt.figure(figsize=(16, 3 * unscaledLemiData.numChans))
unscaledLemiData.view(fig=fig, sampleStop=unscaledLemiData.numSamples)
fig.tight_layout(rect=[0, 0.02, 1, 0.96])
fig.savefig(os.path.join("images", "lemiB423UnscaledWithScaleOption.png"))

physicalLemiData = lemiReader.getPhysicalData(startTime, stopTime)
physicalLemiData.printInfo()
fig = plt.figure(figsize=(16, 3 * physicalLemiData.numChans))
physicalLemiData.view(fig=fig, sampleStop=physicalLemiData.numSamples)
fig.tight_layout(rect=[0, 0.02, 1, 0.96])
fig.savefig(os.path.join("images", "lemiB423PhysicalData.png"))

# write out as the internal format
from resistics.ioHandlers.dataWriterInternal import DataWriterInternal

lemiB423_2internal = os.path.join("timeData", "lemiB423Internal")
writer = DataWriterInternal()
writer.setOutPath(lemiB423_2internal)
writer.writeDataset(lemiReader, physical=True)

# read in the internal format dataset, see comments and plot original lemi data
from resistics.ioHandlers.dataReaderInternal import DataReaderInternal

internalReader = DataReaderInternal(lemiB423_2internal)
internalReader.printComments()
physicalInternalData = internalReader.getPhysicalData(startTime, stopTime)
fig = plt.figure(figsize=(16, 3 * physicalLemiData.numChans))
physicalLemiData.view(fig=fig, sampleStop=500, label="Lemi B423 format")
physicalInternalData.view(fig=fig, sampleStop=500, label="Internal format", legend=True)
fig.tight_layout(rect=[0, 0.02, 1, 0.96])
fig.savefig(os.path.join("images", "lemiB423_vs_internal.png"))

# preparing headers for all measurement folders in a site
folderPath = os.path.join("timeData", "lemiB423_site")
folderB423Headers(
    folderPath, 500, hxSensor=712, hySensor=710, hzSensor=714, hGain=16, dx=60, dy=60.7
)

lemiPath = os.path.join(folderPath, "lemi01")
lemiReader = DataReaderLemiB423(lemiPath)
lemiReader.printInfo()
lemiReader.printDataFileInfo()