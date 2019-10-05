from datapaths import timePath, timeImages
from resistics.time.reader_lemib423 import (
    TimeReaderLemiB423,
    measB423Headers,
    folderB423Headers,
)

lemiPath = timePath / "lemiB423"
measB423Headers(
    lemiPath, 500, hxSensor=712, hySensor=710, hzSensor=714, hGain=16, dx=60, dy=60.7
)

lemiReader = TimeReaderLemiB423(lemiPath)
lemiReader.printInfo()
lemiReader.printDataFileInfo()

# plot data
import matplotlib.pyplot as plt

unscaledLemiData = lemiReader.getUnscaledSamples()
unscaledLemiData.printInfo()
fig = plt.figure(figsize=(16, 3 * unscaledLemiData.numChans))
unscaledLemiData.view(fig=fig, sampleStop=10000)
fig.tight_layout(rect=[0, 0.02, 1, 0.96])
fig.savefig(timeImages / "lemiB423Unscaled.png")

startTime = "2019-05-27 15:00:00"
stopTime = "2019-05-27 15:00:15"
unscaledLemiData = lemiReader.getUnscaledData(startTime, stopTime, scale=True)
unscaledLemiData.printInfo()
fig = plt.figure(figsize=(16, 3 * unscaledLemiData.numChans))
unscaledLemiData.view(fig=fig, sampleStop=unscaledLemiData.numSamples)
fig.tight_layout(rect=[0, 0.02, 1, 0.96])
fig.savefig(timeImages / "lemiB423UnscaledWithScaleOption.png")

physicalLemiData = lemiReader.getPhysicalData(startTime, stopTime)
physicalLemiData.printInfo()
fig = plt.figure(figsize=(16, 3 * physicalLemiData.numChans))
physicalLemiData.view(fig=fig, sampleStop=physicalLemiData.numSamples)
fig.tight_layout(rect=[0, 0.02, 1, 0.96])
fig.savefig(timeImages / "lemiB423PhysicalData.png")

# write out as the internal format
from resistics.time.writer_internal import TimeWriterInternal

lemiB423_2internal = timePath / "lemiB423Internal"
writer = TimeWriterInternal()
writer.setOutPath(lemiB423_2internal)
writer.writeDataset(lemiReader, physical=True)

# read in the internal format dataset, see comments and plot original lemi data
from resistics.time.reader_internal import TimeReaderInternal

internalReader = TimeReaderInternal(lemiB423_2internal)
internalReader.printComments()
physicalInternalData = internalReader.getPhysicalData(startTime, stopTime)
fig = plt.figure(figsize=(16, 3 * physicalLemiData.numChans))
physicalLemiData.view(fig=fig, sampleStop=500, label="Lemi B423 format")
physicalInternalData.view(fig=fig, sampleStop=500, label="Internal format", legend=True)
fig.tight_layout(rect=[0, 0.02, 1, 0.96])
fig.savefig(timeImages / "lemiB423_vs_internal.png")

# preparing headers for all measurement folders in a site
folderPath = timePath / "lemiB423_site"
folderB423Headers(
    folderPath, 500, hxSensor=712, hySensor=710, hzSensor=714, hGain=16, dx=60, dy=60.7
)

lemiPath = folderPath / "lemi01"
lemiReader = TimeReaderLemiB423(lemiPath)
lemiReader.printInfo()
lemiReader.printDataFileInfo()