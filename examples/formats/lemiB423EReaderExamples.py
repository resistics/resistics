import os

from resistics.ioHandlers.dataReaderLemiB423E import (
    DataReaderLemiB423E,
    measB423EHeaders,
    folderB423EHeaders,
)

lemiPath = os.path.join("timeData", "lemiB423E")
measB423EHeaders(lemiPath, 500, dx=60, dy=60.7, ex="E1", ey="E2")

lemiReader = DataReaderLemiB423E(lemiPath)
lemiReader.printInfo()
lemiReader.printDataFileInfo()

# plot data
import matplotlib.pyplot as plt

unscaledLemiData = lemiReader.getUnscaledSamples(
    startSample=0, endSample=10000, scale=False
)
unscaledLemiData.printInfo()
fig = plt.figure(figsize=(16, 2 * unscaledLemiData.numChans))
unscaledLemiData.view(fig=fig, sampleStop=unscaledLemiData.numSamples)
fig.tight_layout(rect=[0, 0.02, 1, 0.96])
fig.savefig(os.path.join("images", "lemiB423EUnscaled.png"))

unscaledLemiData = lemiReader.getUnscaledSamples(
    startSample=0, endSample=10000, scale=True
)
unscaledLemiData.printInfo()
fig = plt.figure(figsize=(16, 2 * unscaledLemiData.numChans))
unscaledLemiData.view(fig=fig, sampleStop=unscaledLemiData.numSamples)
fig.tight_layout(rect=[0, 0.02, 1, 0.96])
fig.savefig(os.path.join("images", "lemiB423EUnscaledWithScaleOption.png"))

physicalLemiData = lemiReader.getPhysicalSamples(
    chans=["Ex", "Ey"], startSample=0, endSample=10000, remaverage=True
)
physicalLemiData.printInfo()
fig = plt.figure(figsize=(16, 3 * physicalLemiData.numChans))
physicalLemiData.view(fig=fig, sampleStop=physicalLemiData.numSamples)
fig.tight_layout(rect=[0, 0.02, 1, 0.96])
fig.savefig(os.path.join("images", "lemiB423EPhysical.png"))

# create headers for all measurements in a folder
folderPath = os.path.join("timeData", "lemiB423E_site")
folderB423EHeaders(folderPath, 500, dx=60, dy=60.7, ex="E1", ey="E2")
lemiPath = os.path.join(folderPath, "lemi01")
lemiReader = DataReaderLemiB423E(lemiPath)
lemiReader.printInfo()
lemiReader.printDataFileInfo()