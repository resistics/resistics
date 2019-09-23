from pathlib import Path
from resistics.project.projectIO import loadProject

projectPath = Path("preprocessProject")
proj = loadProject(projectPath)
proj.printInfo()

from resistics.utilities.utilsPlotter import plotOptionsTime, getPresentationFonts

plotOptions = plotOptionsTime(plotfonts=getPresentationFonts())

from resistics.ioHandlers.dataReaderATS import DataReaderATS

site1 = proj.getSiteData("site1")
readerATS = DataReaderATS(site1.getMeasurementTimePath("meas_2012-02-10_11-05-00"))
# headers of recording
headers = readerATS.getHeaders()
chanHeaders, chanMap = readerATS.getChanHeaders()
# separate out two datasets
timeOriginal1 = readerATS.getPhysicalData(
    "2012-02-10 11:05:00", "2012-02-10 11:09:00", remaverage=False
)
timeOriginal2 = readerATS.getPhysicalData(
    "2012-02-10 11:10:00", "2012-02-10 11:14:00", remaverage=False
)

from resistics.ioHandlers.dataWriterInternal import DataWriterInternal

# create a new site
proj.createSite("site1_gaps")
proj.refresh()
writer = DataWriterInternal()
writer.setOutPath(
    Path(proj.timePath, "site1_gaps", "meas_2012-02-10_11-05-00_section1")
)
writer.writeData(headers, chanHeaders, timeOriginal1, physical=True)
writer.setOutPath(
    Path(proj.timePath, "site1_gaps", "meas_2012-02-10_11-05-00_section2")
)
writer.writeData(headers, chanHeaders, timeOriginal2, physical=True)

from resistics.project.projectTime import viewTime

# now view time
fig = viewTime(
    proj,
    "2012-02-10 11:05:00",
    "2012-02-10 11:14:00",
    sites=["site1", "site1_gaps"],
    filter={"lpfilt": 16},
    chans=["Ex", "Hy"],
    show=False,
    plotoptions=plotOptions,
)
fig.savefig(Path(proj.imagePath, "viewTimeGaps.png"))

from resistics.ioHandlers.dataReaderInternal import DataReaderInternal

siteGaps = proj.getSiteData("site1_gaps")
readerSection1 = DataReaderInternal(
    siteGaps.getMeasurementTimePath("meas_2012-02-10_11-05-00_section1")
)
timeData1 = readerSection1.getPhysicalSamples(remaverage=False)
timeData1.printInfo()

readerSection2 = DataReaderInternal(
    siteGaps.getMeasurementTimePath("meas_2012-02-10_11-05-00_section2")
)
timeData2 = readerSection2.getPhysicalSamples(remaverage=False)
timeData2.printInfo()

from resistics.utilities.utilsInterp import fillGap

timeDataFilled = fillGap(timeData1, timeData2)
timeDataFilled.printInfo()
samplesToView = 14 * 60 * 4096
fig = timeDataFilled.view(sampleStop=samplesToView, chans=["Ex", "Hy"])
fig.savefig(Path(proj.imagePath, "timeDataFilled.png"))

# create a new site to write out to
proj.createSite("site1_filled")
proj.refresh()
# use channel headers from one of the datasets, stop date will be automatically amended
writer = DataWriterInternal()
writer.setOutPath(
    Path(proj.timePath, "site1_filled", "meas_2012-02-10_11-05-00_filled")
)
headers = readerSection1.getHeaders()
chanHeaders, chanMap = readerSection1.getChanHeaders()
writer.writeData(headers, chanHeaders, timeDataFilled, physical=True)
proj.refresh()

# now view time
fig = viewTime(
    proj,
    "2012-02-10 11:05:00",
    "2012-02-10 11:14:00",
    sites=["site1", "site1_filled"],
    filter={"lpfilt": 16},
    chans=["Ex", "Hy"],
    show=False,
    plotoptions=plotOptions,
)
fig.savefig(Path(proj.imagePath, "viewTimeGapsFilled.png"))