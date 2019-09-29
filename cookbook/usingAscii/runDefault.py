from configuration import projectPath, imagePath
from resistics.project.projectIO import loadProject

# load the project
projData = loadProject(projectPath)
fig = projData.view()
fig.savefig(imagePath / "projectTimeline")

# view site data
siteData = projData.getSiteData("site1")
fig = siteData.view()
fig.savefig(imagePath / "siteTimeline")

from resistics.project.projectTime import viewTime
from resistics.utilities.utilsPlotter import plotOptionsTime, getPaperFonts

plotOptions = plotOptionsTime(plotfonts=getPaperFonts())
fig = viewTime(
    projData,
    "2018-01-03 00:00:00",
    "2018-01-05 00:00:00",
    plotoptions=plotOptions,
    save=False,
)
fig.savefig(imagePath / "viewTime")

# calculate spectrum using standard options
from resistics.project.projectSpectra import calculateSpectra

calculateSpectra(projData, calibrate=False)
projData.refresh()

from resistics.project.projectSpectra import viewSpectraStack
from resistics.utilities.utilsPlotter import plotOptionsSpec

plotOptions = plotOptionsSpec(plotfonts=getPaperFonts())
fig = viewSpectraStack(
    projData,
    "site1",
    "meas",
    coherences=[["Ex", "Hy"], ["Ey", "Hx"]],
    plotoptions=plotOptions,
    save=False,
    show=False,
)
fig.savefig(imagePath / "viewSpectraStack")

# process the spectra to estimate the transfer function
from resistics.project.projectTransferFunction import processProject

processProject(projData, outchans=["Ex", "Ey"])

# plot impedance tensor and save the plot
from resistics.project.projectTransferFunction import viewImpedance
from resistics.utilities.utilsPlotter import plotOptionsTransferFunction

plotoptions = plotOptionsTransferFunction(plotfonts=getPaperFonts())
plotoptions["xlim"] = [0.01, 1000000]
plotoptions["phase_ylim"] = [-360, 360]
figs = viewImpedance(
    projData,
    sites=["site1"],
    oneplot=True,
    polarisations=["ExHy", "EyHx"],
    plotoptions=plotoptions,
    save=False,
)
figs[0].savefig(imagePath / "impedance_default")

# calculate the tipper
processProject(projData, outchans=["Ex", "Ey", "Hz"], postpend="withHz")

# plot the tipper
from resistics.project.projectTransferFunction import viewTipper
from resistics.utilities.utilsPlotter import plotOptionsTipper

plotoptions = plotOptionsTipper(plotfonts=getPaperFonts())
plotoptions["xlim"] = [0.01, 1000000]
figs = viewTipper(
    projData, sites=["site1"], postpend="withHz", plotoptions=plotoptions, save=True
)
figs[0].savefig(imagePath / "impedance_default_withHz")