from configuration import projectPath, imagePath
from resistics.project.projectIO import loadProject

#  load the project and also provide a config file
projData = loadProject(projectPath, configFile="asciiconfig.ini")
projData.printInfo()

from resistics.project.projectTime import viewTime
from resistics.utilities.utilsPlotter import plotOptionsTime, getPaperFonts

plotOptions = plotOptionsTime(plotfonts=getPaperFonts())
fig = viewTime(
    projData,
    "2018-01-03 00:00:00",
    "2018-01-05 00:00:00",
    polreverse={"Hy": True},
    plotoptions=plotOptions,
    save=False,
)
fig.savefig(imagePath / "viewTime_polreverse")

# calculate spectrum using the new configuration
from resistics.project.projectSpectra import calculateSpectra

calculateSpectra(projData, calibrate=False, polreverse={"Hy": True})
projData.refresh()

# plot spectra stack
from resistics.project.projectSpectra import viewSpectraStack
from resistics.utilities.utilsPlotter import plotOptionsSpec, getPaperFonts

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
fig.savefig(imagePath / "viewSpectraStack_config_polreverse")

# calculate impedance tensor
from resistics.project.projectTransferFunction import processProject

processProject(projData, outchans=["Ex", "Ey"])

# plot impedance tensor and save the plot
from resistics.project.projectTransferFunction import viewImpedance
from resistics.utilities.utilsPlotter import plotOptionsTransferFunction

plotoptions = plotOptionsTransferFunction(plotfonts=getPaperFonts())
plotoptions["xlim"] = [0.01, 1000000]
plotoptions["phase_ylim"] = [-10, 100]
figs = viewImpedance(
    projData,
    sites=["site1"],
    oneplot=True,
    polarisations=["ExHy", "EyHx"],
    plotoptions=plotoptions,
    save=True,
)
figs[0].savefig(imagePath / "impedance_config")

# process for the tipper
processProject(projData, outchans=["Ex", "Ey", "Hz"], postpend="withHz")

from resistics.project.projectTransferFunction import viewTipper
from resistics.utilities.utilsPlotter import plotOptionsTipper

plotoptions = plotOptionsTipper(plotfonts=getPaperFonts())
plotoptions["xlim"] = [0.01, 1000000]
figs = viewTipper(
    projData, sites=["site1"], postpend="withHz", plotoptions=plotoptions, save=True
)
figs[0].savefig(imagePath / "impedance_config_withHz")