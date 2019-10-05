from datapaths import projectPath, imagePath
from resistics.project.io import loadProject

#  load the project and also provide a config file
projData = loadProject(projectPath, configFile="asciiconfig.ini")
projData.printInfo()

from resistics.project.time import viewTime
from resistics.common.plot import plotOptionsTime, getPaperFonts

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
from resistics.project.spectra import calculateSpectra

calculateSpectra(projData, calibrate=False, polreverse={"Hy": True})
projData.refresh()

# plot spectra stack
from resistics.project.spectra import viewSpectraStack
from resistics.common.plot import plotOptionsSpec, getPaperFonts

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
from resistics.project.transfunc import processProject

processProject(projData, outchans=["Ex", "Ey"])

# plot impedance tensor and save the plot
from resistics.project.transfunc import viewImpedance
from resistics.common.plot import plotOptionsTransferFunction

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

from resistics.project.transfunc import viewTipper
from resistics.common.plot import plotOptionsTipper

plotoptions = plotOptionsTipper(plotfonts=getPaperFonts())
plotoptions["xlim"] = [0.01, 1000000]
figs = viewTipper(
    projData, sites=["site1"], postpend="withHz", plotoptions=plotoptions, save=True
)
figs[0].savefig(imagePath / "impedance_config_withHz")