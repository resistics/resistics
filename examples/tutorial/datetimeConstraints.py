from datapaths import projectPath, imagePath
from resistics.project.io import loadProject

# load the project
projData = loadProject(projectPath, configFile="tutorialConfig.ini")

# define date/time constraints - only time windows within the constraints will be used
datetimes = list()
datetimes.append(
    {"type": "datetime", "start": "2012-02-10 19:00:00", "stop": "2012-02-11 07:00:00"}
)

# process the spectra
from resistics.project.transfunc import processProject

processProject(
    projData, sampleFreqs=[128], datetimes=datetimes, postpend="datetimeConstraint"
)

# plot transfer function and save the plot
from resistics.project.transfunc import viewImpedance

figs = viewImpedance(
    projData,
    sites=["site1"],
    postpend="datetimeConstraint",
    oneplot=False,
    save=False,
    show=False,
)
figs[0].savefig(imagePath / "datetimeConstraints_viewimp")

# process again with a mask too
processProject(
    projData,
    sampleFreqs=[128],
    sites=["site1"],
    outchans=["Ex", "Ey"],
    masks={"site1": "coh70_100_tfConstrained"},
    datetimes=datetimes,
    postpend="coh70_100_tfConstrained_datetimeConstrained",
)

figs = viewImpedance(
    projData,
    sites=["site1"],
    postpend="coh70_100_tfConstrained_datetimeConstrained",
    oneplot=False,
    save=False,
    show=False,
)
figs[0].savefig(imagePath / "datetimeConstraints_viewimp_mask_coh70_100_tfConstrained")
