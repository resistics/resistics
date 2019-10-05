from datapaths import projectPath, imagePath
from resistics.project.io import loadProject

# load the project
projData = loadProject(projectPath)

# calculate spectrum using standard options
from resistics.project.spectra import calculateSpectra

calculateSpectra(projData)
projData.refresh()

# process the spectra
from resistics.project.transfunc import processProject

processProject(projData)

# plot transfer function and save the plot
from resistics.project.transfunc import viewImpedance

figs = viewImpedance(projData, sites=["site1"], show=False, save=False)
figs[0].savefig(imagePath / "simpleRun_viewimp_default")

# or keep the two most important polarisations on the same plot
figs = viewImpedance(
    projData, sites=["site1"], polarisations=["ExHy", "EyHx"], save=True
)
figs[0].savefig(imagePath / "simpleRun_viewimp_polarisations")

# this plot is quite busy, let's plot all the components on separate plots
figs = viewImpedance(projData, sites=["site1"], oneplot=False, save=False)
figs[0].savefig(imagePath / "simpleRun_viewimp_multplot")

# get a transfer function data object
from resistics.project.transfunc import getTransferFunctionData

tfData = getTransferFunctionData(projData, "site1", 128)
fig = tfData.viewImpedance(oneplot=True, polarisations=["ExHy", "EyHx"], save=True)
fig.savefig(imagePath / "simpleRun_tfData_view")
