from datapaths import projectPath, imagePath
from resistics.project.io import loadProject

# load the project
projData = loadProject(projectPath)

# process the spectra with tippers
from resistics.project.transfunc import processProject

processProject(
    projData, sites=["site1"], outchans=["Ex", "Ey", "Hz"], postpend="with_Hz"
)

# plot the tippers
from resistics.project.transfunc import viewTipper

figs = viewTipper(projData, sites=["site1"], postpend="with_Hz", save=True, show=False)
figs[0].savefig(imagePath / "simpleRunWithTipper_viewtip_withHz")

# plot the transfer function
from resistics.project.transfunc import viewImpedance

figs = viewImpedance(
    projData,
    sites=["site1"],
    polarisations=["ExHy", "EyHx"],
    postpend="with_Hz",
    save=False,
    show=False,
)
figs[0].savefig(imagePath / "simpleRunWithTipper_viewimp_withHz_polarisations")

# process only the tippers
processProject(projData, sites=["site1"], outchans=["Hz"], postpend="only_Hz")
figs = viewTipper(projData, sites=["site1"], postpend="only_Hz", save=False, show=False)
figs[0].savefig(imagePath / "simpleRunWithTipper_viewtip_onlyHz")
