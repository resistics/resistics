from datapaths import projectPath, imagePath
from resistics.project.io import loadProject
from resistics.project.spectra import calculateSpectra
from resistics.project.transfunc import processProject, viewImpedance
from resistics.project.statistics import calculateStatistics
from resistics.project.mask import newMaskData, calculateMask

if __name__ == "__main__":
    projData = loadProject(projectPath, "multiconfig.ini")

    # calculate spectrum using standard options
    calculateSpectra(projData)
    projData.refresh()
    calculateStatistics(projData)
    
    # process project with standard options
    processProject(projData)
    figs = viewImpedance(projData, oneplot=False, show=False, save=False)
    figs[0].savefig(imagePath / "multproc_standard_process")

    # calculate mask for 128
    maskData = newMaskData(projData, 128)
    maskData.setStats(["coherence"])
    maskData.addConstraint("coherence", {"cohExHy": [0.7, 1.0], "cohEyHx": [0.7, 1.0]})
    maskData.maskName = "coh70_100"
    calculateMask(projData, maskData, sites=["site1"])
    # calculate mask for 4096
    maskData = newMaskData(projData, 4096)
    maskData.setStats(["coherence"])
    maskData.addConstraint("coherence", {"cohExHy": [0.7, 1.0], "cohEyHx": [0.7, 1.0]})
    maskData.maskName = "coh70_100"
    calculateMask(projData, maskData, sites=["site1"])
    # process with masks
    processProject(projData, masks={"site1": "coh70_100"}, postpend="coh70_100")
    figs = viewImpedance(
        projData, oneplot=False, postpend="coh70_100", show=False, save=False
    )
    figs[0].savefig(imagePath / "multproc_mask_process")
