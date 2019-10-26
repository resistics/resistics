from datapaths import projectPath, imagePath
from resistics.project.io import loadProject
from resistics.project.spectra import calculateSpectra
from resistics.project.transfunc import processProject, viewImpedance
from resistics.project.statistics import calculateStatistics
from resistics.project.mask import newMaskData, calculateMask
from resistics.common.plot import plotOptionsTransferFunction, getPaperFonts

if __name__ == "__main__":
    plotOptions = plotOptionsTransferFunction(plotfonts=getPaperFonts())
    proj = loadProject(projectPath, "multiconfig.ini")

    # calculate spectrum using standard options
    calculateSpectra(proj)
    proj.refresh()
    calculateStatistics(proj)
    processProject(
		proj,
		sites=["M6"],
		sampleFreqs=[128],
		inputsite="M7_split",
		remotesite="Remote_split",
		postpend="rr_m7_split",
	)
    viewImpedance(
		proj,
		sites=["M6"],
		sampleFreqs=[128],
		postpend="rr_split",
		oneplot=False,
		plotoptions=plotOptions,
		show=False,
		save=True,
	)
	
