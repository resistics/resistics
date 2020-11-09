from datapaths import versions_project
from resistics.project.io import loadProject

if __name__ == "__main__":
    projData = loadProject(versions_project, "0.0.6_config.ini")
    projData.printInfo()

    from resistics.project.spectra import calculateSpectra

    calculateSpectra(projData, sites=["site1_mt"])
    calculateSpectra(
        projData, sites=["site2_te"], chans=["Ex", "Ey"], polreverse={"Ey": True}
    )
    projData.refresh()

    # single site processing again
    from resistics.project.transfunc import processProject, processSite, viewImpedance
    from resistics.common.plot import plotOptionsTransferFunction, getPaperFonts

    plotOptions = plotOptionsTransferFunction(plotfonts=getPaperFonts())
    plotOptions["res_ylim"] = [1, 1000000]

    processProject(
        projData,
        sites=["site1_mt"],
        sampleFreqs=[500],
    )
    viewImpedance(
        projData,
        sites=["site1_mt"],
        sampleFreqs=[500],
        polarisations=["ExHy", "EyHx"],
        plotoptions=plotOptions,
        oneplot=True,
        show=True,
        save=True,
    )
