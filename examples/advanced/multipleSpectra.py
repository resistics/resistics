import os
from resistics.project.projectIO import loadProject
from resistics.project.projectSpecCalc import calculateSpectra
from resistics.project.projectView import (
    viewSpectra,
    viewSpectraSection,
    viewSpectraStack,
    viewTime,
    viewTransferFunction,
)
from resistics.project.projectTransferFunctionCalc import processProject
from resistics.utilities.utilsPlotter import plotOptionsSpec, getPaperFonts


"""Remote Reference with multiple spectra

"""
# need the project path for loading
projectPath = os.path.join("advancedProject")
# load the project
projData = loadProject(projectPath, "ex1_04b_config.ini")

# # calculate spectrum using standard options
# calculateSpectra(projData, sites=["M1", "M13", "RemoteResampled"])
# # refresh the project to find the new files
# projData.refresh()

# # initially, perform standard single site processing to have a reference for comparison
# processProject(projData)
# # refresh the project
# projData.refresh()

# # plot the transfer functions
viewTransferFunction(projData, sites=["M1", "M13", "RemoteResampled"], oneplot=False, save=True)

# # perform standard remote reference runs - remember to call the output something else
# processProject(projData, sites=["M1", "M13"], sampleFreqs=[128], remotesite="RemoteResampled", postpend="rr")
# # refresh the project
# projData.refresh()

viewTransferFunction(projData, sites=["M1", "M13"], postpend="rr", oneplot=False, save=True)