from datapaths import projectPath, imagePath
from resistics.project.io import loadProject

projData = loadProject(projectPath)

# calculate another set of spectra for the 128 Hz data with notching at 50Hz and 16.667Hz
from resistics.project.spectra import calculateSpectra

calculateSpectra(projData, sampleFreqs=[128], notch=[50], specdir="notch")
projData.refresh()

# view the spectra
from resistics.common.plot import plotOptionsSpec, getPaperFonts
from resistics.project.spectra import viewSpectra, viewSpectraSection

plotOptions = plotOptionsSpec(plotfonts=getPaperFonts())
fig = viewSpectra(
    projData,
    "site1",
    "meas_2012-02-10_11-30-00",
    specdir="notch",
    plotoptions=plotOptions,
    show=True,
    save=False,
)
fig.savefig(imagePath / "multspec_viewspec_notch_spec")

fig = viewSpectraSection(
    projData,
    "site1",
    "meas_2012-02-10_11-30-00",
    specdir="notch",
    plotoptions=plotOptions,
    show=True,
    save=False,
)
fig.savefig(imagePath / "multspec_viewspec_notch_section")

# process the new set of spectra
from resistics.project.transfunc import processProject

processProject(projData, sites=["site1"], specdir="notch")

# plot the transfer functions, again with specifying the relevant specdir
from resistics.project.transfunc import viewImpedance

figs = viewImpedance(
    projData,
    sites=["site1"],
    sampleFreqs=[128],
    oneplot=False,
    specdir="notch",
    save=True,
)
figs[0].savefig(imagePath / "multspec_viewimp_notch")

# and compare to the original
fig = viewImpedance(
    projData,
    sites=["site1"],
    sampleFreqs=[128],
    oneplot=False,
    specdir="spectra",
    save=True,
)
figs[0].savefig(imagePath / "multspec_viewimp_nonotch")
