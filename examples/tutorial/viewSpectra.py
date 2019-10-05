from datapaths import projectPath, imagePath
from resistics.project.io import loadProject

# load the project
projData = loadProject(projectPath)

# view the spectra at 128 Hz
from resistics.project.spectra import viewSpectra

fig = viewSpectra(projData, "site1", "meas_2012-02-10_11-30-00", show=False, save=False)
fig.savefig(imagePath / "viewSpec_projspec_128_view")

# setting the plot options
from resistics.common.plot import plotOptionsSpec, getPaperFonts

plotOptions = plotOptionsSpec(plotfonts=getPaperFonts())
print(plotOptions)

# view the spectra at 128 Hz
fig = viewSpectra(
    projData,
    "site1",
    "meas_2012-02-10_11-30-00",
    plotoptions=plotOptions,
    show=False,
    save=False,
)
fig.savefig(imagePath / "viewSpec_projspec_128_view_plotoptions")

# view the spectra at 128 Hz for more than a single window
fig = viewSpectra(
    projData,
    "site1",
    "meas_2012-02-10_11-30-00",
    chans=["Ex", "Hy"],
    plotwindow="all",
    plotoptions=plotOptions,
    show=False,
    save=False,
)
fig.savefig(imagePath / "viewSpec_projspec_128_view_plotall_chans")

# spectra sections
from resistics.project.spectra import viewSpectraSection

fig = viewSpectraSection(
    projData,
    "site1",
    "meas_2012-02-10_11-30-00",
    plotoptions=plotOptions,
    show=False,
    save=False,
)
fig.savefig(imagePath / "viewSpec_projspec_128_section")

# spectra stack
from resistics.project.spectra import viewSpectraStack

fig = viewSpectraStack(
    projData,
    "site1",
    "meas_2012-02-10_11-30-00",
    coherences=[["Ex", "Hy"], ["Ey", "Hx"]],
    plotoptions=plotOptions,
    show=False,
    save=False,
)
fig.savefig(imagePath / "viewSpec_projspec_128_stack")

# view the spectra at 4096 Hz
fig = viewSpectra(
    projData,
    "site1",
    "meas_2012-02-10_11-05-00",
    plotwindow="all",
    plotoptions=plotOptions,
    show=False,
    save=False,
)
fig.savefig(imagePath / "viewSpec_projspec_4096_view_plotall")

fig = viewSpectraSection(
    projData,
    "site1",
    "meas_2012-02-10_11-05-00",
    plotoptions=plotOptions,
    show=False,
    save=False,
)
fig.savefig(imagePath / "viewSpec_projspec_4096_view_section")

fig = viewSpectraStack(
    projData,
    "site1",
    "meas_2012-02-10_11-05-00",
    coherences=[["Ex", "Hy"], ["Ey", "Hx"]],
    plotoptions=plotOptions,
    show=True,
    save=True,
)
fig.savefig(imagePath / "viewSpec_projspec_4096_view_stack_coherences")
