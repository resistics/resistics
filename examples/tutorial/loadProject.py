from datapaths import projectPath, imagePath
from resistics.project.io import loadProject

# load the project and print infomation
projData = loadProject(projectPath)
projData.printInfo()

# view the project timeline
fig = projData.view()
fig.savefig(imagePath / "projectTimeline")

# get site data
siteData = projData.getSiteData("site1")
siteData.printInfo()
fig = siteData.view()
fig.savefig(imagePath / "siteTimeline")