import os
from resistics.project.projectIO import loadProject

# define project path
projectPath = os.path.join("tutorialProject")
# load the project and print infomation
projData = loadProject(projectPath)
projData.printInfo()

# view the project timeline
fig = projData.view()
fig.savefig(os.path.join("tutorialProject", "images", "projectTimeline"))

# get site data
siteData = projData.getSiteData("site1")
siteData.printInfo()
fig = siteData.view()
fig.savefig(os.path.join("tutorialProject", "images", "siteTimeline"))