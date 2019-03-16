import os
from resistics.project.projectIO import loadProject

# define project path
projectPath = os.path.join("tutorialProject")
# load the project and print infomation
projData = loadProject(projectPath)
projData.printInfo()

# view the project timeline
projData.view()

# get site data
siteData = projData.getSiteData("site1")
siteData.printInfo()
siteData.view()