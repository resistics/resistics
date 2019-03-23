import os
from resistics.project.projectIO import newProject

# define project path
projectPath = os.path.join("tutorialProject")
# define reference time for project
referenceTime = "2012-02-10 00:00:00"
# create a new project and print infomation
projData = newProject(projectPath, referenceTime)
projData.printInfo()
# create a new site
projData.createSite("site1")
projData.printInfo()

