import os
from resistics.project.projectIO import newProject

# define the project path. The project will be created under this project path.
# If the path does not exist, it will be created
projectPath = os.path.join("remoteProject")
projData = newProject(projectPath, "2016-01-18 00:00:00")

# let's create 2 sites
# M6 is the location of interest. Remote is a remote reference station.
projData.createSite("M6")
projData.createSite("Remote")

