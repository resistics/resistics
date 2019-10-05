from datapaths import projectPath
from resistics.project.io import newProject

referenceTime = "2019-05-25 12:00:00"
proj = newProject(projectPath, referenceTime)
proj.printInfo()