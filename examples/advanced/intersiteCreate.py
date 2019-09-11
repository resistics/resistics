from pathlib import Path
from resistics.project.projectIO import newProject

projectPath = Path("intersiteProject")
refTime = "2019-05-27 00:00:00"
newProject(projectPath, refTime)
