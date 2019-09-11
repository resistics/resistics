from pathlib import Path
from resistics.project.projectIO import newProject

projectPath = Path("preprocessProject")
refTime = "2012-02-10 00:00:00"
newProject(projectPath, refTime)
