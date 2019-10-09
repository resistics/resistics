from datapaths import projectPath, imagePath, docPath
import shutil

# tutorial images
for image in imagePath.glob("*.png"):
    shutil.copy2(image, docPath)
# copy config
shutil.copy2("tutorialConfig.ini", docPath)
shutil.copy2("usingWindowSelector.txt", docPath)
# copy the project file
shutil.copy2(projectPath / "mtProj.prj", docPath)

