from datapaths import projectPath, imagePath, docPath
import shutil

for image in imagePath.glob("*.png"):
    shutil.copy2(image, docPath)
# configuration file
shutil.copy2("customconfig.ini", docPath)


