from datapaths import *
import shutil

# preprocess images
for image in preprocessImages.glob("*.png"):
    shutil.copy2(image, preprocessDoc)
# gapfill comments
gapfillPath = (
    preprocessPath / "timeData" / "site1_filled" / "meas_2012-02-10_11-05-00_filled"
)
shutil.copy2(gapfillPath / "comments.txt", preprocessDoc / "gapfill_comments.txt")

# advanced images
for image in remoteImages.glob("*.png"):
    shutil.copy2(image, remoteDoc)
# configuration files
shutil.copy2("remoteConfig.ini", remoteDoc)
shutil.copy2("manualWindowsConfig.ini", remoteDoc)

# intersite images
for image in intersiteImages.glob("*.png"):
    shutil.copy2(image, intersiteDoc)
# configuration files
shutil.copy2("customconfig.ini", intersiteDoc)

