from datapaths import projectPath, imagePath, docPath
import shutil

# tutorial images
for image in imagePath.glob("*.png"):
    shutil.copy2(image, docPath)
# spectra comments
shutil.copy2(
    projectPath
    / "specData"
    / "site1"
    / "meas_2012-02-10_11-30-00"
    / "spectra"
    / "comments.txt",
    docPath / "viewSpec_comments.txt",
)
shutil.copy2(
    projectPath
    / "specData"
    / "site1"
    / "meas_2012-02-10_11-30-00"
    / "notch"
    / "comments.txt",
    docPath / "multspec_comments.txt",
)
# statistics
shutil.copy2(
    projectPath
    / "statData"
    / "site1"
    / "meas_2012-02-10_11-05-00"
    / "spectra"
    / "coherence"
    / "comments.txt",
    docPath / "usingStats_comments.txt",
)
# transfer functions
shutil.copy2(
    projectPath / "transFuncData" / "site1" / "128_000" / "site1_fs128_000_spectra",
    docPath,
)
shutil.copy2(
    projectPath
    / "transFuncData"
    / "site1"
    / "128_000"
    / "site1_fs128_000_spectra_with_Hz",
    docPath,
)
# copy config
shutil.copy2("tutorialconfig.ini", docPath)
shutil.copy2("multiconfig.ini", docPath)
shutil.copy2("multiconfigSeparate.ini", docPath)
shutil.copy2("usingWindowSelector.txt", docPath)
# copy the project file
shutil.copy2(projectPath / "mtProj.prj", docPath)

