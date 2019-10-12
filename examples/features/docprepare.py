import shutil
from datapaths import (
    projectPath,
    imagePath,
    docPathConfig,
    docPathComments,
    docPathStats,
    docPathRemote,
)


# configuration files
shutil.copy2("resisticsDefaultConfig.ini", docPathConfig)
shutil.copy2("exampleConfig1.ini", docPathConfig)
shutil.copy2("exampleConfig2.ini", docPathConfig)

# comments
timeCommentsPath = (
    projectPath / "timeData" / "Remote_M1" / "run4_2016-03-24_02-35-00" / "comments.txt"
)
shutil.copy2(timeCommentsPath, docPathComments / "features_time_comments.txt")
spectraCommentsPath = (
    projectPath
    / "specData"
    / "Remote_M1"
    / "run4_2016-03-24_02-35-00"
    / "spectra"
    / "comments.txt"
)
shutil.copy2(spectraCommentsPath, docPathComments / "features_spec_comments.txt")
statCommentsPath = (
    projectPath
    / "statData"
    / "Remote_M1"
    / "run4_2016-03-24_02-35-00"
    / "spectra"
    / "partialCoherence"
    / "comments.txt"
)
shutil.copy2(statCommentsPath, docPathComments / "features_stat_comments.txt")

# spectra images

