from pathlib import Path

datapath = Path("E:/", "magnetotellurics", "code", "resisticsdata", "features")
projectPath = datapath / "featureProject"
statImagePath = Path("statImages")
remoteImagePath = Path("remoteImages")

# doc path
docPath = Path("..", "..", "docs", "source", "_static", "examples", "features")
docPathConfig = docPath / "config"
docPathComments = docPath / "comments"
docPathStats = docPath / "stats"
docPathRemote = docPath / "remotestats"