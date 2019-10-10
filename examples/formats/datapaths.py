from pathlib import Path

datapath = Path("E:/", "magnetotellurics", "code", "resisticsdata", "formats")
timePath = datapath / "timeData"
timeImages = Path("timeImages")
calPath = datapath / "calData"
calImages = Path("calImages")

# doc paths
docPath = Path("..", "..", "docs", "source", "_static", "examples", "formats")
docTime = docPath / "time"
docCal = docPath / "cal"