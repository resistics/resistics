def test_getTimeData() -> None:
    """Test getting time data"""
    from datapaths import path_integrated_singlesite
    from resistics.project.io import loadProject
    from resistics.project.io import loadProject
    from resistics.project.time import getTimeReader
    from datetime import datetime

    proj = loadProject(path_integrated_singlesite)
    reader = getTimeReader(proj, "M7_4096", "meas_2016-02-22_02-00-00")
    assert reader.getStartDatetime() == datetime(2016, 2, 22, 2, 00, 00, 000000)
    assert reader.getStopDatetime() == datetime(2016, 2, 22, 2, 29, 59, 999756)
    assert reader.getNumSamples() == 7372800
    assert reader.getSampleFreq() == 4096
    assert reader.getChannels() == ["Ex", "Ey", "Hx", "Hy", "Hz"]