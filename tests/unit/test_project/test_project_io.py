def test_newProject() -> None:
    """Test creating a new project"""
    from datapaths import path_project_new
    from resistics.project.io import newProject

    proj = newProject(path_project_new, "2020-01-01 12:00:00")
    assert proj.refTime.strftime("%Y-%m-%d %H:%M:%S.%f") == "2020-01-01 12:00:00.000000"
    assert proj.sites == []


def test_loadProject() -> None:
    """Test loading the project"""
    from datapaths import path_integrated_singlesite
    from resistics.project.io import loadProject

    proj = loadProject(path_integrated_singlesite)
    assert proj.refTime.strftime("%Y-%m-%d %H:%M:%S.%f") == "2016-02-18 12:00:00.000000"
    assert (
        proj.projStart.strftime("%Y-%m-%d %H:%M:%S.%f") == "2016-02-19 11:22:57.000000"
    )
    assert proj.projEnd.strftime("%Y-%m-%d %H:%M:%S.%f") == "2016-02-25 02:29:59.999756"
    assert proj.sites == ["M7_4096"]
