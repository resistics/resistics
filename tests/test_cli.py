"""Tests for the resistics terminal application launcher."""

from importlib.metadata import entry_points
from pathlib import Path

from resistics.tui import ResisticsTui, main


def test_launcher_accepts_an_optional_project_path(monkeypatch):
    import resistics.tui.app as tui_app

    paths = []
    monkeypatch.setattr(tui_app, "run_tui", paths.append)

    assert main([]) == 0
    assert main(["example/project"]) == 0
    assert paths == [None, Path("example/project")]


def test_package_facade_preserves_application_imports():
    import resistics.tui as tui
    import resistics.tui.app as tui_app

    assert tui.ResisticsTui is ResisticsTui is tui_app.ResisticsTui
    assert tui.main is main is tui_app.main
    assert Path(tui.__file__).name == "__init__.py"


def test_package_facade_preserves_extracted_screen_imports():
    import resistics.tui as tui
    import resistics.tui.app as tui_app
    from resistics.tui.screens.dialogs import (
        ConfirmJobScreen,
        ConfirmProjectDataDeletionScreen,
        CopyYamlFileScreen,
        CreateJobScreen,
        DeleteProjectDataScreen,
        DeleteYamlFileScreen,
        DirectoryPickerScreen,
        ProjectDataDeletionRequest,
    )
    from resistics.tui.screens.launcher import (
        CreateProjectScreen,
        HomeScreen,
        ProjectLoadingScreen,
        TuiHeader,
    )

    extracted = {
        "ConfirmJobScreen": ConfirmJobScreen,
        "ConfirmProjectDataDeletionScreen": ConfirmProjectDataDeletionScreen,
        "CopyYamlFileScreen": CopyYamlFileScreen,
        "CreateJobScreen": CreateJobScreen,
        "DeleteProjectDataScreen": DeleteProjectDataScreen,
        "DeleteYamlFileScreen": DeleteYamlFileScreen,
        "DirectoryPickerScreen": DirectoryPickerScreen,
        "ProjectDataDeletionRequest": ProjectDataDeletionRequest,
        "CreateProjectScreen": CreateProjectScreen,
        "HomeScreen": HomeScreen,
        "ProjectLoadingScreen": ProjectLoadingScreen,
        "TuiHeader": TuiHeader,
    }
    for name, screen_type in extracted.items():
        assert getattr(tui, name) is screen_type
        assert getattr(tui_app, name) is screen_type

    assert ConfirmJobScreen.__module__ == "resistics.tui.screens.dialogs"
    assert HomeScreen.__module__ == "resistics.tui.screens.launcher"


def test_console_entry_point_loads_the_package_facade():
    launcher = next(
        entry
        for entry in entry_points(group="console_scripts")
        if entry.name == "resistics"
    )

    assert launcher.value == "resistics.tui:main"
    assert launcher.load() is main


def test_launcher_rejects_more_than_one_argument(capsys):
    assert main(["tui", "example/project"]) == 2
    assert "Usage: resistics [PROJECT_PATH]" in capsys.readouterr().err
