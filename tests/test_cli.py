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
