"""Tests for the resistics terminal application launcher."""

from pathlib import Path

from resistics.tui import main


def test_launcher_accepts_an_optional_project_path(monkeypatch):
    import resistics.tui

    paths = []
    monkeypatch.setattr(resistics.tui, "run_tui", paths.append)

    assert main([]) == 0
    assert main(["example/project"]) == 0
    assert paths == [None, Path("example/project")]


def test_launcher_rejects_more_than_one_argument(capsys):
    assert main(["tui", "example/project"]) == 2
    assert "Usage: resistics [PROJECT_PATH]" in capsys.readouterr().err
