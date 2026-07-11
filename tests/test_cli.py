"""Tests for the resistics command-line entry point."""

from pathlib import Path

from resistics.cli import build_parser, main


def test_tui_command_parses_project_path():
    args = build_parser().parse_args(["tui", "example/project"])
    assert args.command == "tui"
    assert args.project_path == Path("example/project")


def test_tui_command_reports_project_error(monkeypatch, capsys):
    import resistics.tui

    def fail(project_path):
        raise ValueError("not a project")

    monkeypatch.setattr(resistics.tui, "run_tui", fail)

    assert main(["tui", "missing"]) == 2
    assert "Unable to open project: not a project" in capsys.readouterr().err
