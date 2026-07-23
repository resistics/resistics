"""Protect the permanent combined source and documentation coverage command."""

from pathlib import Path

from scripts.check_coverage import _coverage_commands


def test_coverage_gate_uses_tests_and_authoritative_sphinx_doctests(
    tmp_path: Path,
) -> None:
    commands = _coverage_commands(tmp_path)
    pytest_command = commands[1]
    sphinx_command = commands[2]

    assert pytest_command[-2:] == ["-m", "not performance"]
    assert "--append" in sphinx_command
    assert "sphinx" in sphinx_command
    assert "doctest" in sphinx_command
    assert "nb_execution_mode=off" in sphinx_command
    assert commands[-3][-1] == "report"
    assert commands[-2][-1] == "html"
    assert commands[-1][-1] == "xml"
