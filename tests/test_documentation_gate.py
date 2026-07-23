"""Protect the permanent local documentation command."""

from pathlib import Path

import pytest

from scripts.check_documentation import (
    EXPECTED_PLOT_PREFIXES,
    OUTPUT_MARKER,
    _claim_output_root,
    _sphinx_command,
    _verify_plot_artifacts,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_GUIDES = {
    "contributing": (
        "uv sync --locked --all-groups",
        "python scripts/check_coverage.py",
        "pyrefly check",
        "cached_action_checks_are_fast",
        "(justified-suppressions)=",
    ),
    "docstrings": (
        "(docstring-content-checklist)=",
        "(private-helper-docstrings)=",
        "(fenced-docstring-directives)=",
        "Rich examples and plots belong with their documented objects",
    ),
    "releasing": (
        "uv lock --check",
        "uv build --no-sources",
        "Owner follow-ups: not active or verified",
        "Trusted Publishing",
    ),
}


def _write_expected_plots(output_dir: Path) -> Path:
    plot_dir = output_dir / "plot_directive"
    plot_dir.mkdir(parents=True)
    for prefix in EXPECTED_PLOT_PREFIXES:
        (plot_dir / f"{prefix}.png").touch()
        (plot_dir / f"{prefix}.hires.png").touch()
    return plot_dir


def test_html_command_enforces_nitpicky_warning_clean_build(tmp_path: Path) -> None:
    command = _sphinx_command("html", tmp_path)
    assert "-n" in command
    assert "-W" in command
    assert "--keep-going" in command
    assert command[-1] == str(tmp_path)


@pytest.mark.parametrize("builder", ["doctest", "linkcheck"])
def test_validation_builder_does_not_repeat_notebooks(
    tmp_path: Path, builder: str
) -> None:
    command = _sphinx_command(builder, tmp_path)
    assert command[command.index("-D") + 1] == "nb_execution_mode=off"


def test_expected_plot_artifacts_pass(tmp_path: Path) -> None:
    _write_expected_plots(tmp_path)
    _verify_plot_artifacts(tmp_path)


def test_output_root_requires_ownership_before_replacement(tmp_path: Path) -> None:
    (tmp_path / "html").mkdir()
    with pytest.raises(RuntimeError, match="Refusing to replace unowned"):
        _claim_output_root(tmp_path)

    (tmp_path / "html").rmdir()
    _claim_output_root(tmp_path)
    assert (tmp_path / OUTPUT_MARKER).is_file()


def test_permanent_contributor_guides_remain_complete_and_discoverable() -> None:
    docs_root = REPOSITORY_ROOT / "docs/source"
    index = (docs_root / "index.md").read_text(encoding="utf-8")
    for name, required_content in REQUIRED_GUIDES.items():
        source = (docs_root / f"{name}.md").read_text(encoding="utf-8")
        assert f"\n{name}\n" in index
        for content in required_content:
            assert content in source

    contributing = (REPOSITORY_ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    assert "docs/source/contributing.md" in contributing
    assert "docs/source/releasing.md" in contributing


@pytest.mark.parametrize("change", ["missing", "unexpected"])
def test_plot_artifact_drift_fails(tmp_path: Path, change: str) -> None:
    plot_dir = _write_expected_plots(tmp_path)
    if change == "missing":
        (plot_dir / f"{next(iter(EXPECTED_PLOT_PREFIXES))}.png").unlink()
    else:
        (plot_dir / "unowned.png").touch()

    with pytest.raises(RuntimeError, match="Unexpected documentation plot"):
        _verify_plot_artifacts(tmp_path)
