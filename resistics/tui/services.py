"""UI-neutral helpers shared by the terminal application screens."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from textual.screen import Screen
from textual.widget import Widget

if TYPE_CHECKING:
    from resistics.tui.app import ResisticsTui


def _feature_error(feature: str, error: Exception) -> str:
    """Return actionable detail for a lazily imported feature failure.

    Parameters
    ----------
    feature : str
        User-facing feature name.
    error : Exception
        Import or runtime failure raised at the feature boundary.

    Returns
    -------
    str
        Error detail that preserves ordinary failures and explains how to
        recover from a missing required dependency.
    """
    if isinstance(error, ModuleNotFoundError):
        dependency = error.name or "unknown"
        return (
            f"{feature} requires the missing dependency {dependency!r}. "
            "Reinstall resistics with its required dependencies."
        )
    return str(error)


def _focus_relative(
    controls: Sequence[Widget],
    focused: Widget | None,
    increment: int,
    *,
    move_from_unfocused: bool = True,
) -> None:
    """Move focus within a bounded group of controls.

    Parameters
    ----------
    controls : Sequence[Widget]
        Ordered focusable widgets.
    focused : Widget | None
        Currently focused widget, if any.
    increment : int
        Relative movement through the group.
    move_from_unfocused : bool
        Whether to use the first control as the fallback position.
    """
    if not controls:
        return
    if focused is None or focused not in controls:
        if not move_from_unfocused:
            return
        index = 0
    else:
        index = controls.index(focused)
    controls[(index + increment) % len(controls)].focus()


def _resistics_app(screen: Screen[None]) -> ResisticsTui:
    """Return the application contract required by project screens.

    Parameters
    ----------
    screen : Screen[None]
        Project screen mounted by the application.

    Returns
    -------
    ResisticsTui
        Owning resistics application.

    Raises
    ------
    RuntimeError
        If the screen is mounted by an incompatible Textual application.
    """
    from resistics.tui.app import ResisticsTui

    app = screen.app
    if not isinstance(app, ResisticsTui):
        raise RuntimeError("Project screens require ResisticsTui")
    return app
