"""Private MTH5 handle contract, construction, and ownership lifecycle."""

from __future__ import annotations

from pathlib import Path
from types import TracebackType
from typing import Any, Literal, Protocol, Self, runtime_checkable

from loguru import logger


@runtime_checkable
class _MTH5Handle(Protocol):
    """Minimal live-handle contract owned by a project inspection source."""

    @property
    def channel_summary(self) -> Any:
        """Return the MTH5 channel-summary adapter."""
        ...

    @property
    def file_version(self) -> Any:
        """Return the MTH5 file format version."""
        ...

    def open_mth5(
        self,
        filename: str | Path | None = None,
        mode: str = "a",
        single_writer_multiple_reader: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Open the configured MTH5 file.

        Parameters
        ----------
        filename : str | Path | None
            File assigned to the handle, or its already configured path.
        mode : str
            HDF5 open mode.
        single_writer_multiple_reader : bool
            Whether to enable MTH5's SWMR behavior.
        **kwargs : Any
            Additional HDF5 open options.

        Returns
        -------
        Any
            Third-party MTH5 handle returned by its open operation.
        """
        ...

    def close_mth5(self) -> None:
        """Close the underlying HDF5 handle."""
        ...

    def h5_is_read(self) -> bool:
        """Return whether the HDF5 handle is open for reading.

        Returns
        -------
        bool
            ``True`` when live MTH5 objects may be accessed.
        """
        ...

    def get_survey(self, survey_name: str) -> Any:
        """Return one survey group.

        Parameters
        ----------
        survey_name : str
            Survey identifier.

        Returns
        -------
        Any
            Third-party MTH5 survey group.
        """
        ...

    def get_station(self, station_name: str, survey: str | None = None) -> Any:
        """Return one station group.

        Parameters
        ----------
        station_name : str
            Station identifier.
        survey : str | None
            Owning survey identifier.

        Returns
        -------
        Any
            Third-party MTH5 station group.
        """
        ...

    def get_run(
        self, station_name: str, run_name: str, survey: str | None = None
    ) -> Any:
        """Return one run group.

        Parameters
        ----------
        station_name : str
            Owning station identifier.
        run_name : str
            Run identifier.
        survey : str | None
            Owning survey identifier.

        Returns
        -------
        Any
            Third-party MTH5 run group.
        """
        ...

    def get_channel(
        self,
        station_name: str,
        run_name: str,
        channel_name: str,
        survey: str | None = None,
    ) -> Any:
        """Return one channel group.

        Parameters
        ----------
        station_name : str
            Owning station identifier.
        run_name : str
            Owning run identifier.
        channel_name : str
            Channel component identifier.
        survey : str | None
            Owning survey identifier.

        Returns
        -------
        Any
            Third-party MTH5 channel group.
        """
        ...


class _MTH5HandleOwner:
    """Own one MTH5 handle and expose deterministic resource semantics.

    Attributes
    ----------
    mth5_data : _MTH5Handle
        Live third-party handle owned by this inspection source.
    """

    mth5_data: _MTH5Handle

    @property
    def closed(self) -> bool:
        """Return whether the owned MTH5 handle has been closed.

        Returns
        -------
        bool
            ``True`` when live MTH5 groups and run data are inaccessible.
        """
        return not self.mth5_data.h5_is_read()

    def close(self) -> None:
        """Release the owned MTH5 handle, safely allowing repeated calls."""
        if not self.closed:
            self.mth5_data.close_mth5()

    def __enter__(self) -> Self:
        """Enter an open inspection-source context.

        Returns
        -------
        Self
            This source while its owned MTH5 handle is open.

        Raises
        ------
        RuntimeError
            If the source was already closed.
        """
        if self.closed:
            raise RuntimeError(f"{type(self).__name__} MTH5 handle is closed")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        """Close the source when its context exits.

        Parameters
        ----------
        exc_type : type[BaseException] | None
            Exception type raised inside the context, when present.
        exc_value : BaseException | None
            Exception raised inside the context, when present.
        traceback : TracebackType | None
            Traceback for the exception raised inside the context.

        Returns
        -------
        Literal[False]
            Always ``False`` so exceptions from the context propagate.
        """
        self.close()
        return False

    def _require_open(self) -> None:
        """Require a live MTH5 handle for a group or sample-data operation.

        Raises
        ------
        RuntimeError
            If this source has already released its owned handle.
        """
        if self.closed:
            raise RuntimeError(f"{type(self).__name__} MTH5 handle is closed")


def _open_read_only_mth5(mth5_path: Path) -> _MTH5Handle:
    """Open one MTH5 handle and clean up a partially failed open.

    Parameters
    ----------
    mth5_path : Path
        Existing MTH5 file to open read-only.

    Returns
    -------
    _MTH5Handle
        Open handle whose ownership must transfer to a public source.

    Raises
    ------
    Exception
        If MTH5 cannot open the file. Any partially opened handle is closed
        before the original exception is re-raised.
    """
    mth5_data = _new_mth5(mth5_path)
    try:
        mth5_data.open_mth5(mode="r")
    # MTH5/HDF5 can surface codec, validation, and filesystem errors here. The
    # handle must be released for all of them while preserving the root cause.
    except Exception:
        _close_failed_mth5(mth5_data, mth5_path)
        raise
    return mth5_data


def _new_mth5(mth5_path: Path) -> _MTH5Handle:
    """Construct the third-party handle only at the file-open boundary.

    Parameters
    ----------
    mth5_path : Path
        MTH5 file assigned to the new handle.

    Returns
    -------
    _MTH5Handle
        Unopened third-party MTH5 handle.
    """
    from mth5.mth5 import MTH5

    return MTH5(mth5_path)


def _close_failed_mth5(mth5_data: _MTH5Handle, mth5_path: Path) -> None:
    """Release a handle after failed construction without masking its error.

    Parameters
    ----------
    mth5_data : _MTH5Handle
        Handle whose ownership did not transfer to a public source.
    mth5_path : Path
        File path used if close itself needs to be diagnosed.
    """
    try:
        if mth5_data.h5_is_read():
            mth5_data.close_mth5()
    except Exception:
        logger.exception(f"Unable to close MTH5 handle after failure: {mth5_path}")
