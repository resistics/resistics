"""
A package for the processing of magnetotelluric data

Resistics is a package for the robust processing of magnetotelluric data. It
includes several features focussed on traceability and data investigation. For
more information, visit the package website at: www.resistics.io
"""
from importlib.metadata import version, PackageNotFoundError

__name__ = "resistics"
try:
    __version__ = version(__name__)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"


from resistics.letsgo import new, load  # noqa: F401
