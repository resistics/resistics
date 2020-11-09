"""A package for the processing of magnetotelluric data

resistics is a package for the robust processing of magnetotelluric data. It includes several features focussed on traceability and data investigation. For more information, visit the package website at:

www.resistics.io
"""

__name__ = "resistics"
# short X.Y version
xyversion = "0.0.7"
# release
release = ".dev1"
# combined version
__version__ = "{}{}".format(xyversion, release)


from resistics.common.log import (
    configure_default_logging,
    configure_warning_logging,
    configure_debug_logging,
)

# from resistics.project.io import loadProject, newProject