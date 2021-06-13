"""
Classes and methods to enable a resistics project

A project is an essential element of a resistics environment together with a
configuration.

In particular, this module includes the core Project, Site and Measurement
clasess and some supporting functions.
"""
from loguru import logger
from typing import Iterator, Optional, List, Dict
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from resistics.common import ResisticsModel, WriteableMetadata
from resistics.sampling import HighResDateTime
from resistics.time import TimeMetadata
from resistics.time import TimeReader


PROJ_FILE = "resistics.json"
PROJ_DIRS = {
    "time": "time",
    "calibration": "calibrate",
    "spectra": "spectra",
    "evals": "evals",
    "features": "features",
    "masks": "masks",
    "results": "results",
    "images": "images",
}


def get_calibration_path(proj_dir: Path) -> Path:
    """Get the path to the calibration data"""
    return proj_dir / PROJ_DIRS["calibration"]


def get_meas_time_path(proj_dir: Path, site_name: str, meas_name: str) -> Path:
    """Get path to measurement time data"""
    return proj_dir / PROJ_DIRS["time"] / site_name / meas_name


def get_meas_spectra_path(
    proj_dir: Path, site_name: str, meas_name: str, config_name: str
) -> Path:
    """Get path to measurement spectra data"""
    return proj_dir / PROJ_DIRS["spectra"] / site_name / config_name / meas_name


def get_meas_evals_path(
    proj_dir: Path, site_name: str, meas_name: str, config_name: str
) -> Path:
    """Get path to measurement evaluation frequency spectra data"""
    return proj_dir / PROJ_DIRS["evals"] / site_name / config_name / meas_name


def get_meas_features_path(
    proj_dir: Path, site_name: str, meas_name: str, config_name: str
) -> Path:
    """Get path to measurement features data"""
    return proj_dir / PROJ_DIRS["features"] / site_name / config_name / meas_name


def get_mask_path(proj_dir: Path, site_name: str, config_name: str) -> Path:
    """Get path to mask data"""
    return proj_dir / PROJ_DIRS["masks"] / site_name / config_name


def get_mask_name(fs: float, mask_name: str) -> str:
    """Get the name of a mask file"""
    from resistics.common import fs_to_string

    return f"{fs_to_string(fs)}_{mask_name}.dat"


def get_results_path(proj_dir: Path, site_name: str, config_name: str) -> Path:
    """Get path to solutions"""
    return proj_dir / PROJ_DIRS["results"] / site_name / config_name


def get_solution_name(
    fs: float, tf_name: str, tf_var: str, postfix: Optional[str] = None
) -> str:
    """Get the name of a solution file"""
    from resistics.common import fs_to_string

    solution_name = f"{fs_to_string(fs)}_{tf_name.lower()}"
    if tf_var != "":
        tf_var = tf_var.replace(" ", "_")
        solution_name = solution_name + f"_{tf_var}"
    if postfix is None:
        return solution_name + ".json"
    return solution_name + "_" + postfix + ".json"


class Measurement(ResisticsModel):
    """
    Class for interfacing with a measurement

    The class holds the original time series metadata and can provide
    information about other types of data
    """

    site_name: str
    dir_path: Path
    metadata: TimeMetadata
    reader: TimeReader

    @property
    def name(self) -> str:
        """Get the name of the measurement"""
        return self.dir_path.name


class Site(ResisticsModel):
    """
    Class for describing Sites

    .. note::

        This should essentially describe a single instrument setup. If the same
        site is re-occupied later with a different instrument setup, it is
        suggested to split this into a different site.
    """

    dir_path: Path
    measurements: Dict[str, Measurement]
    begin_time: HighResDateTime
    end_time: HighResDateTime

    def __iter__(self) -> Iterator:
        """Iterator over measurements"""
        return self.measurements.values().__iter__()

    def __getitem__(self, meas_name: str) -> Measurement:
        """Get a measurement"""
        return self.get_measurement(meas_name)

    @property
    def name(self) -> str:
        """The Site name"""
        return self.dir_path.name

    @property
    def n_meas(self) -> int:
        """Get the number of measurements"""
        return len(self.measurements)

    def fs(self) -> List[float]:
        """Get the sampling frequencies in the Site"""
        fs = [x.metadata.fs for x in self.measurements.values()]
        return sorted(list(set(fs)))

    def get_measurement(self, meas_name: str) -> Measurement:
        """Get a measurement"""
        from resistics.errors import MeasurementNotFoundError

        if meas_name not in self.measurements:
            raise MeasurementNotFoundError(self.name, meas_name)
        return self.measurements[meas_name]

    def get_measurements(self, fs: Optional[float] = None) -> Dict[str, Measurement]:
        """Get dictionary of measurements with optional filter by sampling frequency"""
        if fs is None:
            return self.measurements
        return {
            name: meas
            for name, meas in self.measurements.items()
            if meas.metadata.fs == fs
        }

    def plot(self) -> go.Figure:
        """Plot the site timeline"""
        df = self.to_dataframe()
        if len(df.index) == 0:
            logger.error("No measurements found to plot")
            return
        fig = px.timeline(
            df,
            x_start="first_time",
            x_end="last_time",
            y="name",
            color="fs",
            title=self.name,
        )
        return fig

    def to_dataframe(self) -> pd.DataFrame:
        """
        Get measurements list in a pandas DataFrame

        .. note::

            Measurement first and last times are converted to pandas Timestamps
            as these are more universally useful in a pandas DataFrame. However,
            this may result in a loss of precision, especially at high sampling
            frequencies.

        Returns
        -------
        pd.DataFrame
            Site measurement DataFrame
        """
        data = [
            [
                x.name,
                x.metadata.fs,
                x.metadata.first_time.isoformat(),
                x.metadata.last_time.isoformat(),
            ]
            for x in self.measurements.values()
        ]
        df = pd.DataFrame(data=data, columns=["name", "fs", "first_time", "last_time"])
        df["first_time"] = pd.to_datetime(df["first_time"])
        df["last_time"] = pd.to_datetime(df["last_time"])
        df["site"] = self.name
        return df


class ProjectMetadata(WriteableMetadata):
    """Project metadata"""

    ref_time: HighResDateTime
    location: str = ""
    country: str = ""
    year: int = -999
    description: str = ""
    contributors: List[str] = []


class Project(ResisticsModel):
    """
    Class to describe a resistics project

    The resistics Project Class connects all resistics data. It is an essential
    part of processing data with resistics.

    Resistics projects are in directory with several sub-directories. Project
    metadata is saved in the resistics.json file at the top level directory.
    """

    dir_path: Path
    begin_time: HighResDateTime
    end_time: HighResDateTime
    metadata: ProjectMetadata
    sites: Dict[str, Site] = {}

    def __iter__(self) -> Iterator:
        """Iterator over sites"""
        return self.sites.values().__iter__()

    def __getitem__(self, site_name: str) -> Site:
        """Get a Site instance given the name of the Site"""
        return self.get_site(site_name)

    @property
    def n_sites(self) -> int:
        """The number of sites"""
        return len(self.sites)

    def fs(self) -> List[float]:
        """Get sampling frequencies in the Project"""
        fs = set()
        for site in self.sites.values():
            fs = fs.union(set(site.fs()))
        return sorted(list(fs))

    def get_site(self, site_name: str) -> Site:
        """Get a Site object given the Site name"""
        from resistics.errors import SiteNotFoundError

        if site_name not in self.sites:
            raise SiteNotFoundError(site_name)
        return self.sites[site_name]

    def get_sites(self, fs: Optional[float] = None) -> Dict[str, Site]:
        """
        Get sites

        Parameters
        ----------
        fs : Optional[float], optional
            Filter by sites which have at least a single recording at a
            specified sampling frequency, by default None

        Returns
        -------
        Dict[str, Site]
            Dictionary of site name to Site
        """
        if fs is None:
            return self.sites
        return {name: site for name, site in self.sites.items() if fs in site.fs()}

    def get_concurrent(self, site_name: str) -> List[str]:
        """
        Find sites that recorded conscurrently to a specified site

        Parameters
        ----------
        site_name : str
            Search for sites recording concurrently to this site

        Returns
        -------
        List[str]
            List of site names which were recording concurrently
        """
        site_start = self.sites[site_name].begin_time
        site_end = self.sites[site_name].end_time
        concurrent = []
        for site in self.sites.values():
            if site.name == site_name:
                continue
            if site.end_time < site_start:
                continue
            if site.begin_time > site_end:
                continue
            concurrent.append(site)
        return concurrent

    def plot(self) -> go.Figure:
        """Plot a timeline of the project"""
        df = self.to_dataframe()
        df["Site"] = df["site"]
        df["Sampling frequency, Hz"] = df["fs"].values.astype(str)
        if len(df.index) == 0:
            logger.error("No measurements found to plot")
            return
        fig = px.timeline(
            df,
            x_start="first_time",
            x_end="last_time",
            y="Site",
            color="Sampling frequency, Hz",
            title=str(self.dir_path),
        )
        return fig

    def to_dataframe(self) -> pd.DataFrame:
        """Detail Project recordings in a DataFrame"""
        df = pd.DataFrame(columns=["name", "first_time", "last_time", "fs", "site"])
        for site in self.sites.values():
            df = df.append(site.to_dataframe())
        return df
