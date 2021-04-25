"""
Classes and methods for making, loading and using resistics projects

The resistics project connects all resistics types of data. It is an essential
part of processing data with resistics.

Resistics projects are in directory with several sub-directories. Project
metadata is saved in the resistics.json file at the top level directory.
"""
from loguru import logger
from typing import Iterator, Union, Optional, List, Dict, Any
from pathlib import Path
from numbers import Number
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from resistics.common import ResisticsData, ResisticsProcess, ProcessHistory
from resistics.common import Metadata
from resistics.sampling import RSDateTime
from resistics.time import TimeReader


PROJ_FILE = "resistics.json"
PROJ_DIRS = {
    "time_dir": "time",
    "calibrate_dir": "calibrate",
    "spectra_dir": "spectra",
    "features_dir": "features",
    "masks_dir": "masks",
    "results_dir": "results",
    "images_dir": "images",
}

info_metadata_specs = {
    "describes": {"type": str, "default": "project"},
    "location": {"type": str, "default": ""},
    "country": {"type": str, "default": ""},
    "year": {"type": int, "default": -999},
    "summary": {"type": str, "default": ""},
    "reference_time": {"type": RSDateTime, "default": None},
    "contributors": {"type": list, "default": []},
}


def get_project_metadata(info_metadata: Dict[str, Any]) -> Metadata:
    """
    Get project metadata

    Parameters
    ----------
    info_metadata : Dict[str, Any]
        Values for project information metadata

    Returns
    -------
    Metadata
        A project metadata
    """
    return Metadata(info_metadata, info_metadata_specs)


def meas_time_path(time_dir: Path, site_name: str, meas_name: str) -> Path:
    """Get path to measurement time data"""
    return time_dir / site_name / meas_name


def meas_spectra_path(
    spectra_dir: Path, site_name: str, meas_name: str, run: str
) -> Path:
    """Get path to measurement spectra data"""
    return spectra_dir / site_name / meas_name / run


def meas_features_path(
    features_dir: Path, site_name: str, meas_name: str, run: str
) -> Path:
    """Get path to measurement features data"""
    return features_dir / site_name / meas_name / run


def fs_mask_path(mask_dir: Path, site_name: str, run: str, fs: float) -> Path:
    """Get path to sampling frequency mask data"""
    from resistics.common import fs_to_string

    return mask_dir / site_name / run / f"{fs_to_string(fs)}.pkl"


def fs_results_path(results_dir: Path, site_name: str, run: str, fs: float) -> Path:
    """Get path to sampling frequency results"""
    from resistics.common import fs_to_string

    return results_dir / site_name / run / f"{fs_to_string(fs)}.json"


class Measurement(ResisticsData):
    """
    Class for describing a measurement

    This is a lightweight wrapper around a TimeReader which has read the time
    metadata
    """

    def __init__(self, meas_dir: Path, reader: TimeReader):
        """
        Initialise

        Parameters
        ----------
        meas_dir : Path
            The measurement directory
        reader : TimeReader
            A TimeReader
        """
        self.meas_dir = meas_dir
        self.reader = reader

    @property
    def name(self) -> str:
        """
        Get the name of the measurement

        Returns
        -------
        str
            Name of measurement
        """
        return self.meas_dir.name

    @property
    def fs(self) -> float:
        """
        Get sampling frequency of measurement

        Returns
        -------
        float
            Sampling frequency Hz
        """
        return self.reader.metadata["common", "fs"]

    @property
    def first_time(self) -> RSDateTime:
        """
        Get the first time of the recording

        Returns
        -------
        RSDateTime
            First time of recording
        """
        return self.reader.metadata["common", "first_time"]

    @property
    def last_time(self) -> RSDateTime:
        """
        Get the last time of the recording

        Returns
        -------
        RSDateTime
            Last time of recording
        """
        return self.reader.metadata["common", "last_time"]

    def to_string(self) -> str:
        """
        Class details as string

        Returns
        -------
        str
            Class information
        """
        outstr = f"Measurement '{self.name}'\n"
        outstr += f"Sampling frequency [Hz] = {self.fs}\n"
        outstr += f"First sample time = {str(self.first_time)}\n"
        outstr += f"Last sample time = {str(self.last_time)}"
        return outstr


class Site(ResisticsData):
    """
    Class for describing Sites

    This should essentially describe a single instrument setup
    """

    def __init__(
        self,
        site_dir: Path,
        measurements: Dict[str, Measurement],
    ) -> None:
        """
        Initialise Site

        Parameters
        ----------
        site_dir : Path
            The site directory
        measurements : Dict[str, Measurement]
            A list of time series measurements
        """
        from resistics.sampling import to_datetime

        self.site_dir = site_dir
        self.measurements = measurements
        if len(self.measurements) > 0:
            self.start = min([x.first_time for x in self.measurements.values()])
            self.end = max([x.last_time for x in self.measurements.values()])
        else:
            self.start = to_datetime(pd.Timestamp.utcnow())
            self.end = to_datetime(pd.Timestamp.utcnow())
        self.spectra = []
        self.features = []
        self.masks = []
        self.results = []

    def __iter__(self) -> Iterator:
        """Iterator over measurements"""
        return self.measurements.values().__iter__()

    def __getitem__(self, meas_name: str) -> Measurement:
        """
        Get a measurement

        Parameters
        ----------
        meas_name : str
            Measurement name

        Returns
        -------
        Measurement
            The Measurement
        """
        return self.get_measurement(meas_name)

    @property
    def name(self) -> str:
        """
        Site name

        Returns
        -------
        str
            The site name
        """
        return self.site_dir.name

    @property
    def n_meas(self) -> int:
        """
        Get the number of measurements

        Returns
        -------
        int
            Number of measurements
        """
        return len(self.measurements)

    def fs(self) -> List[float]:
        """
        Get the sampling frequencies in the Site

        Returns
        -------
        List[float]
            Sorted list of sampling frequencies
        """
        fs = [x.fs for x in self.measurements.values()]
        return sorted(list(set(fs)))

    def get_measurement(self, meas_name: str) -> Measurement:
        """
        Get a measurement

        Parameters
        ----------
        meas_name : str
            Measurement name

        Returns
        -------
        Measurement
            The Measurement

        Raises
        ------
        MeasurementNotFoundError
            If measurement not found
        """
        from resistics.errors import MeasurementNotFoundError

        if meas_name not in self.measurements:
            raise MeasurementNotFoundError(self.name, meas_name)
        return self.measurements[meas_name]

    def get_measurements(self, fs: Optional[Number] = None) -> Dict[str, Measurement]:
        """
        Get measurements

        Parameters
        ----------
        fs : Optional[Number], optional
            Filter measurements by a sampling frequency, by default None

        Returns
        -------
        Dict[str, Measurement]
            Dictionary with measurement name and matching Measurement
        """
        if fs is None:
            return self.measurements
        return {name: meas for name, meas in self.measurements.items() if meas.fs == fs}

    def plot(self) -> go.Figure:
        """
        Plot the site timeline

        Returns
        -------
        go.Figure
            Plotly figure
        """
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
            as these are more universally useful in a pandas DataFrame

        Returns
        -------
        pd.DataFrame
            Site measurement DataFrame
        """
        data = [
            [x.name, x.first_time.isoformat(), x.last_time.isoformat(), x.fs, self.name]
            for x in self.measurements.values()
        ]
        df = pd.DataFrame(
            data=data, columns=["name", "first_time", "last_time", "fs", "site"]
        )
        df["first_time"] = pd.to_datetime(df["first_time"])
        df["last_time"] = pd.to_datetime(df["last_time"])
        return df

    def to_string(self) -> str:
        """
        Site information as string

        Returns
        -------
        str
            Site info
        """
        from resistics.common import list_to_string

        outstr = f"Site {self.name}\n"
        outstr += f"{self.n_meas} Measurement(s): {list(self.measurements.keys())}\n"
        outstr += f"Sampling frequencies [Hz]: {list_to_string(self.fs())}\n"
        outstr += f"Recording start time: {str(self.start)}\n"
        outstr += f"Recording end time: {str(self.end)}"
        return outstr


class Project(ResisticsData):
    """
    Class to describe a resistics project
    """

    def __init__(
        self,
        proj_dir: Path,
        metadata: Metadata,
        sites: Dict[str, Site],
        setup: Dict[str, Any],
        history: ProcessHistory,
    ):
        """
        Project data object holding information about a resistics project

        Parameters
        ----------
        proj_dir : Path
            The project directory
        metadata : Metadata
            The project metadata
        sites : Dict[str, Site]
            Sites in the project
        setup : Dict[str, Any]
            Setup for resistics
        history : ProcessHistory
            Processing history for the project
        """
        from resistics.sampling import to_datetime

        self.proj_dir = proj_dir
        self.metadata = metadata
        self.sites = sites
        self.setup = setup
        self.history = history
        if len(self.sites) > 0:
            self.start = min([x.start for x in self.sites.values()])
            self.end = max([x.end for x in self.sites.values()])
        else:
            self.start = to_datetime(pd.Timestamp.utcnow())
            self.end = to_datetime(pd.Timestamp.utcnow())

    def __iter__(self) -> Iterator:
        """Iterator over sites"""
        return self.sites.values().__iter__()

    def __getitem__(self, site_name: str) -> Site:
        """
        Get a Site object given the name of a site

        Parameters
        ----------
        site_name : str
            The name

        Returns
        -------
        Site
            The matching Site object
        """
        return self.get_site(site_name)

    @property
    def ref_time(self) -> RSDateTime:
        """
        Get project reference time

        Returns
        -------
        RSDateTime
            Project reference time
        """
        return self.metadata["reference_time"]

    @property
    def n_sites(self) -> int:
        """
        Get the number of sites

        Returns
        -------
        int
            Number of sites
        """
        return len(self.sites)

    def fs(self) -> List[float]:
        """
        Get sampling frequencies in the project

        Returns
        -------
        List[float]
            The sampling frequencies in Hz
        """
        fs = set()
        for site in self.sites.values():
            fs = fs.union(set(site.fs()))
        return sorted(list(fs))

    def get_site(self, site_name: str) -> Site:
        """
        Get a Site object given a site name

        Parameters
        ----------
        site_name : str
            The name

        Returns
        -------
        Site
            The matching Site object

        Raises
        ------
        SiteNotFoundError
            If the site name does not exist
        """
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
        site_start = self.sites[site_name].start
        site_end = self.sites[site_name].end
        concurrent = []
        for site in self.sites.values():
            if site.name == site_name:
                continue
            if site.end < site_start:
                continue
            if site.start > site_end:
                continue
            concurrent.append(site)
        return concurrent

    def to_dataframe(self) -> pd.DataFrame:
        """
        Detail project recordings in a DataFrame

        Returns
        -------
        pd.DataFrame
            Recordings listed in a DataFrame
        """
        df = pd.DataFrame(columns=["name", "first_time", "last_time", "fs", "site"])
        for site in self.sites.values():
            df = df.append(site.to_dataframe())
        return df

    def plot(self) -> go.Figure:
        """
        Plot a timeline of the project

        Returns
        -------
        go.Figure
            Timeline of project
        """
        df = self.to_dataframe()
        if len(df.index) == 0:
            logger.error("No measurements found to plot")
            return
        fig = px.timeline(
            df,
            x_start="first_time",
            x_end="last_time",
            y="site",
            color="fs",
            title=str(self.proj_dir),
        )
        return fig

    def to_string(self) -> str:
        """
        Class details in a string

        Returns
        -------
        str
            Class details as string
        """
        from resistics.common import list_to_string

        outstr = f"{self.type_to_string()}\n"
        outstr += f"Project: {self.proj_dir}\n"
        outstr += f"Start of recording: {str(self.start)}\n"
        outstr += f"End of recording: {str(self.end)}\n"
        outstr += f"Sampling frequencies: {list_to_string(self.fs())}\n"
        outstr += f"Sites: {list(self.sites.keys())}\n"
        for key in self.metadata.keys(describes=False):
            outstr += f"{key.capitalize()}: {str(self.metadata[key])}\n"
        outstr = outstr.rstrip("\n")
        return outstr


class ProjectCreator(ResisticsProcess):
    """
    Process to create a project
    """

    def __init__(self, proj_dir: Path, proj_info: Dict[str, Any]):
        """
        Create a project

        Parameters
        ----------
        proj_dir : Path
            The project directory
        proj_info : Dict[str, Any]
            The project info
        """
        self.proj_dir = proj_dir
        self.proj_path = self.proj_dir / PROJ_FILE
        self.proj_info = proj_info

    def check(self) -> bool:
        """
        Checks before creating a project

        Returns
        -------
        bool
            True if check passed, else False
        """
        if self._is_existing():
            logger.error("Existing project found, use load.")
            return False
        return True

    def run(self):
        """
        Create the project

        This should be run after a successful check

        Raises
        ------
        ProjectCreateError
            If an existing project found
        """
        from resistics.errors import ProjectCreateError
        from resistics.common import assert_dir, metadata_to_json

        if self._is_existing():
            raise ProjectCreateError(
                self.proj_dir, "Existing project found, try loading"
            )
        elif self.proj_dir.exists():
            logger.warning("Directory already exists, the project will be saved here")
            assert_dir(self.proj_dir)
        else:
            logger.info("Making directory for project")
            self.proj_dir.mkdir(parents=True)

        self._make_subdirs()
        metadata = get_project_metadata(self.proj_info)
        metadata_to_json(metadata, self.proj_path)
        logger.info(f"Project created in {self.proj_dir}")

    def _is_existing(self) -> bool:
        """
        Check if existing project found in project directory

        Returns
        -------
        bool
            True if existing project found
        """
        from resistics.common import is_dir

        if self.proj_dir.exists() and is_dir(self.proj_dir) and self.proj_path.exists():
            return True
        return False

    def _make_subdirs(self):
        """Make project subdirectories"""
        for subdir in PROJ_DIRS.values():
            subdir_path = self.proj_dir / subdir
            if not subdir_path.exists():
                logger.info("Making project subdirectory {subdir_path}")
                subdir_path.mkdir()


class ProjectLoader(ResisticsProcess):
    """
    Project loader
    """

    def __init__(self, proj_dir: Path):
        """
        Load a project

        Parameters
        ----------
        proj_dir : Path
            Project directory
        """
        self.proj_dir = proj_dir
        self.proj_path = proj_dir / PROJ_FILE

    def check(self) -> bool:
        """
        Perform checks before loading

        Returns
        -------
        bool
            True if checks passed, else False
        """
        from resistics.common import assert_dir

        assert_dir(self.proj_dir)
        if not self.proj_path.exists():
            logger.error(f"Resistics project file {self.proj_path} not found")
            return False
        return True

    def run(self, setup: Dict[str, Any]) -> Project:
        """
        Load a project

        Parameters
        ----------
        setup : Dict[str, Any]
            A setup for the project

        Returns
        -------
        Project
            Project instance
        """
        from resistics.common import dir_subdirs, json_to_metadata

        metadata = json_to_metadata(self.proj_path)
        metadata = get_project_metadata(metadata.to_dict())
        self._check_subdirs()
        # get sites and measurements
        time_subdirs = dir_subdirs(self.proj_dir / PROJ_DIRS["time_dir"])
        sites = {}
        for site_dir in time_subdirs:
            site = self._load_site(site_dir, setup)
            sites[site_dir.name] = site
        record = self._get_process_record(f"Loaded project in {self.proj_dir}")
        return Project(self.proj_dir, metadata, sites, setup, ProcessHistory([record]))

    def _check_subdirs(self) -> bool:
        """
        Check that project subdirectories exist

        Returns
        -------
        bool
            True if they all exist
        """
        from resistics.common import assert_dir

        for subdir in PROJ_DIRS.values():
            subdir_path = self.proj_dir / subdir
            assert_dir(subdir_path)
            return False
        return True

    def _load_site(self, site_dir: Path, setup: Dict[str, Any]) -> Site:
        """
        Load a Site

        Parameters
        ----------
        site_dir : Path
            Site subdirectory in the time directory
        setup : Dict[str, Any]
            The resistics setup

        Returns
        -------
        Site
            A Site object
        """
        from resistics.common import dir_subdirs

        subdirs = dir_subdirs(site_dir)
        measurements = {}
        for meas_dir in subdirs:
            meas = self._load_measurement(meas_dir, setup)
            if meas is not None:
                measurements[meas_dir.name] = meas
        return Site(site_dir, measurements)

    def _load_measurement(
        self, meas_dir: Path, setup: Dict[str, Any]
    ) -> Union[Measurement, None]:
        """
        Load a measurement

        Parameters
        ----------
        meas_dir : Path
            The measurement subdirectory in the site time directory
        setup : Dict[str, Any]
            The resistics setup

        Returns
        -------
        Union[Measurement, None]
            Measurement if reading was successful, else None
        """
        time_reader = None
        for TReader in setup["time_readers"]:
            reader = TReader(meas_dir)
            if not reader.check():
                continue
            time_reader = reader
            break
        if time_reader is None:
            logger.error(f"Unable to read data in measumrent directory {meas_dir}")
            return None
        return Measurement(meas_dir, time_reader)
