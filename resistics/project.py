from logging import getLogger
from typing import Union, Optional, Type, List, Dict, Any
from numbers import Number
from pathlib import Path
import pandas as pd
import plotly.express as px

from resistics.common import ResisticsData, ResisticsProcess
from resistics.time import TimeReader, TimeReaderNumpy, TimeReaderAscii

logger = getLogger(__name__)


project_subdirs = {
    "time": "time",
    "calibration": "calibration",
    "spectra": "spectra",
    "features": "features",
    "masks": "masks",
    "trfuncs": "trfuncs",
}

default_setup = {
    "time_readers": [TimeReaderAscii, TimeReaderNumpy],
}


def project_time_path(time_dir: Path, site_name: str, meas_name: str) -> Path:
    return time_dir / site_name / meas_name


def project_spectra_path(
    spectra_dir: Path, site_name: str, meas_name: str, run: str
) -> Path:
    return spectra_dir / site_name / meas_name / run


def project_features_path(
    features_dir: Path, site_name: str, meas_name: str, run: str
) -> Path:
    return features_dir / site_name / meas_name / run


def project_mask_path(mask_dir: Path, site_name: str, run: str, fs: float) -> Path:
    from resistics.common import fs_to_string

    return mask_dir / site_name / run / f"{fs_to_string(fs)}.pkl"


def project_trfnc_path(trfnc_dir: Path, site_name: str, run: str, fs: float) -> Path:
    from resistics.common import fs_to_string

    return trfnc_dir / site_name / run / f"{fs_to_string(fs)}.json"


class Measurement(ResisticsData):
    """
    Class for describing a measurement

    This is a lightweight class as much of the work is done by the TimeReader which has read the header
    """

    def __init__(self, meas_dir: Path, reader: TimeReader):
        self.meas_dir = meas_dir
        self.reader = reader

    @property
    def name(self) -> str:
        return self.meas_dir.name

    @property
    def fs(self) -> float:
        return self.reader.headers["fs"]

    @property
    def first_time(self) -> pd.Timestamp:
        return self.reader.headers["first_time"]

    @property
    def last_time(self) -> pd.Timestamp:
        return self.reader.headers["last_time"]

    def to_string(self) -> str:
        outstr = f"Site '{self.name}'"
        outstr += f"Sampling frequency [Hz] = {self.fs}\n"
        outstr += f"First sample time = {self.first_time}\n"
        outstr += f"Last sample time = {self.last_time}\n"
        return outstr


class Site(ResisticsData):
    """
    Class for describing Sites

    This should essentially describe a single instrument setup
    """

    def __init__(
        self,
        site_dir: Path,
        paths: Dict[str, Path],
        measurements: Dict[str, Measurement],
    ) -> None:
        self.site_dir = site_dir
        self.paths = paths
        self.measurements = measurements
        if len(self.measurements) > 0:
            self.start = min([x.first_time for x in self.measurements.values()])
            self.end = max([x.last_time for x in self.measurements.values()])
        else:
            self.start = pd.Timestamp.utcnow()
            self.end = pd.Timestamp.utcnow()
        self.spectra = []
        self.features = []
        self.trfncs = []

    def __iter__(self):
        return self.measurements.values().__iter__()

    def __getitem__(self, meas_name: str) -> Measurement:
        return self.get_measurement(meas_name)

    @property
    def name(self) -> str:
        return self.site_dir.name

    @property
    def n_meas(self) -> int:
        return len(self.measurements)

    def fs(self) -> List[float]:
        fs = [x.fs for x in self.measurements.values()]
        return sorted(list(set(fs)))

    def get_measurement(self, meas_name: str) -> Measurement:
        from resistics.errors import MeasurementNotFoundError

        if not meas_name in self.measurements:
            raise MeasurementNotFoundError(self.name, meas_name)
        return self.measurements[meas_name]

    def get_measurements(self, fs: Optional[Number] = None) -> List[Measurement]:
        if fs is None:
            return list(self.measurements.values())
        return [x for x in self.measurements.values() if x.fs == fs]

    def plot(self) -> Any:
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
        data = [
            [x.name, x.first_time, x.last_time, x.fs, self.name]
            for x in self.measurements.values()
        ]
        return pd.DataFrame(
            data=data, columns=["name", "first_time", "last_time", "fs", "site"]
        )

    def to_string(self) -> str:
        from resistics.common import list_to_string

        outstr = f"Site {self.name}\n"
        outstr += f"{self.n_meas} Measurements\n"
        if self.n_meas > 0:
            outstr += ", ".join(self.measurements.keys())
            outstr += "\n"
        outstr += f"Sampling frequencies [Hz]: {list_to_string(self.fs())}\n"
        outstr += f"Recording start time: {self.start}\n"
        outstr += f"Recording end time: {self.end}"
        return outstr


class Project(ResisticsData):
    """
    Class to describe a resistics project
    """

    def __init__(
        self,
        project_dir: Path,
        project_info: Dict[str, Any],
        sites: Dict[str, Site],
        setup: Dict[str, Any],
    ):
        self.project_dir = project_dir
        self.project_info = project_info
        self.ref_time = pd.Timestamp(project_info["ref_time"])
        self.paths = project_info["subdirs"]
        self.sites = sites
        if len(self.sites) > 0:
            self.start = min([x.start for x in self.sites.values()])
            self.end = max([x.end for x in self.sites.values()])
        else:
            self.start = pd.Timestamp.utcnow()
            self.end = pd.Timestamp.utcnow()
        self.setup = setup

    def __iter__(self):
        return self.sites.values().__iter__()

    def __getitem__(self, site_name: str):
        return self.get_site(site_name)

    @property
    def n_sites(self) -> int:
        return len(self.sites)

    def fs(self) -> List[float]:
        fs = set()
        for site in self.sites.values():
            fs = fs.union(set(site.fs()))
        return sorted(list(fs))

    def get_site(self, site_name: str):
        from resistics.errors import SiteNotFoundError

        if site_name not in self.sites:
            raise SiteNotFoundError(site_name)
        return self.sites[site_name]

    def get_sites(self, fs: Optional[float] = None):
        sites = [x for x in self.sites.values() if fs in x.fs()]
        return sites

    def get_concurrent(self, site_name: str) -> List[str]:
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

    def plot(self) -> Any:
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
            title=str(self.project_dir),
        )
        return fig

    def to_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame(columns=["name", "first_time", "last_time", "fs", "site"])
        for site in self.sites.values():
            df = df.append(site.to_dataframe())
        return df

    def to_string(self) -> str:
        outstr = "Project"
        return outstr


class ProjectLoader(ResisticsProcess):
    def __init__(self, project_dir: Path, setup: Dict[str, Any]):
        self.project_dir = project_dir
        self.setup: Dict[str, Any] = setup
        self.project_info: Union[Dict[str, Any], None] = None

    def parameters(self) -> Dict[str, Any]:
        return {"project_dir": str(self.project_dir), "project_info": self.project_info}

    def check(self) -> bool:
        import json
        from resistics.common import assert_dir
        from resistics.errors import ProjectLoadError

        assert_dir(self.project_dir)
        resistics_path = self.project_dir / "resistics.json"
        if not resistics_path.exists():
            raise ProjectLoadError(
                self.project_dir, "resistics.json file not found in directory"
            )
        logger.info(f"Loading project information from {resistics_path}")
        with resistics_path.open("r") as f:
            project_info = json.load(f)
        for subdir in project_subdirs:
            if subdir not in project_info["subdirs"]:
                raise ProjectLoadError(
                    self.project_dir, f"Required subdirectory {subdir} not found"
                )
        for subdir in project_info["subdirs"]:
            assert_dir(self.project_dir / subdir)
        self.project_info = project_info
        return True

    def run(self) -> Project:
        from resistics.common import dir_subdirs

        assert self.project_info is not None

        time_dir = self.project_dir / self.project_info["subdirs"]["time"]
        subdirs = dir_subdirs(time_dir)
        sites = {}
        for site_dir in subdirs:
            site = self.load_site(site_dir)
            sites[site_dir.name] = site
        return Project(self.project_dir, self.project_info, sites, self.setup)

    def load_site(self, site_dir: Path) -> Site:
        from resistics.common import dir_subdirs

        subdirs = dir_subdirs(site_dir)
        measurements = {}
        for meas_dir in subdirs:
            meas = self.load_measurement(meas_dir)
            if meas is not None:
                measurements[meas_dir.name] = meas
        return Site(site_dir, self.project_info["subdirs"], measurements)

    def load_measurement(self, meas_dir: Path) -> Union[Measurement, None]:
        time_reader = None
        for TReader in self.setup["time_readers"]:
            reader = TReader(meas_dir)
            if not reader.check():
                continue
            time_reader = reader
            break
        if time_reader is None:
            logger.error(f"Unable to read data in measumrent directory {meas_dir}")
            return None
        return Measurement(meas_dir, time_reader)


def load(
    project_dir: Union[Path, str], setup: Optional[Dict[str, Any]] = None
) -> Project:
    """
    Load an existing project
    """
    from resistics.errors import ProjectLoadError

    if isinstance(project_dir, str):
        project_dir = Path(project_dir)
    if setup is None:
        setup = default_setup

    loader = ProjectLoader(project_dir, setup)
    if not loader.check():
        raise ProjectLoadError(project_dir, "Error loading project")
    return loader.run()


def reload(proj: Project):
    """
    Reload a project
    """
    return load(proj.project_dir, proj.setup)


def new(
    project_dir: Union[Path, str],
    ref_time: Union[pd.Timestamp, str],
    time_readers: Optional[List[Type[TimeReader]]] = None,
) -> Project:
    """
    Create a new resistics project by providing a directory for the project

    If the project directory does not exist, a new one will be created. If a project already exists in project path, this project will be loaded and returned.
    """
    import json
    from resistics.common import is_dir, assert_dir, serialize
    from resistics.time import TimeReaderNumpy, TimeReaderAscii
    from resistics.errors import ProjectCreateError

    if isinstance(project_dir, str):
        project_dir = Path(project_dir)
    if isinstance(ref_time, str):
        ref_time = pd.Timestamp(ref_time)
    if time_readers is None:
        time_readers = [TimeReaderNumpy, TimeReaderAscii]

    resistics_path = project_dir / "resistics.json"
    if project_dir.exists() and is_dir(project_dir) and resistics_path.exists():
        raise ProjectCreateError(
            project_dir, "Existing project found, please try loading"
        )
    if project_dir.exists():
        logger.info("Directory already exists, the project will be saved here")
        assert_dir(project_dir)
    else:
        logger.info("Making directory for project")
        project_dir.mkdir(parents=True)
    # save relative paths from project directory
    project_dict = {"ref_time": serialize(ref_time), "subdirs": {}}
    for key, subdir in project_subdirs.items():
        subdir_path = project_dir / subdir
        if not subdir_path.exists():
            subdir_path.mkdir()
        project_dict["subdirs"][key] = subdir
    project_dict["ref_time"] = serialize(ref_time)
    project_dict["created_on_local"] = serialize(pd.Timestamp.now(tz=None))
    project_dict["created_on_utc"] = serialize(pd.Timestamp.utcnow())
    # save project file
    with resistics_path.open("w") as f:
        json.dump(project_dict, f)
    logger.info("Project created, loading with default setup.")
    return load(project_dir, setup=default_setup)
