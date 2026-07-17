"""
Module for gathering data that will be combined to calculate transfer functions

There are two scenarios considered here. The first is the simplest, which is
quick processing outside the project environment. In this case data gathering
is not complicated. This workflow does not involve a data selector, meaining
only a single step is required.

- QuickGather to put together the out_data, in_data and cross_data

When inside the project environment, regardless of whether it is single site or
multi site processing, the workflow follows:

- Selector to select shared windows across all sites for a sampling frequency
- Gather to gather the combined evaluation frequency data

.. warning::

    There may be some confusion in the code with many references to spectra data
    and evaluation frequency data. Evaluation frequency data, referred to below
    as eval_data is actually an instance of Spectra data. However, it is named
    differently to highlight the fact that it is not the complete spectra data,
    but is actually spectra data at a reduced set of frequencies corresponding
    to the evaluation frequncies.

    Within a project, there are separate folders for users who want to save both
    the full spectra data with all the frequencies as well as the evaluation
    frequency spectra data with the smaller subset of frequencies. Only the
    evaluation frequency data is required to calculate the transfer function,
    but the full spectral data might be useful for visualisation and analysis
    reasons.
"""

from loguru import logger
from dataclasses import dataclass
from datetime import datetime
from typing import Any, ClassVar, List, Dict, Literal, Optional, Tuple, Union
from pathlib import Path
import numpy as np
import pandas as pd
from pydantic import ConfigDict, Field, field_validator, model_validator

from resistics.common import (
    ResisticsModel,
    ResisticsProcess,
    ResisticsData,
    validate_output_label,
)
from resistics.common import WriteableMetadata, History
from resistics.project import Project, Site
from resistics.decimate import DecimationParameters
from resistics.spectra import SpectraLevelMetadata, SpectraMetadata, SpectraData
from resistics.spectra import SpectraDataReader
from resistics.transfunc import TransferFunction
from resistics.mask import (
    AbsoluteTimeRange,
    DailyTimeRange,
    WindowMask,
    WindowMaskReader,
    get_run_mask_path,
    validate_mask_name,
)


def _validate_station_path(value: str) -> str:
    parts = value.split("/")
    if len(parts) != 2 or any(not part or part in {".", ".."} for part in parts):
        raise ValueError("Station paths must have the form 'survey/station'")
    return value


class MaskCriteria(ResisticsModel):
    """Named masks and the rule used to combine their keep decisions."""

    model_config = ConfigDict(extra="forbid")
    combine: Literal["and", "or"] = "and"
    names: List[str]

    @field_validator("names")
    @classmethod
    def validate_names(cls, values: List[str]) -> List[str]:
        if not values:
            raise ValueError("Mask criteria must name at least one mask")
        for value in values:
            validate_mask_name(value)
        if len(values) != len(set(values)):
            raise ValueError("Mask criteria cannot contain duplicate names")
        return values


class RateGatherCriteria(ResisticsModel):
    """Gather policy for one station and original sampling frequency."""

    model_config = ConfigDict(extra="forbid")
    remote_references: Optional[Union[Literal["auto"], List[str]]] = None
    masks: Optional[MaskCriteria] = None

    @field_validator("remote_references")
    @classmethod
    def validate_remotes(
        cls, value: Optional[Union[Literal["auto"], List[str]]]
    ) -> Optional[Union[Literal["auto"], List[str]]]:
        if value is None or value == "auto":
            return value
        if not value:
            raise ValueError("remote_references must be 'auto' or a non-empty list")
        for station_path in value:
            _validate_station_path(station_path)
        if len(value) != len(set(value)):
            raise ValueError("remote_references cannot contain duplicates")
        return value


class StationGatherCriteria(ResisticsModel):
    """Sampling-frequency keyed gather policies for one station."""

    model_config = ConfigDict(extra="forbid")
    sampling_frequencies: Dict[float, RateGatherCriteria] = Field(default_factory=dict)

    @field_validator("sampling_frequencies")
    @classmethod
    def validate_sample_rates(
        cls, value: Dict[float, RateGatherCriteria]
    ) -> Dict[float, RateGatherCriteria]:
        if any(sample_rate <= 0 for sample_rate in value):
            raise ValueError("Sampling frequencies must be positive")
        sample_rates = list(value)
        for index, left in enumerate(sample_rates):
            if any(
                np.isclose(left, right, rtol=1e-9, atol=1e-12)
                for right in sample_rates[index + 1 :]
            ):
                raise ValueError("Sampling-frequency keys must be distinct")
        return value


class ResolvedGatherCriteria(ResisticsModel):
    """Resolved rate policy returned uniformly for every level/eval index."""

    remote_references: Optional[Union[Literal["auto"], List[str]]] = None
    masks: Optional[MaskCriteria] = None


class GatherSelection(ResisticsData):
    """Resolved target/rate inputs and admissible global windows for gathering."""

    def __init__(
        self,
        station_rate_batch: Dict[str, Any],
        remote_station: Optional[str],
        criteria: "GatherCriteria",
        automatic_remote: bool = False,
    ) -> None:
        self.station_rate_batch = dict(station_rate_batch)
        self.remote_station = remote_station
        self.criteria = criteria
        self.automatic_remote = automatic_remote


class GatherCriteria(ResisticsProcess):
    """Strict station/rate policy for gathering persisted evaluation data.

    An empty instance is the no-criteria policy: all target windows are kept and
    gathering is single-site.  Policies are currently rate-wide, while
    :meth:`resolve` accepts level and evaluation-frequency indices so callers do
    not need to change when the schema gains more granular overrides.
    """

    model_config = ConfigDict(extra="forbid")

    output_type: ClassVar[str] = "gather_selection"
    runtime_requirements: ClassVar[List[str]] = ["station_rate_batch"]

    name: Optional[str] = Field(default=None, exclude=True)
    stations: Dict[str, StationGatherCriteria] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def reject_flat_schema(cls, value: Any) -> Any:
        if isinstance(value, dict) and {
            "remote_references",
            "absolute_include",
            "absolute_exclude",
            "daily_include",
            "daily_exclude",
            "automatic_remote",
        }.intersection(value):
            raise ValueError(
                "Legacy flat gather criteria are not supported. Use "
                "stations.<survey/station>.sampling_frequencies.<rate> with "
                "remote_references and/or masks; time constraints belong in a "
                "named TimeMask artifact."
            )
        return value

    @field_validator("stations")
    @classmethod
    def validate_stations(
        cls, value: Dict[str, StationGatherCriteria]
    ) -> Dict[str, StationGatherCriteria]:
        for station_path, station_criteria in value.items():
            _validate_station_path(station_path)
            for rate_criteria in station_criteria.sampling_frequencies.values():
                remotes = rate_criteria.remote_references
                if isinstance(remotes, list) and station_path in remotes:
                    raise ValueError(
                        f"Target station {station_path!r} cannot be its own remote"
                    )
        return value

    def resolve(
        self,
        station_path: str,
        sample_rate: float,
        level: int = 0,
        evaluation_frequency_index: int = 0,
    ) -> ResolvedGatherCriteria:
        """Resolve a policy, returning the no-criteria policy when unlisted."""
        if level < 0 or evaluation_frequency_index < 0:
            raise ValueError(
                "Level and evaluation-frequency index must be non-negative"
            )
        station = self.stations.get(station_path)
        if station is None:
            return ResolvedGatherCriteria()
        for configured_rate, policy in station.sampling_frequencies.items():
            if np.isclose(configured_rate, sample_rate, rtol=1e-9, atol=1e-12):
                return ResolvedGatherCriteria(**policy.model_dump())
        return ResolvedGatherCriteria()

    def remote_station_paths(self, station_path: str, sample_rate: float) -> List[str]:
        """Return explicitly configured remotes (``auto`` resolves at runtime)."""
        remotes = self.resolve(station_path, sample_rate).remote_references
        return [] if remotes is None or remotes == "auto" else list(remotes)

    def remote_reference_count(self) -> int:
        """Return the number of explicit remote assignments for summaries."""
        total = 0
        for station in self.stations.values():
            for policy in station.sampling_frequencies.values():
                if policy.remote_references == "auto":
                    total += 1
                elif isinstance(policy.remote_references, list):
                    total += len(policy.remote_references)
        return total

    def run(self, station_rate_batch: Dict[str, Any]) -> GatherSelection:
        """Resolve the remote assignment for one target station/rate batch."""
        station_rate_batch = dict(station_rate_batch)
        target = station_rate_batch.get("station_path")
        if target is None:
            target = f"{station_rate_batch['survey']}/{station_rate_batch['station']}"
            station_rate_batch["station_path"] = target
        sample_rate = station_rate_batch["sample_rate"]
        remotes = self.resolve(target, sample_rate).remote_references
        remote_station = remotes[0] if isinstance(remotes, list) and remotes else None
        return GatherSelection(
            station_rate_batch,
            remote_station,
            self,
            automatic_remote=remotes == "auto",
        )

    def execute(self, inputs: Dict[str, Any], context: Any) -> GatherSelection:
        """Resolve criteria from the station-rate batch supplied by the executor."""
        del inputs
        return self.run(context["station_rate_batch"])

    def includes(self, timestamp: datetime) -> bool:
        """Compatibility hook: time rules now live in named mask artifacts."""
        del timestamp
        return True


class EvaluationFrequencyGather(ResisticsProcess):
    """Gather every persisted target-run evaluation artifact for regression.

    The selection is deliberately a separate input: it makes the persisted
    evaluation artifact boundary explicit and leaves window-selection policy in
    :class:`GatherCriteria`, not in a job or parameter file.  Remote-reference
    alignment is intentionally not guessed; an explicit resolved remote is
    rejected until the cross-station aligner is implemented.
    """

    input_types: ClassVar[Dict[str, str]] = {"selection": "gather_selection"}
    output_type: ClassVar[str] = "gathered_data"
    runtime_requirements: ClassVar[List[str]] = ["project_path"]

    def execute(self, inputs: Dict[str, Any], context: Any) -> "GatheredData":
        """Load and concatenate all selected local-run evaluation artifacts."""
        from resistics.spectra import EvaluationFrequencyReader
        from resistics.transfunc import ImpedanceTensor

        selection = inputs["selection"]
        if not isinstance(selection, GatherSelection):
            raise ValueError("EvaluationFrequencyGather requires GatherSelection")
        if selection.automatic_remote:
            raise NotImplementedError(
                "Automatic remote selection is declared but has not been implemented"
            )
        if selection.remote_station is not None:
            raise NotImplementedError(
                "Remote-reference gathering is not implemented yet; the criteria "
                "mapping was resolved but cannot be applied"
            )
        batch = selection.station_rate_batch
        gathered = []
        for run_path in batch["run_paths"]:
            survey, station, run = run_path.split("/", 2)
            artifact = EvaluationFrequencyReader(
                label=validate_output_label(context["output_label"])
            ).execute(
                {},
                {
                    "project_path": context["project_path"],
                    "run_batch": {"survey": survey, "station": station, "run": run},
                },
            )
            artifact = self._apply_criteria(artifact, selection.criteria)
            gathered.append(
                QuickGather().run(
                    Path(run_path),
                    artifact.decimation_parameters,
                    ImpedanceTensor(),
                    artifact.spectra_data,
                )
            )
        if not gathered:
            raise ValueError("No evaluation-frequency artifacts selected for gather")
        return self._combine(gathered)

    @staticmethod
    def _apply_criteria(artifact: Any, criteria: GatherCriteria) -> Any:
        """Drop persisted evaluation windows excluded by the reusable criteria."""
        data = artifact.spectra_data
        metadata = data.metadata.model_copy(deep=True)
        selected = {}
        for level in range(metadata.n_levels):
            timestamps = data.get_timestamps(level)
            mask = np.array(
                [
                    criteria.includes(timestamp.to_pydatetime())
                    for timestamp in timestamps
                ]
            )
            selected[level] = data.get_level(level)[mask]
            metadata.levels_metadata[level].n_wins = int(mask.sum())
        if not any(level_data.shape[0] for level_data in selected.values()):
            raise ValueError("Gather criteria excludes every evaluation window")
        artifact.spectra_data = SpectraData(metadata, selected)
        return artifact

    @staticmethod
    def _combine(values: List["GatheredData"]) -> "GatheredData":
        def combine(kind: str) -> SiteCombinedData:
            first = getattr(values[0], kind)
            metadata = first.metadata.model_copy(
                update={
                    "measurements": [
                        measurement
                        for value in values
                        for measurement in (
                            getattr(value, kind).metadata.measurements or []
                        )
                    ],
                    "histories": {
                        name: history
                        for value in values
                        for name, history in getattr(
                            value, kind
                        ).metadata.histories.items()
                    },
                }
            )
            keys = first.data
            return SiteCombinedData(
                metadata,
                {
                    key: np.concatenate(
                        [getattr(value, kind).data[key] for value in values]
                    )
                    for key in keys
                },
            )

        return GatheredData(
            combine("out_data"), combine("in_data"), combine("cross_data")
        )


def get_site_evals_metadata(
    config_name: str, proj: Project, site_name: str, fs: float
) -> Dict[str, SpectraMetadata]:
    """
    Get spectra metadata for a given site and sampling frequency

    Parameters
    ----------
    config_name : str
        The configuration name to get the right data
    proj : Project
        The project instance to get the measurements
    site_name : str
        The name of the site for which to gather the SpectraMetadata
    fs : float
        The original recording sampling frequency

    Returns
    -------
    Dict[str, SpectraMetadata]
        Dictionary of measurement name to SpectraMetadata
    """
    from resistics.project import get_meas_evals_path

    site = proj[site_name]
    measurements = site.get_measurements(fs=fs)
    meas_metadata = {}
    for meas_name in measurements:
        meas = site[meas_name]
        evals_path = get_meas_evals_path(
            proj.dir_path, site.name, meas.name, config_name
        )
        try:
            metadata = SpectraDataReader().run(evals_path, metadata_only=True)
        except Exception:
            logger.error(f"No evals data found in path {evals_path}")
            continue
        logger.info(f"Found evals data for {site.name}, {meas.name}, {config_name}")
        meas_metadata[meas.name] = metadata
    return meas_metadata


def get_site_level_wins(
    meas_metadata: Dict[str, SpectraMetadata], level: int
) -> pd.Series:
    """
    Get site windows for a decimation level given a sampling frequency

    Parameters
    ----------
    meas_metadata : Dict[str, SpectraMetadata]
        The measurement spectra metadata for a site
    level : int
        The decimation level

    Returns
    -------
    pd.Series
        A series with an index of global windows for the site and values the
        measurements which have that global window. This is for a single
        decimation level

    See Also
    --------
    get_site_wins : Get windows for all decimation levels

    Examples
    --------
    An example getting the site windows for decimation level 0 when there are
    three measurements in the site.

    >>> from resistics.testing import spectra_metadata_multilevel
    >>> from resistics.gather import get_site_level_wins
    >>> meas_metadata = {}
    >>> meas_metadata["meas1"] = spectra_metadata_multilevel(n_wins=[3, 2, 2], index_offset=[3, 2, 1])
    >>> meas_metadata["meas2"] = spectra_metadata_multilevel(n_wins=[4, 3, 2], index_offset=[28, 25, 22])
    >>> meas_metadata["meas3"] = spectra_metadata_multilevel(n_wins=[2, 2, 1], index_offset=[108, 104, 102])
    >>> get_site_level_wins(meas_metadata, 0)
    3      meas1
    4      meas1
    5      meas1
    28     meas2
    29     meas2
    30     meas2
    31     meas2
    108    meas3
    109    meas3
    dtype: str
    >>> get_site_level_wins(meas_metadata, 1)
    2      meas1
    3      meas1
    25     meas2
    26     meas2
    27     meas2
    104    meas3
    105    meas3
    dtype: str
    >>> get_site_level_wins(meas_metadata, 2)
    1      meas1
    2      meas1
    22     meas2
    23     meas2
    102    meas3
    dtype: str
    """
    level_wins = pd.Series(dtype=str)
    for meas_name, metadata in meas_metadata.items():
        if level >= metadata.n_levels:
            continue
        level_metadata = metadata.levels_metadata[level]
        first_global = level_metadata.index_offset
        last_gobal = level_metadata.index_offset + level_metadata.n_wins - 1
        index = np.arange(first_global, last_gobal + 1)
        level_wins = pd.concat(
            [level_wins, pd.Series(data=meas_name, index=index)], axis=0
        )
    level_wins = level_wins[~level_wins.index.duplicated(keep="first")]
    return level_wins.sort_index()


def get_site_wins(
    config_name: str, proj: Project, site_name: str, fs: float
) -> Dict[int, pd.Series]:
    """
    Get site windows for all levels given a sampling frequency

    Parameters
    ----------
    config_name : str
        The configuration name to get the right data
    proj : Project
        The project instance to get the measurements
    site_name : str
        The site name
    fs : float
        The recording sampling frequency

    Returns
    -------
    Dict[int, pd.Series]
        Dictionary of integer to levels, with one entry for each decimation
        level

    Raises
    ------
    ValueError
        If no matching spectra metadata is found
    """
    logger.debug(f"Getting windows for site {site_name}")
    meas_metadata = get_site_evals_metadata(config_name, proj, site_name, fs)
    if len(meas_metadata) == 0:
        raise ValueError(f"No measurements for site {site_name}, sample frequency {fs}")
    n_levels = max([x.n_levels for x in meas_metadata.values()])
    logger.debug(f"Levels for site {site_name}, sample frequency {fs} = {n_levels}")
    tables = {}
    for ilevel in range(n_levels):
        tables[ilevel] = get_site_level_wins(meas_metadata, ilevel)
    return tables


class Selection(ResisticsData):
    """
    Selections are output by the Selector. They hold information about the data
    that should be gathered for the regression.
    """

    def __init__(
        self,
        sites: List[Site],
        dec_params: DecimationParameters,
        tables: Dict[int, pd.DataFrame],
    ):
        """
        Initialise the selection

        Parameters
        ----------
        sites : List[Site]
            The list of Sites that were included in the Selector
        dec_params : DecimationParameters
            The decimation parameters
        tables : Dict[int, pd.DataFrame]
            The window tables
        """
        self.sites = sites
        self.dec_params = dec_params
        self.tables = tables
        self.n_levels = len(tables)

    def get_n_evals(self) -> int:
        """
        Get the total number of evaluation frequnecies

        Returns
        -------
        int
            The total number of evaluation frequencies that can be calculated
        """
        return self.n_levels * self.dec_params.per_level

    def get_n_wins(self, level: int, eval_idx: int) -> int:
        """
        Get the number of windows for an evaluation frequency

        Parameters
        ----------
        level : int
            The decimation level
        eval_idx : int
            The evaluation frequency index in the decimation level

        Returns
        -------
        int
            The number of windows

        Raises
        ------
        ValueError
            If the level is greater than the maximum level available
        """
        if level >= self.n_levels:
            raise ValueError(f"Level {level} >= n_levels {self.n_levels}")
        level_table = self.tables[level]
        eval_series = level_table[eval_idx]
        return eval_series[eval_series].count()

    def get_measurements(self, site: Site) -> List[str]:
        """
        Get the measurement names to read from a Site

        Parameters
        ----------
        site : Site
            The site for which to get the measurements

        Returns
        -------
        List[str]
            The measurements to read from
        """
        measurements = set()
        for level_table in self.tables.values():
            level_set = set(level_table[site.name].unique())
            measurements = measurements.union(level_set)
        return sorted(list(measurements))

    def get_eval_freqs(self) -> List[float]:
        """
        Get the evaluation frequencies

        Returns
        -------
        List[float]
            The evaluation frequencies as a flat list of floats
        """
        eval_freqs = []
        for ilevel in range(self.n_levels):
            eval_freqs = eval_freqs + self.dec_params.get_eval_freqs(ilevel)
        return eval_freqs

    def get_eval_wins(self, level: int, eval_idx: int) -> pd.DataFrame:
        """
        Limit the level windows to the evaluation frequency

        Parameters
        ----------
        level : int
            The decimation level
        eval_idx : int
            The evalution frequency index in the decimation level

        Returns
        -------
        pd.DataFrame
            pandas DataFrame of the windows and the measurements from each site
            the window can be read from
        """
        cols = [site.name for site in self.sites]
        eval_wins = self.tables[level]
        eval_wins = eval_wins[eval_wins[eval_idx]]
        return eval_wins[cols]


class Selector(ResisticsProcess):
    """
    The Selector takes Sites and tries to find shared windows across them. A
    project instance is required for the Selector to be able to find shared
    windows.

    The Selector should be used for remote reference and intersite processing
    and single site processing when masks are involved.
    """

    def run(
        self,
        config_name: str,
        proj: Project,
        site_names: List[str],
        dec_params: DecimationParameters,
        masks: Optional[Dict[str, str]] = None,
    ) -> Selection:
        """
        Run the selector

        If a site repeats, the selector only considers it once. This might be
        the case when performing intersite or other cross site style processing.

        Parameters
        ----------
        config_name : str
            The configuration name
        proj : Project
            The project instance
        site_names : List[str]
            The names of the sites to get data from
        dec_params : DecimationParameters
            The decimation parameters with number of levels etc.
        masks : Optional[Dict[str, str]], optional
            Any masks to add, by default None

        Returns
        -------
        Selection
            The Selection information defining the measurements and windows to
            read for each site
        """
        # get unique sites
        site_names = sorted(list(set(site_names)))
        fs = dec_params.fs
        sites_wins = {
            site_name: get_site_wins(config_name, proj, site_name, fs)
            for site_name in site_names
        }
        # get the higest decimation level that all sites have
        n_levels = min([len(x) for x in sites_wins.values()])
        logger.info(f"Finding shared windows across {', '.join(sites_wins.keys())}")
        logger.info(f"Max. level across sites = {n_levels - 1}, num. levels {n_levels}")
        tables: Dict[int, pd.DataFrame] = {}
        for ilevel in range(n_levels):
            logger.info(f"Finding shared windows for decimation level {ilevel}")
            data = {x: y[ilevel] for x, y in sites_wins.items()}
            table = pd.DataFrame(data=data).dropna()
            table = self._get_evals(dec_params, table, ilevel)
            if masks is not None:
                table = self._apply_masks(table, masks)
            tables[ilevel] = table
        sites = [proj[site_name] for site_name in site_names]
        return Selection(sites, dec_params, tables)

    def _get_evals(
        self, dec_params: DecimationParameters, table: pd.DataFrame, level: int
    ) -> pd.DataFrame:
        """
        Add a column for each evaluation frequency

        Parameters
        ----------
        dec_params : DecimationParameters
            The decimation parameters
        table : pd.DataFrame
            The window table with measurements from each site
        level : int
            The decimation level

        Returns
        -------
        pd.DataFrame
            pandas DataFrame with boolean column for each evaluation frequency
        """
        for ifreq in range(dec_params.per_level):
            table[ifreq] = True
        return table

    def _apply_masks(self, table: pd.DataFrame, masks: Dict[str, str]) -> pd.DataFrame:
        """Set some windows False based on masks"""
        return table


class SiteCombinedMetadata(WriteableMetadata):
    """
    Metadata for combined data

    Combined metadata stores metadata for measurements that are combined from
    a single site.
    """

    site_name: str
    """The name of the site"""
    site_names: List[str] = Field(default_factory=list)
    """Authoritative station paths, including every pooled remote station"""
    fs: float
    """Recording sampling frequency"""
    system: str = ""
    """The system used for recording"""
    serial: str = ""
    """Serial number of the system"""
    wgs84_latitude: float = -999.0
    """Latitude in WGS84"""
    wgs84_longitude: float = -999.0
    """Longitude in WGS84"""
    easting: float = -999.0
    """The easting of the site in local cartersian coordinates"""
    northing: float = -999.0
    """The northing of the site in local cartersian coordinates"""
    elevation: float = -999.0
    """The elevation of the site"""
    measurements: Optional[List[str]] = None
    """List of measurement names that were included in the combined data"""
    chans: List[str]
    """List of channels, these are common amongst all the measurements"""
    n_evals: int
    """The number of evaluation frequencies"""
    eval_freqs: List[float]
    """The evaluation frequencies"""
    histories: Dict[str, History]
    """Dictionary mapping measurement name to measurement processing history"""

    @model_validator(mode="after")
    def populate_site_names(self) -> "SiteCombinedMetadata":
        if not self.site_names:
            self.site_names = [self.site_name]
        return self


class SiteCombinedData(ResisticsData):
    """
    Combined data is data that is combined from a single site for the purposes
    of regression.

    All of the data that is combined should have the same sampling frequency,
    same evaluation frequencies and some shared channels.

    Data is stored in the data attribute of the class. This is a dictionary
    mapping evaluation frequency index to data for the evaluation frequency
    from all windows in the site. The shape of data for a single evaluation
    frequency is:

    n_wins x n_chans

    The data is complex valued.
    """

    def __init__(self, metadata: SiteCombinedMetadata, data: Dict[int, np.ndarray]):
        """
        Initialise the CombinedData

        Parameters
        ----------
        metadata : CombinedMetadata
            The combined metadata
        data : Dict[int, np.ndarray]
            The data with index the evaluation frequency index and value the
            combined data from a site for the evaluation frequency
        """
        self.metadata = metadata
        self.data = data


class GatheredData(ResisticsData):
    """
    Class to hold data to be used in by Regression preparers

    Gathered data has an out_data, in_data and cross_data. The important thing
    here is that the data is all aligned with regards to windows
    """

    def __init__(
        self,
        out_data: SiteCombinedData,
        in_data: SiteCombinedData,
        cross_data: SiteCombinedData,
    ) -> None:
        self.out_data = out_data
        self.in_data = in_data
        self.cross_data = cross_data


@dataclass(frozen=True)
class _EvaluationLocator:
    """One evaluation row addressed by its station-global window index."""

    run_path: str
    local_index: int
    artifact: Any


class Gather(ResisticsProcess):
    """Gather aligned persisted evaluation data for local or remote regression.

    Target data supplies output and input channels.  With remote references,
    target/remote pairs are pooled and the remote data supplies cross channels.
    The same target window is intentionally repeated when it aligns with more
    than one configured remote.
    """

    input_types: ClassVar[Dict[str, str]] = {
        "selection": "gather_selection",
        "tf": "transfer_function",
    }
    output_type: ClassVar[str] = "gathered_data"
    runtime_requirements: ClassVar[List[str]] = ["project", "project_path"]

    def execute(self, inputs: Dict[str, Any], context: Any) -> GatheredData:
        selection = inputs["selection"]
        if not isinstance(selection, GatherSelection):
            raise ValueError("Gather requires GatherSelection")
        return self.run(
            context["project"],
            Path(context["project_path"]),
            selection,
            inputs["tf"],
            context["output_label"],
        )

    def run(
        self,
        project: Project,
        project_path: Path,
        selection: GatherSelection,
        tf: TransferFunction,
        output_label: str = "default",
    ) -> GatheredData:
        from resistics.spectra import EvaluationFrequencyReader

        batch = selection.station_rate_batch
        target = batch["station_path"]
        sample_rate = float(batch["sample_rate"])
        target_runs = sorted(batch["run_paths"])
        if not target_runs:
            raise ValueError(f"Target {target} at {sample_rate:g} Hz has no runs")

        self._output_label = validate_output_label(output_label)
        self._reader = EvaluationFrequencyReader(label=self._output_label)
        self._project_path = project_path
        self._criteria = selection.criteria
        self._artifact_cache: Dict[str, Any] = {}
        self._mask_cache: Dict[Tuple[str, str], WindowMask] = {}

        target_artifacts = self._load_runs(target_runs, required=True, role="target")
        baseline = next(iter(target_artifacts.values()))
        for run_path, artifact in target_artifacts.items():
            self._validate_compatible(baseline, artifact, f"target run {run_path}")
        level_references = self._level_references(target_artifacts, target)

        resolved = self._criteria.resolve(target, sample_rate)
        remote_setting = resolved.remote_references
        if remote_setting is None:
            candidate_paths: List[str] = []
            automatic = False
        elif remote_setting == "auto":
            candidate_paths = project.get_concurrent(target, sample_rate)
            automatic = True
        else:
            candidate_paths = sorted(remote_setting)
            automatic = False

        remote_artifacts: Dict[str, Dict[str, Any]] = {}
        candidate_reasons: Dict[str, str] = {}
        for station_path in sorted(candidate_paths):
            try:
                run_paths = self._station_runs(project, station_path, sample_rate)
                artifacts = self._load_runs(
                    run_paths,
                    required=not automatic,
                    role=f"remote {station_path}",
                )
                if not artifacts:
                    candidate_reasons[station_path] = "no readable evaluation artifacts"
                    continue
                compatible = {}
                for run_path, artifact in artifacts.items():
                    try:
                        self._validate_compatible(
                            baseline, artifact, f"remote run {run_path}"
                        )
                        for level, reference in level_references.items():
                            if level < artifact.spectra_data.metadata.n_levels:
                                self._validate_level_compatible(
                                    reference,
                                    artifact,
                                    level,
                                    f"remote run {run_path}",
                                )
                    except Exception as exc:
                        if not automatic:
                            raise
                        logger.warning(
                            f"Skipping incompatible automatic remote run "
                            f"{run_path}: {exc}"
                        )
                        continue
                    compatible[run_path] = artifact
                if not compatible:
                    candidate_reasons[station_path] = (
                        "no compatible evaluation artifacts"
                    )
                    continue
                remote_artifacts[station_path] = compatible
            except Exception as exc:
                if not automatic:
                    raise
                candidate_reasons[station_path] = str(exc)
                logger.warning(
                    f"Skipping automatic remote {station_path} for {target}: {exc}"
                )

        out_values: Dict[int, np.ndarray] = {}
        in_values: Dict[int, np.ndarray] = {}
        cross_values: Dict[int, np.ndarray] = {}
        target_used: set[str] = set()
        remote_used: set[str] = set()
        usable_remotes: set[str] = set()
        eval_freqs: List[float] = []

        key = 0
        for level, level_reference in level_references.items():
            level_metadata = level_reference.spectra_data.metadata.levels_metadata[
                level
            ]
            target_catalog = self._catalog(target_artifacts, level, target)
            for evaluation_index, evaluation_frequency in enumerate(
                level_metadata.freqs
            ):
                eval_freqs.append(evaluation_frequency)
                target_valid = self._apply_masks(
                    target,
                    sample_rate,
                    level,
                    evaluation_index,
                    target_catalog,
                )
                pairs: List[Tuple[_EvaluationLocator, _EvaluationLocator]] = []
                if remote_setting is None:
                    pairs = [
                        (target_valid[global_index], target_valid[global_index])
                        for global_index in sorted(target_valid)
                    ]
                else:
                    skipped = []
                    for remote_path in sorted(remote_artifacts):
                        remote_catalog = self._catalog(
                            remote_artifacts[remote_path], level, remote_path
                        )
                        remote_valid = self._apply_masks(
                            remote_path,
                            sample_rate,
                            level,
                            evaluation_index,
                            remote_catalog,
                        )
                        shared = sorted(set(target_valid).intersection(remote_valid))
                        if not shared:
                            skipped.append(f"{remote_path}: no shared admitted windows")
                            continue
                        usable_remotes.add(remote_path)
                        pairs.extend(
                            (target_valid[global_index], remote_valid[global_index])
                            for global_index in shared
                        )
                    if skipped:
                        logger.warning(
                            f"Gather {target}, level {level}, evaluation index "
                            f"{evaluation_index}: {'; '.join(skipped)}"
                        )
                if not pairs:
                    reasons = "; ".join(
                        f"{name}: {reason}"
                        for name, reason in sorted(candidate_reasons.items())
                    )
                    suffix = f" Candidate diagnostics: {reasons}" if reasons else ""
                    raise ValueError(
                        f"No admissible gather windows for target {target}, "
                        f"{sample_rate:g} Hz, level {level}, evaluation-frequency "
                        f"index {evaluation_index}.{suffix}"
                    )
                target_locs = [pair[0] for pair in pairs]
                cross_locs = [pair[1] for pair in pairs]
                out_values[key] = self._extract(
                    target_locs, level, evaluation_index, tf.out_chans
                )
                in_values[key] = self._extract(
                    target_locs, level, evaluation_index, tf.in_chans
                )
                cross_values[key] = self._extract(
                    cross_locs, level, evaluation_index, tf.cross_chans
                )
                target_used.update(locator.run_path for locator in target_locs)
                if remote_setting is not None:
                    remote_used.update(locator.run_path for locator in cross_locs)
                key += 1

        if remote_setting is not None and not usable_remotes:
            raise ValueError(
                f"No usable remote reference remains for target {target} at "
                f"{sample_rate:g} Hz"
            )

        out_metadata = self._combined_metadata(
            target,
            [target],
            target_used,
            tf.out_chans,
            eval_freqs,
            target_artifacts,
        )
        in_metadata = self._combined_metadata(
            target,
            [target],
            target_used,
            tf.in_chans,
            eval_freqs,
            target_artifacts,
        )
        cross_stations = [target] if remote_setting is None else sorted(usable_remotes)
        cross_source = (
            target_artifacts
            if remote_setting is None
            else {
                run: artifact
                for station_artifacts in remote_artifacts.values()
                for run, artifact in station_artifacts.items()
            }
        )
        cross_metadata = self._combined_metadata(
            target if remote_setting is None else "+".join(cross_stations),
            cross_stations,
            target_used if remote_setting is None else remote_used,
            tf.cross_chans,
            eval_freqs,
            cross_source,
        )
        logger.info(
            f"Gathered {target} at {sample_rate:g} Hz using cross stations "
            f"{cross_stations}"
        )
        return GatheredData(
            SiteCombinedData(out_metadata, out_values),
            SiteCombinedData(in_metadata, in_values),
            SiteCombinedData(cross_metadata, cross_values),
        )

    def _station_runs(
        self, project: Project, station_path: str, sample_rate: float
    ) -> List[str]:
        _validate_station_path(station_path)
        rows = project.table[
            (project.table["station_path"] == station_path)
            & np.isclose(project.table["sample_rate"].astype(float), sample_rate)
        ]
        run_paths = sorted(rows["run_path"].unique().tolist())
        if not run_paths:
            raise ValueError(
                f"Remote station {station_path!r} has no runs at {sample_rate:g} Hz"
            )
        return run_paths

    def _load_runs(
        self, run_paths: List[str], required: bool, role: str
    ) -> Dict[str, Any]:
        values = {}
        failures = []
        for run_path in sorted(run_paths):
            if run_path in self._artifact_cache:
                values[run_path] = self._artifact_cache[run_path]
                continue
            survey, station, run = run_path.split("/", 2)
            try:
                artifact = self._reader.execute(
                    {},
                    {
                        "project_path": self._project_path,
                        "run_batch": {
                            "survey": survey,
                            "station": station,
                            "run": run,
                        },
                    },
                )
            except Exception as exc:
                failures.append(f"{run_path}: {exc}")
                continue
            self._artifact_cache[run_path] = artifact
            values[run_path] = artifact
        if failures and required:
            raise ValueError(
                f"Missing or unreadable evaluation artifact for {role}: "
                + "; ".join(failures)
            )
        if failures:
            logger.warning(f"Skipping unreadable {role} runs: {'; '.join(failures)}")
        return values

    @staticmethod
    def _validate_compatible(reference: Any, candidate: Any, description: str) -> None:
        ref_dec = reference.decimation_parameters
        got_dec = candidate.decimation_parameters
        failures = []
        if not np.isclose(ref_dec.fs, got_dec.fs):
            failures.append("original sample rate")
        ref_data = reference.spectra_data
        got_data = candidate.spectra_data
        if str(ref_data.metadata.ref_time) != str(got_data.metadata.ref_time):
            failures.append("reference time")
        for level in range(
            min(ref_data.metadata.n_levels, got_data.metadata.n_levels)
        ):
            failures.extend(
                Gather._level_compatibility_failures(reference, candidate, level)
            )
        if failures:
            raise ValueError(
                f"Incompatible {description}: {', '.join(dict.fromkeys(failures))}"
            )

    @staticmethod
    def _level_compatibility_failures(
        reference: Any, candidate: Any, level: int
    ) -> List[str]:
        """Compare one realised evaluation level using spectra metadata only."""
        left = reference.spectra_data.metadata.levels_metadata[level]
        right = candidate.spectra_data.metadata.levels_metadata[level]
        failures = []
        if not np.isclose(left.fs, right.fs):
            failures.append(f"level {level} sample rate")
        if left.win_size != right.win_size or left.olap_size != right.olap_size:
            failures.append(f"level {level} window signature")
        if (
            left.n_freqs != right.n_freqs
            or len(left.freqs) != len(right.freqs)
            or not np.allclose(left.freqs, right.freqs)
        ):
            failures.append(f"level {level} evaluation frequencies")
        return failures

    @staticmethod
    def _validate_level_compatible(
        reference: Any, candidate: Any, level: int, description: str
    ) -> None:
        failures = Gather._level_compatibility_failures(
            reference, candidate, level
        )
        if failures:
            raise ValueError(
                f"Incompatible {description}: {', '.join(failures)}"
            )

    @staticmethod
    def _level_references(
        artifacts: Dict[str, Any], station_path: str
    ) -> Dict[int, Any]:
        """Choose and validate an authoritative target artifact for each level."""
        n_levels = max(
            artifact.spectra_data.metadata.n_levels
            for artifact in artifacts.values()
        )
        if n_levels == 0:
            raise ValueError(
                f"Target station {station_path} has no realised evaluation levels"
            )
        references = {}
        for level in range(n_levels):
            available = {
                run_path: artifact
                for run_path, artifact in artifacts.items()
                if level < artifact.spectra_data.metadata.n_levels
            }
            reference_run = sorted(available)[0]
            reference = available[reference_run]
            for run_path, artifact in available.items():
                Gather._validate_level_compatible(
                    reference,
                    artifact,
                    level,
                    f"target run {run_path} for station {station_path}",
                )
            references[level] = reference
        return references

    @staticmethod
    def _catalog(
        artifacts: Dict[str, Any], level: int, station_path: str
    ) -> Dict[int, _EvaluationLocator]:
        catalog = {}
        for run_path in sorted(artifacts):
            artifact = artifacts[run_path]
            metadata = artifact.spectra_data.metadata
            if level >= metadata.n_levels:
                continue
            level_meta = metadata.levels_metadata[level]
            for local_index in range(level_meta.n_wins):
                global_index = level_meta.index_offset + local_index
                if global_index in catalog:
                    raise ValueError(
                        f"Ambiguous global window {global_index} for station "
                        f"{station_path}, level {level}: {catalog[global_index].run_path} "
                        f"and {run_path}"
                    )
                catalog[global_index] = _EvaluationLocator(
                    run_path, local_index, artifact
                )
        return catalog

    def _apply_masks(
        self,
        station_path: str,
        sample_rate: float,
        level: int,
        evaluation_index: int,
        catalog: Dict[int, _EvaluationLocator],
    ) -> Dict[int, _EvaluationLocator]:
        policy = self._criteria.resolve(
            station_path, sample_rate, level, evaluation_index
        )
        if policy.masks is None:
            return catalog
        result = {}
        for global_index, locator in catalog.items():
            decisions = []
            for name in policy.masks.names:
                mask = self._load_mask(
                    station_path,
                    locator.run_path,
                    name,
                    locator.artifact,
                    level,
                    evaluation_index,
                )
                decisions.append(
                    bool(mask.get_keep(level, evaluation_index).loc[global_index])
                )
            keep = all(decisions) if policy.masks.combine == "and" else any(decisions)
            if keep:
                result[global_index] = locator
        return result

    def _load_mask(
        self,
        station_path: str,
        run_path: str,
        name: str,
        artifact: Any,
        level: int,
        evaluation_index: int,
    ) -> WindowMask:
        cache_key = (run_path, name)
        if cache_key in self._mask_cache:
            return self._mask_cache[cache_key]
        survey, station, run = run_path.split("/", 2)
        path = get_run_mask_path(
            self._project_path,
            {"survey": survey, "station": station, "run": run},
            name,
            self._output_label,
        )
        try:
            mask = WindowMaskReader().run(path)
        except Exception as exc:
            raise ValueError(
                f"Required mask {name!r} is unavailable for target/remote "
                f"{station_path}, run {run_path}, level {level}, "
                f"evaluation-frequency index {evaluation_index}: {exc}"
            ) from exc
        self._validate_mask(mask, artifact, run_path, name)
        self._mask_cache[cache_key] = mask
        return mask

    @staticmethod
    def _validate_mask(
        mask: WindowMask, artifact: Any, run_path: str, name: str
    ) -> None:
        data = artifact.spectra_data
        dec_params = artifact.decimation_parameters
        failures = []
        if not np.isclose(mask.metadata.sample_rate, dec_params.fs):
            failures.append("original sample rate")
        if str(mask.metadata.ref_time) != str(data.metadata.ref_time):
            failures.append("reference time")
        if len(mask.metadata.levels) != data.metadata.n_levels:
            failures.append("number of levels")
        else:
            for item, source in zip(
                mask.metadata.levels, data.metadata.levels_metadata
            ):
                if item.n_evaluation_frequencies != source.n_freqs:
                    failures.append(f"level {item.level} evaluation count")
                expected_freqs = source.freqs
                if len(item.evaluation_frequencies) != len(
                    expected_freqs
                ) or not np.allclose(item.evaluation_frequencies, expected_freqs):
                    failures.append(f"level {item.level} evaluation frequencies")
                if (
                    not np.isclose(item.fs, source.fs)
                    or item.n_wins != source.n_wins
                    or item.win_size != source.win_size
                    or item.olap_size != source.olap_size
                    or item.index_offset != source.index_offset
                ):
                    failures.append(f"level {item.level} window signature")
        if failures:
            raise ValueError(
                f"Mask {name!r} for run {run_path} is incompatible: "
                + ", ".join(dict.fromkeys(failures))
            )

    @staticmethod
    def _extract(
        locators: List[_EvaluationLocator],
        level: int,
        evaluation_index: int,
        channels: List[str],
    ) -> np.ndarray:
        """Extract selected windows in batches for each source artifact.

        Selecting channels from a whole spectra level uses NumPy advanced
        indexing and therefore copies the entire level.  Doing that once per
        window made a large standard-window gather quadratic in its window
        count.  Grouping by run lets us select only the requested rows for
        each artifact, as the legacy :class:`ProjectGather` does.
        """
        from resistics.errors import ChannelNotFoundError

        values = np.empty((len(locators), len(channels)), dtype=np.complex128)
        grouped: Dict[str, Tuple[Any, List[int], List[int]]] = {}
        for output_index, locator in enumerate(locators):
            if locator.run_path not in grouped:
                grouped[locator.run_path] = (locator.artifact, [], [])
            _, output_indices, local_indices = grouped[locator.run_path]
            output_indices.append(output_index)
            local_indices.append(locator.local_index)

        for artifact, output_indices, local_indices in grouped.values():
            spectra = artifact.spectra_data
            channel_indices = []
            for channel in channels:
                if channel not in spectra.metadata.chans:
                    raise ChannelNotFoundError(channel, spectra.metadata.chans)
                channel_indices.append(spectra.metadata.chans.index(channel))
            level_data = spectra.get_level(level)
            values[np.asarray(output_indices)] = level_data[
                np.asarray(local_indices), :, evaluation_index
            ][:, channel_indices]
        return values

    def _combined_metadata(
        self,
        site_name: str,
        site_names: List[str],
        used_runs: set[str],
        channels: List[str],
        eval_freqs: List[float],
        artifacts: Dict[str, Any],
    ) -> SiteCombinedMetadata:
        first_run = sorted(used_runs)[0]
        first_artifact = artifacts[first_run]
        first = first_artifact.spectra_data.metadata
        histories = {}
        for run_path in sorted(used_runs):
            history = artifacts[run_path].spectra_data.metadata.history.model_copy(
                deep=True
            )
            history.add_record(
                self._get_record(
                    [f"Gathered evaluation-frequency data from {run_path}"]
                )
            )
            histories[run_path] = history
        return SiteCombinedMetadata(
            site_name=site_name,
            site_names=site_names,
            fs=float(first_artifact.decimation_parameters.fs),
            system=first.system,
            serial=first.serial,
            wgs84_latitude=first.wgs84_latitude,
            wgs84_longitude=first.wgs84_longitude,
            easting=first.easting,
            northing=first.northing,
            elevation=first.elevation,
            measurements=sorted(used_runs),
            chans=channels,
            n_evals=len(eval_freqs),
            eval_freqs=eval_freqs,
            histories=histories,
        )


class ProjectGather(ResisticsProcess):
    """
    Gather aligned data from a single or multiple sites in the project

    Aligned data means that the same index of data across multiple sites points
    to data covering the same global window (i.e. the same time window). This
    is essential for calculating intersite or remote reference transfer
    functions.
    """

    def run(
        self,
        config_name: str,
        proj: Project,
        selection: Selection,
        tf: TransferFunction,
        out_name: str,
        in_name: Optional[str] = None,
        cross_name: Optional[str] = None,
    ) -> GatheredData:
        """
        Gather data for input into the regression preparer

        Parameters
        ----------
        config_name : str
            The config name for getting the correct evals data
        proj : Project
            The project instance
        selection : Selection
            The selection
        tf : TransferFunction
            The transfer function
        out_name : str
            The name of the output site
        in_name : Optional[str], optional
            The name of the input site, by default None
        cross_name : Optional[str], optional
            The name of the cross site, by default None

        Returns
        -------
        GatheredData
            The data gathered for the regression preparer
        """
        if in_name is None:
            in_name = out_name
        if cross_name is None:
            cross_name = in_name

        logger.info(f"Collecting data for out site {out_name}")
        out_data = self._get_site_data(
            config_name, proj, selection, out_name, tf.out_chans
        )
        logger.info(f"Collecting data for in site {in_name}")
        in_data = self._get_site_data(
            config_name, proj, selection, in_name, tf.in_chans
        )
        logger.info(f"Collecting data for cross site {cross_name}")
        cross_data = self._get_site_data(
            config_name, proj, selection, cross_name, tf.cross_chans
        )
        return GatheredData(out_data, in_data, cross_data)

    def _get_site_data(
        self,
        config_name: str,
        proj: Project,
        selection: Selection,
        site_name: str,
        chans: List[str],
    ) -> SiteCombinedData:
        """
        Collect the evals data for the site. This is only for the shared
        windows.

        Parameters
        ----------
        config_name : str
            The configuration name to fetch the correct data
        proj : Project
            The project instance
        selection : Selection
            The window selection
        site_name : str
            The site name
        chans : List[str]
            The channels to get for the site

        Returns
        -------
        SiteCombinedData
            A combined data instance
        """
        from resistics.project import get_meas_evals_path

        logger.debug(f"Collecting site data for {site_name}, channels {chans}")
        site = proj[site_name]
        measurements = selection.get_measurements(site)
        data = self._get_empty_data(selection, chans)
        histories = {}
        metadata = None
        for meas in measurements:
            evals_path = get_meas_evals_path(
                proj.dir_path, site.name, meas, config_name
            )
            eval_data = SpectraDataReader().run(evals_path)
            self._populate_data(selection, site, meas, eval_data, chans, data)
            histories[meas] = eval_data.metadata.history
            if metadata is None:
                metadata = eval_data.metadata
        combined_metadata = SiteCombinedMetadata(
            site_name=site.name,
            fs=selection.dec_params.fs,
            system=metadata.system,
            serial=metadata.serial,
            wgs84_latitude=metadata.wgs84_latitude,
            wgs84_longitude=metadata.wgs84_longitude,
            easting=metadata.easting,
            northing=metadata.northing,
            elevation=metadata.elevation,
            measurements=measurements,
            chans=chans,
            n_evals=len(data),
            eval_freqs=selection.get_eval_freqs(),
            histories=histories,
        )
        return SiteCombinedData(combined_metadata, data)

    def _get_empty_data(
        self, selection: Selection, chans: List[str]
    ) -> Dict[int, np.ndarray]:
        """
        Get dictionary of empty arrays to put the data in

        This is a dictionary with n_evals entries
        Each evaluation frequency entry is of size:

        n_wins * n_chans * 1

        Parameters
        ----------
        selection : Selection
            The window selection information
        chans : List[str]
            The channels to define the shape of the array

        Returns
        -------
        Dict[int, np.ndarray]
            Dictionary mapping evaluation frequency index to the combined
            evaluation frequency data
        """
        per_level = selection.dec_params.per_level
        empty_data = {}
        for ilevel in range(selection.n_levels):
            for ifreq in range(per_level):
                n_wins = selection.get_n_wins(ilevel, ifreq)
                key = per_level * ilevel + ifreq
                empty_data[key] = np.empty(
                    shape=(n_wins, len(chans)), dtype=np.complex128
                )
        return empty_data

    def _populate_data(
        self,
        selection: Selection,
        site: Site,
        meas: str,
        eval_data: SpectraData,
        chans: List[str],
        data: Dict[int, np.ndarray],
    ) -> Dict[int, np.ndarray]:
        """
        Populate a measurement's evaluation frequency data into combined data

        There is some complexity here regarding mapping of the right windows to
        the correct indices for the site.

        Parameters
        ----------
        selection : Selection
            The selection data
        site : Site
            The site
        meas : str
            The name of the measurement
        eval_data : SpectraData
            The evaluation frequency data for the measurement
        chans : List[str]
            The channels to get from the data
        data : Dict[int, np.ndarray]
            The data dictionary which will be populated

        Returns
        -------
        Dict[int, np.ndarray]
            The data dictionary with data for the measurement populated
        """
        per_level = selection.dec_params.per_level
        for ilevel in range(selection.n_levels):
            if ilevel >= eval_data.metadata.n_levels:
                logger.debug(f"Measurement {meas} has no level {ilevel}")
                break
            level_data = eval_data.get_chans(ilevel, chans)
            for ifreq in range(per_level):
                key = per_level * ilevel + ifreq
                eval_wins = selection.get_eval_wins(ilevel, ifreq)
                eval_data_indices, combined_indices = self._get_indices(
                    eval_wins, site, meas, eval_data.metadata.levels_metadata[ilevel]
                )
                data[key][combined_indices] = level_data[eval_data_indices, ..., ifreq]
        return data

    def _get_indices(
        self,
        eval_wins: pd.DataFrame,
        site: Site,
        meas_name: str,
        level_metadata: SpectraLevelMetadata,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get two arrays to help align windows

        - eval_data_indices are the the global indices relative to measurement
        - combined_indices are indices relative to the combined data

        Breaking this down even more, suppose there are three measurements in
        a site at a sampling frequency of 10 Hz. Together, these measurements
        will span a range of global indices with gaps in between, for example:

        - Measurement 1: 0, 1, 2, 3, 4
        - Measurement 2: 20, 21, 22, 23, 24
        - Measurement 3: 30, 31, 32, 33, 34

        Relative to their own start times, the indices will be:

        - Measurement 1: 0, 1, 2, 3, 4
        - Measurement 2: 0, 1, 2, 3, 4
        - Measurement 3: 0, 1, 2, 3, 4

        Finally, when combining data for a site, these need to be reindexed
        relative to the number of contributing windows in the site.

        - Measurement 1: 0, 1, 2, 3, 4
        - Measurement 2: 5, 6, 7, 8, 9
        - Measurement 3: 10, 11, 12, 13, 14

        This method returns the mapping from the index relative to the
        measurement to the index relative to the site. For Measurement 2, this
        would be:

        0, 1, 2, 3, 4 -> 5, 6, 7, 8, 9

        Parameters
        ----------
        eval_wins : pd.DataFrame
            The global windows for the evaluation frequencies
        site : Site
            The site instance
        meas_name : str
            The name of the measurement
        level_metadata : SpectraLevelMetadata
            The spectra level metadata

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Indices relative to the spectra data and indices relative to the
            site combined data
        """
        eval_wins["combined_index"] = np.arange(len(eval_wins))
        eval_meas_wins = eval_wins[eval_wins[site.name] == meas_name]
        # this is the local window indices for this measurement
        eval_data_indices = eval_meas_wins.index.values - level_metadata.index_offset
        # this is the combined indices for this measurement
        combined_indices = eval_meas_wins["combined_index"].values
        return eval_data_indices, combined_indices


class QuickGather(ResisticsProcess):
    """
    Processor to gather data outside of a resistics environment

    This is intended for use when quickly calculating out a transfer function
    for a single measurement and only a single spectra data instance is accepted
    as input.

    Remote reference or intersite processing is not possible using QuickGather

    See Also
    --------
    ProjectGather : For more advanced gathering of data in a project
    """

    def run(
        self,
        dir_path: Path,
        dec_params: DecimationParameters,
        tf: TransferFunction,
        eval_data: SpectraData,
    ) -> GatheredData:
        """
        Generate the GatheredData object for input into regression preparation

        The input is a single spectra data instance and is used to populate the
        in_data, out_data and cross_data.

        Parameters
        ----------
        dir_path : Path
            The directory path to the measurement
        dec_params : DecimationParameters
            The decimation parameters
        tf : TransferFunction
            The transfer function
        eval_data : SpectraData
            The spectra data at the evaluation frequencies

        Returns
        -------
        GatheredData
            GatheredData for regression preparer
        """
        metadata = eval_data.metadata
        out_data = {}
        in_data = {}
        cross_data = {}
        eval_freqs = []
        logger.info("Quick gathering data for regression prepartion")
        for ilevel in range(metadata.n_levels):
            level_metadata = metadata.levels_metadata[ilevel]
            eval_freqs = eval_freqs + level_metadata.freqs
            out_level = eval_data.get_chans(ilevel, tf.out_chans)
            in_level = eval_data.get_chans(ilevel, tf.in_chans)
            cross_level = eval_data.get_chans(ilevel, tf.cross_chans)
            for ifreq in range(level_metadata.n_freqs):
                key = dec_params.per_level * ilevel + ifreq
                out_data[key] = out_level[..., ifreq]
                in_data[key] = in_level[..., ifreq]
                cross_data[key] = cross_level[..., ifreq]
        # make combined data
        fs = dec_params.fs
        out_combined = self._get_combined_data(
            dir_path.name, fs, tf.out_chans, eval_freqs, metadata, out_data
        )
        in_combined = self._get_combined_data(
            dir_path.name, fs, tf.in_chans, eval_freqs, metadata, in_data
        )
        cross_combined = self._get_combined_data(
            dir_path.name, fs, tf.cross_chans, eval_freqs, metadata, cross_data
        )
        return GatheredData(
            out_data=out_combined, in_data=in_combined, cross_data=cross_combined
        )

    def _get_combined_data(
        self,
        meas: str,
        fs: float,
        chans: List[str],
        eval_freqs: List[float],
        metadata: SpectraMetadata,
        data: Dict[int, np.ndarray],
    ) -> SiteCombinedData:
        """Get the combined metadata"""
        combined_metadata = SiteCombinedMetadata(
            site_name=meas,
            fs=fs,
            system=metadata.system,
            serial=metadata.serial,
            wgs84_latitude=metadata.wgs84_latitude,
            wgs84_longitude=metadata.wgs84_longitude,
            easting=metadata.easting,
            northing=metadata.northing,
            elevation=metadata.elevation,
            measurements=[meas],
            chans=chans,
            n_evals=len(eval_freqs),
            eval_freqs=eval_freqs,
            histories={meas: metadata.history},
        )
        return SiteCombinedData(combined_metadata, data)
