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

    Within a project instance, they have separate folders for users who want to
    save both the full spectra data with all the frequencies as well as the
    evaluation frequency spectra data with the smaller subset of frequencies.
    Only the evaluation frequency data is required to calculate the transfer
    function, but the full spectral data might be useful for visualisation and
    analysis reasons.
"""
from loguru import logger
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import numpy as np
import pandas as pd

from resistics.common import ResisticsProcess, ResisticsData
from resistics.common import WriteableMetadata, History
from resistics.project import Project, Site
from resistics.decimate import DecimationParameters
from resistics.spectra import SpectraLevelMetadata, SpectraMetadata, SpectraData
from resistics.spectra import SpectraDataReader
from resistics.transfunc import TransferFunction


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
    dtype: object
    >>> get_site_level_wins(meas_metadata, 1)
    2      meas1
    3      meas1
    25     meas2
    26     meas2
    27     meas2
    104    meas3
    105    meas3
    dtype: object
    >>> get_site_level_wins(meas_metadata, 2)
    1      meas1
    2      meas1
    22     meas2
    23     meas2
    102    meas3
    dtype: object
    """
    level_wins = pd.Series(dtype=str)
    for meas_name, metadata in meas_metadata.items():
        if level >= metadata.n_levels:
            continue
        level_metadata = metadata.levels_metadata[level]
        first_global = level_metadata.index_offset
        last_gobal = level_metadata.index_offset + level_metadata.n_wins - 1
        index = np.arange(first_global, last_gobal + 1)
        level_wins = level_wins.append(pd.Series(data=meas_name, index=index))
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

        # collect the data
        out_data = self._get_site_data(
            config_name, proj, selection, out_name, tf.out_chans
        )
        in_data = self._get_site_data(
            config_name, proj, selection, in_name, tf.in_chans
        )
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
        Collect the evals data for the site

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
        eval_data_indices = eval_meas_wins.index.values - level_metadata.index_offset
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
