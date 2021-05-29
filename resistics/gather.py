"""
Module for gathering data that will be combined to calculate transfer functions
"""
from loguru import logger
from typing import List, Dict, Optional
import numpy as np
import pandas as pd

from resistics.common import ResisticsProcess, ResisticsData
from resistics.decimate import DecimationParameters
from resistics.spectra import SpectraMetadata
from resistics.letsgo import Project, Site


def get_site_spectra_metadata(
    proj: Project, site: Site, fs
) -> Dict[str, SpectraMetadata]:
    """Get the spectra metadata for a site and specific sampling frequency"""
    from resistics.letsgo import get_meas_spectra_path
    from resistics.spectra import SpectraDataReader

    config_name = proj.config.name
    measurements = site.get_measurements(fs=fs)
    meas_metadata = {}
    for meas in measurements:
        meas = site[meas]
        spectra_path = get_meas_spectra_path(
            proj.dir_path, site.name, meas.name, config_name
        )
        try:
            metadata = SpectraDataReader().run(spectra_path, metadata_only=True)
        except Exception:
            logger.error(f"No spectra data found in path {spectra_path}")
            continue
        logger.info(f"Found spectra data for {site.name}, {meas.name}, {config_name}")
        meas_metadata[meas.name] = metadata
    return meas_metadata


def get_site_level_wins(
    meas_metadata: Dict[str, SpectraMetadata], fs: float, level: int
) -> pd.Series:
    """Get site windows for a decimation level given a sampling frequency"""
    level_wins = pd.Series()
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


def get_site_wins(proj: Project, site: Site, fs: float) -> Dict[int, pd.Series]:
    """Get site windows for all levels"""
    meas_metadata = get_site_spectra_metadata(proj, site, fs)
    n_levels = max([x.n_levels for x in meas_metadata.values()])
    tables = {}
    for ilevel in range(n_levels):
        tables[ilevel] = get_site_level_wins(meas_metadata, fs, ilevel)
    return tables


class Selection(ResisticsData):
    def __init__(
        self,
        sites: List[Site],
        dec_params: DecimationParameters,
        tables: Dict[int, pd.DataFrame],
        per_level: int,
    ):
        self.sites = sites
        self.dec_params = dec_params
        self.tables = tables
        self.n_levels = len(tables)
        self.per_level = per_level


class Selector(ResisticsProcess):
    def run(
        self,
        proj: Project,
        sites: List[Site],
        dec_params: DecimationParameters,
        masks: Optional[Dict[str, str]] = None,
    ) -> Selection:

        fs = dec_params.fs
        sites_wins = {site.name: get_site_wins(proj, site, fs) for site in sites}
        # get the higest decimation level that all sites have
        n_levels = min([len(x) for x in sites_wins.values()])
        logger.info(f"Finding shared windows across {', '.join(sites_wins.keys())}")
        tables: Dict[int, pd.DataFrame] = {}
        for ilevel in range(n_levels):
            logger.info(f"Finding shared windows for decimation level {ilevel}")
            data = {x: y[ilevel] for x, y in sites_wins.items()}
            table = pd.DataFrame(data=data).dropna()
            table = self._get_evals(dec_params, table, ilevel)
            if masks is not None:
                table = self._apply_masks(table, masks)
            tables[ilevel] = table
        return Selection(sites, dec_params, tables, dec_params.per_level)

    def _get_evals(
        self, dec_params: DecimationParameters, table: pd.DataFrame, level: int
    ) -> pd.DataFrame:
        """Add a column for each evaluation frequency"""
        for ifreq in range(dec_params.per_level):
            table[ifreq] = True
        return table

    def _apply_masks(self, table: pd.DataFrame, masks: Dict[str, str]) -> pd.DataFrame:
        return table


class GatheredData(ResisticsData):
    def __init__(
        self,
        out_data: Dict[int, np.ndarray],
        in_data: Optional[Dict[int, np.ndarray]] = None,
        cross_data: Optional[Dict[int, np.ndarray]] = None,
    ) -> None:
        self.out_data = out_data
        self.in_data = out_data if in_data is None else in_data
        self.cross_data = out_data if cross_data is None else cross_data


class Gather(ResisticsProcess):
    def run(
        self,
        selection: Selection,
        out_site: Site,
        in_site: Optional[Site] = None,
        cross_site: Optional[Site] = None,
    ) -> GatheredData:
        pass
