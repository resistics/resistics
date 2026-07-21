"""Public gathered-data models and evaluation-data assembly processes."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from loguru import logger
from pydantic import Field, model_validator

from resistics.common import (
    History,
    ResisticsData,
    ResisticsProcess,
    WriteableMetadata,
    validate_output_label,
)
from resistics.decimate import DecimationParameters
from resistics.gather_criteria import GatherCriteria, GatherSelection
from resistics.gather_plan import _EvaluationLocator, _GatherPlan
from resistics.spectra import SpectraData, SpectraMetadata
from resistics.transfunc import TransferFunction

__all__ = [
    "EvaluationFrequencyGather",
    "GatheredData",
    "QuickGather",
    "SiteCombinedData",
    "SiteCombinedMetadata",
]


def _cross_channels(tf: TransferFunction) -> list[str]:
    return tf.cross_chans


class EvaluationFrequencyGather(ResisticsProcess):
    """Gather every persisted target-run evaluation artifact for regression.

    The selection is deliberately a separate input: it makes the persisted
    evaluation artifact boundary explicit and leaves window-selection policy in
    :class:`GatherCriteria`, not in a job or parameter file.  Remote-reference
    alignment is intentionally not guessed; an explicit resolved remote is
    rejected until the cross-station aligner is implemented.
    """

    input_types: ClassVar[dict[str, str]] = {"selection": "gather_selection"}
    output_type: ClassVar[str] = "gathered_data"
    runtime_requirements: ClassVar[list[str]] = ["project_path"]

    def execute(self, inputs: dict[str, Any], context: Any) -> GatheredData:
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
    def _combine(values: list[GatheredData]) -> GatheredData:
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


class SiteCombinedMetadata(WriteableMetadata):
    """
    Metadata for combined data

    Combined metadata stores metadata for measurements that are combined from
    a single site.
    """

    site_name: str
    """The name of the site"""
    site_names: list[str] = Field(default_factory=list)
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
    measurements: list[str] | None = None
    """List of measurement names that were included in the combined data"""
    chans: list[str]
    """List of channels, these are common amongst all the measurements"""
    n_evals: int
    """The number of evaluation frequencies"""
    eval_freqs: list[float]
    """The evaluation frequencies"""
    histories: dict[str, History]
    """Dictionary mapping measurement name to measurement processing history"""

    @model_validator(mode="after")
    def populate_site_names(self) -> SiteCombinedMetadata:
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

    def __init__(self, metadata: SiteCombinedMetadata, data: dict[int, np.ndarray]):
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


class _GatherAssembler:
    """Assemble one immutable gather plan into regression-ready arrays.

    Parameters
    ----------
    record_factory : Callable[[list[str]], Any]
        Factory used to append gather records to copied histories.
    """

    def __init__(self, record_factory: Callable[[list[str]], Any]):
        self.record_factory = record_factory

    def assemble(
        self,
        plan: _GatherPlan,
        target: str,
        tf: TransferFunction,
        target_artifacts: dict[str, Any],
        remote_artifacts: dict[str, dict[str, Any]],
    ) -> GatheredData:
        """Materialise selected rows and combined metadata from a gather plan.

        Parameters
        ----------
        plan : _GatherPlan
            Immutable aligned window selections.
        target : str
            Canonical target station path.
        tf : TransferFunction
            Transfer function defining output, input, and cross channels.
        target_artifacts : dict[str, Any]
            Validated target artifacts keyed by run path.
        remote_artifacts : dict[str, dict[str, Any]]
            Compatible remote artifacts keyed by station and run path.

        Returns
        -------
        GatheredData
            Window-aligned arrays and source metadata ready for regression.
        """
        cross_channels = _cross_channels(tf)
        out_values = {}
        in_values = {}
        cross_values = {}
        for evaluation in plan.evaluations:
            out_values[evaluation.key] = self._extract(
                evaluation.target_locators,
                evaluation.level,
                evaluation.evaluation_index,
                tf.out_chans,
            )
            in_values[evaluation.key] = self._extract(
                evaluation.target_locators,
                evaluation.level,
                evaluation.evaluation_index,
                tf.in_chans,
            )
            cross_values[evaluation.key] = self._extract(
                evaluation.cross_locators,
                evaluation.level,
                evaluation.evaluation_index,
                cross_channels,
            )

        eval_freqs = [evaluation.frequency for evaluation in plan.evaluations]
        out_metadata = self._combined_metadata(
            target,
            [target],
            plan.target_runs,
            tf.out_chans,
            eval_freqs,
            target_artifacts,
        )
        in_metadata = self._combined_metadata(
            target,
            [target],
            plan.target_runs,
            tf.in_chans,
            eval_freqs,
            target_artifacts,
        )
        cross_stations = (
            [target] if not plan.remote_enabled else list(plan.usable_remotes)
        )
        cross_source = (
            target_artifacts
            if not plan.remote_enabled
            else {
                run: artifact
                for station_artifacts in remote_artifacts.values()
                for run, artifact in station_artifacts.items()
            }
        )
        cross_metadata = self._combined_metadata(
            target if not plan.remote_enabled else "+".join(cross_stations),
            cross_stations,
            plan.target_runs if not plan.remote_enabled else plan.remote_runs,
            cross_channels,
            eval_freqs,
            cross_source,
        )
        return GatheredData(
            SiteCombinedData(out_metadata, out_values),
            SiteCombinedData(in_metadata, in_values),
            SiteCombinedData(cross_metadata, cross_values),
        )

    @staticmethod
    def _extract(
        locators: tuple[_EvaluationLocator, ...],
        level: int,
        evaluation_index: int,
        channels: list[str],
    ) -> np.ndarray:
        """Extract selected rows once per source artifact.

        Parameters
        ----------
        locators : tuple[_EvaluationLocator, ...]
            Ordered source rows selected by the planner.
        level : int
            Realised decimation level.
        evaluation_index : int
            Evaluation-frequency index within the level.
        channels : list[str]
            Channel names to extract.

        Returns
        -------
        np.ndarray
            Complex array shaped as selected windows by channels.

        Raises
        ------
        ChannelNotFoundError
            If an artifact does not contain a requested channel.
        """
        from resistics.errors import ChannelNotFoundError

        values = np.empty((len(locators), len(channels)), dtype=np.complex128)
        grouped: dict[str, tuple[Any, list[int], list[int]]] = {}
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
        site_names: list[str],
        used_runs: frozenset[str],
        channels: list[str],
        eval_freqs: list[float],
        artifacts: dict[str, Any],
    ) -> SiteCombinedMetadata:
        """Build source-aware metadata for one gathered channel role.

        Parameters
        ----------
        site_name : str
            Display name for the contributing station set.
        site_names : list[str]
            Canonical contributing station paths.
        used_runs : frozenset[str]
            Run paths contributing selected rows.
        channels : list[str]
            Channels represented by the assembled arrays.
        eval_freqs : list[float]
            Ordered realised evaluation frequencies.
        artifacts : dict[str, Any]
            Source artifacts keyed by run path.

        Returns
        -------
        SiteCombinedMetadata
            Combined source identity, channel, frequency, and history metadata.
        """
        first_run = sorted(used_runs)[0]
        first_artifact = artifacts[first_run]
        first = first_artifact.spectra_data.metadata
        histories = {}
        for run_path in sorted(used_runs):
            history = artifacts[run_path].spectra_data.metadata.history.model_copy(
                deep=True
            )
            history.add_record(
                self.record_factory(
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


class QuickGather(ResisticsProcess):
    """
    Processor to gather data outside of a resistics environment

    This is intended for use when quickly calculating out a transfer function
    for a single measurement and only a single spectra data instance is accepted
    as input.

    Remote reference or intersite processing is not possible using QuickGather

    See Also
    --------
    Gather : For persisted, aligned project gathering.
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
        cross_channels = _cross_channels(tf)
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
            cross_level = eval_data.get_chans(ilevel, cross_channels)
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
            dir_path.name, fs, cross_channels, eval_freqs, metadata, cross_data
        )
        return GatheredData(
            out_data=out_combined, in_data=in_combined, cross_data=cross_combined
        )

    def _get_combined_data(
        self,
        meas: str,
        fs: float,
        chans: list[str],
        eval_freqs: list[float],
        metadata: SpectraMetadata,
        data: dict[int, np.ndarray],
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
