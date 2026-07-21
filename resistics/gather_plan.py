"""Immutable window-alignment planning for persisted gather operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from loguru import logger

from resistics.gather_project import _GatherProjectSource, _RemoteDiscovery


@dataclass(frozen=True)
class _EvaluationLocator:
    """One evaluation row addressed by its station-global window index.

    Attributes
    ----------
    run_path : str
        Persisted artifact run path.
    local_index : int
        Row index within that run's level array.
    artifact : Any
        Loaded evaluation-frequency artifact.
    """

    run_path: str
    local_index: int
    artifact: Any


@dataclass(frozen=True)
class _GatherEvaluationPlan:
    """Aligned rows for one realised evaluation frequency.

    Attributes
    ----------
    key : int
        Contiguous gathered-data evaluation key.
    level : int
        Realised decimation level.
    evaluation_index : int
        Evaluation index within the level.
    frequency : float
        Evaluation frequency in Hz.
    target_locators : tuple[_EvaluationLocator, ...]
        Target rows supplying output and input channels.
    cross_locators : tuple[_EvaluationLocator, ...]
        Local or remote rows supplying cross channels.
    """

    key: int
    level: int
    evaluation_index: int
    frequency: float
    target_locators: tuple[_EvaluationLocator, ...]
    cross_locators: tuple[_EvaluationLocator, ...]


@dataclass(frozen=True)
class _GatherPlan:
    """Complete immutable execution plan for gathered-data assembly.

    Attributes
    ----------
    evaluations : tuple[_GatherEvaluationPlan, ...]
        Ordered per-frequency row selections.
    target_runs : frozenset[str]
        Target artifacts that contribute at least one row.
    remote_runs : frozenset[str]
        Remote artifacts that contribute at least one row.
    usable_remotes : tuple[str, ...]
        Remote stations contributing aligned admitted windows.
    remote_enabled : bool
        Whether cross channels are sourced from remote artifacts.
    """

    evaluations: tuple[_GatherEvaluationPlan, ...]
    target_runs: frozenset[str]
    remote_runs: frozenset[str]
    usable_remotes: tuple[str, ...]
    remote_enabled: bool


class _GatherPlanner:
    """Plan mask admission and global-window alignment without assembling arrays.

    Parameters
    ----------
    source : _GatherProjectSource
        Source responsible for persisted mask admission.
    """

    def __init__(self, source: _GatherProjectSource):
        self.source = source

    def build(
        self,
        target: str,
        sample_rate: float,
        target_artifacts: dict[str, Any],
        level_references: dict[int, Any],
        remotes: _RemoteDiscovery,
    ) -> _GatherPlan:
        """Build an immutable plan for every realised evaluation frequency.

        Parameters
        ----------
        target : str
            Canonical target station path.
        sample_rate : float
            Original sampling frequency.
        target_artifacts : dict[str, Any]
            Validated target artifacts keyed by run path.
        level_references : dict[int, Any]
            Authoritative target artifact for each realised level.
        remotes : _RemoteDiscovery
            Compatible remote artifacts and candidate diagnostics.

        Returns
        -------
        _GatherPlan
            Ordered admitted row pairs and contributing source identities.

        Raises
        ------
        ValueError
            If a level/frequency has no admissible rows or no configured remote
            can contribute.
        """
        evaluations = []
        target_used: set[str] = set()
        remote_used: set[str] = set()
        usable_remotes: set[str] = set()
        remote_enabled = remotes.setting is not None

        for level, level_reference in level_references.items():
            level_metadata = level_reference.spectra_data.metadata.levels_metadata[
                level
            ]
            target_catalog = self._catalog(target_artifacts, level, target)
            for evaluation_index, evaluation_frequency in enumerate(
                level_metadata.freqs
            ):
                target_valid = self.source.admit_windows(
                    target,
                    sample_rate,
                    level,
                    evaluation_index,
                    target_catalog,
                )
                pairs = self._align_frequency(
                    target,
                    sample_rate,
                    level,
                    evaluation_index,
                    target_valid,
                    remotes,
                    usable_remotes,
                )
                if not pairs:
                    reasons = "; ".join(
                        f"{name}: {reason}"
                        for name, reason in sorted(remotes.candidate_reasons.items())
                    )
                    suffix = f" Candidate diagnostics: {reasons}" if reasons else ""
                    raise ValueError(
                        f"No admissible gather windows for target {target}, "
                        f"{sample_rate:g} Hz, level {level}, evaluation-frequency "
                        f"index {evaluation_index}.{suffix}"
                    )
                target_locs = tuple(pair[0] for pair in pairs)
                cross_locs = tuple(pair[1] for pair in pairs)
                evaluations.append(
                    _GatherEvaluationPlan(
                        key=len(evaluations),
                        level=level,
                        evaluation_index=evaluation_index,
                        frequency=evaluation_frequency,
                        target_locators=target_locs,
                        cross_locators=cross_locs,
                    )
                )
                target_used.update(locator.run_path for locator in target_locs)
                if remote_enabled:
                    remote_used.update(locator.run_path for locator in cross_locs)

        if remote_enabled and not usable_remotes:
            raise ValueError(
                f"No usable remote reference remains for target {target} at "
                f"{sample_rate:g} Hz"
            )
        return _GatherPlan(
            evaluations=tuple(evaluations),
            target_runs=frozenset(target_used),
            remote_runs=frozenset(remote_used),
            usable_remotes=tuple(sorted(usable_remotes)),
            remote_enabled=remote_enabled,
        )

    def _align_frequency(
        self,
        target: str,
        sample_rate: float,
        level: int,
        evaluation_index: int,
        target_valid: dict[int, _EvaluationLocator],
        remotes: _RemoteDiscovery,
        usable_remotes: set[str],
    ) -> list[tuple[_EvaluationLocator, _EvaluationLocator]]:
        """Align admitted local or remote rows for one evaluation frequency.

        Parameters
        ----------
        target : str
            Canonical target station path.
        sample_rate : float
            Original sampling frequency.
        level : int
            Realised decimation level.
        evaluation_index : int
            Evaluation-frequency index within the level.
        target_valid : dict[int, _EvaluationLocator]
            Mask-admitted target rows keyed by global window.
        remotes : _RemoteDiscovery
            Compatible remote artifacts and the resolved policy.
        usable_remotes : set[str]
            Mutable operation-local accumulator of contributing stations.

        Returns
        -------
        list[tuple[_EvaluationLocator, _EvaluationLocator]]
            Ordered target/cross row pairs.
        """
        if remotes.setting is None:
            return [
                (target_valid[global_index], target_valid[global_index])
                for global_index in sorted(target_valid)
            ]
        pairs = []
        skipped = []
        for remote_path in sorted(remotes.artifacts):
            remote_catalog = self._catalog(
                remotes.artifacts[remote_path], level, remote_path
            )
            remote_valid = self.source.admit_windows(
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
        return pairs

    @staticmethod
    def _catalog(
        artifacts: dict[str, Any], level: int, station_path: str
    ) -> dict[int, _EvaluationLocator]:
        """Index realised artifact rows by station-global window number.

        Parameters
        ----------
        artifacts : dict[str, Any]
            Evaluation artifacts keyed by run path.
        level : int
            Realised decimation level to index.
        station_path : str
            Canonical station path used in diagnostics.

        Returns
        -------
        dict[int, _EvaluationLocator]
            Unique global-window-indexed source rows.

        Raises
        ------
        ValueError
            If two runs claim the same station-global window.
        """
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
                        f"{station_path}, level {level}: "
                        f"{catalog[global_index].run_path} and {run_path}"
                    )
                catalog[global_index] = _EvaluationLocator(
                    run_path, local_index, artifact
                )
        return catalog
