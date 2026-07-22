"""Project discovery and persisted-input loading for gather planning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from loguru import logger

from resistics.common import validate_output_label
from resistics.gather_criteria import GatherCriteria, _validate_station_path
from resistics.mask import WindowMask, WindowMaskReader, get_run_mask_path
from resistics.spectra import EvaluationFrequencyReader

if TYPE_CHECKING:
    from resistics.project import Project


@dataclass(frozen=True)
class _RemoteDiscovery:
    """Compatible remote artifacts and diagnostics for one gather target.

    **Attributes**

    - **setting** — Resolved remote-reference policy.
    - **artifacts** — Compatible artifacts keyed by station and run path.
    - **candidate_reasons** — Rejection reason for each unusable automatic candidate.
    """

    setting: Literal["auto"] | list[str] | None
    artifacts: dict[str, dict[str, Any]]
    candidate_reasons: dict[str, str]


class _GatherProjectSource:
    """Own project discovery, persisted artifact reads, and mask reads.

    :param project: Open project used for run and concurrent-station discovery.
    :param project_path: Root containing persisted evaluation and mask artifacts.
    :param criteria: Resolved policy source used when admitting masked windows.
    :param output_label: Persisted artifact namespace.
    """

    def __init__(
        self,
        project: Project,
        project_path: Path,
        criteria: GatherCriteria,
        output_label: str,
    ) -> None:
        self.project = project
        self.project_path = project_path
        self.criteria = criteria
        self.output_label = validate_output_label(output_label)
        self.reader = EvaluationFrequencyReader(label=self.output_label)
        self.artifact_cache: dict[str, Any] = {}
        self.mask_cache: dict[tuple[str, str], WindowMask] = {}

    def load_target(
        self,
        run_paths: list[str],
        station_path: str,
    ) -> tuple[dict[str, Any], dict[int, Any]]:
        """Load and validate target artifacts and per-level references.

        :param run_paths: Target run paths selected by the job batch.
        :param station_path: Canonical target station path.

        :return: Artifacts by run and compatible reference artifact by level.

        :raises ValueError: If no run is selected, an artifact is unreadable, or target
            metadata is incompatible.
        """
        if not run_paths:
            raise ValueError(f"Target {station_path} has no runs")
        artifacts = self._load_runs(run_paths, required=True, role="target")
        baseline = next(iter(artifacts.values()))
        for run_path, artifact in artifacts.items():
            self._validate_compatible(baseline, artifact, f"target run {run_path}")
        return artifacts, self._level_references(artifacts, station_path)

    def discover_remotes(
        self,
        target: str,
        sample_rate: float,
        setting: Literal["auto"] | list[str] | None,
        baseline: Any,
        level_references: dict[int, Any],
    ) -> _RemoteDiscovery:
        """Discover readable, compatible artifacts for the remote policy.

        :param target: Target station path.
        :param sample_rate: Original sampling frequency.
        :param setting: Resolved remote-reference policy.
        :param baseline: Target artifact used for whole-run compatibility.
        :param level_references: Authoritative target artifact for each realised level.

        :return: Compatible remote artifacts plus automatic-candidate diagnostics.

        :raises ValueError: If an explicit remote is absent, unreadable, or incompatible.
        """
        if setting is None:
            return _RemoteDiscovery(setting, {}, {})
        automatic = setting == "auto"
        candidate_paths = (
            self.project.get_concurrent(target, sample_rate)
            if automatic
            else sorted(setting)
        )
        remote_artifacts: dict[str, dict[str, Any]] = {}
        candidate_reasons: dict[str, str] = {}
        for station_path in sorted(candidate_paths):
            try:
                artifacts, failure_reason = self._candidate_artifacts(
                    station_path,
                    sample_rate,
                    baseline,
                    level_references,
                    automatic,
                )
            except ValueError as exc:
                if not automatic:
                    raise
                candidate_reasons[station_path] = str(exc)
                logger.warning(
                    f"Skipping automatic remote {station_path} for {target}: {exc}"
                )
                continue
            if artifacts:
                remote_artifacts[station_path] = artifacts
            else:
                candidate_reasons[station_path] = failure_reason or (
                    "no compatible evaluation artifacts"
                )
        return _RemoteDiscovery(setting, remote_artifacts, candidate_reasons)

    def admit_windows(
        self,
        station_path: str,
        sample_rate: float,
        level: int,
        evaluation_index: int,
        catalog: dict[int, Any],
    ) -> dict[int, Any]:
        """Apply every configured persisted mask to a window catalog.

        :param station_path: Target or remote station owning the catalog.
        :param sample_rate: Original sampling frequency.
        :param level: Decimation level.
        :param evaluation_index: Evaluation-frequency index within the level.
        :param catalog: Global-window-indexed evaluation locators.

        :return: Locators admitted by the configured mask policy.
        """
        policy = self.criteria.resolve(
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

    def _candidate_artifacts(
        self,
        station_path: str,
        sample_rate: float,
        baseline: Any,
        level_references: dict[int, Any],
        automatic: bool,
    ) -> tuple[dict[str, Any], str | None]:
        """Return compatible artifacts for one remote candidate.

        :param station_path: Canonical candidate station path.
        :param sample_rate: Original sampling frequency.
        :param baseline: Target artifact used for whole-run compatibility.
        :param level_references: Authoritative target artifact for each realised level.
        :param automatic: Whether incompatibility should skip this candidate.

        :return: Compatible candidate artifacts and an automatic-candidate rejection reason when no artifact remains.

        :raises ValueError: If an explicit candidate has no usable runs or is incompatible.
        """
        run_paths = self._station_runs(station_path, sample_rate)
        artifacts = self._load_runs(
            run_paths,
            required=not automatic,
            role=f"remote {station_path}",
        )
        if not artifacts:
            return {}, "no readable evaluation artifacts"
        compatible = {}
        for run_path, artifact in artifacts.items():
            try:
                self._validate_compatible(baseline, artifact, f"remote run {run_path}")
                for level, reference in level_references.items():
                    if level < artifact.spectra_data.metadata.n_levels:
                        self._validate_level_compatible(
                            reference,
                            artifact,
                            level,
                            f"remote run {run_path}",
                        )
            except ValueError as exc:
                if not automatic:
                    raise
                logger.warning(
                    f"Skipping incompatible automatic remote run {run_path}: {exc}"
                )
                continue
            compatible[run_path] = artifact
        if not compatible:
            return {}, "no compatible evaluation artifacts"
        return compatible, None

    def _station_runs(self, station_path: str, sample_rate: float) -> list[str]:
        """Discover run paths for one station and rate from the project index.

        :param station_path: Canonical station path.
        :param sample_rate: Original sampling frequency.

        :return: Sorted matching run paths.

        :raises ValueError: If the station path is invalid or has no runs at the rate.
        """
        _validate_station_path(station_path)
        rows = self.project.table[
            (self.project.table["station_path"] == station_path)
            & np.isclose(self.project.table["sample_rate"].astype(float), sample_rate)
        ]
        run_paths = sorted(rows["run_path"].unique().tolist())
        if not run_paths:
            raise ValueError(
                f"Remote station {station_path!r} has no runs at {sample_rate:g} Hz"
            )
        return run_paths

    def _load_runs(
        self, run_paths: list[str], required: bool, role: str
    ) -> dict[str, Any]:
        """Load cached evaluation artifacts and contextualise read failures.

        :param run_paths: Run paths to load.
        :param required: Whether any unreadable run makes the operation fail.
        :param role: Target or remote description used in diagnostics.

        :return: Readable artifacts keyed by run path.

        :raises ValueError: If a required artifact cannot be read.
        """
        values = {}
        failures = []
        for run_path in sorted(run_paths):
            if run_path in self.artifact_cache:
                values[run_path] = self.artifact_cache[run_path]
                continue
            survey, station, run = run_path.split("/", 2)
            try:
                artifact = self.reader.execute(
                    {},
                    {
                        "project_path": self.project_path,
                        "run_batch": {
                            "survey": survey,
                            "station": station,
                            "run": run,
                        },
                    },
                )
            # Reader plugins and codecs can raise non-I/O exceptions. This is
            # the boundary that converts them into run-specific diagnostics.
            except Exception as exc:
                failures.append(f"{run_path}: {exc}")
                continue
            self.artifact_cache[run_path] = artifact
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
        """Validate whole-artifact and realised-level compatibility.

        :param reference: Authoritative target artifact.
        :param candidate: Target or remote artifact to validate.
        :param description: Source identity used in diagnostics.

        :raises ValueError: If sampling, time, or realised-level metadata differs.
        """
        ref_dec = reference.decimation_parameters
        got_dec = candidate.decimation_parameters
        failures = []
        if not np.isclose(ref_dec.fs, got_dec.fs):
            failures.append("original sample rate")
        ref_data = reference.spectra_data
        got_data = candidate.spectra_data
        if str(ref_data.metadata.ref_time) != str(got_data.metadata.ref_time):
            failures.append("reference time")
        for level in range(min(ref_data.metadata.n_levels, got_data.metadata.n_levels)):
            failures.extend(
                _GatherProjectSource._level_compatibility_failures(
                    reference, candidate, level
                )
            )
        if failures:
            raise ValueError(
                f"Incompatible {description}: {', '.join(dict.fromkeys(failures))}"
            )

    @staticmethod
    def _level_compatibility_failures(
        reference: Any, candidate: Any, level: int
    ) -> list[str]:
        """Compare one realised evaluation level using spectra metadata only.

        :param reference: Authoritative target artifact.
        :param candidate: Artifact to compare.
        :param level: Realised decimation level.

        :return: Human-readable incompatible fields.
        """
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
        """Raise with context when one realised level is incompatible.

        :param reference: Authoritative target artifact.
        :param candidate: Artifact to compare.
        :param level: Realised decimation level.
        :param description: Source identity used in diagnostics.

        :raises ValueError: If level sampling, windows, or evaluation frequencies differ.
        """
        failures = _GatherProjectSource._level_compatibility_failures(
            reference, candidate, level
        )
        if failures:
            raise ValueError(f"Incompatible {description}: {', '.join(failures)}")

    @staticmethod
    def _level_references(
        artifacts: dict[str, Any], station_path: str
    ) -> dict[int, Any]:
        """Choose and validate an authoritative target artifact for each level.

        :param artifacts: Validated target artifacts keyed by run path.
        :param station_path: Canonical target station path used in diagnostics.

        :return: Authoritative artifact for each realised level.

        :raises ValueError: If no level is realised or target level metadata differs.
        """
        n_levels = max(
            artifact.spectra_data.metadata.n_levels for artifact in artifacts.values()
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
                _GatherProjectSource._validate_level_compatible(
                    reference,
                    artifact,
                    level,
                    f"target run {run_path} for station {station_path}",
                )
            references[level] = reference
        return references

    def _load_mask(
        self,
        station_path: str,
        run_path: str,
        name: str,
        artifact: Any,
        level: int,
        evaluation_index: int,
    ) -> WindowMask:
        """Load and validate one required mask with full gather context.

        :param station_path: Target or remote station owning the mask.
        :param run_path: Run whose windows are being admitted.
        :param name: Persisted mask name.
        :param artifact: Evaluation artifact the mask must describe.
        :param level: Realised decimation level.
        :param evaluation_index: Evaluation-frequency index within the level.

        :return: Compatible cached or newly read mask.

        :raises ValueError: If the required mask is unavailable or incompatible.
        """
        cache_key = (run_path, name)
        if cache_key in self.mask_cache:
            return self.mask_cache[cache_key]
        survey, station, run = run_path.split("/", 2)
        path = get_run_mask_path(
            self.project_path,
            {"survey": survey, "station": station, "run": run},
            name,
            self.output_label,
        )
        try:
            mask = WindowMaskReader().run(path)
        # Mask readers may surface JSON, validation, and filesystem failures;
        # convert all of them here into the full window-selection context.
        except Exception as exc:
            raise ValueError(
                f"Required mask {name!r} is unavailable for target/remote "
                f"{station_path}, run {run_path}, level {level}, "
                f"evaluation-frequency index {evaluation_index}: {exc}"
            ) from exc
        self._validate_mask(mask, artifact, run_path, name)
        self.mask_cache[cache_key] = mask
        return mask

    @staticmethod
    def _validate_mask(
        mask: WindowMask, artifact: Any, run_path: str, name: str
    ) -> None:
        """Validate mask identity and every realised level signature.

        :param mask: Persisted mask to validate.
        :param artifact: Evaluation artifact the mask must describe.
        :param run_path: Source run used in diagnostics.
        :param name: Mask name used in diagnostics.

        :raises ValueError: If sampling, time, level, frequency, or window metadata differs.
        """
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
                mask.metadata.levels, data.metadata.levels_metadata, strict=False
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
