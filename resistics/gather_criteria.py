"""Criteria models and validation for persisted evaluation-data gathering."""

from __future__ import annotations

from typing import Any, ClassVar, Literal

import numpy as np
from pydantic import ConfigDict, Field, field_validator, model_validator

from resistics.common import ResisticsData, ResisticsModel, ResisticsProcess
from resistics.mask import validate_mask_name

__all__ = [
    "GatherCriteria",
    "GatherSelection",
    "MaskCriteria",
    "RateGatherCriteria",
    "ResolvedGatherCriteria",
    "StationGatherCriteria",
]


def _validate_station_path(value: str) -> str:
    parts = value.split("/")
    if len(parts) != 2 or any(not part or part in {".", ".."} for part in parts):
        raise ValueError("Station paths must have the form 'survey/station'")
    return value


class MaskCriteria(ResisticsModel):
    """Named masks and the rule used to combine their keep decisions."""

    model_config = ConfigDict(extra="forbid")
    combine: Literal["and", "or"] = "and"
    names: list[str]

    @field_validator("names")
    @classmethod
    def validate_names(cls, values: list[str]) -> list[str]:
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
    remote_references: Literal["auto"] | list[str] | None = None
    masks: MaskCriteria | None = None

    @field_validator("remote_references")
    @classmethod
    def validate_remotes(
        cls, value: Literal["auto"] | list[str] | None
    ) -> Literal["auto"] | list[str] | None:
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
    sampling_frequencies: dict[float, RateGatherCriteria] = Field(default_factory=dict)

    @field_validator("sampling_frequencies")
    @classmethod
    def validate_sample_rates(
        cls, value: dict[float, RateGatherCriteria]
    ) -> dict[float, RateGatherCriteria]:
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

    remote_references: Literal["auto"] | list[str] | None = None
    masks: MaskCriteria | None = None


class GatherSelection(ResisticsData):
    """Resolved target/rate inputs and admissible global windows for gathering.

    :param station_rate_batch: Selected station, sample rate, and run paths.
    :param remote_station: Selected remote station path, if remote reference is enabled.
    :param criteria: Gathering policy resolved for the batch.
    :param automatic_remote: Whether the remote station must be selected at runtime.
    """

    def __init__(
        self,
        station_rate_batch: dict[str, Any],
        remote_station: str | None,
        criteria: GatherCriteria,
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
    {py:meth}`resolve` accepts level and evaluation-frequency indices so callers do
    not need to change when the schema gains more granular overrides.

    **Examples**

    Resolve an automatic remote-reference policy for one target and rate.

    ```{doctest}
    >>> from resistics.gather import GatherCriteria
    >>> criteria = GatherCriteria(
    ...     stations={
    ...         "survey/target": {
    ...             "sampling_frequencies": {
    ...                 128: {"remote_references": "auto"}
    ...             }
    ...         }
    ...     }
    ... )
    >>> criteria.resolve("survey/target", 128).remote_references
    'auto'

    ```
    """

    model_config = ConfigDict(extra="forbid")

    output_type: ClassVar[str] = "gather_selection"
    runtime_requirements: ClassVar[list[str]] = ["station_rate_batch"]

    name: str = Field(default="", exclude=True)
    stations: dict[str, StationGatherCriteria] = Field(default_factory=dict)

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
        cls, value: dict[str, StationGatherCriteria]
    ) -> dict[str, StationGatherCriteria]:
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
        """Resolve a policy, returning the no-criteria policy when unlisted.

        :param station_path: Canonical survey/station path.
        :param sample_rate: Sample rate of the station batch.
        :param level: Decimation level to select.
        :param evaluation_frequency_index: Index of the evaluation frequency within the level.
        :return: The value produced when this operation completes.
        :raises ValueError: If the requested operation cannot satisfy its contract.
        """
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

    def remote_station_paths(self, station_path: str, sample_rate: float) -> list[str]:
        """Return explicitly configured remotes (``auto`` resolves at runtime).

        :param station_path: Canonical survey/station path.
        :param sample_rate: Sample rate of the station batch.
        :return: Explicitly configured remotes (``auto`` resolves at runtime).
        """
        remotes = self.resolve(station_path, sample_rate).remote_references
        return [] if remotes is None or remotes == "auto" else list(remotes)

    def remote_reference_count(self) -> int:
        """Return the number of explicit remote assignments for summaries.

        :return: The number of explicit remote assignments for summaries.
        """
        total = 0
        for station in self.stations.values():
            for policy in station.sampling_frequencies.values():
                if policy.remote_references == "auto":
                    total += 1
                elif isinstance(policy.remote_references, list):
                    total += len(policy.remote_references)
        return total

    def run(self, station_rate_batch: dict[str, Any]) -> GatherSelection:
        """Resolve the remote assignment for one target station/rate batch.

        :param station_rate_batch: Selected station, sample rate, and run paths.
        :return: The value produced when this operation completes.
        """
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

    def execute(self, inputs: dict[str, Any], context: Any) -> GatherSelection:
        """Resolve criteria from the station-rate batch supplied by the executor.

        :param inputs: Named upstream values supplied to the process.
        :param context: Runtime values supplied by the flow executor.
        :return: The value produced when this operation completes.
        """
        del inputs
        return self.run(context["station_rate_batch"])
