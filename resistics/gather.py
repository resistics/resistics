"""Public facade and orchestration for evaluation-data gathering.

Criteria models, project discovery, immutable planning, and array assembly live
in dedicated implementation modules. Existing imports from {py:mod}`resistics.gather`
remain the supported public API.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from loguru import logger

from resistics.common import ResisticsProcess
from resistics.gather_criteria import (
    GatherCriteria,
    GatherSelection,
    MaskCriteria,
    RateGatherCriteria,
    ResolvedGatherCriteria,
    StationGatherCriteria,
)
from resistics.gather_data import (
    EvaluationFrequencyGather,
    GatheredData,
    QuickGather,
    SiteCombinedData,
    SiteCombinedMetadata,
    _GatherAssembler,
)
from resistics.gather_plan import _GatherPlanner
from resistics.gather_project import _GatherProjectSource
from resistics.transfunc import TransferFunction

if TYPE_CHECKING:
    from resistics.project import Project

__all__ = [
    "EvaluationFrequencyGather",
    "Gather",
    "GatherCriteria",
    "GatherSelection",
    "GatheredData",
    "MaskCriteria",
    "QuickGather",
    "RateGatherCriteria",
    "ResolvedGatherCriteria",
    "SiteCombinedData",
    "SiteCombinedMetadata",
    "StationGatherCriteria",
]

# These names have long-lived YAML, pickle, and autodoc identities. Their
# implementations are split by responsibility, while the facade remains their
# canonical public owner.
for _public_type in (
    EvaluationFrequencyGather,
    GatherCriteria,
    GatherSelection,
    GatheredData,
    MaskCriteria,
    QuickGather,
    RateGatherCriteria,
    ResolvedGatherCriteria,
    SiteCombinedData,
    SiteCombinedMetadata,
    StationGatherCriteria,
):
    _public_type.__module__ = __name__
del _public_type


class Gather(ResisticsProcess):
    """Gather aligned persisted evaluation data for local or remote regression.

    Target data supplies output and input channels. With remote references,
    target/remote pairs are pooled and the remote data supplies cross channels.
    The same target window is intentionally repeated when it aligns with more
    than one configured remote.
    """

    input_types: ClassVar[dict[str, str]] = {
        "selection": "gather_selection",
        "tf": "transfer_function",
    }
    output_type: ClassVar[str] = "gathered_data"
    runtime_requirements: ClassVar[list[str]] = ["project", "project_path"]

    def execute(self, inputs: dict[str, Any], context: Any) -> GatheredData:
        """Gather the flow inputs using the project supplied at runtime.

        :param inputs: Named upstream values supplied to the process.
        :param context: Runtime values supplied by the flow executor.
        :return: Gather the flow inputs using the project supplied at runtime.
        :raises ValueError: If the requested operation cannot satisfy its contract.
        """
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
        """Gather aligned evaluation data for one target station and rate.

        :param project: Open project containing target and remote station data.
        :param project_path: Project root containing persisted evaluation artifacts.
        :param selection: Resolved target, rate, masks, and remote-reference policy.
        :param tf: Transfer function defining output, input, and cross channels.
        :param output_label: Namespace containing persisted inputs.

        :return: Window-aligned arrays ready for regression.

        :raises ValueError: If required inputs are absent, unreadable, incompatible, or have no
            admitted aligned windows.
        """
        batch = selection.station_rate_batch
        target = batch["station_path"]
        sample_rate = float(batch["sample_rate"])
        target_runs = sorted(batch["run_paths"])
        if not target_runs:
            raise ValueError(f"Target {target} at {sample_rate:g} Hz has no runs")

        source = _GatherProjectSource(
            project,
            project_path,
            selection.criteria,
            output_label,
        )
        target_artifacts, level_references = source.load_target(target_runs, target)
        baseline = next(iter(target_artifacts.values()))
        remote_setting = selection.criteria.resolve(
            target, sample_rate
        ).remote_references
        remotes = source.discover_remotes(
            target,
            sample_rate,
            remote_setting,
            baseline,
            level_references,
        )
        plan = _GatherPlanner(source).build(
            target,
            sample_rate,
            target_artifacts,
            level_references,
            remotes,
        )
        gathered = _GatherAssembler(self._get_record).assemble(
            plan,
            target,
            tf,
            target_artifacts,
            remotes.artifacts,
        )
        cross_stations = (
            [target] if not plan.remote_enabled else list(plan.usable_remotes)
        )
        logger.info(
            f"Gathered {target} at {sample_rate:g} Hz using cross stations "
            f"{cross_stations}"
        )
        return gathered
