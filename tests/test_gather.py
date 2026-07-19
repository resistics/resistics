"""Tests for the MTH5-compatible in-memory gather path."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from resistics.decimate import DecimationSetup
from resistics.flow import model_from_yaml
from resistics.gather import (
    GatherCriteria,
    MaskCriteria,
    QuickGather,
    RateGatherCriteria,
    StationGatherCriteria,
)
from resistics.mask import (
    WindowMask,
    WindowMaskLevelMetadata,
    WindowMaskMetadata,
    WindowMaskWriter,
    get_run_mask_path,
)
from resistics.spectra import (
    EvaluationFrequencyData,
    EvaluationFrequencyReader,
    EvaluationFrequencyWriter,
    SpectraData,
    SpectraLevelMetadata,
    SpectraMetadata,
)
from resistics.testing import evaluation_data, solution_mt, time_metadata_general
from resistics.transfunc import ImpedanceTensor


def test_quick_gather_run():
    solution = solution_mt()
    decimation_parameters = DecimationSetup(
        n_levels=1,
        per_level=len(solution.freqs),
        eval_freqs=solution.freqs,
    ).run(256)
    evaluation = evaluation_data(decimation_parameters, 20, solution)

    gathered = QuickGather().run(
        Path("survey/station/run"),
        decimation_parameters,
        solution.tf,
        evaluation,
    )

    assert gathered.out_data.metadata.site_name == "run"
    assert gathered.out_data.metadata.chans == solution.tf.out_chans
    assert gathered.in_data.metadata.chans == solution.tf.in_chans


def test_empty_and_unlisted_criteria_resolve_to_single_site_unmasked():
    criteria = GatherCriteria()

    assert criteria.resolve("survey/target", 128).remote_references is None
    assert criteria.resolve("survey/target", 128).masks is None


def test_nested_criteria_resolves_rate_for_every_level_and_eval_index():
    criteria = GatherCriteria(
        stations={
            "survey/target": StationGatherCriteria(
                sampling_frequencies={
                    128: RateGatherCriteria(
                        remote_references=["survey/remote-a", "survey/remote-b"],
                        masks=MaskCriteria(combine="or", names=["night", "quiet"]),
                    )
                }
            )
        }
    )

    resolved = criteria.resolve("survey/target", 128.0, 7, 3)

    assert resolved.remote_references == ["survey/remote-a", "survey/remote-b"]
    assert resolved.masks.combine == "or"
    assert resolved.masks.names == ["night", "quiet"]
    assert criteria.resolve("survey/target", 4).remote_references is None


def test_criteria_yaml_accepts_auto_and_numeric_sample_rate_keys():
    criteria = model_from_yaml(
        GatherCriteria,
        """
stations:
  survey/target:
    sampling_frequencies:
      128.0:
        remote_references: auto
        masks:
          combine: and
          names: [night]
""",
    )

    assert criteria.resolve("survey/target", 128).remote_references == "auto"


def test_legacy_flat_criteria_has_actionable_migration_error():
    with pytest.raises(ValueError, match="Legacy flat gather criteria"):
        GatherCriteria(remote_references={"survey/target": "survey/remote"})


@pytest.mark.parametrize("name", ["../outside", "contains/slash", "has space"])
def test_mask_names_are_filename_safe(name):
    with pytest.raises(ValueError, match="Mask names"):
        MaskCriteria(names=[name])


class ProjectStub:
    def __init__(self, project_path, rows):
        self.project_path = project_path
        self.table = pd.DataFrame(rows)

    def get_concurrent(self, station_path, fs):
        return sorted(
            self.table[
                (self.table.station_path != station_path)
                & np.isclose(self.table.sample_rate, fs)
            ].station_path.unique()
        )


def write_evaluation(
    project_path,
    run_path,
    *,
    offset,
    value,
    n_wins=1,
    actual_levels=1,
    planned_levels=None,
    output_label="default",
):
    chans = ["ex", "ey", "hx", "hy"]
    base = time_metadata_general(chans, fs=128).model_dump()
    planned_levels = planned_levels or actual_levels
    evaluation_frequencies = [
        frequency
        for level in range(planned_levels)
        for frequency in (32 / (4**level), 16 / (4**level))
    ]
    decimation = DecimationSetup(
        n_levels=planned_levels,
        per_level=2,
        eval_freqs=evaluation_frequencies,
    ).run(128)
    levels = [
        SpectraLevelMetadata(
            fs=decimation.get_fs(level),
            n_wins=n_wins,
            win_size=20,
            olap_size=5,
            index_offset=offset,
            n_freqs=2,
            freqs=decimation.get_eval_freqs(level),
        )
        for level in range(actual_levels)
    ]
    base.update(
        fs=[level.fs for level in levels],
        n_levels=actual_levels,
        levels_metadata=levels,
        ref_time=base["first_time"],
    )
    spectra = SpectraData(
        SpectraMetadata(**base),
        {
            level: np.full((n_wins, 4, 2), value + level, dtype=np.complex128)
            for level in range(actual_levels)
        },
    )
    survey, station, run = run_path.split("/")
    EvaluationFrequencyWriter().execute(
        {
            "eval_data": EvaluationFrequencyData(
                spectra_data=spectra, decimation_parameters=decimation
            )
        },
        {
            "project_path": project_path,
            "run_batch": {"survey": survey, "station": station, "run": run},
            "output_label": output_label,
        },
    )
    return spectra, decimation


def test_evaluation_artifacts_use_the_runtime_output_label(tmp_path):
    spectra, decimation = write_evaluation(
        tmp_path,
        "survey/station/run",
        offset=0,
        value=1,
        output_label="field",
    )

    artifact = EvaluationFrequencyReader(label="default").execute(
        {},
        {
            "project_path": tmp_path,
            "run_batch": {"survey": "survey", "station": "station", "run": "run"},
            "output_label": "field",
        },
    )

    assert (tmp_path / "data/survey/station/run/evals/field/metadata.json").is_file()
    assert artifact.spectra_data.metadata.chans == spectra.metadata.chans
    assert artifact.spectra_data.metadata.ref_time == spectra.metadata.ref_time
    assert artifact.decimation_parameters == decimation


def project_with_evaluations(tmp_path, values):
    rows = []
    artifacts = {}
    for run_path, offset, value in values:
        survey, station, _ = run_path.split("/")
        spectra, decimation = write_evaluation(
            tmp_path, run_path, offset=offset, value=value
        )
        artifacts[run_path] = (spectra, decimation)
        rows.append(
            {
                "survey": survey,
                "station": station,
                "station_path": f"{survey}/{station}",
                "sample_rate": 128.0,
                "run_path": run_path,
            }
        )
    return ProjectStub(tmp_path, rows), artifacts


def test_gather_concatenates_multiple_target_runs_by_global_window(tmp_path):
    from resistics.gather import Gather

    project, _ = project_with_evaluations(
        tmp_path,
        [("survey/target/run-a", 0, 1), ("survey/target/run-b", 1, 2)],
    )
    selection = GatherCriteria().run(
        {
            "survey": "survey",
            "station": "target",
            "sample_rate": 128,
            "run_paths": ["survey/target/run-b", "survey/target/run-a"],
        }
    )

    gathered = Gather().run(project, tmp_path, selection, ImpedanceTensor())

    assert gathered.out_data.data[0].shape == (2, 2)
    assert gathered.out_data.data[0][:, 0].tolist() == [1 + 0j, 2 + 0j]
    assert gathered.cross_data.metadata.site_names == ["survey/target"]
    assert gathered.out_data.metadata.measurements == [
        "survey/target/run-a",
        "survey/target/run-b",
    ]


def test_gather_batches_selected_rows_without_full_level_channel_copies(
    tmp_path, monkeypatch
):
    """A standard-window gather must not select every channel per window."""
    from resistics.gather import Gather

    run_path = "survey/target/run"
    write_evaluation(tmp_path, run_path, offset=0, value=1, n_wins=3)
    project = ProjectStub(
        tmp_path,
        [
            {
                "survey": "survey",
                "station": "target",
                "station_path": "survey/target",
                "sample_rate": 128.0,
                "run_path": run_path,
            }
        ],
    )
    selection = GatherCriteria().run(
        {
            "survey": "survey",
            "station": "target",
            "sample_rate": 128,
            "run_paths": [run_path],
        }
    )

    def fail_full_level_selection(*args, **kwargs):
        raise AssertionError("Gather must select only its requested rows")

    monkeypatch.setattr(SpectraData, "get_chans", fail_full_level_selection)

    gathered = Gather().run(project, tmp_path, selection, ImpedanceTensor())

    assert gathered.out_data.data[0].shape == (3, 2)


def test_gather_uses_realised_metadata_levels_and_allows_shorter_runs(tmp_path):
    from resistics.gather import Gather

    run_a = "survey/target/run-a"
    run_b = "survey/target/run-b"
    write_evaluation(
        tmp_path,
        run_a,
        offset=0,
        value=1,
        actual_levels=1,
        planned_levels=3,
    )
    write_evaluation(
        tmp_path,
        run_b,
        offset=1,
        value=2,
        actual_levels=2,
        planned_levels=3,
    )
    project = ProjectStub(
        tmp_path,
        [
            {
                "survey": "survey",
                "station": "target",
                "station_path": "survey/target",
                "sample_rate": 128.0,
                "run_path": run_path,
            }
            for run_path in (run_a, run_b)
        ],
    )
    selection = GatherCriteria().run(
        {
            "survey": "survey",
            "station": "target",
            "sample_rate": 128,
            "run_paths": [run_a, run_b],
        }
    )

    gathered = Gather().run(project, tmp_path, selection, ImpedanceTensor())

    assert gathered.out_data.metadata.n_evals == 4
    assert gathered.out_data.metadata.eval_freqs == [32, 16, 8, 4]
    assert gathered.out_data.data[0].shape == (2, 2)
    assert gathered.out_data.data[2].shape == (1, 2)
    assert gathered.out_data.data[2][0, 0] == 3 + 0j


def test_gather_pools_multi_remote_pairs_and_repeats_target_rows(tmp_path):
    from resistics.gather import Gather

    project, _ = project_with_evaluations(
        tmp_path,
        [
            ("survey/target/run", 5, 1),
            ("survey/remote-a/run", 5, 10),
            ("survey/remote-b/run", 5, 20),
        ],
    )
    criteria = GatherCriteria(
        stations={
            "survey/target": StationGatherCriteria(
                sampling_frequencies={
                    128: RateGatherCriteria(
                        remote_references=["survey/remote-b", "survey/remote-a"]
                    )
                }
            )
        }
    )
    selection = criteria.run(
        {
            "survey": "survey",
            "station": "target",
            "sample_rate": 128,
            "run_paths": ["survey/target/run"],
        }
    )

    gathered = Gather().run(project, tmp_path, selection, ImpedanceTensor())

    assert gathered.out_data.data[0][:, 0].tolist() == [1 + 0j, 1 + 0j]
    assert gathered.cross_data.data[0][:, 0].tolist() == [10 + 0j, 20 + 0j]
    assert gathered.cross_data.metadata.site_names == [
        "survey/remote-a",
        "survey/remote-b",
    ]


def test_gather_applies_frequency_dependent_persisted_mask(tmp_path):
    from resistics.gather import Gather

    run_path = "survey/target/run"
    project, artifacts = project_with_evaluations(tmp_path, [(run_path, 5, 1)])
    spectra, decimation = artifacts[run_path]
    source = spectra.metadata.levels_metadata[0]
    metadata = WindowMaskMetadata(
        survey="survey",
        station="target",
        run="run",
        sample_rate=128,
        ref_time=spectra.metadata.ref_time,
        levels=[
            WindowMaskLevelMetadata(
                level=0,
                fs=source.fs,
                n_wins=source.n_wins,
                win_size=source.win_size,
                olap_size=source.olap_size,
                index_offset=source.index_offset,
                n_evaluation_frequencies=2,
                evaluation_frequencies=decimation.eval_freqs,
            )
        ],
    )
    mask = WindowMask(
        metadata,
        {0: pd.DataFrame([[True, False]], index=pd.RangeIndex(5, 6), columns=[0, 1])},
    )
    WindowMaskWriter().run(
        get_run_mask_path(
            tmp_path,
            {"survey": "survey", "station": "target", "run": "run"},
            "TimeMask",
            "default",
        ),
        mask,
    )
    criteria = GatherCriteria(
        stations={
            "survey/target": StationGatherCriteria(
                sampling_frequencies={
                    128: RateGatherCriteria(masks=MaskCriteria(names=["TimeMask"]))
                }
            )
        }
    )
    selection = criteria.run(
        {
            "survey": "survey",
            "station": "target",
            "sample_rate": 128,
            "run_paths": [run_path],
        }
    )

    with pytest.raises(ValueError, match="evaluation-frequency index 1"):
        Gather().run(project, tmp_path, selection, ImpedanceTensor())
