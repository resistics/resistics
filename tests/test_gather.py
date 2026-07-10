"""Tests for the MTH5-compatible in-memory gather path."""

from pathlib import Path

from resistics.decimate import DecimationSetup
from resistics.gather import QuickGather
from resistics.testing import evaluation_data, solution_mt


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
