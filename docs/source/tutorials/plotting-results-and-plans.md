---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

(tutorial-plotting-results-and-plans)=
# Plot results, flows, and jobs

Resistics plotting methods return Plotly figures. In these text notebooks the
figures are emitted through notebook MIME, so the built HTML retains zoom,
hover, pan, and legend controls.

## Plot a transfer-function result

The small solution below is deterministic test data with the same public
`Solution` shape as a completed regression.

```{code-cell} ipython3
from IPython.display import HTML

from resistics.testing import solution_mt

solution = solution_mt()
result_figure = solution.tf.plot(solution.freqs, solution.components)
HTML(
    result_figure.to_html(
        full_html=False,
        include_plotlyjs=True,
        div_id="transfer-function-result",
    )
)
```

## Plot a flow

Flow plots expose stage boundaries, process types, and dependencies before the
flow is bound to a project.

```{code-cell} ipython3
from resistics.flow import FlowDefinition, FlowNode, FlowStage, ParameterSet
from resistics.plot import plot_flow, plot_job

flow = FlowDefinition(
    id="plot-tutorial",
    name="Plot tutorial flow",
    stages=[
        FlowStage(
            stage_id="prepare-runs",
            scope="run",
            nodes=[
                FlowNode(id="read", process="resistics.time.MTH5TimeReader"),
                FlowNode(
                    id="remove-mean",
                    process="resistics.time.RemoveMean",
                    inputs={"time_data": "read"},
                ),
            ],
        )
    ],
)
flow_figure = plot_flow(flow)
HTML(
    flow_figure.to_html(
        full_html=False,
        include_plotlyjs=False,
        div_id="flow-plan",
    )
)
```

## Plot a resolved job

The job plot adds effective parameters, selected stages, scope, and planned
batches. As in the execution tutorial, validation creates the resolved plan.

```{code-cell} ipython3
:tags: [remove-output]

from pathlib import Path
from tempfile import TemporaryDirectory

from _fixtures import create_demo_mth5
from resistics.flow import model_to_yaml_file
from resistics.job import JobDefinition, JobScope, ProjectJobs
from resistics.project import init, load

workspace = TemporaryDirectory(prefix="resistics-tutorial-")
root = Path(workspace.name)
mth5_path = create_demo_mth5(root)
project_path = root / "demo-project"
init(project_path, mth5_path, ref_time="2020-01-01T00:00:00+00:00")
project = load(project_path)

processing_path = project_path / "processing"
model_to_yaml_file(flow, processing_path / "flows" / "plot-tutorial.yaml")
model_to_yaml_file(
    ParameterSet(name="plot-tutorial"),
    processing_path / "parameters" / "plot-tutorial.yaml",
)
jobs = ProjectJobs(project)
job_path = jobs.create_template(
    JobDefinition(
        name="plot-tutorial",
        flow="plot-tutorial",
        parameters="plot-tutorial",
        scope=JobScope(stations=["target"]),
    )
)
resolved_job = jobs.validate(job_path).resolved_job
```

```{code-cell} ipython3
job_figure = plot_job(resolved_job)
HTML(
    job_figure.to_html(
        full_html=False,
        include_plotlyjs=False,
        div_id="job-plan",
    )
)
```

```{code-cell} ipython3
:tags: [remove-cell]

project.close()
workspace.cleanup()
```
