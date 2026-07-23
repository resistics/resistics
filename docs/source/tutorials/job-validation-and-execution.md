---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

(tutorial-job-validation-and-execution)=
# Validate and execute a job

A job binds a flow and parameter set to project scope. Validation resolves the
files, checks the graph, and expands the scope into deterministic work batches
before any processing begins.

```{code-cell} ipython3
:tags: [remove-output]

from pathlib import Path
from tempfile import TemporaryDirectory

from _fixtures import create_demo_mth5
from resistics.flow import (
    FlowDefinition,
    FlowNode,
    FlowStage,
    ParameterSet,
    model_to_yaml_file,
)
from resistics.job import (
    JobDefinition,
    JobRunner,
    JobScope,
    ProjectJobs,
)
from resistics.project import init, load

workspace = TemporaryDirectory(prefix="resistics-tutorial-")
root = Path(workspace.name)
mth5_path = create_demo_mth5(root)
project_path = root / "demo-project"
init(project_path, mth5_path, ref_time="2020-01-01T00:00:00+00:00")
project = load(project_path)
```

For a short documentation run, use a two-node run stage and select only the
target station.

```{code-cell} ipython3
flow = FlowDefinition(
    id="tutorial-job-flow",
    name="Read and centre MTH5 runs",
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
parameters = ParameterSet(name="tutorial-defaults")

processing_path = project_path / "processing"
model_to_yaml_file(flow, processing_path / "flows" / "tutorial-job-flow.yaml")
model_to_yaml_file(
    parameters, processing_path / "parameters" / "tutorial-defaults.yaml"
)

jobs = ProjectJobs(project)
job_path = jobs.create_template(
    JobDefinition(
        name="tutorial-job",
        flow="tutorial-job-flow",
        parameters="tutorial-defaults",
        scope=JobScope(surveys=["demo"], stations=["target"]),
    )
)
validation = jobs.validate(job_path)
{
    "valid": validation.ok,
    "errors": validation.errors,
    "batches": [batch.model_dump() for batch in validation.resolved_job.batches],
}
```

Run the already-resolved plan and consume structured progress events. Warnings
raised by dependencies are retained in the project job log.

```{code-cell} ipython3
events = []
state = JobRunner(project, progress_callback=events.append).run(
    validation.resolved_job
)
{
    "state": state.value,
    "progress": [(event.state.value, event.message) for event in events],
    "log_created": (project_path / "logs" / "tutorial-job.log").exists(),
}
```

```{code-cell} ipython3
:tags: [remove-cell]

project.close()
workspace.cleanup()
```
