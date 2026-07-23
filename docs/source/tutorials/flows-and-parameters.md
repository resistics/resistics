---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

(tutorial-flows-and-parameters)=
# Create flows and parameters

A flow is a portable directed graph of concrete process classes. Parameters
are stored separately, so several projects or jobs can reuse the same graph
with different numerical choices.

```{code-cell} ipython3
from resistics.flow import (
    FlowDefinition,
    FlowNode,
    FlowStage,
    ParameterSet,
    ProcessingJob,
    FlowValidator,
    model_to_yaml,
)

flow = FlowDefinition(
    id="tutorial-time",
    name="Tutorial time preparation",
    description="Read an MTH5 run, centre it, and prepare decimation levels.",
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
                FlowNode(
                    id="decimation-setup",
                    process="resistics.decimate.DecimationSetup",
                    inputs={"time_data": "remove-mean"},
                ),
            ],
        )
    ],
)

parameters = ParameterSet(
    name="tutorial-fast",
    processes={
        "resistics.decimate.DecimationSetup": {
            "n_levels": 2,
            "per_level": 2,
            "min_samples": 128,
        }
    },
)
```

Both models serialize directly to the YAML documents used by projects and the
terminal interface.

```{code-cell} ipython3
print(model_to_yaml(flow))
print(model_to_yaml(parameters))
```

Validation resolves every process class, checks graph ports and types, and
constructs each process from its parameter mapping without executing data.

```{code-cell} ipython3
processing_job = ProcessingJob(
    name="tutorial-check",
    flow=flow,
    parameters=parameters,
    runtime={"project": object(), "run_batch": {}},
)
validation = FlowValidator().validate(processing_job)
validation.model_dump()
```
