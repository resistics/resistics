---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

(tutorial-project-discovery)=
# Discover and navigate a project

Project discovery reads compact MTH5 summaries before it reads any sample
arrays. The same UI-neutral explorer index supports library applications and
the terminal interface.

```{code-cell} ipython3
:tags: [remove-output]

from pathlib import Path
from tempfile import TemporaryDirectory

from _fixtures import create_demo_mth5
from resistics.explorer import ProjectExplorerIndex
from resistics.project import init, load

workspace = TemporaryDirectory(prefix="resistics-tutorial-")
root = Path(workspace.name)
mth5_path = create_demo_mth5(root)
project_path = root / "demo-project"
init(project_path, mth5_path, ref_time="2020-01-01T00:00:00+00:00")
project = load(project_path)
index = ProjectExplorerIndex(project)
```

List runs without loading their arrays.

```{code-cell} ipython3
[
    {
        "path": run.run_path,
        "sample_rate": run.sample_rate,
        "samples": run.n_samples,
        "channels": run.channels,
    }
    for run in index.runs()
]
```

The explorer also indexes current flow and parameter resources installed when
the project was created.

```{code-cell} ipython3
{
    kind: [resource.path.name for resource in index.resources(kind)]
    for kind in ("flows", "parameters", "criteria", "jobs")
}
```

Read samples only after choosing a concrete survey, station, and run. Bounds
are inclusive, so the following request returns 128 samples.

```{code-cell} ipython3
time_data = project.read_run(
    "demo", "target", "run001", from_sample=0, to_sample=127
)
{
    "identity": (
        time_data.metadata.survey,
        time_data.metadata.station,
        time_data.metadata.run,
    ),
    "channels": time_data.metadata.chans,
    "shape": time_data.data.shape,
    "sample_rate": time_data.metadata.fs,
}
```

```{code-cell} ipython3
:tags: [remove-cell]

project.close()
workspace.cleanup()
```
