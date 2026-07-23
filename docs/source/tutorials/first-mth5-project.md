---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

(tutorial-first-mth5-project)=
# Create your first MTH5 project

A Resistics project points at an existing MTH5 file and keeps its processing
resources and derived results in a separate directory. This tutorial creates a
small local MTH5 fixture so that the example is executable and offline; replace
that fixture path with your recording in real work.

```{code-cell} ipython3
:tags: [remove-output]

from pathlib import Path
from tempfile import TemporaryDirectory

from _fixtures import create_demo_mth5
from resistics.project import init, load

workspace = TemporaryDirectory(prefix="resistics-tutorial-")
root = Path(workspace.name)
mth5_path = create_demo_mth5(root)
project_path = root / "demo-project"

init(project_path, mth5_path, ref_time="2020-01-01T00:00:00+00:00")
```

Load projects as context managers. This makes ownership of the read-only MTH5
handle explicit and guarantees that it is closed, including when an exception
occurs.

```{code-cell} ipython3
:tags: [remove-output]

with load(project_path) as project:
    summary = project.file_summary()
    overview = {
        "surveys": summary.n_surveys,
        "stations": summary.n_stations,
        "runs": summary.n_runs,
        "channels": summary.n_channels,
        "sample_rates": summary.sample_rates,
    }
```

```{code-cell} ipython3
overview
```

The project directory contains editable processing resources, results, and
logs. The source MTH5 file remains separate and is opened read-only by
`load()`.

```{code-cell} ipython3
sorted(path.name for path in project_path.iterdir())
```

```{code-cell} ipython3
:tags: [remove-cell]

workspace.cleanup()
```
