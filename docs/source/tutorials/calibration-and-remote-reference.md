---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

(tutorial-calibration-and-remote-reference)=
# Calibration and remote reference

Calibration responses and remote-reference choices are explicit, validated
models. They can be inspected and serialized independently of a long
processing run.

## Read calibration data

JSON calibration files use the same schema as `CalibrationData`. Phase values
are radians and the response contains one magnitude for each frequency.

```{code-cell} ipython3
from pathlib import Path
from tempfile import TemporaryDirectory

from IPython.display import HTML

from resistics.calibrate import CalibrationData, SensorCalibrationJSON
from resistics.time import ChanMetadata

workspace = TemporaryDirectory(prefix="resistics-tutorial-")
root = Path(workspace.name)
calibration_path = root / "IC_MFS06.json"

expected = CalibrationData(
    sensor="MFS06",
    serial="1234",
    static_gain=1.0,
    magnitude_unit="mV/nT",
    frequency=[0.01, 0.1, 1.0, 10.0],
    magnitude=[0.1, 1.0, 10.0, 100.0],
    phase=[1.55, 1.50, 1.40, 1.20],
)
calibration_path.write_text(expected.model_dump_json(indent=2))

calibration = SensorCalibrationJSON().read_calibration_data(
    calibration_path,
    ChanMetadata(
        name="hx", chan_type="magnetic", sensor="MFS06", serial="1234"
    ),
)
calibration.to_dataframe()
```

## Choose a remote reference

Gather criteria are keyed by canonical `survey/station` path and original
sampling frequency. An explicit list is reproducible; `"auto"` asks the
project to discover concurrent stations at execution time.

```{code-cell} ipython3
from resistics.gather import (
    GatherCriteria,
    RateGatherCriteria,
    StationGatherCriteria,
)

criteria = GatherCriteria(
    stations={
        "demo/target": StationGatherCriteria(
            sampling_frequencies={
                16.0: RateGatherCriteria(
                    remote_references=["demo/remote"]
                )
            }
        )
    }
)

resolved = criteria.resolve("demo/target", 16.0)
{
    "remote_references": resolved.remote_references,
    "unconfigured_station": criteria.resolve("demo/remote", 16.0).model_dump(),
}
```

Calibration data has an interactive Plotly representation; hover to inspect
the frequency response. `HTML` keeps the figure as notebook MIME output and
embeds Plotly locally, so the built page does not depend on a CDN.

```{code-cell} ipython3
figure = calibration.plot()
HTML(
    figure.to_html(
        full_html=False,
        include_plotlyjs=True,
        div_id="calibration-response",
    )
)
```

```{code-cell} ipython3
:tags: [remove-cell]

workspace.cleanup()
```
