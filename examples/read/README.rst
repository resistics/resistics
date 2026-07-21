Reading data
------------

Time data enters resistics through an MTH5 ``RunTS`` dataset. Open an
MTH5-backed :class:`resistics.project.Project` and use ``Project.read_run`` to
obtain channel-labelled :class:`resistics.time.TimeData`.

For calibration data, resistics supports:

- Text file calibration data
- JSON calibration data

The structure of these two calibration formats can be seen in the examples.
