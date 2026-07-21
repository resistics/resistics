Using projects
--------------

Projects pair a processing directory with one read-only MTH5 source. They are
the standard way to work with multiple runs and stations and enable:

- Multiple recordings at the same sampling frequency can be used to calculate transfer functions
- Processing which combines data from different stations and aligns windows
- Calculation of statistics and deeper analysis of recordings
- Use of multiple configurations (useful for experimentation or mixed instrument surveys)

Use ``Project`` as a context manager so its owned MTH5 handle is released
deterministically.
