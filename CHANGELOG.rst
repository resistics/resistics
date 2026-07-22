Next release
^^^^^^^^^^^^

- **Added:** ``Project`` and ``MTH5File`` now support deterministic ``close()``
  and context-manager ownership of their read-only MTH5 handles.
- **Added:** A locked OSV dependency audit covers every supported Python minor
  and enforces expiring, owned records for any temporarily accepted risk.
- **Changed:** Project loading closes partially opened MTH5 files when opening,
  channel-summary preparation, or model construction fails.
- **Changed:** ``resistics.testing`` now contains only the small data builders
  used by executable documentation; regression-only factories and comparisons
  are owned by the test suite.
- **Changed:** Resistics now targets Python 3.12-3.14 and uses the current
  locked dependency releases as its supported floors. Dependency requirements
  remain open above those floors, including ``tsdownsample``; no speculative
  runtime upper bounds are retained.
- **Removed:** The pre-MTH5 ``Site``/``Measurement`` directory path and file-name
  helpers, the ``Project.dir_path`` and ``Project.metadata`` compatibility
  aliases, their unused exception classes, ``init(force=...)``, and
  ``close_mth5()``. Use canonical survey/station/run paths, ``overwrite=...``,
  and ``close()`` or ``with``. Gallery examples for the already-removed
  ``config`` and ``letsgo`` modules were also removed.
- **Removed:** The empty ``resistics.resq`` placeholder, obsolete commented
  ``Join`` implementation, unmaintained exploratory notebooks and their unused
  FDSN request file, unreferenced evaluation-data fixture module, no-op
  timestamp gather hook, and the unused ``standard_mt_flow``,
  ``time_to_evals_flow``, ``evals_to_tf_flow``, and mask-example flow aliases.
- **Removed:** Unused direct declarations for ``obspy``, ``prettyprinter``, and
  ``scikit-learn``; the standard library now renders model summaries and
  ``regressioninc`` owns its scikit-learn requirement. Unused notebook,
  pytest-HTML, IPython, nbformat, seedir, and emoji development dependencies
  were also removed.

Resistics 0.0.6
^^^^^^^^^^^^^^^^^^^^
*March 15, 2020*

- **Added:** None.
- **Changed:** The way windows are defined when selecting statistics from statistic data. Currently, the statistic data is in raw arrays. In future, might be best to be able to export as Pandas dataframe to allow people to use statistics easier.
- **Deprecated:** None.
- **Removed:** None.
- **Fixed:** Change the way data is selected in StatisticData to fix `Issue #5 <https://github.com/resistics/resistics/issues/5>`_. The was due to injudicious use to bools and Lists for truth checking. Now increasing the use of None in the code where no results are found.
- **Fixed:** Fixed issue where some window parameters are not passed from configuration file into WindowParameters. For more information, see `Issue #4 <https://github.com/resistics/resistics/issues/4>`_.
- **Fixed:** Fixed bug in SpectrumReader when requesting 0 windows from batch reader. This can happen when selecting windows shared across multiple sites and a recording at one site has not shared windows with another site. In this case, 0 windows will be requested from this recording and it would previously fail. See `Issue #3 <https://github.com/resistics/resistics/issues/3>`_.

Resistics 0.0.6rc1
^^^^^^^^^^^^^^^^^^^^
*Nov 11, 2019*

- **Added:** Multiprocessing added for spectra calculations, statistic calculations and transfer function estimation (predominantly the power spectra calculations).
- **Added:** Multiprocessing documentation.
- **Added:** Spectra batches in WindowSelector allowing batches of overlapping spectra data to be pre-read for processing. This speeds up transfer function estimation by reducing disk reads.
- **Changed:** Speed up transfer function estimation by using WindowSelector spectra batches.
- **Changed:** Statistics and spectra readers return None when no file is found rather than False.
- **Deprecated:** None.
- **Removed:** None.
- **Fixed:** None.

Resistics 0.0.6.dev3
^^^^^^^^^^^^^^^^^^^^
*Oct 13, 2019*

- **Added:** Modules common, config, time, spectra, statistics, mask, transfunc, calibration, window, decimate, regression, project, site as part of a restructure of the resistics code base to give more meaningful imports.
- **Added:** Statistic documentation in the features section
- **Changed:** Tutorial changed to reflect new code structure.
- **Changed:** Advanced examples to reflect new code structure.
- **Changed:** Time series and calibration format examples changed to reflect new code structure.
- **Changed:** Cookbook changed to reflect new code structure.
- **Deprecated:** All functionality remains but has been re-organised.
- **Removed:** Modules ioHandlers, dataObjects, calculators and utilities all removed as part of a restructure of resistics to give more meaningful imports and to simplify the organisation of functionality.
- **Fixed:** Fixed a bug in resistics.project.transfunc.processSite and resistics.project.shortcuts.getWindowSelector which ignores user supplied specdir option.

Resistics 0.0.6.dev2
^^^^^^^^^^^^^^^^^^^^
*Sep 29, 2019*

- **Added:** Modules to read in Lemi B423 and B423E data and documenation to show usage.
- **Added:** Documentation for advanced methods include remote reference, pre-processing and intersite processing.
- **Changed:** None
- **Deprecated:** None
- **Removed:** None
- **Fixed:** Fixed a bug in resistics.ioHandlers.spectrumWriter.SpectrumWriter with a mismatch between channel ordering in the info file and channel ordering in the data. This could lead to incorrect usage of channel data in transfer function calculations if the data had been written out in resistics internal format.
