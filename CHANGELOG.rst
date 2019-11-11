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
