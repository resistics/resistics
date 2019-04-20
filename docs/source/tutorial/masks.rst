.. |Ex| replace:: E\ :sub:`x`
.. |Ey| replace:: E\ :sub:`y`
.. |Hx| replace:: H\ :sub:`x`
.. |Hy| replace:: H\ :sub:`y`
.. |Hz| replace:: H\ :sub:`z`
.. |Zxy| replace:: Z\ :sub:`xy`
.. |Zxx| replace:: Z\ :sub:`xx`
.. |Zyx| replace:: Z\ :sub:`yx`
.. |Zyy| replace:: Z\ :sub:`yy`
.. |fs| replace:: f\ :sub:`s`

Masks
-----

:doc:`Statistics <statistics>` give information about individual time windows. In resistics, masks are the way this information can be used.

A mask is used to exclude time windows from the transfer function calculation. Mask data is stored under the maskData folder. Masks are not individual to a specific time series measurement. Instead they are indexed by:

1. The site
2. The specdir
3. The sampling frequency

For example:

.. code-block:: text

    exampleProject
    ├── calData 
    ├── timeData   
    │   └── site1
    |       |── dataFolder1
    │       |── dataFolder2
    |       |──     .     
    |       |──     .
    |       |──     .
    |       └── dataFolderN     
    ├── specData
    │   └── site1
    |       |── dataFolder1
    |       |   |── dec8_5
    |       |   └── spectra
    |       |   
    │       |── dataFolder2
    |       |   |── dec8_5
    |       |   └── spectra    
    |       |──     .     
    |       |──     .
    |       |──     .
    |       └── dataFolderN
    |           |── dec8_5
    |           └── spectra        
    ├── statData
    │   └── site1
    |       |── dataFolder1
    |       |   |── dec8_5
    |       |   |   |── coherence
    |       |   |   |──     .
    |       |   |   |──     .
    |       |   |   |──     .        
    |       |   |   |── resPhase
    |       |   |   └── transferFunction            
    |       |   └── spectra
    |       |       |── coherence
    |       |       |──     .
    |       |       |──     .
    |       |       |──     .        
    |       |       |── resPhase
    |       |       └── transferFunction       
    |       |   
    │       |── dataFolder2
    |       |   |── dec8_5
    |       |   |   |── coherence
    |       |   |   |──     .
    |       |   |   |──     .
    |       |   |   |──     .        
    |       |   |   |── resPhase
    |       |   |   └── transferFunction     
    |       |   └── spectra
    |       |       |── coherence
    |       |       |──     .
    |       |       |──     .
    |       |       |──     .        
    |       |       |── resPhase
    |       |       └── transferFunction          
    |       |──     .     
    |       |──     .
    |       |──     .
    |       └── dataFolderN
    |           |── dec8_5
    |           |   |── coherence
    |           |   |──     .
    |           |   |──     .
    |           |   |──     .        
    |           |   |── resPhase
    |           |   └── transferFunction     
    |           └── spectra
    |               |── coherence
    |               |──     .
    |               |──     .
    |               |──     .        
    |               |── resPhase
    |               └── transferFunction           
    ├── maskData
    │   └── site1
    |       └── dec8_5  
    |           |── {coh70_100_128_000.npy, coh70_100_128_000.info}    
    |           └── {coh70_100_4096_000.npy, coh70_100_4096_000.info}        
    ├── transFuncData 
    ├── images
    └── mtProj.prj

Mask data is made up of two files:

1. The info file: coh70_100_128_000.info
2. The mask data file: coh70_100_128_000.npy 

The info file holds information about the statistics and constraints used to generate the mask file. The mask data is simply numpy array data holding information about which time windows to exclude.

The easiest way to understand masking is to step through an example. Begin, as usual, by loading the project. 

.. literalinclude:: ../../../examples/tutorial/usingMasks.py
    :linenos:
    :language: python
    :lines: 1-6
    :lineno-start: 1

To generate new mask data, the method  :meth:`~resistics.project.projectMask.newMaskData` in module :mod:`~resistics.project.projectMask` is used. To create a new mask, the sampling frequency to which the mask applies must be specified. 

.. literalinclude:: ../../../examples/tutorial/usingMasks.py
    :linenos:
    :language: python
    :lines: 8-11
    :lineno-start: 8

The method :meth:`~resistics.project.projectMask.newMaskData` returns a :class:`~resistics.dataObjects.maskData.MaskData` object, which will hold information about the statistics to use and the constraints. 

The next step is to set the statistics to use for excluding windows. These statistics must already be calculated and were so in the :doc:`Statistics <statistics>` section. 

.. literalinclude:: ../../../examples/tutorial/usingMasks.py
    :linenos:
    :language: python
    :lines: 12-13
    :lineno-start: 12

After the statistics are chosen, the constraints can be defined. 

.. literalinclude:: ../../../examples/tutorial/usingMasks.py
    :linenos:
    :language: python
    :lines: 14-15
    :lineno-start: 14

In the above example, a mask is being created that requires the |Ex|-|Hy| coherence **and** the |Ey|-|Hx| coherence to be between 0.70 and 1.00 for a time window. If this is not the case, then the time window will be masked (excluded) for the transfer funciton calculation. 

Before writing out a mask dataset, the :class:`~resistics.dataObjects.maskData.MaskData` instance should be given a name.

.. literalinclude:: ../../../examples/tutorial/usingMasks.py
    :linenos:
    :language: python
    :lines: 16-17
    :lineno-start: 16

The naming of mask files is made up of two parts:

- The mask name
- The sampling frequency written to 3 decimal places and using a _ rather than . to indicate the decimal point (i.e. 128.000 becomes 128_000)

Therefore, for the above example, the mask files will be:

- The info file: coh70_100_128_000.info
- The mask data file: coh70_100_128_000.npy 

The :class:`~resistics.dataObjects.maskData.MaskData` parameters can be viewed by using the :meth:`~resistics.dataObjects.dataObject.DataObject.printInfo` method of the :class:`~resistics.dataObjects.dataObject.DataObject` parent class. The masking constraints can be printed to the terminal using the :meth:`~resistics.dataObjects.maskData.MaskData.printConstraints` method.

.. literalinclude:: ../../../examples/tutorial/usingMasks.py
    :linenos:
    :language: python
    :lines: 18-20
    :lineno-start: 18

So far, the following information has been defined:

- The sampling frequency 
- The statistics to use for masking
- The constraints for the statistics

However, the windows to mask have not yet been calculated. This is achieved by using the :meth:`~resistics.project.projectMask.calculateMask` method of the module :mod:`~resistics.project.projectMask`. This method runs through the pre-calculated statistic data for all data of the given sampling frequency in a site and finds and saves the windows which do not meet the constraints.

.. literalinclude:: ../../../examples/tutorial/usingMasks.py
    :linenos:
    :language: python
    :lines: 22-26
    :lineno-start: 22

As statistics are calculated on an evaluation frequency basis, masked windows are calculated for each evaluation frequency too. 

Once the masked windows are calculated, they can then be viewed using the :meth:`~resistics.dataObjects.maskData.MaskData.view` method of :class:`~resistics.dataObjects.maskData.MaskData`. When using the :meth:`~resistics.dataObjects.maskData.MaskData.view` method, the decimation level has to be specified. In the example above, this has been set to 0, which is the first decimation level. The resultant plot is shown below.

.. figure:: ../_images/stat_statData_coherence_view.png
    :align: center
    :alt: alternate text
    :figclass: align-center

    Mask data plot for 128 Hz data using coherence constraints of 0.70-1.00

The bar chart at the bottom shows the number of masked windows for each evaluation frequency in the decimation level, plus the total number of masked windows for the decimation level. The top plot shows all the masked windows for the decimation level and which ones of those are masked for the individual evaluation frequencies. Poor quality time windows will generally be masked across all the evaluation frequencies. 

The same process can be repeated for the 4096 Hz data.

.. literalinclude:: ../../../examples/tutorial/usingMasks.py
    :linenos:
    :language: python
    :lines: 28-36
    :lineno-start: 28

The mask data plot for the 4096 Hz data is shown below.

.. figure:: ../_images/stat_statData_coherence_view.png
    :align: center
    :alt: alternate text
    :figclass: align-center

    Mask data plot for 4096 Hz data using coherence constraints of 0.70-1.00

Calculating masks with more than a single statistic can be done too. However, it is important to remember that **only time windows which satisfy all the constraints** will be used in the transfer function calculation. In the below example, a new mask is calculated using two statistics, coherence and transfer function.

.. literalinclude:: ../../../examples/tutorial/usingMasks.py
    :linenos:
    :language: python
    :lines: 38-53
    :lineno-start: 38

This produces the plot:

.. figure:: ../_images/stat_statData_coherence_view.png
    :align: center
    :alt: alternate text
    :figclass: align-center

    Mask data plot for 128 Hz data using coherence constraints of 0.70-1.00 and constraints on the transfer function values calculated on an individual window basis. 

Notice that adding more constraints has increased the number of masked windows as expected. Repeating the exercise with the 4096 Hz data:

.. literalinclude:: ../../../examples/tutorial/usingMasks.py
    :linenos:
    :language: python
    :lines: 55-69
    :lineno-start: 55

This produces the plot:

.. figure:: ../_images/stat_statData_coherence_view.png
    :align: center
    :alt: alternate text
    :figclass: align-center

    Mask data plot for 4096 Hz data using coherence constraints of 0.70-1.00 and constraints on the transfer function values calculated on an individual window basis. 

Here, the constraints have masked nearly all the windows for evaluation frequencies xx and xx. This is almost certainly over-zealous. Some statistic constraints can be provided on a global basis whereas others should be more targeted. More targeted constraints can be specified using the:

1. :meth:`~resistics.dataObjects.maskData.MaskData.addConstraintLevel` method of :class:`~resistics.dataObjects.maskData.MaskData`, which allows setting constraints just for a specific decimation level.
2. :meth:`~resistics.dataObjects.maskData.MaskData.addConstraintFreq` method of :class:`~resistics.dataObjects.maskData.MaskData`, which allows setting constraints for a specific evaluation frequency.

Using the second option requires specification of decimation level and evaluation frequency. In resistics, these are generally specified using indices. To understand what that means, consider the evaluation frequencies in this set up using 8 decimation levels and 5 evaluation frequencies per level with a data sampling frequency of 128 Hz:

.. code-block:: text

    Decimation Level = 0: 32.00000000, 22.62741700, 16.00000000, 11.31370850, 8.00000000
    Decimation Level = 1: 5.65685425, 4.00000000, 2.82842712, 2.00000000, 1.41421356
    Decimation Level = 2: 1.00000000, 0.70710678, 0.50000000, 0.35355339, 0.25000000
    Decimation Level = 3: 0.17677670, 0.12500000, 0.08838835, 0.06250000, 0.04419417
    Decimation Level = 4: 0.03125000, 0.02209709, 0.01562500, 0.01104854, 0.00781250
    Decimation Level = 5: 0.00552427, 0.00390625, 0.00276214, 0.00195312, 0.00138107
    Decimation Level = 6: 0.00097656, 0.00069053, 0.00048828, 0.00034527, 0.00024414
    Decimation Level = 7: 0.00017263, 0.00012207, 0.00008632, 0.00006104, 0.00004316

Decimation level numbering starts from 0 (and with 8 decimation levels, extends to 7). Evaluation frequency numbering begins from 0 (and with 5 evaluation frequencies per decimation level, extends to 4).

The decimation and evaluation frequency indices can be best demonstrated using a few of examples:

- Evaluation frequency 32 Hz, decimation level = 0, evaluation frequency index = 0
- Evaluation frequency 1 Hz, decimation level = 2, evaluation frequency index = 0
- Evaluation frequency 0.35355339 Hz, decimation level = 2, evaluation frequency index = 3

The main motivation behind this is the difficulty in manually specifying evaluation frequencies such as 0.35355339 Hz. 

The impact of masks on statistics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A more useful visualisation of masks is to see how removing selected time windows effects the statistics. In the :doc:`Statistics <statistics>` section, the plotting of statistics was demonstrated. Plotting statistics with masks applied is similar.

Load the project and use the :meth:`~resistics.project.projectMask.getMaskData` method of the :mod:`~resistics.project.projectMask` module to open up a previous calculated mask dataset.

.. literalinclude:: ../../../examples/tutorial/masksAndStatistics.py
    :linenos:
    :language: python
    :lines: 1-11
    :lineno-start: 1

Next, get a set of masked windows for an evaluation frequency of 32 Hz, which translates to decimation level 0 and evaluation frequency index 0.

.. literalinclude:: ../../../examples/tutorial/masksAndStatistics.py
    :linenos:
    :language: python
    :lines: 12-13
    :lineno-start: 12

To be able to plot the statistic data, this needs to be loaded to and can be by using the :meth:`~resistics.project.projectStatistics.getStatisticData` method of module :mod:`~resistics.project.projectStatistics`. For more information, see the :doc:`Statistics <statistics>` section.

.. literalinclude:: ../../../examples/tutorial/masksAndStatistics.py
    :linenos:
    :language: python
    :lines: 15-20
    :lineno-start: 15

Now the statistic data can be plotted with the mask data. In the examples below, the statistic data is plotted both with and without masking to demonstrate the difference.

Viewing transfer function statistic data:

.. literalinclude:: ../../../examples/tutorial/masksAndStatistics.py
    :linenos:
    :language: python
    :lines: 22-24
    :lineno-start: 22

.. figure:: ../_images/stat_statData_coherence_view.png
    :align: center
    :alt: alternate text
    :figclass: align-center

    Transfer function statistic data without masking 

.. figure:: ../_images/stat_statData_coherence_view.png
    :align: center
    :alt: alternate text
    :figclass: align-center

    Transfer function statistic data with masking (constraints on only coherence)

Histograms of transfer function statistic data:

.. literalinclude:: ../../../examples/tutorial/masksAndStatistics.py
    :linenos:
    :language: python
    :lines: 25-27
    :lineno-start: 22

.. figure:: ../_images/stat_statData_coherence_view.png
    :align: center
    :alt: alternate text
    :figclass: align-center

    Transfer function statistic histogram without masking 

.. figure:: ../_images/stat_statData_coherence_view.png
    :align: center
    :alt: alternate text
    :figclass: align-center

    Transfer function statistic histogram with masking (constraints on only coherence) 

Crossplots of transfer function statistic data:

.. literalinclude:: ../../../examples/tutorial/masksAndStatistics.py
    :linenos:
    :language: python
    :lines: 28-51
    :lineno-start: 28

.. figure:: ../_images/stat_statData_coherence_view.png
    :align: center
    :alt: alternate text
    :figclass: align-center

    Transfer function statistic crossplot without masking 

.. figure:: ../_images/stat_statData_coherence_view.png
    :align: center
    :alt: alternate text
    :figclass: align-center

    Transfer function statistic crossplot with masking (constraints on only coherence)

Repeating the same process with the second mask that was calculated (using both coherence and transfer function constraints) is easily done. 

.. literalinclude:: ../../../examples/tutorial/masksAndStatistics.py
    :linenos:
    :language: python
    :lines: 53-69
    :lineno-start: 53

.. figure:: ../_images/stat_statData_coherence_view.png
    :align: center
    :alt: alternate text
    :figclass: align-center

    Transfer function statistic with masking (constraints on both coherence and transfer function)

.. figure:: ../_images/stat_statData_coherence_view.png
    :align: center
    :alt: alternate text
    :figclass: align-center

    Transfer function statistic histogram with masking (constraints on both coherence and transfer function)

.. figure:: ../_images/stat_statData_coherence_view.png
    :align: center
    :alt: alternate text
    :figclass: align-center

    Transfer function statistic crossplot with masking (constraints on both coherence and transfer function)

Remember that statistic data is saved for each individual time series measurement directory. Hence, currently only the statistic values from one time series recordings are being considered. To really understand how masking windows will influence the transfer function calculation, the effect on the statistics for all time series measurements at a single sampling frequency in a site needs to be considered. This information can be plotted using the :meth:`~resistics.project.projectStatistics.viewStatistic` and  :meth:`~resistics.project.projectStatistics.viewStatisticHistogram` of the :mod:`~resistics.project.projectStatistics` module. An example is provided below. 

.. literalinclude:: ../../../examples/tutorial/masksAndStatistics.py
    :linenos:
    :language: python
    :lines: 71-94
    :lineno-start: 71

.. warning::

    These plots can be quite intensive due to the number of time windows in the data. Therefore, it is usually best to save them rather than show them. 

The result plots are shown below.

.. figure:: ../_images/stat_statData_coherence_view.png
    :align: center
    :alt: alternate text
    :figclass: align-center

    Transfer function statistic histogram with masking (constraints on both coherence and transfer function)

.. figure:: ../_images/stat_statData_coherence_view.png
    :align: center
    :alt: alternate text
    :figclass: align-center

    Transfer function statistic crossplot with masking (constraints on both coherence and transfer function)


Complete example script
~~~~~~~~~~~~~~~~~~~~~~~

For the purposes of clarity, the complete example scripts are provided below.

For calculating masks from statistics:

.. literalinclude:: ../../../examples/tutorial/usingMasks.py
    :linenos:
    :language: python

To see the impact of masks on statistics:

.. literalinclude:: ../../../examples/tutorial/masksAndStatistics.py
    :linenos:
    :language: python