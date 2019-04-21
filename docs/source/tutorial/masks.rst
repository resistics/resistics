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
    :lines: 22-27
    :lineno-start: 22

As statistics are calculated on an evaluation frequency basis, masked windows are calculated for each evaluation frequency too. 

Once the masked windows are calculated, they can then be viewed using the :meth:`~resistics.dataObjects.maskData.MaskData.view` method of :class:`~resistics.dataObjects.maskData.MaskData`. When using the :meth:`~resistics.dataObjects.maskData.MaskData.view` method, the decimation level has to be specified. In the example above, this has been set to 0, which is the first decimation level. The resultant plot is shown below.

.. figure:: ../../../examples/tutorial/tutorialProject/images/maskData_128_coh_dec0.png
    :align: center
    :alt: alternate text
    :figclass: align-center

    Mask data plot for 128 Hz data using coherence constraints of 0.70-1.00

The bar chart at the bottom shows the number of masked windows for each evaluation frequency in the decimation level, plus the total number of masked windows for the decimation level. The top plot shows all the masked windows for the decimation level and which ones of those are masked for the individual evaluation frequencies. Poor quality time windows will generally be masked across all the evaluation frequencies. 

The same process can be repeated for the 4096 Hz data.

.. literalinclude:: ../../../examples/tutorial/usingMasks.py
    :linenos:
    :language: python
    :lines: 29-38
    :lineno-start: 29

The mask data plot for the 4096 Hz data is shown below.

.. figure:: ../../../examples/tutorial/tutorialProject/images/maskData_4096_coh_dec0.png
    :align: center
    :alt: alternate text
    :figclass: align-center

    Mask data plot for 4096 Hz data using coherence constraints of 0.70-1.00

Calculating masks with more than a single statistic can be done too. However, it is important to remember that **only time windows which satisfy all the constraints** will be used in the transfer function calculation. In the below example, a new mask is calculated using two statistics, coherence and transfer function.

.. literalinclude:: ../../../examples/tutorial/usingMasks.py
    :linenos:
    :language: python
    :lines: 40-56
    :lineno-start: 40

This produces the plot:

.. figure:: ../../../examples/tutorial/tutorialProject/images/maskData_128_coh_tf_dec0.png
    :align: center
    :alt: alternate text
    :figclass: align-center

    Mask data plot for 128 Hz data using coherence constraints of 0.70-1.00 and constraints on the transfer function values calculated on an individual window basis. 

Notice that adding more constraints has increased the number of masked windows as expected. Repeating the exercise with the 4096 Hz data:

.. literalinclude:: ../../../examples/tutorial/usingMasks.py
    :linenos:
    :language: python
    :lines: 58-73
    :lineno-start: 58

This produces the plot:

.. figure:: ../../../examples/tutorial/tutorialProject/images/maskData_4096_coh_tf_dec0.png
    :align: center
    :alt: alternate text
    :figclass: align-center

    Mask data plot for 4096 Hz data using coherence constraints of 0.70-1.00 and constraints on the transfer function values calculated on an individual window basis. 

Here, the constraints have masked many windows for evaluation frequencies 1024 Hz and 724.077344 Hz. This is almost certainly over-zealous for these two evaluation frequencies (but to better understand how over-zealous read further down). 

Up until now, all mask constraints have been specified globally (i.e. as applying to all evaluation frequencies). However, statistic constraints can be defined in a more targeted manner, at evaluation frequencies that are noisier than others. More targeted constraints can be specified using the listed methods.

1. :meth:`~resistics.dataObjects.maskData.MaskData.addConstraintLevel` method of :class:`~resistics.dataObjects.maskData.MaskData`, which allows setting constraints just for a specific decimation level.
2. :meth:`~resistics.dataObjects.maskData.MaskData.addConstraintFreq` method of :class:`~resistics.dataObjects.maskData.MaskData`, which allows setting constraints for a specific evaluation frequency.

Using the second option requires specification of decimation level and evaluation frequency. In resistics, these are generally specified using indices as shown in the :doc:`Statistics <statistics>` section. 

.. important::

    It is worth repeating here the meaning of the decimation level index and evaluation frequency index, which was previously covered in :doc:`Statistics <statistics>`. 
    
    .. include:: decimation-eval-indices.rst 

The impact of masks on statistics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A more useful visualisation of masks is to see how removing selected time windows effects the statistics. In the :doc:`Statistics <statistics>` section, the plotting of statistics was demonstrated. Plotting statistics with masks applied is similar.

Load the project and use the :meth:`~resistics.project.projectMask.getMaskData` method of the :mod:`~resistics.project.projectMask` module to open up a previous calculated mask dataset at a sampling frequency of 4096 Hz.

.. literalinclude:: ../../../examples/tutorial/masksAndStatistics.py
    :linenos:
    :language: python
    :lines: 1-11
    :lineno-start: 1

Next, get a set of masked windows for an evaluation frequency of 1024 Hz, which translates to decimation level 0 and evaluation frequency index 0.

.. literalinclude:: ../../../examples/tutorial/masksAndStatistics.py
    :linenos:
    :language: python
    :lines: 12-13
    :lineno-start: 12

To be able to plot the statistic data, this needs to be loaded too and can be by using the :meth:`~resistics.project.projectStatistics.getStatisticData` method of module :mod:`~resistics.project.projectStatistics`. For more information, see the :doc:`Statistics <statistics>` section.

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
    :lines: 22-26
    :lineno-start: 22

.. figure:: ../../../examples/tutorial/tutorialProject/images/statistic_4096_nomask_view.png
    :align: center
    :alt: alternate text
    :figclass: align-center

    Transfer function statistic data without masking 

.. figure:: ../../../examples/tutorial/tutorialProject/images/statistic_4096_maskcoh_view.png
    :align: center
    :alt: alternate text
    :figclass: align-center

    Transfer function statistic data with masking (constraints on only coherence)

Ideally, the effect of masking should lead to more consistency in the various components of the transfer function, with less scatter. In this case, coherence based time window masking has reduced the scatter around the average value, which will lead to an improvement in the robust regression for transfer funciton estimation. 

Histograms of transfer function statistic data:

.. literalinclude:: ../../../examples/tutorial/masksAndStatistics.py
    :linenos:
    :language: python
    :lines: 27-31
    :lineno-start: 27

.. figure:: ../../../examples/tutorial/tutorialProject/images/statistic_4096_nomask_hist.png
    :align: center
    :alt: alternate text
    :figclass: align-center

    Transfer function statistic histogram without masking 

.. figure:: ../../../examples/tutorial/tutorialProject/images/statistic_4096_maskcoh_hist.png
    :align: center
    :alt: alternate text
    :figclass: align-center

    Transfer function statistic histogram with masking (constraints on only coherence) 

For the transfer function statistic, Gaussian distributions across the windows is an indication that the overall transfer function estimate will be good. In situations where window-by-window transfer function estimates are not near Gaussian distributed, the resultant overall transfer function estimation will normally be poor. Here, the data is quite well distributed.  

Crossplots of transfer function statistic data:

.. literalinclude:: ../../../examples/tutorial/masksAndStatistics.py
    :linenos:
    :language: python
    :lines: 32-57
    :lineno-start: 32

.. figure:: ../../../examples/tutorial/tutorialProject/images/statistic_4096_nomask_crossplot.png
    :align: center
    :alt: alternate text
    :figclass: align-center

    Transfer function statistic crossplot without masking 

.. figure:: ../../../examples/tutorial/tutorialProject/images/statistic_4096_maskcoh_crossplot.png
    :align: center
    :alt: alternate text
    :figclass: align-center

    Transfer function statistic crossplot with masking (constraints on only coherence)

Plotting the window-by-window transfer function estimation in the complex plane should ideally yield a tight scatter of points around an average point. Where this is not the case, the overall transfer function estimate will tend to be poor. In this case, the data is good.

Repeating the same process with the second mask that was calculated (using both coherence and transfer function constraints) is easily done. 

.. literalinclude:: ../../../examples/tutorial/masksAndStatistics.py
    :linenos:
    :language: python
    :lines: 59-78
    :lineno-start: 59

.. figure:: ../../../examples/tutorial/tutorialProject/images/statistic_4096_maskcoh_tf_view.png
    :align: center
    :alt: alternate text
    :figclass: align-center

    Transfer function statistic with masking (constraints on both coherence and transfer function)

.. figure:: ../../../examples/tutorial/tutorialProject/images/statistic_4096_maskcoh_tf_hist.png
    :align: center
    :alt: alternate text
    :figclass: align-center

    Transfer function statistic histogram with masking (constraints on both coherence and transfer function)

.. figure:: ../../../examples/tutorial/tutorialProject/images/statistic_4096_maskcoh_tf_crossplot.png
    :align: center
    :alt: alternate text
    :figclass: align-center

    Transfer function statistic crossplot with masking (constraints on both coherence and transfer function)

As suggested earlier, adding the extra mask constraints for this evaluation frequency was over-zealous, mainly due to the ExHyImag and EyHxImag constraints. Only a handful of time windows meet these constraints and the transfer function estimate at this evaluation frequency is likely to be poorer than when simply using the coherence mask.

Remember that statistic data is saved for each individual time series measurement directory. Hence, currently only the statistic values from one time series recordings are being considered. To really understand how masking windows will influence the transfer function calculation, the effect on the statistics for all time series measurements at a single sampling frequency in a site needs to be considered. This information can be plotted using the :meth:`~resistics.project.projectStatistics.viewStatistic` and  :meth:`~resistics.project.projectStatistics.viewStatisticHistogram` of the :mod:`~resistics.project.projectStatistics` module. An example is provided below. 

.. literalinclude:: ../../../examples/tutorial/masksAndStatistics.py
    :linenos:
    :language: python
    :lines: 80-107
    :lineno-start: 71

.. warning::

    These plots can be quite intensive due to the number of time windows in the data. Therefore, it is usually best to save them rather than show them. 

The result plots are shown below.

.. figure:: ../../../examples/tutorial/tutorialProject/images/stat_coherence_site1_128_000_dec0_efreq0_dec8_5_coh70_100.png
    :align: center
    :alt: alternate text
    :figclass: align-center

    Transfer function statistic histogram with masking (constraints on both coherence and transfer function)

.. figure:: ../../../examples/tutorial/tutorialProject/images/statHist_coherence_site1_128_000_dec0_efreq0_dec8_5_coh70_100.png
    :align: center
    :alt: alternate text
    :figclass: align-center

    Transfer function statistic crossplot with masking (constraints on both coherence and transfer function)

Now that masks have been calculated, the next stage is to use the masks in the transfer function estimation, which is demonstrated in the :doc:`Processing with masks <mask-processing>` section.

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