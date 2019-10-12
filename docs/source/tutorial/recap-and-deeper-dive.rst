Recap and deeper dive
---------------------

Now that many of the basic features of resistics have been introduced, it is a useful time to recap some theory and consider some lower level resistics API. The :class:`~resistics.decimate.parameters.DecimationParameters`, :class:`~resistics.window.parameters.WindowParameters` and :class:`~resistics.window.selector.WindowSelector` classes are central to how resistics processes magnetotelluric data. They are lower level elements of the resistics package but are useful to know and understand, particularly before approaching the :doc:`Advanced <../advanced>` and :doc:`Cookbook <../cookbook>` sections. 

Recap
~~~~~

Ahead of introducing the window selector, it is useful to do a quick recap of how resistics processes magnetotelluric data. Ultimately, the point of processing magnetotelluric data is to understand the electrical properties of the subsurface. 

In the standard magnetotelluric setup, the electrical properties of the subsurface are described by the impedance tensor, which relates the input magnetic fields to the output electric fields. This is usually posed in the frequency domain, as shown below. Please see recommended literature if this is unfamiliar.

.. math::
    :nowrap:

    \begin{eqnarray}
    E_{x}(\omega) & = & Z_{xx}(\omega) H_{x}(\omega) + Z_{xy}(\omega) H_{y}(\omega) \\
    E_{y}(\omega) & = & Z_{yx}(\omega) H_{x}(\omega) + Z_{yy}(\omega) H_{y}(\omega) 
    \end{eqnarray} 

Here, the dependency on frequency (:math:`\omega`) is shown explicitly. 

In the field, :math:`E_{x}`, :math:`E_{y}`, :math:`H_{x}`, :math:`H_{y}` (and often :math:`H_{z}`) are recorded. These are then converted into the frequency domain using the Fourier transform, or in resistics, calculating spectra. To read more about resistics spectra, see :doc:`Up and running <up-and-running>` and :doc:`Viewing spectra <viewing-spectra>`.

Resistics then processes the magnetotelluric spectra data to try and estimate the four unknown values of :math:`Z_{xx}`, :math:`Z_{yy}`, :math:`Z_{xy}`, :math:`Z_{yx}`. But in the above, there are two equations and four unknowns, so somehow, more equations need to be generated. This is normally done by by using cross-powers as shown in the below set of equations,

.. math::
    :nowrap:

    \begin{eqnarray}
    <E_{x}(\omega) E_{x}^{*}(\omega)> & = & Z_{xx}(\omega) <H_{x}(\omega) E_{x}^{*}(\omega)> + Z_{xy}(\omega) <H_{y}(\omega) E_{x}^{*}(\omega)> \\
    <E_{x}(\omega) E_{y}^{*}(\omega)> & = & Z_{xx}(\omega) <H_{x}(\omega) E_{y}^{*}(\omega)> + Z_{xy}(\omega) <H_{y}(\omega) E_{y}^{*}(\omega)> \\
    <E_{x}(\omega) H_{x}^{*}(\omega)> & = & Z_{xx}(\omega) <H_{x}(\omega) H_{x}^{*}(\omega)> + Z_{xy}(\omega) <H_{y}(\omega) H_{x}^{*}(\omega)> \\
    <E_{x}(\omega) H_{y}^{*}(\omega)> & = & Z_{xx}(\omega) <H_{x}(\omega) H_{y}^{*}(\omega)> + Z_{xy}(\omega) <H_{y}(\omega) H_{y}^{*}(\omega)> \\              
    \end{eqnarray} 

.. math::
    :nowrap:

    \begin{eqnarray}    
    <E_{y}(\omega) E_{x}^{*}(\omega)> & = & Z_{yx}(\omega) <H_{x}(\omega) E_{x}^{*}(\omega)> + Z_{yy}(\omega) <H_{y}(\omega) E_{x}^{*}(\omega)> \\ 
    <E_{y}(\omega) E_{y}^{*}(\omega)> & = & Z_{yx}(\omega) <H_{x}(\omega) E_{y}^{*}(\omega)> + Z_{yy}(\omega) <H_{y}(\omega) E_{y}^{*}(\omega)> \\ 
    <E_{y}(\omega) H_{x}^{*}(\omega)> & = & Z_{yx}(\omega) <H_{x}(\omega) H_{x}^{*}(\omega)> + Z_{yy}(\omega) <H_{y}(\omega) H_{x}^{*}(\omega)> \\ 
    <E_{y}(\omega) H_{y}^{*}(\omega)> & = & Z_{yx}(\omega) <H_{x}(\omega) H_{y}^{*}(\omega)> + Z_{yy}(\omega) <H_{y}(\omega) H_{y}^{*}(\omega)> \\ 
    \end{eqnarray}     

where :math:`*` represents the complex conjugate.

This equations are solved at a set of frequencies called the **evaluation frequencies**. These are normally chosen on a log scale. 

So theoretically, a magnetotelluric processing system could do the following:

#. Transform full electric and magnetic field time series measurements to frequency domain
#. Calculate cross-power spectra
#. Select a set of evaluation frequencies
#. For each evaluation frequency, perform linear regression to solve for :math:`Z_{xx}`, :math:`Z_{yy}`, :math:`Z_{xy}`, :math:`Z_{yx}`

However, the impedance tensor estimates can be improved by providing more equations or by stacking cross-power spectra, especially in the presence of electromagnetic noise. This is usually done by windowing the time series, in which case the processing flow looks like this:

#. Choose window parameters (window size and overlap size between windows)
#. Transform each window into frequency domain (calculate spectra)
#. Calculate cross-power spectra at a set of evaluation frequencies
#. Stack the cross-power spectra for the different windows
#. For each evaluation frequency, perform linear regression to solve for :math:`Z_{xx}`, :math:`Z_{yy}`, :math:`Z_{xy}`, :math:`Z_{yx}`

There are other variants on this and stacking strategies. It is not necessary for stacking to be performed and instead cross-powers from a set of randomly selected windows can be used. Further, linear regression can be replaced by robust regression.

For the purposes of calculating out the longer period evaluation frequencies, the time data is usually decimated before the process is repeated. Explicitly, this means:

- For decimation level in number of decimation levels 

    #. Decimate time series by the decimation factor
    #. Choose window parameters for decimation level
    #. Select evaluation frequencies for decimation level    
    #. Transform each window into frequency domain (calculate spectra)
    #. Calculate cross-power spectra at the evaluation frequencies
    #. Stack the cross-power spectra for the different windows
    #. For each evaluation frequency, perform linear regression to solve for :math:`Z_{xx}`, :math:`Z_{yy}`, :math:`Z_{xy}`, :math:`Z_{yx}`

In the following sections, the resistics approach to decimation parameters, window parameters and window selection will be detailed. It is not necessary to know this, but this information will be a useful pre-cursor to exploring the :doc:`Advanced <../advanced>` and :doc:`Cookbook <../cookbook>` sections.

Decimation parameters
~~~~~~~~~~~~~~~~~~~~~

The :class:`~resistics.decimate.parameters.DecimationParameters` class holds the decimation information. Decimation parameters can be set using configuration files (see :doc:`Using configuration files <configuration-files>`). To see what the :class:`~resistics.decimate.parameters.DecimationParameters` class holds, consider an example.

.. literalinclude:: ../../../examples/tutorial/usingWindowSelector.py
    :linenos:
    :language: python
    :lines: 1-11
    :lineno-start: 1

The project has been loaded along with the configuration file, which has specified 8 decimation levels and 5 evaluation frequencies per level. The decimation paramters can be printed to the terminal using the parent class :meth:`~resistics.common.base.ResisticsBase.printInfo` method to give:

.. literalinclude:: ../_static/examples/tutorial/printDecimationParameters.txt
    :linenos:
    :language: text

The information provides all the decimation factors and the sampling frequencies at each decimation level. It also provides the 5 evaluation frequencies for each decimation level. 

Window parameters
~~~~~~~~~~~~~~~~~

The :class:`~resistics.window.parameters.WindowParameters` class contains information about the windowing parameters. Window parameters can be set using configuration files (see :doc:`Using configuration files <configuration-files>`).

Again, consider an example which continues from the decimation paramters example:

.. literalinclude:: ../../../examples/tutorial/usingWindowSelector.py
    :linenos:
    :language: python
    :lines: 13-17
    :lineno-start: 13

The project has been loaded along with the configuration file, which has specified a minimum window size of 256 samples and a minimum overlap of 64 samples. However, resistics is free to use window and overlap sizes greater than these if it makes sense. Windowing parameter information can be printed to the terminal using the :meth:`~resistics.common.base.ResisticsBase.printInfo` method.

.. literalinclude:: ../_static/examples/tutorial/printWindowParameters.txt
    :linenos:
    :language: text

In this case, for the 4096 Hz data, the for the lower decimation levels, larger window sizes are used but as the decimation increases, the 256 minimum window sample size and 64 minimum overlap sample size is used. 

Window selector
~~~~~~~~~~~~~~~

The :class:`~resistics.window.selector.WindowSelector` class is a key component of resistics. Before processing time windows (and their respective spectra) to calculate transfer function estimates, the windows to use have to be selected. The :class:`~resistics.window.selector.WindowSelector` class is the how windows are selected. 

Below are a few scenarios under which windows might have to be selected:

#. Single site processing, use all the data
#. Single site processing with masks and/or date and time constraints
#. Remote reference processing with two sites with masks and/or date and time constraints
#. Intersite processing with two sites
#. Intersite processing with a remote reference as a third site

Let's compare 1 and 2 as simple examples. The others will be introduced in the :doc:`Advanced <../advanced>` section.

Begin by getting a :class:`~resistics.window.selector.WindowSelector` instance. In this case, the 128 Hz data is being used. 

.. literalinclude:: ../../../examples/tutorial/usingWindowSelector.py
    :linenos:
    :language: python
    :lines: 19-24
    :lineno-start: 24

The :class:`~resistics.window.selector.WindowSelector` needs to know the :class:`~resistics.decimate.parameters.DecimationParameters` and :class:`~resistics.window.parameters.WindowParameters`. Information can be printed to the terminal using the :meth:`~resistics.common.base.ResisticsBase.printInfo` method. This gives:

.. literalinclude:: ../_static/examples/tutorial/printWindowSelector_1.txt
    :linenos:
    :language: text

Currently, no sites have been specified for the :class:`~resistics.window.selector.WindowSelector` instance. Therefore, there are yet no windows to select from. The next step then is to specify a list of sites. 

.. literalinclude:: ../../../examples/tutorial/usingWindowSelector.py
    :linenos:
    :language: python
    :lines: 26-28
    :lineno-start: 26

Printing information gives the following, which details the recordings and the global windows for those recordings. The global windows are the window numbers beginning from the project reference time. Here there is nothing for decimation level 7 as the time series was not long enough to give sufficient windows after decimation. 

.. literalinclude:: ../_static/examples/tutorial/printWindowSelector_2.txt
    :linenos:
    :language: text

Now the shared windows can be calculated using the :meth:`~resistics.window.selector.WindowSelector.calcSharedWindows` method. In this case, only a single site is specified so there are no other sites to share windows with. When more than a single site is specified, the shared windows between the sites will be calculated. **This means the times where both sites were recording**. This is an essential step for remote reference and intersite processing (see :doc:`Advanced <../advanced>`). 

.. literalinclude:: ../../../examples/tutorial/usingWindowSelector.py
    :linenos:
    :language: python
    :lines: 30-32
    :lineno-start: 30

The shared windows can be printed to the terminal using the :meth:`~resistics.window.selector.WindowSelector.printSharedWindows` method.

.. literalinclude:: ../_static/examples/tutorial/printWindowSelector_3.txt
    :linenos:
    :language: text

The :class:`~resistics.window.selector.WindowSelector` has functionality to limit windows using date and time constraints. Date and time constraints can be provided using the :meth:`~resistics.window.selector.WindowSelector.addDateConstraint`, :meth:`~resistics.window.selector.WindowSelector.addTimeConstraint` and :meth:`~resistics.window.selector.WindowSelector.addDatetimeConstraint` methods. 

.. important::

    Providing a date or time constraint means that only windows which are inside these date/time constraints will be selected. Any windows which are outside the date/time constraint will be discarded.

Below is an example of providing a date constraint using the :meth:`~resistics.window.selector.WindowSelector.addDateConstraint` method. After running the :meth:`~resistics.window.selector.WindowSelector.calcSharedWindows` method, only time windows within this date/time constraint will be selected. 

.. literalinclude:: ../../../examples/tutorial/usingWindowSelector.py
    :linenos:
    :language: python
    :lines: 34-38
    :lineno-start: 34

The datetime constraints can be printed to the terminal using the :meth:`~resistics.window.selector.WindowSelector.printDatetimeConstraints` method. The various print methods give the following: 

.. literalinclude:: ../_static/examples/tutorial/printWindowSelector_4.txt
    :linenos:
    :language: text

Comparing the number of shared windows to the previous results, it is clear that the number of shared windows has now reduced. This is due to the application of the date constraint. To reset the date constraint, the :meth:`~resistics.window.selector.WindowSelector.resetDatetimeConstraints` method can be used as in the following example, where the current date/time constraint is cleared and a new one added. 

.. literalinclude:: ../../../examples/tutorial/usingWindowSelector.py
    :linenos:
    :language: python
    :lines: 40-45
    :lineno-start: 40

Calculating the shared windows again with this new datetime constraint and printing the information to the terminal gives:

.. literalinclude:: ../_static/examples/tutorial/printWindowSelector_5.txt
    :linenos:
    :language: text

The number of shared windows has reduced even more. Instead of using a full day (24 hours) of data as in the previous example with the date constraint, now only the evening and night has been specified, totalling 13 hours.  

Masks can also be specified in the :class:`~resistics.window.selector.WindowSelector` using the :meth:`~resistics.window.selector.WindowSelector.addWindowMask` method.

.. literalinclude:: ../../../examples/tutorial/usingWindowSelector.py
    :linenos:
    :language: python
    :lines: 47-51
    :lineno-start: 47

And printing the information to the terminal gives:

.. literalinclude:: ../_static/examples/tutorial/printWindowSelector_6.txt
    :linenos:
    :language: text

However, there has been no change in the number of shared windows, despite the application of the masks. This is because:

.. important::

    Recall that statistics are calculated for each evaluation frequency. Therefore masks are applied on an evaluation frequency basis rather than decimation level. 
    
The loop below can demonstrate the effect of masks on the number of windows for each evaluation frequency.

.. literalinclude:: ../../../examples/tutorial/usingWindowSelector.py
    :linenos:
    :language: python
    :lines: 53-70
    :lineno-start: 53

The output from this loop is shown below.

.. literalinclude:: ../../../examples/tutorial/usingWindowSelector.txt
    :linenos:
    :language: text

.. note::

    To see how to use date and time constraints in the processing of magnetotelluric data, see  :doc:`Date and time constraints <date-time-constraints>`.


Complete example script
~~~~~~~~~~~~~~~~~~~~~~~
For the purposes of clarity, the complete example script is provided below.

.. literalinclude:: ../../../examples/tutorial/usingWindowSelector.py
    :linenos:
    :language: python