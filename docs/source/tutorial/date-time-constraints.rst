Date and time constraints
-------------------------

Another way of constraining the data before evaluation frequency calculation is through date and time constraints. In many cases, it can be advantageous to explicitly define the dates and times of the time windows (and their respective spectra) to include in the transfer function calculation. 

There are three types of date and time constraints:

- A time constraint which recurs on a daily basis
- A date constraint to define dates to use
- A combined date and time constraint

.. note::

    For a better understanding of how window selection works, see the :doc:`Recap and deeper dive <recap-and-deeper-dive>`.

As always, the means of providing date and time constraints will be demonstrated through an example. Begin as always by loading the project.

.. literalinclude:: ../../../examples/tutorial/datetimeConstraints.py
    :linenos:
    :language: python
    :lines: 1-6
    :lineno-start: 1

The next step is define a list to hold the date/time constraints and add a single date/time constraint, which is in the form of a dictionary.

.. literalinclude:: ../../../examples/tutorial/datetimeConstraints.py
    :linenos:
    :language: python
    :lines: 8-12
    :lineno-start: 8

As stated above, there are three types of datetime constraints. They are each specified in the following way:

- Time constraint as a dictionary: {"type": "time", "start": "HH:MM:SS", "stop": "HH:MM:SS"}. If the stop time is less than the start time, the time constraint is assumed to cross days. For example, {"type": "time", "start": "20:00:00", "stop": "07:00:00"} will take only night time data.
- Date constraint as a dictionary: {"type": "date", "date": "YYYY-MM-DD"}.  
- Datetime constraint as a dictionary: {"type": "datetime", "start": "YYYY-MM-DD HH:MM:SS", "stop": "YYYY-MM-DD HH:MM:SS"}

Multiple date/time constraints can be provided and they will be combined and only windows which fall within these date/time constraints will be used. Below is an example of processing with date/time constraints. 

.. literalinclude:: ../../../examples/tutorial/datetimeConstraints.py
    :linenos:
    :language: python
    :lines: 14-31
    :lineno-start: 14

There is little new in this example apart from the addition of the *datetimes* keyword to specify the date/time constraints.

The processing gives the resultant transfer function (note, only processing 128 Hz time series data here).

.. figure:: ../../../examples/tutorial/tutorialProject/images/transFunction_site1_dec8_5_datetimeConstraint.png
    :align: center
    :alt: alternate text
    :figclass: align-center

    Impedance tensor estimate with date/time constraints

Compare this to the same processing but without datetime constraints.

.. figure:: ../../../examples/tutorial/tutorialProject/images/transFunction_site1_dec8_5.png
    :align: center
    :alt: alternate text
    :figclass: align-center

    Impedance tensor estimate without date/time constraints

Finally, date/time constraints and masks can be combined as shown in the following example. 

.. literalinclude:: ../../../examples/tutorial/datetimeConstraints.py
    :linenos:
    :language: python
    :lines: 33-51
    :lineno-start: 33

This gives the below transfer function.

.. figure:: ../../../examples/tutorial/tutorialProject/images/transFunction_site1_dec8_5_coh70_100_tfConstrained_datetimeConstrained.png
    :align: center
    :alt: alternate text
    :figclass: align-center

    Impedance tensor estimate with date/time constraints and masks based on coherence and transfer function statistics

In comparison, the transfer function with just the mask.

.. figure:: ../../../examples/tutorial/tutorialProject/images/transFunction_site1_dec8_5_coh70_100_tfConstrained.png
    :align: center
    :alt: alternate text
    :figclass: align-center

    Impedance tensor estimate with just the masking based on coherence and transfer function statistics

Complete example script
~~~~~~~~~~~~~~~~~~~~~~~
For the purposes of clarity, the complete example script is provided below.

.. literalinclude:: ../../../examples/tutorial/datetimeConstraints.py
    :linenos:
    :language: python