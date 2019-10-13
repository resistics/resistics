.. |Ex| replace:: E\ :sub:`x`
.. |Ey| replace:: E\ :sub:`y`
.. |Hx| replace:: H\ :sub:`x`
.. |Hy| replace:: H\ :sub:`y`
.. |Hz| replace:: H\ :sub:`z`

Remote reference resistivity and phase
--------------------------------------

Looking at the transfer function components for each evaluation frequency is useful, but in many cases, it is easier to consider the apparent resistivity and phase. Apparent resitivty and phase are more physically relateable and can be a better choice for restricting window selection.

.. important::

    The resistics name for the remote reference resistivity and phase statistic is: **RR_resPhase**.

    The components of the remote reference resistivity and phase statistic are:

    - |Ex| |Hx| ResRR
    - |Ex| |Hx| PhaseRR
    - |Ex| |Hy| ResRR
    - |Ex| |Hy| PhaseRR
    - |Ey| |Hx| ResRR
    - |Ey| |Hx| PhaseRR
    - |Ey| |Hy| ResRR
    - |Ey| |Hy| PhaseRR

Examples are shown below for window-by-window remote reference apparent resistivity and phase. 

.. figure:: ../../_static/examples/features/remotestats/M1_RR_resphase_view_128.png
    :align: center
    :alt: alternate text
    :figclass: align-center

    Remote reference apparent resistivty and phase plotted over time for evaluation frequency 2.83 Hz

There are numerous outliers. This is clear from the time plot. The histogram below shows the existence of outliers by the automatic x axis limits. However, the number of outliers is small in comparison to the number of potentially good windows.

.. figure:: ../../_static/examples/features/remotestats/M1_RR_resphase_histogram_128.png
    :align: center
    :alt: alternate text
    :figclass: align-center

    Remote reference apparent resistivty and phase plotted as a histogram for evaluation frequency 2.83 Hz
