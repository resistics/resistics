Coherence
---------

Coherence is a measure of the causality between input and output in a linear system. In the magnetotelluric linear system stated below,

.. math::
    :nowrap:

    \begin{eqnarray}
    E_x & = & Z_{xx} H_x + Z_{xy} H_y \\
    E_y & = & Z_{yx} H_x + Z_{yy} H_y 
    \end{eqnarray} 

the inputs are usually the magnetic channels and the outputs the electric channels.

In most cases, the off-diagonals are expected to be small and the coherences of interest are between |Ey|-|Hx| and |Ex|-|Hy|. However, the coherence between all the channels is calculated out.

The coherence is a value between 0 and 1. A larger value indicates a stronger the relationship between signals. A useful way to visualise coherence is by plotting the histogram of coherences.

.. figure:: ../../_images/histCoherence.png
    :align: center
    :alt: alternate text
    :figclass: align-center

    A histogram of coherence values for the various coherence pairs

In this example, the coherences are clearly higher between channels |Ey|-|Hx| and |Ex|-|Hy| as expected. If the windows were limited to only those with high coherences for this evaluation frequency, it is clear that there will be still a large set of windows.

.. figure:: ../../_images/timeCoherence.png
    :align: center
    :alt: alternate text
    :figclass: align-center

    An example of coherence pairs plotted over time

An alternative way to plot the values is with respect to time. In the above plot, it is clear there are periods of time where the coherence between the channels |Ey|-|Hx| and |Ex|-|Hy| drops. These periods indicate noisier sections of time data.

For more information about coherence, see: https://en.wikipedia.org/wiki/Coherence_(signal_processing) 


.. |Ex| replace:: E\ :sub:`x`
.. |Ey| replace:: E\ :sub:`y`
.. |Hx| replace:: H\ :sub:`x`
.. |Hy| replace:: H\ :sub:`y`
.. |Hz| replace:: H\ :sub:`z`