Consider the evaluation frequencies in this set up using 8 decimation levels and 5 evaluation frequencies per level with a data sampling frequency of 128 Hz:

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