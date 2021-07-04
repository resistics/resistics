"""
These are spectra data dictionaries and metadata for testing of combining data

Whilst the data is generally called eval(uation frequency) data, it is in
reality the same as spectra data.
"""
import numpy as np

from resistics.testing import spectra_metadata_multilevel
from resistics.spectra import SpectraMetadata, SpectraData

# eval data dictionarys
# each level should be an array with shape n_wins * n_chans (2) * n_freqs (2)

# level 0 windows: 5
# level 1 windows: 4
# level 3 windows: 3
SITE1_MEAS1_DATA = {
    0: np.array(
        [
            [[1 + 1j, 2 + 2j], [1 + 3j, 2 + 4j]],
            [[2 + 1j, 3 + 2j], [1 + 1j, 2 + 2j]],
            [[4 + 3j, 3 + 4j], [3 + 2j, 2 + 3j]],
            [[2 + 2j, 3 + 3j], [5 + 5j, 5 + 4j]],
            [[5 + 1j, 1 + 6j], [4 + 1j, 6 + 1j]],
        ]
    ),
    1: np.array(
        [
            [[1 + 2j, 2 + 1j], [3 + 3j, 3 + 2j]],
            [[1 + 3j, 3 + 1j], [4 + 1j, 4 + 2j]],
            [[5 + 3j, 3 + 2j], [2 + 2j, 2 + 3j]],
            [[4 + 4j, 2 + 1j], [1 + 6j, 6 + 3j]],
        ]
    ),
    2: np.array(
        [
            [[1 + 2j, 2 + 1j], [3 + 3j, 3 + 2j]],
            [[1 + 3j, 3 + 1j], [4 + 1j, 4 + 2j]],
            [[5 + 3j, 3 + 2j], [2 + 2j, 2 + 3j]],
        ]
    ),
}

# level 0 windows: 12
# level 1 windows: 9
# level 3 windows: 7
SITE1_MEAS2_DATA = {
    0: np.array(
        [
            [[-1 + 1j, -2 + 2j], [-1 + 3j, -2 + 4j]],
            [[-2 + 1j, -3 + 2j], [-1 + 1j, -2 + 2j]],
            [[-4 + 3j, -3 + 4j], [-3 + 2j, -2 + 3j]],
            [[-2 + 2j, -3 + 3j], [-5 + 5j, -5 + 4j]],
            [[-5 + 1j, -1 + 6j], [-4 + 1j, -3 + 1j]],
            [[-2 + 1j, -2 + 2j], [-5 + 3j, -5 + 4j]],
            [[-3 + 1j, -4 + 2j], [-5 + 1j, -4 + 2j]],
            [[-4 + 3j, -5 + 4j], [-3 + 2j, -3 + 3j]],
            [[-2 + 2j, -4 + 3j], [-2 + 5j, -1 + 4j]],
            [[-1 + 1j, -3 + 6j], [-5 + 1j, -1 + 1j]],
            [[-6 + 1j, -1 + 2j], [-5 + 3j, -1 + 4j]],
            [[-3 + 1j, -2 + 2j], [-6 + 1j, -2 + 2j]],
        ]
    ),
    1: np.array(
        [
            [[-1 + 2j, -2 + 1j], [-3 + 3j, -3 + 2j]],
            [[-1 + 3j, -3 + 1j], [-4 + 1j, -4 + 2j]],
            [[-5 + 3j, -3 + 2j], [-2 + 2j, -2 + 3j]],
            [[-4 + 4j, -1 + 1j], [-1 + 6j, -6 + 3j]],
            [[-4 + 2j, -2 + 1j], [-4 + 3j, -3 + 2j]],
            [[-5 + 3j, -2 + 1j], [-5 + 1j, -2 + 2j]],
            [[-4 + 3j, -3 + 2j], [-6 + 2j, -1 + 3j]],
            [[-5 + 4j, -2 + 1j], [-4 + 6j, -1 + 3j]],
            [[-3 + 2j, -1 + 1j], [-2 + 3j, -1 + 2j]],
        ]
    ),
    2: np.array(
        [
            [[-1 + 2j, -2 + 1j], [-3 + 3j, -3 + 2j]],
            [[-1 + 3j, -3 + 1j], [-4 + 1j, -4 + 2j]],
            [[-5 + 3j, -3 + 2j], [-2 + 2j, -2 + 3j]],
            [[-1 + 3j, -2 + 1j], [-4 + 1j, -2 + 2j]],
            [[-2 + 3j, -2 + 2j], [-2 + 2j, -1 + 3j]],
            [[-2 + 4j, -3 + 1j], [-3 + 6j, -2 + 3j]],
            [[-1 + 2j, -4 + 1j], [-4 + 3j, -3 + 2j]],
        ]
    ),
}

# level 0 windows: 7
# level 1 windows: 4
# level 3 windows: 2
SITE1_MEAS3_DATA = {
    0: np.array(
        [
            [[1 - 1j, 2 - 2j], [1 - 3j, 2 - 4j]],
            [[2 - 1j, 3 - 2j], [1 - 1j, 2 - 2j]],
            [[4 - 3j, 3 - 1j], [3 - 3j, 2 - 3j]],
            [[2 - 2j, 3 - 2j], [5 - 4j, 5 - 3j]],
            [[5 - 1j, 1 - 4j], [4 - 2j, 6 - 3j]],
            [[6 - 2j, 3 - 1j], [5 - 5j, 5 - 2j]],
            [[3 - 1j, 1 - 2j], [4 - 3j, 6 - 0j]],
        ]
    ),
    1: np.array(
        [
            [[1 - 2j, 2 - 1j], [3 - 3j, 3 - 2j]],
            [[1 - 3j, 3 - 1j], [4 - 1j, 4 - 2j]],
            [[5 - 3j, 3 - 2j], [2 - 2j, 2 - 3j]],
            [[4 - 0j, 2 - 2j], [1 - 5j, 6 - 3j]],
        ]
    ),
    2: np.array(
        [
            [[1 - 2j, 2 - 1j], [3 - 3j, 3 - 2j]],
            [[1 - 3j, 3 - 1j], [4 - 1j, 4 - 2j]],
        ]
    ),
}


# the site combined data
SITE1_COMBINED_DATA = {
    0: np.array(
        [
            [1 + 1j, 1 + 3j],  # meas1 0
            [2 + 1j, 1 + 1j],  # meas1 1
            [4 + 3j, 3 + 2j],  # meas1 2
            [2 + 2j, 5 + 5j],  # meas1 3
            [5 + 1j, 4 + 1j],  # meas1 4
            [-2 + 1j, -1 + 1j],  # meas2 1
            [-4 + 3j, -3 + 2j],  # meas2 2
            [-2 + 2j, -5 + 5j],  # meas2 3
            [-5 + 1j, -4 + 1j],  # meas2 4
            [-2 + 1j, -5 + 3j],  # meas2 5
            [-3 + 1j, -5 + 1j],  # meas2 6
            [-4 + 3j, -3 + 2j],  # meas2 7
            [-2 + 2j, -2 + 5j],  # meas2 8
            [-1 + 1j, -5 + 1j],  # meas2 9
            [-6 + 1j, -5 + 3j],  # meas2 10
        ]
    ),
    1: np.array(
        [
            [2 + 2j, 2 + 4j],  # meas1 0
            [3 + 2j, 2 + 2j],  # meas1 1
            [3 + 4j, 2 + 3j],  # meas1 2
            [3 + 3j, 5 + 4j],  # meas1 3
            [1 + 6j, 6 + 1j],  # meas1 4
            [-3 + 2j, -2 + 2j],  # meas2 1
            [-3 + 4j, -2 + 3j],  # meas2 2
            [-3 + 3j, -5 + 4j],  # meas2 3
            [-1 + 6j, -3 + 1j],  # meas2 4
            [-2 + 2j, -5 + 4j],  # meas2 5
            [-4 + 2j, -4 + 2j],  # meas2 6
            [-5 + 4j, -3 + 3j],  # meas2 7
            [-4 + 3j, -1 + 4j],  # meas2 8
            [-3 + 6j, -1 + 1j],  # meas2 9
            [-1 + 2j, -1 + 4j],  # meas2 10
        ]
    ),
    2: np.array(
        [
            [1 + 3j, 4 + 1j],  # meas1 1
            [-1 + 2j, -3 + 3j],  # meas2 0
            [-1 + 3j, -4 + 1j],  # meas2 1
            [-5 + 3j, -2 + 2j],  # meas2 2
            [-4 + 4j, -1 + 6j],  # meas2 3
        ]
    ),
    3: np.array(
        [
            [3 + 1j, 4 + 2j],  # meas1 1
            [-2 + 1j, -3 + 2j],  # meas2 0
            [-3 + 1j, -4 + 2j],  # meas2 1
            [-3 + 2j, -2 + 3j],  # meas2 2
            [-1 + 1j, -6 + 3j],  # meas2 3
        ]
    ),
}


# level 0 windows: 6
# level 1 windows: 3
SITE2_RUN1_DATA = {
    0: np.array(
        [
            [[1 + 1j, 2 + 1j], [3 + 1j, 4 + 1j]],
            [[1 + 2j, 2 + 2j], [3 + 2j, 4 + 2j]],
            [[1 + 3j, 2 + 3j], [3 + 3j, 4 + 3j]],
            [[1 + 4j, 2 + 4j], [3 + 4j, 4 + 4j]],
            [[1 + 5j, 2 + 5j], [3 + 5j, 4 + 5j]],
            [[1 + 6j, 2 + 6j], [3 + 6j, 4 + 6j]],
        ]
    ),
    1: np.array(
        [
            [[0 + 3j, 1 + 3j], [2 + 3j, 3 + 3j]],
            [[0 + 2j, 1 + 2j], [2 + 2j, 3 + 2j]],
            [[0 + 1j, 1 + 1j], [2 + 1j, 3 + 1j]],
        ]
    ),
}


# level 0 windows: 10
# level 1 windows: 5
SITE2_RUN2_DATA = {
    0: np.array(
        [
            [[-1 + 1j, -2 + 1j], [-3 + 1j, -4 + 1j]],
            [[-1 + 2j, -2 + 2j], [-3 + 2j, -4 + 2j]],
            [[-1 + 3j, -2 + 3j], [-3 + 3j, -4 + 3j]],
            [[-1 + 4j, -2 + 4j], [-3 + 4j, -4 + 4j]],
            [[-1 + 5j, -2 + 5j], [-3 + 5j, -4 + 5j]],
            [[-1 + 6j, -2 + 6j], [-3 + 6j, -4 + 6j]],
            [[-1 + 7j, -2 + 7j], [-3 + 7j, -4 + 7j]],
            [[-1 + 8j, -2 + 8j], [-3 + 8j, -4 + 8j]],
            [[-1 + 9j, -2 + 9j], [-3 + 9j, -4 + 9j]],
            [[-1 + 0j, -2 + 0j], [-3 + 0j, -4 + 0j]],
        ]
    ),
    1: np.array(
        [
            [[0 - 4j, 1 - 4j], [2 - 4j, 3 - 4j]],
            [[0 - 3j, 1 - 3j], [2 - 3j, 3 - 3j]],
            [[0 - 2j, 1 - 2j], [2 - 2j, 3 - 2j]],
            [[0 - 1j, 1 - 1j], [2 - 1j, 3 - 1j]],
            [[0 - 0j, 1 - 0j], [2 - 0j, 3 - 0j]],
        ]
    ),
}


SITE2_COMBINED_DATA = {
    0: np.array(
        [
            [1 + 2j, 3 + 2j],  # run1 1
            [1 + 3j, 3 + 3j],  # run1 2
            [1 + 4j, 3 + 4j],  # run1 3
            [1 + 5j, 3 + 5j],  # run1 4
            [1 + 6j, 3 + 6j],  # run1 5
            [-1 + 1j, -3 + 1j],  # run2 0
            [-1 + 2j, -3 + 2j],  # run2 1
            [-1 + 3j, -3 + 3j],  # run2 2
            [-1 + 4j, -3 + 4j],  # run2 3
            [-1 + 5j, -3 + 5j],  # run2 4
            [-1 + 6j, -3 + 6j],  # run2 5
            [-1 + 7j, -3 + 7j],  # run2 6
            [-1 + 8j, -3 + 8j],  # run2 7
            [-1 + 9j, -3 + 9j],  # run2 8
            [-1 + 0j, -3 + 0j],  # run2 9
        ]
    ),
    1: np.array(
        [
            [2 + 2j, 4 + 2j],  # run1 1
            [2 + 3j, 4 + 3j],  # run1 2
            [2 + 4j, 4 + 4j],  # run1 3
            [2 + 5j, 4 + 5j],  # run1 4
            [2 + 6j, 4 + 6j],  # run1 5
            [-2 + 1j, -4 + 1j],  # run2 0
            [-2 + 2j, -4 + 2j],  # run2 1
            [-2 + 3j, -4 + 3j],  # run2 2
            [-2 + 4j, -4 + 4j],  # run2 3
            [-2 + 5j, -4 + 5j],  # run2 4
            [-2 + 6j, -4 + 6j],  # run2 5
            [-2 + 7j, -4 + 7j],  # run2 6
            [-2 + 8j, -4 + 8j],  # run2 7
            [-2 + 9j, -4 + 9j],  # run2 8
            [-2 + 0j, -4 + 0j],  # run2 9
        ]
    ),
    2: np.array(
        [
            [0 + 1j, 2 + 1j],  # run1 2
            [0 - 3j, 2 - 3j],  # run2 1
            [0 - 2j, 2 - 2j],  # run2 2
            [0 - 1j, 2 - 1j],  # run2 3
            [0 - 0j, 2 - 0j],  # run2 4
        ]
    ),
    3: np.array(
        [
            [1 + 1j, 3 + 1j],  # run1 2
            [1 - 3j, 3 - 3j],  # run2 1
            [1 - 2j, 3 - 2j],  # run2 2
            [1 - 1j, 3 - 1j],  # run2 3
            [1 - 0j, 3 - 0j],  # run2 4
        ]
    ),
}


# level 0 windows: 25
# level 1 windows: 17
# level 2 windows: 12
SITE3_DATA1_DATA = {
    0: np.array(
        [
            [[0 + 0j, 1 + 0j], [2 + 0j, 3 + 0j]],
            [[0 + 1j, 1 + 1j], [2 + 1j, 3 + 1j]],
            [[0 + 2j, 1 + 2j], [2 + 2j, 3 + 2j]],
            [[0 + 3j, 1 + 3j], [2 + 3j, 3 + 3j]],
            [[0 + 4j, 1 + 4j], [2 + 4j, 3 + 4j]],
            [[0 + 5j, 1 + 5j], [2 + 5j, 3 + 5j]],
            [[0 + 6j, 1 + 6j], [2 + 6j, 3 + 6j]],
            [[0 + 7j, 1 + 7j], [2 + 7j, 3 + 7j]],
            [[0 + 8j, 1 + 8j], [2 + 8j, 3 + 8j]],
            [[0 + 9j, 1 + 9j], [2 + 9j, 3 + 9j]],
            [[0 + 10j, 1 + 10j], [2 + 10j, 3 + 10j]],
            [[0 + 11j, 1 + 11j], [2 + 11j, 3 + 11j]],
            [[0 + 12j, 1 + 12j], [2 + 12j, 3 + 12j]],
            [[0 + 13j, 1 + 13j], [2 + 13j, 3 + 13j]],
            [[0 + 14j, 1 + 14j], [2 + 14j, 3 + 14j]],
            [[0 + 15j, 1 + 15j], [2 + 15j, 3 + 15j]],
            [[0 + 16j, 1 + 16j], [2 + 16j, 3 + 16j]],
            [[0 + 17j, 1 + 17j], [2 + 17j, 3 + 17j]],
            [[0 + 18j, 1 + 18j], [2 + 18j, 3 + 18j]],
            [[0 + 19j, 1 + 19j], [2 + 19j, 3 + 19j]],
            [[0 + 20j, 1 + 20j], [2 + 20j, 3 + 20j]],
            [[0 + 21j, 1 + 21j], [2 + 21j, 3 + 21j]],
            [[0 + 22j, 1 + 22j], [2 + 22j, 3 + 22j]],
            [[0 + 23j, 1 + 23j], [2 + 23j, 3 + 23j]],
            [[0 + 24j, 1 + 24j], [2 + 24j, 3 + 24j]],
        ]
    ),
    1: np.array(
        [
            [[-0 + 0j, -1 + 0j], [-2 + 0j, -3 + 0j]],
            [[-0 + 1j, -1 + 1j], [-2 + 1j, -3 + 1j]],
            [[-0 + 2j, -1 + 2j], [-2 + 2j, -3 + 2j]],
            [[-0 + 3j, -1 + 3j], [-2 + 3j, -3 + 3j]],
            [[-0 + 4j, -1 + 4j], [-2 + 4j, -3 + 4j]],
            [[-0 + 5j, -1 + 5j], [-2 + 5j, -3 + 5j]],
            [[-0 + 6j, -1 + 6j], [-2 + 6j, -3 + 6j]],
            [[-0 + 7j, -1 + 7j], [-2 + 7j, -3 + 7j]],
            [[-0 + 8j, -1 + 8j], [-2 + 8j, -3 + 8j]],
            [[-0 + 9j, -1 + 9j], [-2 + 9j, -3 + 9j]],
            [[-0 + 10j, -1 + 10j], [-2 + 10j, -3 + 10j]],
            [[-0 + 11j, -1 + 11j], [-2 + 11j, -3 + 11j]],
            [[-0 + 12j, -1 + 12j], [-2 + 12j, -3 + 12j]],
            [[-0 + 13j, -1 + 13j], [-2 + 13j, -3 + 13j]],
            [[-0 + 14j, -1 + 14j], [-2 + 14j, -3 + 14j]],
            [[-0 + 15j, -1 + 15j], [-2 + 15j, -3 + 15j]],
            [[-0 + 16j, -1 + 16j], [-2 + 16j, -3 + 16j]],
        ]
    ),
    2: np.array(
        [
            [[-0 - 0j, -1 - 0j], [-2 - 0j, -3 - 0j]],
            [[-0 - 1j, -1 - 1j], [-2 - 1j, -3 - 1j]],
            [[-0 - 2j, -1 - 2j], [-2 - 2j, -3 - 2j]],
            [[-0 - 3j, -1 - 3j], [-2 - 3j, -3 - 3j]],
            [[-0 - 4j, -1 - 4j], [-2 - 4j, -3 - 4j]],
            [[-0 - 5j, -1 - 5j], [-2 - 5j, -3 - 5j]],
            [[-0 - 6j, -1 - 6j], [-2 - 6j, -3 - 6j]],
            [[-0 - 7j, -1 - 7j], [-2 - 7j, -3 - 7j]],
            [[-0 - 8j, -1 - 8j], [-2 - 8j, -3 - 8j]],
            [[-0 - 9j, -1 - 9j], [-2 - 9j, -3 - 9j]],
            [[-0 - 10j, -1 - 10j], [-2 - 10j, -3 - 10j]],
            [[-0 - 11j, -1 - 11j], [-2 - 11j, -3 - 11j]],
        ]
    ),
}


SITE3_COMBINED_DATA = {
    0: np.array(
        [
            [0 + 0j, 2 + 0j],
            [0 + 1j, 2 + 1j],
            [0 + 2j, 2 + 2j],
            [0 + 3j, 2 + 3j],
            [0 + 4j, 2 + 4j],
            [0 + 12j, 2 + 12j],
            [0 + 13j, 2 + 13j],
            [0 + 14j, 2 + 14j],
            [0 + 15j, 2 + 15j],
            [0 + 16j, 2 + 16j],
            [0 + 17j, 2 + 17j],
            [0 + 18j, 2 + 18j],
            [0 + 19j, 2 + 19j],
            [0 + 20j, 2 + 20j],
            [0 + 21j, 2 + 21j],
        ]
    ),
    1: np.array(
        [
            [1 + 0j, 3 + 0j],
            [1 + 1j, 3 + 1j],
            [1 + 2j, 3 + 2j],
            [1 + 3j, 3 + 3j],
            [1 + 4j, 3 + 4j],
            [1 + 12j, 3 + 12j],
            [1 + 13j, 3 + 13j],
            [1 + 14j, 3 + 14j],
            [1 + 15j, 3 + 15j],
            [1 + 16j, 3 + 16j],
            [1 + 17j, 3 + 17j],
            [1 + 18j, 3 + 18j],
            [1 + 19j, 3 + 19j],
            [1 + 20j, 3 + 20j],
            [1 + 21j, 3 + 21j],
        ]
    ),
    2: np.array(
        [
            [-0 + 0j, -2 + 0j],
            [-0 + 7j, -2 + 7j],
            [-0 + 8j, -2 + 8j],
            [-0 + 9j, -2 + 9j],
            [-0 + 10j, -2 + 10j],
        ]
    ),
    3: np.array(
        [
            [-1 + 0j, -3 + 0j],
            [-1 + 7j, -3 + 7j],
            [-1 + 8j, -3 + 8j],
            [-1 + 9j, -3 + 9j],
            [-1 + 10j, -3 + 10j],
        ]
    ),
}


# testing data for quick gather
# this uses SITE2_RUN2_DATA
SITE2_RUN2_QUICK_OUT = {
    0: np.array(
        [
            [-3 + 1j],
            [-3 + 2j],
            [-3 + 3j],
            [-3 + 4j],
            [-3 + 5j],
            [-3 + 6j],
            [-3 + 7j],
            [-3 + 8j],
            [-3 + 9j],
            [-3 + 0j],
        ]
    ),
    1: np.array(
        [
            [-4 + 1j],
            [-4 + 2j],
            [-4 + 3j],
            [-4 + 4j],
            [-4 + 5j],
            [-4 + 6j],
            [-4 + 7j],
            [-4 + 8j],
            [-4 + 9j],
            [-4 + 0j],
        ]
    ),
    2: np.array(
        [
            [2 - 4j],
            [2 - 3j],
            [2 - 2j],
            [2 - 1j],
            [2 - 0j],
        ]
    ),
    3: np.array(
        [
            [3 - 4j],
            [3 - 3j],
            [3 - 2j],
            [3 - 1j],
            [3 - 0j],
        ]
    ),
}


SITE2_RUN2_QUICK_IN = {
    0: np.array(
        [
            [-1 + 1j],
            [-1 + 2j],
            [-1 + 3j],
            [-1 + 4j],
            [-1 + 5j],
            [-1 + 6j],
            [-1 + 7j],
            [-1 + 8j],
            [-1 + 9j],
            [-1 + 0j],
        ]
    ),
    1: np.array(
        [
            [-2 + 1j],
            [-2 + 2j],
            [-2 + 3j],
            [-2 + 4j],
            [-2 + 5j],
            [-2 + 6j],
            [-2 + 7j],
            [-2 + 8j],
            [-2 + 9j],
            [-2 + 0j],
        ]
    ),
    2: np.array(
        [
            [0 - 4j],
            [0 - 3j],
            [0 - 2j],
            [0 - 1j],
            [0 - 0j],
        ]
    ),
    3: np.array(
        [
            [1 - 4j],
            [1 - 3j],
            [1 - 2j],
            [1 - 1j],
            [1 - 0j],
        ]
    ),
}


SITE2_RUN2_QUICK_CROSS = {
    0: np.array(
        [
            [-1 + 1j, -3 + 1j],
            [-1 + 2j, -3 + 2j],
            [-1 + 3j, -3 + 3j],
            [-1 + 4j, -3 + 4j],
            [-1 + 5j, -3 + 5j],
            [-1 + 6j, -3 + 6j],
            [-1 + 7j, -3 + 7j],
            [-1 + 8j, -3 + 8j],
            [-1 + 9j, -3 + 9j],
            [-1 + 0j, -3 + 0j],
        ]
    ),
    1: np.array(
        [
            [-2 + 1j, -4 + 1j],
            [-2 + 2j, -4 + 2j],
            [-2 + 3j, -4 + 3j],
            [-2 + 4j, -4 + 4j],
            [-2 + 5j, -4 + 5j],
            [-2 + 6j, -4 + 6j],
            [-2 + 7j, -4 + 7j],
            [-2 + 8j, -4 + 8j],
            [-2 + 9j, -4 + 9j],
            [-2 + 0j, -4 + 0j],
        ]
    ),
    2: np.array(
        [
            [0 - 4j, 2 - 4j],
            [0 - 3j, 2 - 3j],
            [0 - 2j, 2 - 2j],
            [0 - 1j, 2 - 1j],
            [0 - 0j, 2 - 0j],
        ]
    ),
    3: np.array(
        [
            [1 - 4j, 3 - 4j],
            [1 - 3j, 3 - 3j],
            [1 - 2j, 3 - 2j],
            [1 - 1j, 3 - 1j],
            [1 - 0j, 3 - 0j],
        ]
    ),
}


def get_evals_metadata_site1(meas_name: str) -> SpectraMetadata:
    """Get evals metadata for site1"""
    if meas_name == "meas1":
        # level 0 windows: 4, 5, 6, 7, 8
        # level 1 windows: 2, 3, 4, 5
        # level 3 windows: 1, 2, 3
        return spectra_metadata_multilevel(
            n_levels=3, n_wins=[5, 4, 3], index_offset=[4, 2, 1], chans=["Ex", "Ey"]
        )
    if meas_name == "meas2":
        # level 0 windows: 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26
        # level 1 windows: 10, 11, 12, 13, 14, 15, 16, 17, 18
        # level 3 windows: 8, 9, 10, 11, 12, 13, 14
        return spectra_metadata_multilevel(
            n_levels=3, n_wins=[12, 9, 7], index_offset=[15, 10, 8], chans=["Ex", "Ey"]
        )
    if meas_name == "meas3":
        # level 0 windows: 41, 42, 43, 44, 45, 46, 47
        # level 1 windows: 38, 39, 40, 41
        # level 3 windows: 35, 36
        return spectra_metadata_multilevel(
            n_levels=3, n_wins=[7, 4, 2], index_offset=[41, 38, 35], chans=["Ex", "Ey"]
        )
    raise ValueError("Unknown measurement for site1")


def get_evals_metadata_site2(meas_name: str) -> SpectraMetadata:
    """Get evals metadata for site2"""
    if meas_name == "run1":
        # level 0 windows: 3, 4, 5, 6, 7, 8
        # level 1 windows: 1, 2, 3
        return spectra_metadata_multilevel(
            n_levels=3, n_wins=[6, 3], index_offset=[3, 1], chans=["Hx", "Hy"]
        )
    if meas_name == "run2":
        # level 0 windows: 16, 17, 18, 19, 20, 21, 22, 23, 24, 25
        # level 1 windows: 9, 10, 11, 12, 13
        return spectra_metadata_multilevel(
            n_levels=2, n_wins=[10, 5], index_offset=[16, 9], chans=["Hx", "Hy"]
        )
    raise ValueError("Unknown measurement for site2")


def get_evals_metadata_site3(meas_name: str) -> SpectraMetadata:
    """Get evals metadata for site3"""
    if meas_name == "data1":
        # level 0 windows: 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28
        # level 1 windows: 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19
        # level 2 windows: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12
        return spectra_metadata_multilevel(
            n_levels=2, n_wins=[25, 17, 12], index_offset=[4, 3, 1], chans=["Hx", "Hy"]
        )
    raise ValueError("Unknown measurement for site3")


def get_evals_data_site1(meas_name: str) -> SpectraData:
    """Get evals data for site 1"""
    metadata = get_evals_metadata_site1(meas_name)
    data = None
    if meas_name == "meas1":
        data = SITE1_MEAS1_DATA
    if meas_name == "meas2":
        data = SITE1_MEAS2_DATA
    if meas_name == "meas3":
        data = SITE1_MEAS3_DATA
    if data is None:
        raise ValueError("Problem getting spectra data")
    return SpectraData(metadata, data)


def get_evals_data_site2(meas_name: str) -> SpectraData:
    """Get evals data for site 2"""
    metadata = get_evals_metadata_site2(meas_name)
    data = None
    if meas_name == "run1":
        data = SITE2_RUN1_DATA
    if meas_name == "run2":
        data = SITE2_RUN2_DATA
    if data is None:
        raise ValueError("Problem getting spectra data")
    return SpectraData(metadata, data)


def get_evals_data_site3(meas_name: str) -> SpectraData:
    """Get evals data for site 3"""
    metadata = get_evals_metadata_site3(meas_name)
    data = None
    if meas_name == "data1":
        data = SITE3_DATA1_DATA
    if data is None:
        raise ValueError("Problem getting spectra data")
    return SpectraData(metadata, data)
