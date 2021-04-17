"""Test resistics.math module"""
from typing import List
import pytest


def test_intdiv() -> None:
    """Test integer division that raise an error if not exact"""
    from resistics.math import intdiv

    assert intdiv(6, 3) == 2
    assert intdiv(5.0, 2.5) == 2
    with pytest.raises(Exception):
        intdiv(6, 4)


@pytest.mark.parametrize("nsamples, expected", [(8, 0), (9, 7), (100, 28)])
def test_pad_to_power2(nsamples: int, expected: int) -> None:
    """Testing padding to next power of 2"""
    from resistics.math import pad_to_power2

    assert pad_to_power2(nsamples) == expected


@pytest.mark.parametrize(
    "data, expected",
    [
        (
            [1, 0, 0, 0, 0, 0, 0, 0],
            [1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j],
        ),
        (
            [0, 1, 0, 0, 0, 0, 0, 0],
            [1 + 0j, 0.707107 - 0.707107j, 0 - 1j, -0.707107 - 0.707107j, -1 + 0j],
        ),
    ],
)
def test_fft(data: List, expected: List) -> None:
    from resistics.math import fft
    import numpy as np

    data = np.array(data)
    expected = np.array(expected)
    # default norm
    data_fft = fft(data)
    np.testing.assert_array_almost_equal(data_fft, expected / np.sqrt(data.size))
    # norm
    data_fft = fft(data, norm=True)
    np.testing.assert_array_almost_equal(data_fft, expected / np.sqrt(data.size))
    # no norm
    data_fft = fft(data, norm=False)
    np.testing.assert_array_almost_equal(data_fft, expected)


@pytest.mark.parametrize(
    "data",
    [
        ([1, 0, 0, 0, 0, 0, 0, 0]),
        ([0, 1, 0, 0, 0, 0, 0, 0]),
    ],
)
def test_ifft(data: List) -> None:
    from resistics.math import fft, ifft
    import numpy as np

    data = np.array(data)
    # norm default
    data_fft = fft(data)
    data_inv = ifft(data_fft, data.size)
    np.testing.assert_array_almost_equal(data, data_inv)
    # norm
    data_fft = fft(data, norm=True)
    data_inv = ifft(data_fft, data.size, norm=True)
    np.testing.assert_array_almost_equal(data, data_inv)
    # no norm
    data_fft = fft(data, norm=False)
    data_inv = ifft(data_fft, data.size, norm=False)
    np.testing.assert_array_almost_equal(data, data_inv)


@pytest.mark.parametrize("fs, nsamples", [(480, 40)])
def test_frequency_array(fs: float, nsamples: int) -> None:
    """Test getting of frequency array"""
    from resistics.math import frequency_array
    import numpy as np

    nyquist = fs / 2
    assert np.array_equal(np.linspace(0, nyquist, nsamples), frequency_array(fs, 40))


@pytest.mark.parametrize(
    "nsamples, proportion, expected",
    [(64, 16, 5), (128, 16, 9), (1598, 16, 99), (1598, 21, 77)],
)
def test_smooth_length(nsamples: int, proportion: float, expected: int) -> None:
    """Test calculating the smoothing length"""
    from resistics.math import smooth_length

    assert smooth_length(nsamples, proportion) == expected


def test_smoother() -> None:
    """Test single dimension smoothing"""
    from resistics.math import Smoother
    import numpy as np

    with pytest.raises(ValueError):
        sm = Smoother(11)
        data = np.array([10])
        sm.smooth(data)

    with pytest.raises(ValueError):
        sm = Smoother(7)
        data = np.array([10, 5, 7])
        sm.smooth(data)

    # window length < 3
    sm = Smoother(1)
    data = np.array([10, 5, 7, 9, 7, 8])
    assert np.array_equal(data, sm.smooth(data))

    # do a smooth
    sm = Smoother(5, window="boxcar")
    data = np.array([10, 5, 7, 9, 7, 8, 6, 7, 4, 2, 4, 6])
    expected = np.array([8.4, 8.2, 7.6, 7.2, 7.4, 7.4, 6.4, 5.4, 4.6, 4.6, 4.4, 4.8])
    np.testing.assert_array_almost_equal(sm.smooth(data), expected)
