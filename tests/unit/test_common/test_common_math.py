"""Test resistics.common.math module"""
import pytest
from typing import List


def test_eps() -> None:
    """Test epsilon is a small number"""
    from resistics.common.math import eps

    assert eps() == 0.0001


def test_intdiv() -> None:
    """Test integer division that raise an error if not exact"""
    from resistics.common.math import intdiv

    assert intdiv(6, 3) == 2
    assert intdiv(5.0, 2.5) == 2
    with pytest.raises(Exception):
        intdiv(6, 4)


@pytest.mark.parametrize("fs, nsamples", [(480, 40)])
def test_getFrequencyArray(fs: float, nsamples: int) -> None:
    """Test getting of frequency array"""
    from resistics.common.math import frequency_array
    import numpy as np

    nyquist = fs / 2
    assert np.array_equal(np.linspace(0, nyquist, nsamples), frequency_array(fs, 40))


@pytest.mark.parametrize("nsamples, expected", [(8, 0), (9, 7), (100, 28)])
def test_pad_to_power2(nsamples: int, expected: int) -> None:
    """Testing padding to next power of 2"""
    from resistics.common.math import pad_to_power2

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
    from resistics.common.math import fft
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
    from resistics.common.math import fft, ifft
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
