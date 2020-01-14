"""Test resistics.common.math"""

def test_intdiv() -> None:
    from resistics.common.math import intdiv
    import pytest

    assert intdiv(6, 3) == 2
    assert intdiv(5.0, 2.5) == 2
    with pytest.raises(SystemExit): 
        intdiv(6, 4)


def test_getFrequencyArray() -> None:
    from resistics.common.math import getFrequencyArray
    import numpy as np
    fs = 480
    nyquist = 240
    samples = 40
    
    assert np.array_equal(np.linspace(0, nyquist, samples), getFrequencyArray(fs, 40))


# def test_forwardFFT() -> None:
#     from resistics.common.math import forwardFFT

#     if not norm:
#         return fft.irfft(data, n=length)
#     return fft.irfft(data, n=length, norm="ortho")


# def test_inverseFFT() -> None:
#     from resistics.common.math import inverseFFT

#     if not norm:
#         return fft.irfft(data, n=length)
#     return fft.irfft(data, n=length, norm="ortho")


def test_padNextPower2() -> None:
    from resistics.common.math import padNextPower2

    assert padNextPower2(8) == 0
    assert padNextPower2(9) == 7
    assert padNextPower2(100) == 28
