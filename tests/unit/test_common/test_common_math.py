"""Test resistics.common.math module"""


def test_eps() -> None:
    """Test epsilon is a small number"""
    from resistics.common.math import eps

    assert eps() < 0.001


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


def test_forwardFFT_norm() -> None:
    from resistics.common.math import forwardFFT
    import numpy as np

    x = np.array([1, 0, 0, 0, 0, 0, 0, 0])
    y = np.array([0, 1, 0, 0, 0, 0, 0, 0])
    xfft = forwardFFT(x)
    yfft = forwardFFT(y)
    # using norm divides the output by sqrt(n) for both fft and ifft
    np.testing.assert_array_almost_equal(
        xfft, [1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j] / np.sqrt(8),
    )
    np.testing.assert_array_almost_equal(
        yfft,
        [1 + 0j, 0.707107 - 0.707107j, 0 - 1j, -0.707107 - 0.707107j, -1 + 0j,]
        / np.sqrt(8),
    )


def test_forwardFFT_nonorm() -> None:
    from resistics.common.math import forwardFFT
    import numpy as np

    x = np.array([1, 0, 0, 0, 0, 0, 0, 0])
    y = np.array([0, 1, 0, 0, 0, 0, 0, 0])
    xfft = forwardFFT(x, norm=False)
    yfft = forwardFFT(y, norm=False)
    # using norm = False means no scaling on the fft, but scaling 1/n on ifft
    np.testing.assert_array_almost_equal(
        xfft, [1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j],
    )
    np.testing.assert_array_almost_equal(
        yfft, [1 + 0j, 0.707107 - 0.707107j, 0 - 1j, -0.707107 - 0.707107j, -1 + 0j,],
    )


def test_inverseFFT_norm() -> None:
    from resistics.common.math import forwardFFT, inverseFFT
    import numpy as np

    x = np.array([1, 0, 0, 0, 0, 0, 0, 0])
    y = np.array([0, 1, 0, 0, 0, 0, 0, 0])
    # using norm divides the output by sqrt(n) for both fft and ifft
    xfft = forwardFFT(x)
    yfft = forwardFFT(y)
    x_test = inverseFFT(xfft, 8)
    y_test = inverseFFT(yfft, 8)
    np.testing.assert_array_almost_equal(x, x_test)
    np.testing.assert_array_almost_equal(y, y_test)


def test_inverseFFT_nonorm() -> None:
    from resistics.common.math import forwardFFT, inverseFFT
    import numpy as np

    x = np.array([1, 0, 0, 0, 0, 0, 0, 0])
    y = np.array([0, 1, 0, 0, 0, 0, 0, 0])
    # using norm = False means no scaling on the fft, but scaling 1/n on ifft
    xfft = forwardFFT(x, norm=False)
    yfft = forwardFFT(y, norm=False)
    x_test = inverseFFT(xfft, 8, norm=False)
    y_test = inverseFFT(yfft, 8, norm=False)
    np.testing.assert_array_almost_equal(x, x_test)
    np.testing.assert_array_almost_equal(y, y_test)


def test_padNextPower2() -> None:
    from resistics.common.math import padNextPower2

    assert padNextPower2(8) == 0
    assert padNextPower2(9) == 7
    assert padNextPower2(100) == 28
