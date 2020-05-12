def test_spectrumdata():
    """Test spectra data"""
    from resistics.spectra.data import SpectrumData
    from resistics.common.format import datetimeFormat
    import numpy as np
    from datetime import datetime

    startTime = "2020-01-01 00:00:00.000000"
    stopTime = "2020-01-01 00:00:00.062500"
    data = {}
    data["Ex"] = np.array([1 + 3j, 2 + 5j, 7 + 6j, 3 + 2j])
    data["Hy"] = np.array([2 + 9j, 9 + 1j, 8 + 8j, 6 + 2j])
    dataArray = np.array(
        [[1 + 3j, 2 + 5j, 7 + 6j, 3 + 2j], [2 + 9j, 9 + 1j, 8 + 8j, 6 + 2j]]
    )
    specData = SpectrumData(8, 4, 128, startTime, stopTime, data)
    assert specData.windowSize == 8
    assert specData.dataSize == 4
    assert specData.startTime == datetime.strptime(startTime, datetimeFormat(ns=True))
    assert specData.stopTime == datetime.strptime(stopTime, datetimeFormat(ns=True))
    assert specData.sampleFreq == 128
    assert specData.numChans == 2
    assert specData.chans == ["Ex", "Hy"]
    assert specData.getComments() == []
    specData.addComment("This is a comment")
    assert specData.getComments() == ["This is a comment"]
    np.testing.assert_equal(specData.data["Ex"], data["Ex"])
    np.testing.assert_equal(specData.data["Hy"], data["Hy"])
    # check __getitem__ accessor
    np.testing.assert_equal(specData["Ex"], data["Ex"])
    np.testing.assert_equal(specData["Hy"], data["Hy"])
    # check toArray
    specDataArray, specDataChans = specData.toArray()
    np.testing.assert_almost_equal(specDataArray, dataArray)
    assert specDataChans == ["Ex", "Hy"]
    specDataArray, specDataChans = specData.toArray(["Hy"])
    np.testing.assert_almost_equal(specDataArray.squeeze(), np.array(data["Hy"]))
    assert specDataChans == ["Hy"]
    # check nyquist and frequency array
    assert specData.nyquist == 64
    np.testing.assert_almost_equal(specData.freqArray, [0, (64 / 3), (64 / 3) * 2, 64])


def test_powerdata():
    "Test PowerData"
    from resistics.spectra.data import PowerData
    import numpy as np

    data = np.array(
        [[[0 + 4j, 1 + 3j, 6 + 6j, 7 + 3j], [0 + 1j, 5 + 2j, 5 + 4j, 2 + 3j]]]
    )
    print(data.shape)
    pData = PowerData(["Ex"], ["Hx", "Hy"], data, 4096)
    # make sure that it matches autopower
    assert pData.dataSize == 4
    assert pData.sampleFreq == 4096
    assert pData.numPowers == 2
    assert pData.powers == [["Ex", "Hx"], ["Ex", "Hy"]]
    assert np.array_equal(pData.getPower("Ex", "Hx"), data[0, 0])
    assert np.array_equal(pData.getPower("Ex", "Hy", fIdx=2), data[0, 1, 2])
    # test getitem
    assert np.array_equal(pData["Ex", "Hx"], data[0, 0])
    assert np.array_equal(pData["Ex", "Hy", 2], data[0, 1, 2])    
    assert pData.nyquist == 2048
    assert np.array_equal(pData.freqArray, [0, 2048 / 3, 2048 * 2 / 3, 2048])


def test_powerdata_smooth():
    """Test PowerData smooth"""
    from resistics.spectra.data import PowerData
    from resistics.common.smooth import smooth1d
    import numpy as np

    data = np.array(
        [
            [
                [
                    0 + 4j,
                    -1 + 3j,
                    6 + 6j,
                    7 + 3j,
                    -6 - 1j,
                    5 + 5j,
                    6 + 2j,
                    1 - 3j,
                    -7 + 8j,
                    1 - 9j,
                    3 + 4j,
                ],
                [
                    0 + 1j,
                    5 + 2j,
                    5 - 4j,
                    2 + 3j,
                    4 + 3j,
                    1 - 6j,
                    -2 + 4j,
                    -2 + 7j,
                    4 + 4j,
                    3 - 2j,
                    6 + 3j,
                ],
            ]
        ]
    )
    pData = PowerData(["Ex"], ["Hx", "Hy"], data, 128)
    # make sure that it matches autopower
    assert pData.dataSize == 11
    assert pData.sampleFreq == 128
    assert pData.numPowers == 2
    assert pData.powers == [["Ex", "Hx"], ["Ex", "Hy"]]
    assert np.array_equal(pData.getPower("Ex", "Hx"), data[0, 0])
    assert np.array_equal(pData.getPower("Ex", "Hy"), data[0, 1])
    assert pData.nyquist == 64
    assert np.array_equal(pData.freqArray, np.linspace(0, 64, 11))
    smooth = pData.smooth(3, "boxcar")
    for chan1 in smooth.primaryChans:
        for chan2 in smooth.secondaryChans:
            print("{}-{}".format(chan1, chan2), smooth.getPower(chan1, chan2))
    assert False


def test_powerdata_interpolate():
    """Test PowerData interpolate"""
    from resistics.spectra.data import PowerData
    import numpy as np

    data = np.array(
        [
            [
                [
                    0 + 4j,
                    -1 + 3j,
                    6 + 6j,
                    7 + 3j,
                    -6 - 1j,
                    5 + 5j,
                    6 + 2j,
                    1 - 3j,
                    -7 + 8j,
                    1 - 9j,
                    3 + 4j,
                ],
                [
                    0 + 1j,
                    5 + 2j,
                    5 - 4j,
                    2 + 3j,
                    4 + 3j,
                    1 - 6j,
                    -2 + 4j,
                    -2 + 7j,
                    4 + 4j,
                    3 - 2j,
                    6 + 3j,
                ],
            ]
        ]
    )
    pData = PowerData(["Ex"], ["Hx", "Hy"], data, 128)
    # make sure that it matches autopower
    assert pData.dataSize == 11
    assert pData.sampleFreq == 128
    assert pData.numPowers == 2
    assert pData.powers == [["Ex", "Hx"], ["Ex", "Hy"]]
    assert np.array_equal(pData.getPower("Ex", "Hx"), data[0, 0])
    assert np.array_equal(pData.getPower("Ex", "Hy"), data[0, 1])
    assert pData.nyquist == 64
    assert np.array_equal(pData.freqArray, np.linspace(0, 64, 11))
    interpolated = pData.interpolate([5, 10, 15, 20])
    for chan1 in interpolated.primaryChans:
        for chan2 in interpolated.secondaryChans:
            print("{}-{}".format(chan1, chan2), interpolated.getPower(chan1, chan2))
    assert False
