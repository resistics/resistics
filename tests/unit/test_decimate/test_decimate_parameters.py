def test_parameters_default() -> None:
    """Test the default parameters"""
    from resistics.decimate.parameters import DecimationParameters

    decParams = DecimationParameters(128)
    assert decParams.numLevels == 7
    assert decParams.freqPerLevel == 7
    assert decParams.sampleFreq == 128
    # calculate some decimation parameters


def test_parameters_custom() -> None:
    """Test custom decimation parameters"""
    from resistics.decimate.parameters import DecimationParameters

    decParams = DecimationParameters(4096)
    decParams.numLevels = 6
    decParams.freqPerLevel = 4
    