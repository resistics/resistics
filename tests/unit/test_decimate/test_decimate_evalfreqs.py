def test_getEvaluationFreqs() -> None:
    """Test getting the evaluation frequencies with a given minimum frequency"""
    from resistics.decimate.evalfreqs import getEvaluationFreq

    evalFreqs = getEvaluationFreq(128, 0.01)
    assert evalFreqs[0] == 32
    assert evalFreqs[-1] > 0.01

    evalFreqs = getEvaluationFreq(64000, 1000)
    assert evalFreqs[0] == 16000
    assert evalFreqs[-1] > 1000


def test_getEvaluationFreqSize() -> None:
    """Test getting the evaluation frequencies with a given size"""
    from resistics.decimate.evalfreqs import getEvaluationFreqSize

    evalFreqs = getEvaluationFreqSize(128, 21)
    assert evalFreqs.size == 21
    assert evalFreqs[0] == 32

    evalFreqs = getEvaluationFreqSize(64000, 3)
    assert evalFreqs.size == 3
    assert evalFreqs[0] == 16000