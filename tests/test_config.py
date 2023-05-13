from resistics.config import Configuration
from resistics.time import ShiftTimestamps, TimeReaderAscii, TimeReaderNumpy
from resistics.time import InterpolateNans, RemoveMean, Add, Multiply, LowPass, HighPass
from resistics.time import BandPass, Notch, Resample, Decimate
from resistics.decimate import DecimationSetup, Decimator
from resistics.window import WindowSetup, Windower
from resistics.spectra import FourierTransform, EvaluationFreqs, SpectraSmootherGaussian
from resistics.calibrate import SensorCalibrator
from resistics.calibrate import SensorCalibrationJSON, SensorCalibrationTXT
from resistics.transfunc import TransferFunction
from resistics.regression import RegressionPreparerGathered, SolverOLS


def test_config_json():
    """Test converting configuration to JSON and then loading back in"""
    import json

    time_readers = [TimeReaderNumpy(), TimeReaderAscii()]
    time_processors = [
        InterpolateNans(),
        RemoveMean(),
        Add(add={"Ex": 5, "Hy": -3}),
        Multiply(multiplier={"Ey": 2, "Hx": 6}),
        LowPass(cutoff=4),
        HighPass(cutoff=60),
        BandPass(cutoff_low=15, cutoff_high=30),
        Notch(notch=16.6),
        Resample(new_fs=16),
        Decimate(factor=16, max_single_factor=4),
        ShiftTimestamps(shift=0.05),
    ]
    dec_setup = DecimationSetup(n_levels=15, per_level=12)
    decimator = Decimator(resample=False, max_single_factor=12)
    win_setup = WindowSetup(min_size=100, win_factor=8, olap_proportion=0.5)
    windower = Windower()
    fft = FourierTransform(win_fnc="hanning", detrend=None)
    spectra_processors = [SpectraSmootherGaussian(sigma=5)]
    evals = EvaluationFreqs()
    calibrator = SensorCalibrator(
        readers=[SensorCalibrationJSON(), SensorCalibrationTXT()]
    )
    tf = TransferFunction(
        out_chans=["a", "b", "c"], in_chans=["x", "y", "z"], cross_chans=["a", "x", "z"]
    )
    regression_preparer = RegressionPreparerGathered()
    solver = SolverOLS(fit_intercept=False)

    config = Configuration(
        name="testing",
        time_readers=time_readers,
        time_processors=time_processors,
        dec_setup=dec_setup,
        decimator=decimator,
        win_setup=win_setup,
        windower=windower,
        fourier=fft,
        spectra_processors=spectra_processors,
        evals=evals,
        sensor_calibrator=calibrator,
        tf=tf,
        regression_preparer=regression_preparer,
        solver=solver,
    )

    # test
    json_data = config.json()
    json_loaded = json.loads(json_data)
    config_test = Configuration(**json_loaded)
    assert config_test.name == "testing"
    assert config_test.time_readers == time_readers
    assert config_test.time_processors == time_processors
    assert config_test.dec_setup == dec_setup
    assert config_test.decimator == decimator
    assert config_test.win_setup == win_setup
    assert config_test.windower == windower
    assert config_test.fourier == fft
    assert config_test.spectra_processors == spectra_processors
    assert config_test.evals == evals
    assert config_test.sensor_calibrator == calibrator
    assert config_test.tf == tf
    assert config_test.regression_preparer == regression_preparer
    assert config_test.solver == solver
    assert config_test == config
