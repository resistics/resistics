def test_io() -> None:
    """Read some data an"""
    from resistics.config.io import getDefaultConfigFilepath
    import resistics as rs
    from pathlib import Path

    filepath = getDefaultConfigFilepath()
    configPath = Path(rs.__file__).parent / "resisticsConfig.ini"
    assert filepath == str(configPath)


def test_loadConfig() -> None:
    """Test loading config"""
    from resistics.config.io import loadConfig

    config = loadConfig()
    assert config["name"] == "default"
    assert config["ncores"] == -1
    
    assert config["Calibration"]["extend"] == True 
    assert config["Calibration"]["usetheoretical"] == False
    
    assert config["Decimation"]["numlevels"] == 7 
    assert config["Decimation"]["minsamples"] == 100 
    
    assert config["Frequencies"]["frequencies"] == []
    assert config["Frequencies"]["perlevel"] == 7
    
    assert config["Window"]["minwindows"] == 5
    assert config["Window"]["windowfactor"] == 2.0 
    assert config["Window"]["minwindowsize"] == 512 
    assert config["Window"]["minoverlapsize"] == 128 
    assert config["Window"]["overlapfraction"] == 0.25 
    assert config["Window"]["windowsizes"] == [] 
    assert config["Window"]["overlapsizes"] == []
    
    assert config["Spectra"]["ncores"] == -1 
    assert config["Spectra"]["specdir"] == "spectra" 
    assert config["Spectra"]["applywindow"] == True 
    assert config["Spectra"]["windowfunc"] == "hann" 
    
    assert config["Statistics"]["ncores"] == -1
    assert config["Statistics"]["stats"] == ["coherence", "transferFunction"] 
    assert config["Statistics"]["remotestats"] == ["RR_coherence", "RR_transferFunction"] 
    
    assert config["Solver"]["ncores"] == -1
    assert config["Solver"]["method"] == "cm"
    assert config["Solver"]["intercept"] == False
    assert config["Solver"]["smoothfunc"] == "hann" 
    assert config["Solver"]["smoothlen"] == 9
    assert config["Solver"]["OLS"] == {}
    assert config["Solver"]["MM"]["weightfnc1"] == "huber"
    assert config["Solver"]["MM"]["weightfnc2"] == "bisquare"
    assert config["Solver"]["CM"] == {}