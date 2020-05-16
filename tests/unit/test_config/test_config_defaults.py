def test_getDefaultParameter() -> None:
    """Test loading of default parameters"""
    from resistics.config.defaults import getDefaultParameter

    name = getDefaultParameter("name")
    assert name == "default"
    ncores = getDefaultParameter("ncores")
    assert ncores == -1
    window = getDefaultParameter("Window")
    assert window == {
        "minwindows": 5,
        "windowfactor": 2.0,
        "minwindowsize": 512,
        "minoverlapsize": 128,
        "overlapfraction": 0.25,
        "windowsizes": [],
        "overlapsizes": [],
    }
    spectra = getDefaultParameter("Spectra")
    assert spectra == {
        "specdir": "spectra",
        "applywindow": True,
        "windowfunc": "hann",
        "ncores": -1,
    }
    statistics = getDefaultParameter("Statistics")
    assert statistics == {
        "ncores": -1,
        "stats": ["coherence", "transferFunction"],
        "remotestats": ["RR_coherence", "RR_transferFunction"],
    }
    solver = getDefaultParameter("Solver")
    assert solver == {
        "ncores": -1,
        "smoothfunc": "hann",
        "smoothlen": 9,
        "intercept": False,
        "method": "cm",
        "OLS": {},
        "MM": {"weightfnc1": "huber", "weightfnc2": "bisquare"},
        "CM": {},
    }


def test_copyDefaultConfig() -> None:
    """Test copying of default config"""
    from datapaths import path_config_copy
    from resistics.config.defaults import copyDefaultConfig
    from resistics.config.io import loadConfig

    copyDefaultConfig(path_config_copy)
    # load in this file
    config = loadConfig(str(path_config_copy))
    assert config["name"] == "global copy"
    defaultConfig = loadConfig()
    for key in config.keys():
        if key == "name":
            continue
        assert config[key] == defaultConfig[key]
