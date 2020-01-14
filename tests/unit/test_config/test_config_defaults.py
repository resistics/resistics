def test_getDefaultParameter() -> None:
    """Test loading of default parameters"""
    from resistics.config.defaults import getDefaultParameter

    name = getDefaultParameter("name")
    assert name == "default"
    ncores = getDefaultParameter("ncores")
    assert ncores == -1
    spectra = getDefaultParameter("Spectra")
    assert spectra == {'specdir': 'spectra', 'applywindow': True, 'windowfunc': 'hamming', 'ncores': -1}


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

