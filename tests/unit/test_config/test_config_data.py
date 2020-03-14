from resistics.config.data import ConfigData


def test_config_data() -> None:
    """Test ConfigData initialiser"""
    from resistics.config.io import loadConfig

    # load default config
    config = ConfigData()
    assert config.configFile == ""
    configParams = config.configParams
    flags = config.flags
    defaultConfig = loadConfig()
    for key in defaultConfig.keys():
        assert configParams[key] == defaultConfig[key]
    assert flags["customfrequencies"] is False
    assert flags["customwindows"] is False
    assert config.getSpectraCores() == 0
    assert config.getStatisticCores() == 0
    assert config.getSolverCores() == 0
    assert config.getConfigComment() == "Using default configuration"


def test_getSectionCores() -> None:
    """Test calculating the spectra cores"""
    from datapaths import path_config_section_cores

    config = ConfigData(str(path_config_section_cores))
    assert config.getSpectraCores() == 4
    assert config.getStatisticCores() == 7
    assert config.getSolverCores() == 8
    commentStr = "Using configuration with name {} in configuration file {}".format("sectioncores", str(path_config_section_cores))
    assert config.getConfigComment() == commentStr
