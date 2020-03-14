from pathlib import Path

datapath = Path("E:/", "magnetotellurics", "code", "resisticstest")

# paths for testing config subpackage
test_config = datapath / "config"
path_config_copy = test_config / "configCopy.ini"
path_config_section_cores = test_config / "configSectionCores.ini"
path_config_window_explicit = test_config / "configWindowExplicit.ini"
path_config_window_factor = test_config / "configWindowFactor.ini"
path_config_window_min = test_config / "configWindowMin.ini"
# path_config_decimate = test_config / "configDecimate.ini"
# path_config_frequencies = test_config / "configFrequencies.ini"

# paths for testing project subpackage
test_project = datapath / "project"
path_project_new = test_project / "testNew"

# paths for testing time readers
test_time = datapath / "time"
path_time_ats = test_time / "ats"

# paths for testing integrated
test_integrated = datapath / "integrated"
path_integrated_singlesite = test_integrated / "singlesite"
path_integrated_singlesite_config = test_integrated / "singlesiteconfig.ini"