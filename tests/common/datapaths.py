from pathlib import Path

datapath = Path("E:/", "magnetotellurics", "code", "resisticstest")

# paths for testing config subpackage
test_config = datapath / "config"
path_config_copy = test_config / "configCopy.ini"

# paths for testing project subpackage
test_project = datapath / "project"
path_project_new = test_project / "testNew"

# paths for testing time readers
test_time = datapath / "time"
path_time_ats = test_time / "ats"

# paths for testing integrated
test_integrated = datapath / "integrated"
path_integrated_singlesite = test_integrated / "singlesite"