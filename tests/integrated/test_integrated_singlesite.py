def test_default_processing() -> None:
    """Test loading the project"""
    from datapaths import path_integrated_singlesite
    from resistics.project.io import loadProject
    from resistics.project.spectra import calculateSpectra
    from resistics.project.transfunc import processProject
    from resistics.project.transfunc import getTransferFunctionData

    # load project
    proj = loadProject(path_integrated_singlesite)
    # calculateSpectra(proj)
    # processProject(proj)
    # tf = getTransferFunctionData(proj, "M7_4096", 4096)
    # test the transfer function
    

def test_masked_processing() -> None:
    """Test masked processing the project"""
    from datapaths import path_integrated_singlesite
    from resistics.project.io import loadProject
    from resistics.project.spectra import calculateSpectra
    from resistics.project.statistics import calculateStatistics
    from resistics.project.transfunc import processProject
    from resistics.project.transfunc import getTransferFunctionData

    # load project
    proj = loadProject(path_integrated_singlesite)
    # calculateSpectra(proj)
    # processProject(proj)
    # tf = getTransferFunctionData(proj, "M7_4096", 4096)
    # test the transfer function    
    




