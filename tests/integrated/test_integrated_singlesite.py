def test_default_processing() -> None:
    """Test loading the project"""
    from datapaths import path_integrated_singlesite
    from resistics.project.io import loadProject
    from resistics.project.spectra import calculateSpectra
    from resistics.project.transfunc import processProject
    from resistics.project.transfunc import getTransferFunctionData

    # load project
    # proj = loadProject(path_integrated_singlesite)
    # calculateSpectra(proj)
    # processProject(proj)
    # tf = getTransferFunctionData(proj, "M7_4096", 4096)
    # test the transfer function

    return


def test_window_selector() -> None:
    """Test selecting masked windows"""
    from datapaths import path_integrated_singlesite, path_integrated_singlesite_config
    from resistics.project.io import loadProject
    from resistics.project.shortcuts import (
        getWindowSelector,
        getDecimationParameters,
        getWindowParameters,
    )
    import numpy as np

    proj = loadProject(
        path_integrated_singlesite, str(path_integrated_singlesite_config)
    )
    sites = ["M7_4096"]
    # get decimation, window parameters and selector
    decParams = getDecimationParameters(4096, proj.config)
    winParams = getWindowParameters(decParams, proj.config)
    selector = getWindowSelector(
        proj, decParams, winParams, proj.config.configParams["Spectra"]["specdir"]
    )
    # now add a mask
    selector.setSites(sites)
    selector.addWindowMask(sites[0], "coh_tf")
    selector.calcSharedWindows()

    declevel = 0
    unmaskedWindows = selector.getUnmaskedWindowsLevel(declevel)
    spectraBatches = selector.getSpecReaderBatches(declevel)
    for batch in spectraBatches:
        batch = spectraBatches[0]
        reader = batch[sites[0]]
        batchedWindows = unmaskedWindows.intersection(
            set(range(batch["globalrange"][0], batch["globalrange"][1] + 1))
        )
        # read the batch
        batchData, batchGlobalIndices = reader.readBinaryBatchGlobal(batchedWindows)

        # for each window, check to make sure all correct
        for testGlobalIndex in list(batchedWindows):
            winData1 = reader.readBinaryWindowGlobal(testGlobalIndex)
            # matching readBatch index
            matchingBatchIndex = list(batchGlobalIndices).index(testGlobalIndex)
            winData2 = batchData[matchingBatchIndex]
            # test winData1 and winData2
            chans = winData1.chans
            for chan in chans:
                assert np.array_equal(winData1.data[chan], winData2.data[chan])


def test_local_regression() -> None:
    """Test selecting masked windows"""
    from datapaths import path_integrated_singlesite, path_integrated_singlesite_config
    from resistics.project.io import loadProject
    from resistics.project.transfunc import getTransferFunctionData, viewImpedance
    from resistics.project.shortcuts import (
        getWindowSelector,
        getDecimationParameters,
        getWindowParameters,
    )
    from localtest2 import LocalRegressor
    # from localtest import LocalRegressor
    import numpy as np

    proj = loadProject(
        path_integrated_singlesite, str(path_integrated_singlesite_config)
    )
    sites = ["M7_4096"]
    decParams = getDecimationParameters(4096, proj.config)
    winParams = getWindowParameters(decParams, proj.config)
    selector = getWindowSelector(
        proj, decParams, winParams, proj.config.configParams["Spectra"]["specdir"]
    )
    # now add a mask
    selector.setSites(sites + sites)
    selector.addWindowMask(sites[0], "coh_tf")
    selector.calcSharedWindows()
    # add the input and output site
    processor = LocalRegressor(selector, "")
    processor.setInput(sites[0], ["Hx", "Hy"])
    processor.setOutput(sites[0], ["Ex", "Ey"])
    processor.postpend = "stacked"
    processor.printInfo()
    processor.process(4)


def test_transfunc_read_internal() -> None:
    from resistics.transfunc.io import TransferFunctionReader
    from pathlib import Path

    for iWin in range(0, 1000, 10):
        filepath = Path("4096_000", "M7_4096_fs4096_000_dec8_5_stacked".format(iWin))
        reader = TransferFunctionReader(str(filepath))
        tfData = reader.tfData
        tfData.viewImpedance(polarisations=["ExHy", "EyHx"])


def test_masked_processing() -> None:
    """Test masked processing the project"""
    from datapaths import path_integrated_singlesite, path_integrated_singlesite_config
    from resistics.project.io import loadProject
    from resistics.project.spectra import calculateSpectra
    from resistics.project.statistics import calculateStatistics
    from resistics.project.mask import newMaskData, calculateMask
    from resistics.project.transfunc import processProject
    from resistics.project.transfunc import getTransferFunctionData, viewImpedance

    # load project
    sites = ["M7_4096"]
    proj = loadProject(
        path_integrated_singlesite, str(path_integrated_singlesite_config)
    )
    # calculateSpectra(proj)
    # proj.refresh()
    # calculateStatistics(proj)
    # calculate out a mask
    maskData = newMaskData(proj, 4096)
    maskData.setStats(["coherence", "transferFunction", "resPhase"])
    maskData.addConstraintLevel(
        "coherence", {"cohExHy": [0.8, 1.0], "cohEyHx": [0.8, 1.0]}, 0
    )
    maskData.addConstraintLevel(
        "coherence", {"cohExHy": [0.8, 1.0], "cohEyHx": [0.8, 1.0]}, 1
    )
    maskData.addConstraintLevel(
        "coherence", {"cohExHy": [0.8, 1.0], "cohEyHx": [0.7, 1.0]}, 2
    )
    maskData.addConstraintLevel(
        "coherence", {"cohExHy": [0.7, 1.0], "cohEyHx": [0.7, 1.0]}, 3
    )
    maskData.addConstraintLevel(
        "coherence", {"cohExHy": [0.7, 1.0], "cohEyHx": [0.7, 1.0]}, 4
    )
    maskData.addConstraintFreq(
        "transferFunction", {"EyHxReal": [0, 150], "EyHxImag": [0, 220]}, 0, 4
    )
    maskData.addConstraintFreq(
        "transferFunction", {"ExHyReal": [-120, 0], "ExHyImag": [-200, -30]}, 1, 0
    )
    maskData.addConstraintFreq(
        "transferFunction", {"ExHyReal": [-100, 0], "ExHyImag": [-160, -30]}, 1, 1
    )
    maskData.addConstraintFreq(
        "transferFunction", {"ExHyReal": [-90, 0], "ExHyImag": [-120, -30]}, 1, 2
    )
    maskData.addConstraintFreq(
        "transferFunction", {"ExHyReal": [-60, -25], "ExHyImag": [-90, -75]}, 1, 3
    )
    maskData.addConstraintFreq(
        "transferFunction", {"ExHyReal": [-80, 0], "ExHyImag": [-100, -20]}, 1, 4
    )
    # finally, lets give maskData a name, which will relate to the output file
    maskData.maskName = "coh_tf"
    calculateMask(proj, maskData, sites=sites)
    # process
    processProject(
        proj, sites=sites, masks={"M7_4096": "coh_tf"}, postpend="coh_tf",
    )
    # tf = getTransferFunctionData(proj, "M7_4096", 4096)
    # test the transfer function
    viewImpedance(
        proj,
        sites=sites,
        postpend="coh_tf",
        polarisations=["ExHy", "EyHx"],
        oneplot=False,
        show=False,
        save=True,
    )


def test_statistic_transfunc() -> None:
    """Get the transfer functions statistic"""
    from datapaths import path_integrated_singlesite, path_integrated_singlesite_config
    from resistics.project.io import loadProject
    from resistics.project.statistics import getStatisticData

    # load project
    sites = ["M7_4096"]
    proj = loadProject(
        path_integrated_singlesite, str(path_integrated_singlesite_config)
    )
    statData = getStatisticData(proj, sites[0], "meas_2016-02-25_02-00-00", "transferFunction", 1)
    data = statData.getStatLocal(0)
    print(statData.winStats)
    print(data)

    # now do the same window through the
    from resistics.project.transfunc import getTransferFunctionData
    from resistics.project.shortcuts import (
        getWindowSelector,
        getDecimationParameters,
        getWindowParameters,
    )
    from localtest3 import LocalRegressor
    import numpy as np

    proj = loadProject(
        path_integrated_singlesite, str(path_integrated_singlesite_config)
    )
    sites = ["M7_4096"]
    decParams = getDecimationParameters(4096, proj.config)
    winParams = getWindowParameters(decParams, proj.config)
    selector = getWindowSelector(
        proj, decParams, winParams, proj.config.configParams["Spectra"]["specdir"]
    )
    # now add a mask
    selector.setSites(sites + sites)
    selector.calcSharedWindows()
    # add the input and output site
    processor = LocalRegressor(selector, "")
    processor.setInput(sites[0], ["Hx", "Hy"])
    processor.setOutput(sites[0], ["Ex", "Ey"])
    processor.postpend = "check"
    processor.process(4)