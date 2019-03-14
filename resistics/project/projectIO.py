import os
from datetime import datetime
import glob
from typing import Union

# import from package
from resistics.dataObjects.projectData import ProjectData
from resistics.dataObjects.configData import ConfigData
from resistics.utilities.utilsConfig import loadConfig
from resistics.utilities.utilsIO import checkAndMakeDir
from resistics.utilities.utilsProject import (
    projectText,
    projectBlock,
    projectWarning,
    projectError,
)


def newProject(
    projectPath: str,
    refTime: Union[str, datetime],
    configFile: str = "",
    name: str = "mtProj",
) -> ProjectData:
    """Create a new project in project path

    A new project will be created in project path. If the project path directory does not exist, a new one will be made. If a project already exists in project path, this project will be loaded and returned.

    Parameters
    ----------
    projectPath : str
        Path for the project directory
    refTime : datetime
        The reference time for the project
    configFile : str
        Path to a configuration file
    name : str, optional (default is "mtProj")
        The name of the project file

    Returns
    -------
    ProjectData
        A project data object
    """

    # check if a project file already exists and if so, load it

    # create project path directory
    checkAndMakeDir(projectPath)
    # reference time
    if isinstance(refTime, str):
        refTime = datetime.strptime(refTime, "%Y-%m-%d %H:%M:%S")
    # print info
    textLst = [
        "Creating a new project in path: {}".format(projectPath),
        "Project name: {}".format(name),
    ]
    projectBlock(textLst)

    # create the subdirectories
    projectFile = os.path.join(projectPath, "{}.prj".format(name))
    timePath = os.path.join(projectPath, "timeData")
    specPath = os.path.join(projectPath, "specData")
    statPath = os.path.join(projectPath, "statData")
    maskPath = os.path.join(projectPath, "maskData")
    transFuncPath = os.path.join(projectPath, "transFuncData")
    calPath = os.path.join(projectPath, "calData")
    imagePath = os.path.join(projectPath, "images")
    saveProjectFile(
        projectFile,
        refTime,
        calPath,
        timePath,
        specPath,
        statPath,
        maskPath,
        transFuncPath,
        imagePath,
    )

    # configuration file
    config = ConfigData(configFile)

    proj = ProjectData(
        projectFile,
        refTime,
        calPath,
        timePath,
        specPath,
        statPath,
        maskPath,
        transFuncPath,
        imagePath,
        config=config,
    )
    proj.printInfo()
    proj.config.printInfo()

    return proj


def loadProject(projectPath: str, configFile: str = "") -> ProjectData:
    """Load an existing project

    Parameters
    ----------
    projectPath : str
        Path for the project directory
    configFile : str
        Path to a configuration file

    Returns
    -------
    ProjectData
        A project data object
    """

    # search for the .prj file (hopefully only one)
    gl = glob.glob(os.path.join(projectPath, "*.prj"))
    if len(gl) == 0:
        projectError("Unable to find project file in path: {}".format(projectPath))
    projectFile: str = os.path.basename(gl[0])
    projectText(
        "Loading project file: {}".format(os.path.join(projectPath, projectFile))
    )
    projectPaths = loadProjectFile(os.path.join(projectPath, projectFile))

    # check the configuration file
    config = ConfigData(configFile)

    proj = ProjectData(
        projectFile,
        projectPaths["refTime"],
        projectPaths["calPath"],
        projectPaths["timePath"],
        projectPaths["specPath"],
        projectPaths["statPath"],
        projectPaths["maskPath"],
        projectPaths["transFuncPath"],
        projectPaths["imagePath"],
        config=config,
    )
    proj.printInfo()
    proj.config.printInfo()

    return proj


def saveProjectFile(
    filepath: str,
    refTime: datetime,
    calPath: str,
    timePath: str,
    specPath: str,
    statPath: str,
    maskPath: str,
    transFuncPath: str,
    imagePath: str,
) -> None:
    """Save the project file to filepath

    Parameters
    ----------
    filepath : str
        Path to project file
    """

    f = open(filepath, "w")
    f.write("Calibration data path = {}\n".format(calPath))
    f.write("Time data path = {}\n".format(timePath))
    f.write("Spectra data path = {}\n".format(specPath))
    f.write("Statistics data path = {}\n".format(statPath))
    f.write("Mask data path = {}\n".format(maskPath))
    f.write("TransFunc data path = {}\n".format(transFuncPath))
    f.write("Image data path = {}\n".format(imagePath))
    f.write("Reference time = {}\n".format(refTime.strftime("%Y-%m-%d %H:%M:%S")))
    f.close()


def loadProjectFile(filepath: str) -> None:
    """Load project from path to project file

    Parameters
    ----------
    filepath : str
        Path to project file

    Returns
    -------
    projPath : Dict[str, Union[str, datetime]]
        A dictionary with various project paths
    """

    f = open(filepath, "r")
    lines = f.readlines()
    f.close()
    projPaths = {}
    for l in lines:
        split = l.strip().split("=")
        if "Calibration data path" in l:
            projPaths["calPath"] = split[1].strip()
        if "Time data path" in l:
            projPaths["timePath"] = split[1].strip()
        if "Spectra data path" in l:
            projPaths["specPath"] = split[1].strip()
        if "Statistics data path" in l:
            projPaths["statPath"] = split[1].strip()
        if "Mask data path" in l:
            projPaths["maskPath"] = split[1].strip()
        if "TransFunc data path" in l:
            projPaths["transFuncPath"] = split[1].strip()
        if "Image data path" in l:
            projPaths["imagePath"] = split[1].strip()
        if "Reference time" in l:
            projPaths["refTime"] = datetime.strptime(
                split[1].strip(), "%Y-%m-%d %H:%M:%S"
            )
    return projPaths

