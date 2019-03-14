from magpy.project.project import Project
from magpy.utilities.utilsIO import *


def setupTestProject():
    breakPrint()
    generalPrint("testsProject", "Running test function: setupTestProject")    
    projectPath = os.path.join("testData", "testProject")
    projectName = "test"
    # the reftime for the test project
    refTime = "2016-01-18 00:00:00"

    # need to check and make project path
    checkAndMakeDir(projectPath)

    proj = Project()
    proj.setTimeDataPath(os.path.join(projectPath, "timeData"))
    proj.setSpecDataPath(os.path.join(projectPath, "specData"))
    proj.setStatDataPath(os.path.join(projectPath, "statData"))
    proj.setTransDataPath(os.path.join(projectPath, "transFuncData"))
    proj.setCalDataPath(os.path.join(projectPath, "calData"))
    proj.setRefTime(refTime)
    proj.initialiseProject()
    proj.saveProjectFile(
        os.path.join(projectPath, "{}.prj".format(projectName)))
    proj.printInfo()
    sites = proj.getAllSites()

    # lets print out information about the sites
    for s in sites:
        proj.printSiteInfo(s)
        timeFiles = proj.getSiteTimeFiles(s)
        for tFile in timeFiles:
            proj.printMeasInfo(s, tFile)


def setupEthiopiaProject():
    breakPrint()  
    generalPrint("testsProject", "Running test function: setupEthiopiaProject")      
    projectPath = os.path.join("..", "..", "..", "general data", "ethiopiaProject")
    projectName = "ethiopia"
    # the reftime for ethiopia project
    refTime = "2012-02-10 00:00:00"

    # need to check and make project path
    checkAndMakeDir(projectPath)

    proj = Project()
    proj.setTimeDataPath(os.path.join(projectPath, "timeData"))
    proj.setSpecDataPath(os.path.join(projectPath, "specData"))
    proj.setStatDataPath(os.path.join(projectPath, "statData"))
    proj.setTransDataPath(os.path.join(projectPath, "transFuncData"))
    proj.setCalDataPath(os.path.join(projectPath, "calData"))
    proj.setRefTime(refTime)
    proj.initialiseProject()
    proj.saveProjectFile(
        os.path.join(projectPath, "{}.prj".format(projectName)))
    proj.printInfo()
    sites = proj.getAllSites()

    # lets print out information about the sites
    for s in sites:
        proj.printSiteInfo(s)
        timeFiles = proj.getSiteTimeFiles(s)
        for tFile in timeFiles:
            proj.printMeasInfo(s, tFile)


setupTestProject()
setupEthiopiaProject()