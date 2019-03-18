import os
from resistics.project.projectIO import newProject


"""Example 2 - Create project

Example 2 covers advanced topics such as remote refence processing, interoperability of different data formats and remote reference statistics.

There are three sites:
M1 : A site of interest 
M13 : 
RemoteRef : The remote reference site 

The remote reference data is in SPAM format, whilst M1 and M13 are in ATS format. 
"""

# define the project path. The project will be created under this project path.
# If the path does not exist, it will be created
projectPath = os.path.join("advancedProject")
projData = newProject(projectPath, "2016-01-18 00:00:00")

# let's create 3 sites
# M1 and M13 are at locations of interest. RemoteRef is a remote reference site which will be used for remote reference processing
projData.createSite("M1")
projData.createSite("M13")
projData.createSite("RemoteRef")

