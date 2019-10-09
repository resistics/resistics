from datapaths import timePath, timeImages, calPath, calImages, docTime, docCal
import shutil

# time data format images
for image in timeImages.glob("*.png"):
    shutil.copy2(image, docTime)
# copy ascii headers
asciiPath = timePath / "ascii"
shutil.copy2(asciiPath / "global.hdr", docTime / "ascii_global.hdr")
shutil.copy2(asciiPath / "chan_00.hdr", docTime / "ascii_chan_00.hdr")
asciiInternalPath = timePath / "asciiInternal"
shutil.copy2(asciiInternalPath / "comments.txt", docTime / "asciiInternal_comments.txt")
# copy ats
atsInternal = timePath / "atsInternal"
shutil.copy2(atsInternal / "global.hdr", docTime / "atsInternal_global.hdr")
shutil.copy2(atsInternal / "chan_00.hdr", docTime / "atsInternal_chan_00.hdr")
shutil.copy2(atsInternal / "comments.txt", docTime / "atsInternal_comments.txt")
atsAscii = timePath / "atsAscii"
shutil.copy2(asciiInternalPath / "comments.txt", docTime / "atsAscii_comments.txt")
# copy spam
spamInternalPath = timePath / "spamInternal"
shutil.copy2(spamInternalPath / "comments.txt", docTime / "spamInternal_comments.txt")
spamFilteredPath = timePath / "spamInternalFiltered"
shutil.copy2(spamFilteredPath / "comments.txt", docTime / "spamInternalFiltered_comments.txt")
spamInterpPath = timePath / "spamInterp"
shutil.copy2(spamInterpPath / "comments.txt", docTime / "spamInterp_comments.txt")
# copy phoenix
phoenixPath = timePath / "phoenixInternal" / "meas_ts5_2011-11-13-17-04-02_2011-11-14-14-29-46"
shutil.copy2(phoenixPath / "comments.txt", docTime / "phoenixInternal_comments.txt")
# copy lemi
lemiB423Path = timePath / "lemiB423"
shutil.copy2(lemiB423Path / "global.h423", docTime / "lemib423_global.h423")
shutil.copy2(lemiB423Path / "chan_00.h423", docTime / "lemib423_chan_00.h423")
shutil.copy2(lemiB423Path / "chan_03.h423", docTime / "lemib423_chan_03.h423")
lemiInternalPath = timePath / "lemiB423Internal"
shutil.copy2(lemiInternalPath / "comments.txt", docTime / "lemiB423Internal_comments.txt")
lemiB423EPath = timePath / "lemiB423E"
shutil.copy2(lemiB423EPath / "global.h423E", docTime / "lemib423_global.h423E")
shutil.copy2(lemiB423EPath / "chan_01.h423E", docTime / "lemib423_chan_01.h423E")
shutil.copy2(lemiB423EPath / "chan_03.h423E", docTime / "lemib423_chan_03.h423E")

# cal data format images
for image in calImages.glob("*.png"):
    shutil.copy2(image, docCal)
# ascii
shutil.copy2(calPath / "ascii.TXT", docCal)
shutil.copy2(calPath / "asciiWithData.TXT", docCal)
# metronix
shutil.copy2(calPath / "metronix2ascii.TXT", docCal)
shutil.copy2(calPath / "Hx_MFS06365.txt", docCal)
# rsp
shutil.copy2(calPath / "rsp2ascii.TXT", docCal)
shutil.copy2(calPath / "Metronix_Coil-----TYPE-006_BB-ID-000365.RSP", docCal)
# rspx
shutil.copy2(calPath / "rspx2ascii.TXT", docCal)
shutil.copy2(calPath / "Metronix_Coil-----TYPE-006_HF-ID-000133.RSPX", docCal)
