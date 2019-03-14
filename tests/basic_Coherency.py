import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import copy
import sys, os
from magpy.utilties.utilsFreq import *


def smooth(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.
	
	This method is based on the convolution of a scaled window with the signal.
	The signal is prepared by introducing reflected copies of the signal 
	(with the window size) in both ends so that transient parts are minimized
	in the begining and end part of the output signal.
	
	input:
		x: the input signal 
		window_len: the dimension of the smoothing window; should be an odd integer
		window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
			flat window will produce a moving average smoothing.
	
	output:
		the smoothed signal
		
	example:
	
	t=linspace(-2,2,0.1)
	x=sin(t)+randn(len(t))*0.1
	y=smooth(x)
	
	see also: 
	
	numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
	scipy.signal.lfilter
	
	TODO: the window parameter could be the window itself if an array instead of a string
	NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
	"""

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in [
            'flat', 'hanning', 'hamming', 'bartlett', 'blackman', 'parzen'
    ]:
        raise ValueError(
            "Window is one of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
        )

    s = np.pad(x, (window_len, window_len), mode="edge")
    if window == 'flat':  #moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('signal.' + window + '(window_len)')

    off = window_len + (window_len - 1) / 2
    if window_len % 2 == 0:
        off = window_len + (window_len / 2)

    y = np.convolve(s, w / w.sum(), mode='full')
    return y[off:off + x.size]


def smooth2d(x, window_len=11, window='hanning'):
    # window_len[0] is smoothing along windows
    # window_len[1] is smoothing in a single window
    kernel = np.outer(
        signal.hanning(window_len[0], 8), signal.gaussian(window_len[1], 8))
    # pad to help the boundaries
    padded = np.pad(
        x, ((window_len[0], window_len[0]), (window_len[1], window_len[1])),
        mode="edge")
    # 2d smoothing
    blurred = signal.fftconvolve(padded, kernel, mode='same')
    print(x.shape)
    print(padded.shape)
    print(blurred.shape)
    return blurred[window_len[0]:window_len[0] +
                   x.shape[0], window_len[1]:window_len[1] + x.shape[1]]


np.random.seed(0)

fs = 10e3
N = 1e5
amp = 20
freq = 1234.0
noise_power = 0.001 * fs / 2
# let's try estimate
windowSize = int(N / 100)
windows = int(N / windowSize)
print(windowSize)
print(windows)
fftSize = windowSize / 2 + 1  #assuming even and real
print(fftSize)

time = np.arange(N) / fs
b, a = signal.butter(2, 0.25, 'low')
x = np.random.normal(scale=np.sqrt(noise_power), size=time.shape)
y = signal.lfilter(b, a, x)
x += amp * np.sin(2 * np.pi * freq * time)
y += np.random.normal(scale=0.1 * np.sqrt(noise_power), size=time.shape)
x2 = np.array(x)
y2 = np.array(y)

fx, xPower = signal.welch(
    x, fs, nperseg=windowSize, noverlap=0, scaling="spectrum")
fy, yPower = signal.welch(
    y, fs, nperseg=windowSize, noverlap=0, scaling="spectrum")
fxy, xyPower = signal.csd(
    x, y, fs, nperseg=windowSize, noverlap=0, scaling="spectrum")

welchCoh1 = np.power(np.absolute(xyPower), 2)
welchCoh2 = xPower * yPower
welchCoh = welchCoh1 / welchCoh2

f, Cxy = signal.coherence(x, y, fs, nperseg=windowSize, noverlap=N / 200)
f2, Cxy2 = signal.coherence(x, y, fs, nperseg=windowSize, noverlap=0)

# my method
powXX = np.zeros(shape=(fftSize), dtype="complex")
powYY = np.zeros(shape=(fftSize), dtype="complex")
powXY = np.zeros(shape=(fftSize), dtype="complex")
# win
winFnc = signal.hanning(windowSize)
winFnc = winFnc / winFnc.sum()

# save stacked
stackX = np.zeros(shape=(fftSize), dtype="complex")
stackY = np.zeros(shape=(fftSize), dtype="complex")
arrX = np.zeros(shape=(windows, fftSize), dtype="complex")
arrY = np.zeros(shape=(windows, fftSize), dtype="complex")
for iW in range(0, windows):
    start = iW * windowSize
    end = start + windowSize
    dataX = signal.detrend(x2[start:end]) * winFnc
    specX = forwardFFT(dataX, norm=False)
    dataY = signal.detrend(y2[start:end]) * winFnc
    specY = forwardFFT(dataY, norm=False)

    # stack
    arrX[iW] = specX
    arrY[iW] = specY

    # calculate auto and cross power spec
    powXX += specX * np.conjugate(specX)
    powYY += specY * np.conjugate(specY)
    powXY += specX * np.conjugate(specY)

# powers averaged out over windows
powXX = 2 * powXX / windows
powXX[[0, -1]] = powXX[[0, -1]] / 2
powYY = 2 * powYY / windows
powYY[[0, -1]] = powYY[[0, -1]] / 2
powXY = 2 * powXY / windows
powXY[[0, -1]] = powXY[[0, -1]] / 2

# calculate coherence
cohNom = np.power(np.absolute(powXY), 2)
cohDenom = powXX * powYY
coherence = cohNom / cohDenom
coherenceWin = coherence.real

# now let's try and calculate the coherency
arrPowXX = arrX * np.conjugate(arrX)
arrPowYY = arrY * np.conjugate(arrY)
arrPowXY = arrX * np.conjugate(arrY)
arrCoh = np.zeros(shape=(windows, fftSize), dtype="complex")

# let's calculate coherency for each window
# winSmoothers = [11,15,19,23,27]
# winTypes = ["parzen", "hamming"]
winSmoothers = [11]
winTypes = ["parzen"]
for iW in range(0, windows):
    count = 0
    for winSmooth in winSmoothers:
        for winType in winTypes:
            # need to smooth the cross power
            absSmooth = smooth(
                np.absolute(arrPowXY[iW]), winSmooth, window=winType)
            testSmooth = np.absolute(
                smooth(arrPowXY[iW], winSmooth, window=winType))
            cohNom = np.power(testSmooth, 2)
            cohDenom = smooth(
                arrPowXX[iW], winSmooth, window=winType) * smooth(
                    arrPowYY[iW], winSmooth, window=winType)
            coherence = cohNom / cohDenom
            arrCoh[iW] += coherence.real
            count = count + 1

    arrCoh[iW] = arrCoh[iW] / count

# let's try 2d smoothing
testXX = smooth2d(arrPowXX, (5, 5), "hamming")
testYY = smooth2d(arrPowYY, (5, 5), "hamming")
testXY = smooth2d(arrPowXY, (5, 5), "hamming")
testCoh = np.zeros(shape=(windows, fftSize), dtype="complex")
for iW in range(0, windows):
    cohNom = np.power(np.absolute(testXY[iW]), 2)
    cohDenom = testXX[iW] * testYY[iW]
    coherence = cohNom / cohDenom
    testCoh[iW] = coherence.real

# plot
plt.figure()
plt.semilogy(f, np.absolute(powXX), label="my XX")
plt.semilogy(f, np.absolute(powYY), label="my YY")
plt.semilogy(f, np.absolute(powXY), label="my XY")
plt.semilogy(f, np.absolute(xPower), label="welch XX")
plt.semilogy(f, np.absolute(yPower), label="welch YY")
plt.semilogy(f, np.absolute(xyPower), label="welch XY")
plt.legend()

plt.figure()
plt.plot(f, Cxy, label="Overlap")
plt.plot(f2, Cxy2, label="No Overlap")
plt.plot(f, welchCoh, label="Welch Coh")
plt.plot(f, coherenceWin, label="My Coh Win", ls="dashed")
# plt.plot(f, coherenceFreq, label="My Coh Freq")
plt.ylim(0, 1.2)
plt.legend()
plt.xlabel('frequency [Hz]')
plt.ylabel('Coherence')

plt.figure()
# for iW in xrange(0, windows):
# plt.plot(f, arrCoh[iW])
plt.plot(f, coherenceWin, label="My Coh Win", lw=2)
plt.plot(f, np.average(arrCoh, axis=0), label="Averaged", lw=2)
plt.plot(f, np.average(testCoh, axis=0), label="Averaged 2d", lw=2)
plt.ylim(-0.1, 1.1)
plt.legend()
plt.xlabel('frequency [Hz]')
plt.ylabel('Coherence')

plt.show()
