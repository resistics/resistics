import sys, os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from magpy.utilities.utilsRobust import *

def test_mestimate():   
    breakPrint()
    generalPrint("basic_Robust",
                 "Running test function: test_mestimate")         
    mean = 0
    std = 5
    x = np.arange(1000)
    y = np.random.normal(mean, std, x.size)
    ones = np.ones(shape=(x.size))
    # add large outliers
    numOutliers = 450
    for i in range(0, numOutliers):
        index = np.random.randint(0, x.size)
        y[index] = np.random.randint(std * 4, std * 20)
    # compute mean
    mean = np.average(y)
    standardDev = np.std(y)
    # compute mad
    med = sampleMedian(y)
    mad = sampleMAD(y)
    # mestimates
    mestLocation, mestScale = mestimate(y)
    # plot
    plt.figure()
    plt.scatter(x, y, color='y')
    plt.plot(x, ones * mean, lw=2, color="b", label="mean")
    plt.plot(x, ones * standardDev, lw=2, color="b", ls="dashed")
    plt.plot(x, ones * med, lw=2, color="g", label="median")
    plt.plot(x, ones * mad, lw=2, color="g", ls="dashed")
    plt.plot(x, ones * mestLocation, lw=2, color="r", label="mest")
    plt.plot(x, ones * mestScale, lw=2, color="r", ls="dashed")
    plt.legend()
    plt.show()


def test_mestimateModel():
    breakPrint()
    generalPrint("basic_Robust",
                 "Running test function: test_mestimateModel")      
    # let's generate some data
    x = np.arange(1000)
    y = np.arange(-50, 50, 0.1)
    # create a linear function of this
    z = 2.5 * x + y
    # let's add some noise
    mean = 0
    std = 3
    noise = np.random.normal(0, 3, x.size)
    # print noise.shape
    z = z + noise
    # now add some outliers
    numOutliers = 80
    for i in range(0, numOutliers):
        index = np.random.randint(0, x.size)
        z[index] = np.random.randint(std * 4, std * 20)

    A = np.transpose(np.vstack((x, y)))
    # now try and do a robust regression
    components = mestimateModel(A, z)
    print(components)
    # plt.figure()
    # plt.plot()


def testRobustRegression():
    breakPrint()
    generalPrint("basic_Robust",
                 "Running test function: testRobustRegression")     
    # random seed
    np.random.seed(0)
    # the function
    x = np.arange(150)
    y = 12 + 0.5 * x
    # noise
    mean = 0
    std = 3
    noise = np.random.normal(mean, 3 * std, x.size)
    # add noise
    yNoise = y + noise
    # now add some outliers
    numOutliers = 30
    for i in range(0, numOutliers):
        index = np.random.randint(0, x.size)
        yNoise[index] = yNoise[index] + np.random.randint(-1000, 1000)

    # now add some outliers
    xNoise = np.array(x)
    numOutliers = 30
    for i in range(0, numOutliers):
        index = np.random.randint(0, x.size)
        xNoise[index] = x[index] + np.random.randint(-5000, 5000)
    xNoise = xNoise.reshape((x.size, 1))

    # lets use m estimate
    paramsM, residsM, scaleM, weightsM = mestimateModel(
        xNoise, yNoise, intercept=True)
    # lets use mm estimate
    paramsMM, residsMM, scaleMM, weightsMM = mmestimateModel(
        xNoise, yNoise, intercept=True)
    # lets test chatterjee machler
    paramsCM, residsCM, weightsCM = chatterjeeMachler(
        xNoise, yNoise, intercept=True)
    # lets test chatterjee machler mod
    paramsModCM, residsModCM, weightsModCM = chatterjeeMachlerMod(
        xNoise, yNoise, intercept=True)

    # let's plot Pdiag
    # plt.figure()
    # n, bins, patches = plt.hist(
    #     Pdiag, 50, normed=0, facecolor='green', alpha=0.75)

    # try and predict
    yM = paramsM[0] + paramsM[1] * x
    yMM = paramsMM[0] + paramsMM[1] * x
    yCM = paramsCM[0] + paramsCM[1] * x
    yCM_mod = paramsModCM[0] + paramsModCM[1] * x

    plt.figure()
    plt.scatter(x, y, marker="s", color="black")
    plt.scatter(xNoise, yNoise)
    plt.plot(x, yM)
    plt.plot(x, yMM)
    plt.plot(x, yCM)
    plt.plot(x, yCM_mod)
    plt.legend([
        "M estimate", "MM estimate", "chatterjeeMachler",
        "chatterjeeMachlerMod"
    ],
               loc="lower left")
    plt.show()


def testRobustRegression2D():
    breakPrint()
    generalPrint("basic_Robust",
                 "Running test function: testRobustRegression2D")        
    # random seed
    np.random.seed(0)
    numPts = 300
    # the function
    x1 = np.arange(numPts, dtype="float")
    x2 = 10 * np.cos(2.0 * np.pi * 10 * x1 / np.max(x1))
    y = 12 + 0.5 * x1 + 3 * x2
    # noise
    mean = 0
    std = 3
    noise = np.random.normal(mean, 3 * std, numPts)
    # add noise
    yNoise = y + noise
    # now add some outliers
    numOutliers = 140
    for i in range(0, numOutliers):
        index = np.random.randint(0, numPts)
        yNoise[index] = yNoise[index] + np.random.randint(-100, 100)

    # now add some outliers
    x1Noise = np.array(x1)
    x2Noise = np.array(x2)
    numOutliers = 5
    for i in range(0, numOutliers):
        index = np.random.randint(0, numPts)
        x1Noise[index] = x1[index] + np.random.randint(-500, 500)
        index = np.random.randint(0, numPts)
        x2Noise[index] = x2[index] + np.random.randint(-500, 500)

    x1Noise = x1Noise.reshape((x1.size, 1))
    x2Noise = x2Noise.reshape((x2.size, 1))
    X = np.hstack((x1Noise, x2Noise))

    # lets use m estimate
    paramsM, residsM, scaleM, weightsM = mestimateModel(
        X, yNoise, intercept=True)
    # lets use mm estimate
    paramsMM, residsMM, scaleMM, weightsMM = mmestimateModel(
        X, yNoise, intercept=True)
    # lets test chatterjee machler
    paramsCM, residsCM, weightsCM = chatterjeeMachler(
        X, yNoise, intercept=True)
    # lets test chatterjee machler mod
    paramsModCM, residsModCM, weightsModCM = chatterjeeMachlerMod(
        X, yNoise, intercept=True)
    # lets test chatterjee machler hadi
    paramsCMHadi, residsCMHadi, weightsCMHadi = chatterjeeMachlerHadi(
        X, yNoise, intercept=True)

    # try and predict
    yM = paramsM[0] + paramsM[1] * x1 + paramsM[2] * x2
    yMM = paramsMM[0] + paramsMM[1] * x1 + paramsMM[2] * x2
    yCM = paramsCM[0] + paramsCM[1] * x1 + paramsCM[2] * x2
    yCM_mod = paramsModCM[0] + paramsModCM[1] * x1 + paramsModCM[2] * x2
    yCM_Hadi = paramsCMHadi[0] + paramsCMHadi[1] * x1 + paramsCMHadi[2] * x2

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x1, x2, y, marker="s", color="black")
    ax.scatter(x1Noise, x2Noise, yNoise, marker="*", s=50, color="goldenrod")
    # plt.plot(x1, x2, zs=yM)
    plt.plot(x1, x2, zs=yMM)
    # plt.plot(x1, x2, zs=yCM)
    plt.plot(x1, x2, zs=yCM_mod)
    # plt.plot(x1, x2, zs=yCM_Hadi)
    # plt.legend(["M estimate", "MM estimate", "chatterjeeMachler", "chatterjeeMachlerMod", "chatterjeeMachlerHadi"], loc="lower left")
    plt.legend(["MM estimate", "chatterjeeMachlerMod"], loc="lower left")
    plt.show()


test_mestimate()
test_mestimateModel()
testRobustRegression()
testRobustRegression2D()