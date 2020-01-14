
from statsmodels.compat import lmap
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.api import abline_plot
from statsmodels.formula.api import ols, rlm
import matplotlib.pyplot as plt
import os, sys
from magpy.utilties.utilsRobust import *


def plot_weights(support, weights, xlabels, xticks):
    #fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111)
    #ax.plot(support, weights_func(support))
    ax.plot(support, weights)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=16)
    ax.set_ylim(-.1, 1.1)
    return ax

# get the data
prestige = sm.datasets.get_rdataset("Duncan", "car", cache=True).data
print(prestige.head(10))

# getting the variables for me
obs = np.array(prestige.prestige)
predictor = np.empty(shape=(obs.size,2))
predictor[:,0] = prestige.income
predictor[:,1] = prestige.education

# plot if wanted
fig = plt.figure(figsize=(12,12))
ax1 = fig.add_subplot(211, xlabel='Income', ylabel='Prestige')
ax1.scatter(prestige.income, prestige.prestige)
xy_outlier = prestige.ix['minister'][['income','prestige']]
ax1.annotate('Minister', xy_outlier, xy_outlier+1, fontsize=16)
ax2 = fig.add_subplot(212, xlabel='Education',
                           ylabel='Prestige')
ax2.scatter(prestige.education, prestige.prestige)
# plt.show()

ols_model = ols('prestige ~ income + education', prestige).fit()
print(ols_model.summary())

print("######################")
print("Built in OLS")
print("######################")
# now get the robust estimate using huber
params, resids, squareResid, rank, s = olsModel(predictor, obs, intercept=True)
print(params)

print("######################")
print("Checking M-estimate")
print("######################")
rlm_model = rlm('prestige ~ income + education', prestige, M=sm.robust.norms.HuberT(t=1.345)).fit()
print(rlm_model.summary())
print("######################")
print("Built in M-estimate")
print("######################")
params, resids, scale, weights = mestimateModel(predictor, obs, weights="huber", intercept=True)
print(params)		

	
print("######################")
print("Checking MAD")
print("######################")
np.random.seed(12345)
fat_tails = stats.t(6).rvs(40)
print(sm.robust.scale.mad(fat_tails))
print(sampleMAD(fat_tails))
print(sm.robust.scale.mad(fat_tails) - sampleMAD(fat_tails))


print("######################")
print("Checking weighting function")
print("######################")
print("Checking Huber Weights")
t = 1.345
support = np.linspace(-3*t, 3*t, 1000)
huber = sm.robust.norms.HuberT(t=t)
huberSM = huber.weights(support)
huberMine = huberLocationWeights(support, t)
print(np.sum(huberMine-huberSM))

print("Checking Bisquare Weights")
c = 4.685
support = np.linspace(-3*c, 3*c, 1000)
tukey = sm.robust.norms.TukeyBiweight(c=c)
tukeySM = tukey.weights(support)
tukeyMine = bisquareLocationWeights(support, c)
print(np.sum(tukeyMine - tukeySM))

print("Checking Hampel Weights")
c = 8 
support = np.linspace(-3*c, 3*c, 1000)
hampel = sm.robust.norms.Hampel(c=c)
hampelSM = hampel.weights(support)
hampelMine = hampelLocationWeights(support, c)
print(np.sum(hampelMine - hampelSM))


print("Checking Trimmed Mean Weights")
c = 2
support = np.linspace(-3*c, 3*c, 1000)
tmean = sm.robust.norms.TrimmedMean(c=c)
tmeanSM = tmean.weights(support)
tmeanMine = trimmedMeanLocationWeights(support, c)
print(np.sum(tmeanMine - tmeanSM))
