# -*- coding: utf-8 -*-

import numpy as np
import glob
import matplotlib.pyplot as plt
from scipy.ndimage import center_of_mass
from scipy import signal
from scipy import stats

realFile = np.load(r"S:\NewAnalysisOutput\spikePairAnalysisOutput\baseData.npz", allow_pickle=True)
rescaledDiff = realFile['rescaledDiff'].T

realFilePeak = np.unravel_index(np.argmax(rescaledDiff), rescaledDiff.shape)
realFileTrough = np.unravel_index(np.argmin(rescaledDiff), rescaledDiff.shape)

circAngsTemp = np.deg2rad(np.arange(0,360,6))
circAngsCalculationMatrix = np.tile([circAngsTemp], (52,1)).T


# realFilePeak = np.array(center_of_mass(rescaledDiff*(rescaledDiff>0.01)))
# realFileTrough = np.array(center_of_mass(rescaledDiff*(rescaledDiff<0.01)))
rescaledDiff[rescaledDiff==0] = np.nan
realFileSparisty = np.nansum(np.square(rescaledDiff)/(np.nanmean(rescaledDiff)**2))
# realFileCoherence = np.corrcoef(), np.convolve(rescaledDiff, [[1,1,1],[1,0,1],[1,1,1]], mode='valid')/8
realFileCoherence = signal.convolve2d(realFile['rescaledDiff'].T, [[1,1,1],[1,0,1],[1,1,1]], mode='same', boundary='wrap')/8
realFileCoherence = stats.pearsonr(realFile['rescaledDiff'].T.ravel(), realFileCoherence.ravel()).statistic
print(realFilePeak, realFileTrough)


rescaledDiff = realFile['rescaledDiff'].T
posRescaledDiff = rescaledDiff*(rescaledDiff>(np.max(rescaledDiff)*0.8))
negRescaledDiff = rescaledDiff*(rescaledDiff<(np.min(rescaledDiff)*0.8))*-1

sinTemp = np.sum(np.sin(circAngsCalculationMatrix)*posRescaledDiff)
cosTemp = np.sum(np.cos(circAngsCalculationMatrix)*posRescaledDiff)
curAng = np.rad2deg(np.arctan2(sinTemp, cosTemp))/6
distCOM = center_of_mass(posRescaledDiff)[1]
realPeakCOM = np.abs([curAng, distCOM, rescaledDiff[int(curAng), int(distCOM)]])

sinTemp = np.sum(np.sin(circAngsCalculationMatrix)*negRescaledDiff)
cosTemp = np.sum(np.cos(circAngsCalculationMatrix)*negRescaledDiff)
curAng = np.rad2deg(np.arctan2(sinTemp, cosTemp))/6
distCOM = center_of_mass(negRescaledDiff)[1]
realTroughCOM = np.abs([curAng, distCOM, rescaledDiff[int(curAng), int(distCOM)]])



f, ax = plt.subplots(1,3)

ax[0].imshow(rescaledDiff)
ax[1].imshow(posRescaledDiff)
ax[2].imshow(negRescaledDiff)





filenames = glob.glob(r"S:\NewAnalysisOutput\spikePairAnalysisOutput\shuffle_??????????.*.npz")
plots = np.zeros((len(filenames), 60,52))
sparsity = np.zeros(len(filenames))
spatinfo = np.zeros(len(filenames))
coherence = np.zeros(len(filenames))
peaks = np.zeros((len(filenames),2))
troughs = np.zeros((len(filenames),2))
peakCOMS = np.zeros((len(filenames),3))
troughCOMS = np.zeros((len(filenames),3))

for i in range(len(filenames)):
    curFile = np.load(filenames[i], allow_pickle=True)
    rescaledDiff = curFile['rescaledDiff'].T
    plots[i] = rescaledDiff
    # peaks[i] = np.array(center_of_mass(rescaledDiff*(rescaledDiff>0.01)))
    # troughs[i] = np.array(center_of_mass(rescaledDiff*(rescaledDiff<0.01)))
    peaks[i] = np.unravel_index(np.argmax(rescaledDiff), rescaledDiff.shape)
    troughs[i] = np.unravel_index(np.argmin(rescaledDiff), rescaledDiff.shape)
    
    posRescaledDiff = rescaledDiff*(rescaledDiff>(np.max(rescaledDiff)*0.8))
    negRescaledDiff = rescaledDiff*(rescaledDiff<(np.min(rescaledDiff)*0.8))*-1
    
    sinTemp = np.sum(np.sin(circAngsCalculationMatrix)*posRescaledDiff)
    cosTemp = np.sum(np.cos(circAngsCalculationMatrix)*posRescaledDiff)
    curAng = np.rad2deg(np.arctan2(sinTemp, cosTemp))/6
    distCOM = center_of_mass(posRescaledDiff)[1]
    peakCOMS[i] = np.abs([curAng, distCOM, rescaledDiff[int(curAng), int(distCOM)]])

    sinTemp = np.sum(np.sin(circAngsCalculationMatrix)*negRescaledDiff)
    cosTemp = np.sum(np.cos(circAngsCalculationMatrix)*negRescaledDiff)
    curAng = np.rad2deg(np.arctan2(sinTemp, cosTemp))/6
    distCOM = center_of_mass(negRescaledDiff)[1]
    troughCOMS[i] = np.abs([curAng, distCOM, rescaledDiff[int(curAng), int(distCOM)]])
    
    
    

    rescaledDiff[rescaledDiff==0] = np.nan
    meanval = np.nanmean(rescaledDiff)
    sparsity[i] = np.nansum(np.square(rescaledDiff)/(np.nanmean(rescaledDiff)**2))
    spatinfo[i] = np.nansum(np.square(rescaledDiff)/(np.nanmean(rescaledDiff)**2))
    
    # bigger = np.zeros((62,54))
    # bigger[1:61, 1:53] = rescaledDiff
    # plt.imshow(np.convolve(rescaledDiff, [[1,1,1],[1,0,1],[1,1,1]]))
    # plt.show()
    # break
    
    curCoherenceStep = signal.convolve2d(plots[i], [[1,1,1],[1,0,1],[1,1,1]], mode='same', boundary='wrap')/8
    coherence[i] = stats.pearsonr(plots[i].ravel(), curCoherenceStep.ravel()).statistic
    # print(curCoherenceStep.shape)
    # plt.imshow(curCoherenceStep)
    # plt.show()
    # break


    # coherenceVals = []
    
    # for ang in range(-1,57):
    #     for dist in range(1,51):
    #         coherenceVals.append(np.nanmean())
    
    # plots[i].ravel()[np.argmin(curFile['rescaledDiff'].T)] = 1
    # plt.imshow(curFile['rescaledDiff'])
    # plt.show()


# plt.plot(centerVal[:,0], coherence, '.')
# plt.plot(realFileCenter[0], realFileCoherence, 'r.')

np.square(peaks[:,0])-30**2

# plt.imshow(np.histogram2d(peakCOMS[:,0], peakCOMS[:,1], bins=[60,52], range=[[0,60],[0,52]]))

# plt.figure()

# plt.plot(peakCOMS[:,0], peakCOMS[:,1], '.')
# plt.plot(troughCOMS[:,0], troughCOMS[:,1], '.')


# plt.plot(realFileCenter[0], realFileCenter[1], 'r.')

# sumPlot = np.nansum(plots, axis=0)
# sumPlot[sumPlot==0] = np.nan
# plt.imshow(sumPlot)

# ax = plt.figure().add_subplot(projection='3d')
# plt.plot(peakCOMS[:,0], peakCOMS[:,1], peakCOMS[:,2], 'b.')
# plt.plot(troughCOMS[:,0], troughCOMS[:,1], troughCOMS[:,2], 'r.')

# plt.plot(realPeakCOM[0], realPeakCOM[1], realPeakCOM[2], 'c.')
# plt.plot(realTroughCOM[0], realTroughCOM[1], realTroughCOM[2], 'y.')

# ax = plt.figure().add_subplot(projection='3d')
# plt.plot(peakCOMS[:,0], peakCOMS[:,1], sparsity, 'b.')
# plt.plot(troughCOMS[:,0], troughCOMS[:,1], sparsity, 'r.')

# plt.plot(realPeakCOM[0], realPeakCOM[1], realFileSparisty, 'c.')
# plt.plot(realTroughCOM[0], realTroughCOM[1], realFileSparisty, 'y.')

print(realFileSparisty)
print(sparsity<realFileSparisty)
closerPeakAng = np.abs(peakCOMS[:,0]-30)<np.abs(realPeakCOM[0]-30)
closerPeakDist = np.abs(peakCOMS[:,1]-12)<np.abs(realPeakCOM[1]-12)
closerTroughAng = np.abs(troughCOMS[:,0]-30)<np.abs(realTroughCOM[0]-30)
closerTroughDist = np.abs(troughCOMS[:,1]-31)<np.abs(realTroughCOM[1]-31)
# print(np.sum((sparsity<realFileSparisty))/len(sparsity))
print(np.sum(closerPeakAng*closerPeakDist*closerTroughAng*closerTroughDist), len(peakCOMS))

plt.figure()
centerDist = np.sqrt(np.square(peakCOMS[:,0]-30)+np.square(peakCOMS[:,1]-12))
periDist = np.sqrt(np.square(troughCOMS[:,0]-30)+np.square(troughCOMS[:,1]-31))

realCenterDist = np.sqrt(np.square(realPeakCOM[0]-30)+np.square(realPeakCOM[1]-12))
realPeriDist = np.sqrt(np.square(realTroughCOM[0]-30)+np.square(realTroughCOM[1]-31))

plt.plot(centerDist, periDist, '.')
plt.plot(realCenterDist, realPeriDist, '.')

print(np.sum((centerDist<realCenterDist)*(periDist<realPeriDist)))
print(realPeakCOM, realTroughCOM)


plt.show()