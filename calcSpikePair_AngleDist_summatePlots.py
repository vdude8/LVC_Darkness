# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 07:07:05 2022

@author: vyash
"""

import matplotlib.pyplot as plt
import numpy as np
import glob
import sys

import pandas as pd
import seaborn as sns
import analysisParms as clp
import re
import statsmodels.api as sm
import statsmodels.formula.api as smf
import random

from scipy.ndimage import center_of_mass

from tqdm import tqdm

from scipy import stats
import time
import json
import pandas as pd

from scipy import signal

    
#%%

#20-40, 12-18

# filenames = glob.glob("S:\\NewAnalysisOutput 10-06-2020\\*-*_t*_c*_m*_spikePairHist.npz")
# origClustFilenames = glob.glob("S:\\NewAnalysisOutput 6-27-16\\*\\*\\*-*_t*_c*.npz")

filenames = glob.glob("S:\\NewAnalysisOutput\\spikePairAnalysisOutput\\*-*_t*_c*_m*_spikePairHist.npz")
filenames = glob.glob("S:\\NewAnalysisOutput\\spikePairAnalysisOutput\\00*-*_t*_c*_m*_spikePairHist.npz")
# filenames = pd.read_pickle("S:\\NewAnalysisOutput\\spikePairAnalysisOutput\\df_tetrode_filtered.pkl")['Filename']
# filenames = pd.read_pickle("S:\\NewAnalysisOutput\\spikePairAnalysisOutput\\df_session_filtered.pkl")['Filename']
origClustFilenames = glob.glob("S:\\NewAnalysisOutput\\*\\*\\*-*_t*_c*.npz")

origClustFilenameSplit = [x.split("\\")[-1] for x in origClustFilenames]


def loadHists(doCentral):
    hists = []
    angles = []
    dists = []
    
    for filename in tqdm(filenames):
        # print(filename)
        fp = getFilenameElements(filename)
        base = filename.split("\\")[-1]
        # ratday = base[:6]
        # rat = ratday[:3]
        # day = ratday[-2:]
        # sessOrder = parms.dayList[int(rat)].index(int(day))
        # tet = base.split("_")[1][1:]
        # clust = base.split("_")[2][1:]
        # maze = base.split("_")[3][1:]
        # isCentralSess = parms.modeList[int(rat)][sessOrder]
        absDay = fp['absDay']
        rat = fp['rat']
        maze = fp['curMaze']
        isCentralSess = bool(fp['mode'])
        isExpDay = fp['isExpDay']
    
        if isCentralSess!=doCentral:
            continue
        if rat == '580':
            continue
        if int(maze)!=1:
            # print(maze, rat)
            continue

        # if absDay!=3:
        #     continue
    
        # if ((sessOrder+1)%4)==0:
        #     # print(sessOrder)
        #     continue
        if isExpDay:
            continue
        if fp['ca1pos']==-1:
            continue
    
        try:
            curFile = np.load(filename)
            rating = np.load(origClustFilenames[origClustFilenameSplit.index(base.split("_m")[0]+".npz")])['rating'].flatten()[0]
            if rating<=0:
                continue
            curFile['hist'] #test to see if file is okay
            curFile['angles'] #test to see if file is okay
            curFile['dists'] #test to see if file is okay
        except KeyError:
            print("Missing components in file:", filename)
            continue
        except EOFError:
            print("Corrupt File:", filename)
            continue
        except:
            print("Unknown error:", filename)
            
        hists.append(curFile['hist'])
        angles.append(curFile['angles'])
        dists.append(curFile['dists'])
        
    return (hists, angles, dists)

def getFilenameElements(filename, parmDict=True):
    basePath = "\\".join(filename.split("\\")[:-1])
    filename = filename.split("\\")[-1]
    filename = filename.split("/")[-1]
    filename = re.split(r"[a-zA-Z_\-\.]+", filename)[:-1]
    filenameI = list(map(int, filename))
    absDay = clp.dayList[filenameI[0]].index(filenameI[1])
    mode = clp.modeList[filenameI[0]][absDay]
    nMazes = clp.numMazes[filenameI[0]][absDay]
    if parmDict:
       return {"rat":filename[0], "day":filename[1], "absDay":absDay, "tetrode":filename[2], "cluster":filename[3], "mode":mode, "curMaze":filename[4], "numMazes":nMazes, "expSess":((absDay+1)%4==0), 'basePath':basePath, 'ca1pos':clp.ca1pos[int(filename[0])][int(filename[2])-1], 'isExpDay':(absDay+1)%4==0}
    else:
        return filename[0], filename[1], absDay, filename[2], filename[3], mode, nMazes, ((absDay+1)%4==0)

def loadHistsShuffled(doCentral):
    hists = []
    angles = []
    dists = []
    
    
    if doCentral:
        shuffFilenames = glob.glob("S:\\NewAnalysisOutput\\spikePairAnalysisOutput\\001-00_t*_c*_m*.npz")
    else:
        shuffFilenames = glob.glob("S:\\NewAnalysisOutput\\spikePairAnalysisOutput\\000-00_t*_c*_m*.npz")
    for filename in tqdm(shuffFilenames):
        curFile = np.load(filename)            
        hists.append(curFile['hist'])
        angles.append(curFile['angles'])
        dists.append(curFile['dists'])
        
    return (hists, angles, dists) 

#%%

# centHists, centAngles, centDists = loadHists(doCentral=1)
# periHists, periAngles, periDists = loadHists(doCentral=0)

centHists, centAngles, centDists = loadHistsShuffled(doCentral=1)
periHists, periAngles, periDists = loadHistsShuffled(doCentral=0)

centHists = np.array(centHists)
periHists = np.array(periHists)

# #%%

# np.savez("S:\\NewAnalysisOutput\\calcSpikePair_summations.npz", centHists=centHists, periHists=periHists)

# #%%

# centHists = np.load("S:\\NewAnalysisOutput\\calcSpikePair_summations.npz")['centHists']
# periHists = np.load("S:\\NewAnalysisOutput\\calcSpikePair_summations.npz")['periHists']

sumCentHist = np.sum(centHists, axis=0)
sumPeriHist = np.sum(periHists, axis=0)

#%%
# # plt.figure()
# centDistsFlattened = np.concatenate(centDists).ravel()
# centAnglesFlattened = np.concatenate(centAngles).ravel()
# # plt.plot(centDistsFlattened, centAnglesFlattened, '.')
# # plt.plot(periDistsFlattened, periDistsFlattened, '.')
# # plt.show()

# plt.figure()
# hist, xedges, yedges, img = plt.hist2d(centDistsFlattened, centAnglesFlattened, bins=[52,60], range=[[0,520], [0, 60]]) 
# plt.imshow(hist.T)

#%%

centHists2 = centHists.copy()
periHists2 = periHists.copy()

for i in range(len(centHists2)):
    centHists2[i]/=np.max(centHists2[i])

for i in range(len(periHists2)):
    periHists2[i]/=np.max(periHists2[i])
    
sumCentHistRescaled = np.sum(centHists2, axis=0)
sumPeriHistRescaled = np.sum(periHists2, axis=0)


#%%

# count = 10000

# bsCentPlots = np.empty((count,52,60))
# bsPeriPlots = np.empty((count,52,60))
# bsDiffPlots = np.empty((count,52,60))
# bsMins = np.empty((count, 2))
# bsMaxs = np.empty((count, 2))

# for i in tqdm(range(count)):    
#     centSampleIdxs = np.random.randint(len(centHists2), size=len(centHists2)) 
#     periSampleIdxs = np.random.randint(len(periHists2), size=len(periHists2))
    
#     bsCentPlots[i] = np.sum(centHists2[centSampleIdxs], axis=0)
#     bsPeriPlots[i] = np.sum(periHists2[periSampleIdxs], axis=0)
    
#     bsCentPlots[i]/=np.max(bsCentPlots[i])
#     bsPeriPlots[i]/=np.max(bsPeriPlots[i])
    
#     bsDiffPlots[i] = bsCentPlots[i]-bsPeriPlots[i]
#     bsMins[i] = np.unravel_index(np.argmin(bsDiffPlots[i]), (52,60))
#     bsMaxs[i] = np.unravel_index(np.argmax(bsDiffPlots[i]), (52,60))

#%%

# print(np.percentile(bsMins[:,0], 2.5), np.percentile(bsMins[:,0], 97.5))
# print(np.percentile(bsMins[:,1], 2.5), np.percentile(bsMins[:,1], 97.5))
# print(np.percentile(bsMaxs[:,0], 2.5), np.percentile(bsMaxs[:,0], 97.5))
# print(np.percentile(bsMaxs[:,1], 2.5), np.percentile(bsMaxs[:,1], 97.5))


#%%

# f, ax = plt.subplots(3)

# ax[0].set_aspect('equal')
# ax[1].set_aspect('equal')

# ax[0].imshow(np.sum(bsCentPlots, axis=0).T, vmax=6000)
# ax[1].imshow(np.sum(bsPeriPlots, axis=0).T, vmax=6000)
# ax[2].imshow(np.sum(bsDiffPlots, axis=0).T)

# bsDiffPlotSum = np.sum(bsDiffPlots, axis=0)

# print(np.unravel_index(np.argmin(bsDiffPlotSum), bsDiffPlotSum.shape), np.unravel_index(np.argmax(bsDiffPlotSum), bsDiffPlotSum.shape))


# plt.show()

#%%

# plt.hist(bsMins[:,0], cumulative=True, histtype='step', bins=count)
# plt.hist(bsMins[:,1], cumulative=True, histtype='step', bins=count)
# plt.show()

#%%
# plt.hist(bsMaxs[:,0], cumulative=True, histtype='step', bins=count)
# plt.hist(bsMaxs[:,1], cumulative=True, histtype='step', bins=count)
# plt.show()


#%%
# f, ax = plt.subplots(2)
# ax[0].set_aspect('equal')
# ax[1].set_aspect('equal')

# ax[0].hist2d(bsMins[:,0], bsMins[:,1], bins=[52,60], range=[[0,52], [0,60]])
# ax[1].hist2d(bsMaxs[:,0], bsMaxs[:,1], bins=[52,60], range=[[0,52], [0,60]])

# plt.show()

#%%
rescaledDiff = (sumCentHist/np.max(sumCentHist))-(sumPeriHist/np.max(sumPeriHist))


#%%
f, ax = plt.subplots(1,5)
ax[0].imshow(sumCentHist.T)
ax[1].imshow(sumPeriHist.T)
ax[2].imshow(sumCentHistRescaled.T)
ax[3].imshow(sumPeriHistRescaled.T)
ax[4].imshow(rescaledDiff.T)
# print(np.unravel_index(np.argmin(rescaledDiff), rescaledDiff.shape), np.unravel_index(np.argmax(rescaledDiff), rescaledDiff.shape))
# plt.show()

#%%
# f, ax = plt.subplots(1,5)
# ax[0].imshow(sumCentHist.T)
# ax[1].imshow(sumPeriHist.T)
# ax[2].imshow(sumCentHistRescaled.T)
# ax[3].imshow(sumPeriHistRescaled.T)
# rescaledDiff = (sumCentHistRescaled/np.max(sumCentHistRescaled))-(sumPeriHistRescaled/np.max(sumPeriHistRescaled))
# rescaledDiff = (sumPeriHistRescaled/np.max(sumPeriHistRescaled))-(sumCentHistRescaled/np.max(sumCentHistRescaled))
# ax[4].imshow(rescaledDiff.T)
# plt.show()

#%%

print(np.unravel_index(np.argmin(rescaledDiff), rescaledDiff.shape), np.unravel_index(np.argmax(rescaledDiff), rescaledDiff.shape))
print(center_of_mass(rescaledDiff*(rescaledDiff>0.01)), center_of_mass(rescaledDiff*(rescaledDiff<0.01)))

#%%

f, ax = plt.subplots(1,6, frameon=False)
f.suptitle("Spike Pair Angles vs Distances")
ax[0].imshow(sumCentHist.T)
ax[0].set_title("All Central Cells")
ax[1].imshow(sumPeriHist.T)
ax[1].set_title("All Peripheral Cells")
ax[2].imshow(rescaledDiff.T)
ax[2].set_title("Peripheral - Central")

# rescaledDiff = rescaledDiff/np.median(rescaledDiff)

# rescaledDiff/=np.median(rescaledDiff)
zscoredDiff = (rescaledDiff-np.mean(rescaledDiff))/np.std(rescaledDiff)
ax[3].imshow(zscoredDiff.T)
f.add_subplot(111, frameon=False)
ax[4].imshow(stats.norm.sf(abs(zscoredDiff.T))<0.05)
zScorePValPlot = (stats.norm.sf(abs(zscoredDiff.T))<0.005)*zscoredDiff.T
# ax[3].imshow(zScorePValPlot)
print(center_of_mass(zScorePValPlot*(zScorePValPlot>0)))
print(center_of_mass(zScorePValPlot*(zScorePValPlot<0)))

def makeGausKernel(size, sigma):
    #generate the gaussian kernel, based on fspecial from MATLAB
    size = (size-1)/2
    x, y = np.mgrid[-size:(size+1), -size:(size+1)]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/g.sum()

gausKernel = makeGausKernel(5,1)

smoothedDiff = signal.convolve2d(rescaledDiff, gausKernel, mode='same', boundary='symm')
ax[5].imshow(smoothedDiff.T)

print(center_of_mass(smoothedDiff*(smoothedDiff>0)))
print(center_of_mass(smoothedDiff*(smoothedDiff<0)))


# neg = np.ma.MaskedArray(rescaledDiff, rescaledDiff<0, fill_vale=np.nan)
# pos = np.ma.MaskedArray(rescaledDiff, rescaledDiff>0, fill_vale=np.nan)


plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.ylabel("Relative Spike Angle")
plt.xlabel("Relative Spike Distance")
plt.tight_layout()



# plt.show()

                        #%%

f, ax = plt.subplots()
im = ax.imshow(rescaledDiff.T, interpolation=None)
ax.set_aspect('equal')
ax.set_yticks([0,10,20,30,40,50, 59.1])
ax.set_yticklabels(["-30","-20","-10","0","10","20", "30"])
savetime = time.time()
np.savez(r"S:\NewAnalysisOutput\spikePairAnalysisOutput\shuffle_{}.npz".format(savetime), rescaledDiff = rescaledDiff, zScorePValPlot = zScorePValPlot)
# print([[np.unravel_index(np.argmin(rescaledDiff), rescaledDiff.shape), np.unravel_index(np.argmax(rescaledDiff), rescaledDiff.shape)], [center_of_mass(rescaledDiff*(rescaledDiff>0.01)), center_of_mass(rescaledDiff*(rescaledDiff<0.01))], [center_of_mass(zScorePValPlot*(zScorePValPlot>0)), center_of_mass(zScorePValPlot*(zScorePValPlot<0))]])
# json.dump([[np.unravel_index(np.argmin(rescaledDiff), rescaledDiff.shape), np.unravel_index(np.argmax(rescaledDiff), rescaledDiff.shape)], [center_of_mass(rescaledDiff*(rescaledDiff>0.01)), center_of_mass(rescaledDiff*(rescaledDiff<0.01))], [center_of_mass(zScorePValPlot*(zScorePValPlot>0)), center_of_mass(zScorePValPlot*(zScorePValPlot<0))]], open(r"S:\NewAnalysisOutput\spikePairAnalysisOutput\shuffle_{}.json".format(savetime), 'w'))
f.savefig(r"S:\NewAnalysisOutput\spikePairAnalysisOutput\shuffle_{}.png".format(savetime))
# plt.show()
