# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 13:20:40 2019

@author: vyash
"""

#calcSpikeAngleDist


import analysisParms as parms
import numpy as np
import glob
import scipy.spatial.distance as dist
import scipy.interpolate as interpol
import matplotlib.pyplot as plt
import sys
import traceback

import json

from scipy.ndimage.measurements import label

from multiprocessing import Pool
import os

np.seterr(all="ignore")

inPath = "S:\\NewAnalysisOutput\\"
outPathAddition = "SpikePairAnalysisOutput_0.3thresh_2regions\\"
outPath = inPath + outPathAddition

filenames = glob.glob(inPath+"*\\*\\*-*_t*_c*.npz")
# filenames = glob.glob(inPath+"302\\10\\*-*_t*_c*.npz")
filenames = [x for x in filenames if '\\580\\' not in x] #remove rat 580
# print(filenames)
mazes = ['m1', 'm2', 'm3', 'm4', 'm5'] #do all mazes, because why not
#mazes = ['m2', 'm3', 'm4']
#mazes = ['m4']

def getSpikePairsBetweenFields(spikeSets, name):
    if len(spikeSets)<2:
        # print(name, ' has less than 3 fields')
        return
    fieldSizes = []
    for field in spikeSets:
        fieldSizes.append(len(field))
    # if max(fieldSizes)>20000:
    #     return
    
#    spikes = spikes[1000:1010]
#    dists = dist.pdist(spikes)
#    angles = dist.pdist(spikes, metric='cosine')
    # numPairs = int((len(spikes)*(len(spikes)-1))/2)

    numPairs = 0    
    for fieldAIdx in range(len(spikeSets)):
        for fieldBIdx in range(fieldAIdx+1, len(spikeSets)):

            numPairs += fieldSizes[fieldAIdx]*fieldSizes[fieldBIdx]
            
    
    angles = np.zeros(numPairs)
    dists = np.zeros(numPairs)
    curIdx = 0;
    for fieldAIdx in range(len(spikeSets)):
        for fieldBIdx in range(fieldAIdx+1, len(spikeSets)):
            fieldA = spikeSets[fieldAIdx]
            fieldB = spikeSets[fieldBIdx]
            # print(np.array(np.meshgrid(fieldA[:,0], fieldB[:,0])).dtype)
            # sys.exit()
            xpair = np.array(np.meshgrid(fieldA[:,0], fieldB[:,0])).T.reshape((-1,2))
            ypair = np.array(np.meshgrid(fieldA[:,1], fieldB[:,1])).T.reshape((-1,2))
            xdiff = xpair[:,1]-xpair[:,0]
            ydiff = ypair[:,1]-ypair[:,0]
            curNumComparisons = fieldSizes[fieldAIdx]*fieldSizes[fieldBIdx]
            # print(len(fieldA), len(fieldB), fieldSizes)
            angles[curIdx:curIdx+curNumComparisons] = np.arctan2(xdiff, ydiff)
            dists[curIdx:curIdx+curNumComparisons] = np.sqrt(xdiff**2+ydiff**2)
            curIdx+=curNumComparisons

    # sys.exit()

            
#    print(spikes)
#    print(dists)
#    print(angles)
    # angles = np.zeros(numPairs)
    # dists = np.zeros(numPairs)
    # curIdx = 0;
    # for i in range(len(spikes)-1):

    #     curSpikes = spikes[i+1:]

    #     diff = curSpikes-spikes[i]

    #     angles[curIdx:curIdx+len(curSpikes)] = np.arctan2(diff[:,0],diff[:,1])
    #     dists[curIdx:curIdx+len(curSpikes)] = np.sqrt(diff[:,0]**2+diff[:,1]**2)
    #     curIdx+=len(curSpikes)
        
    angles = angles*180/np.pi
    angles[angles<0] = angles[angles<0]+180
    angles = np.mod(angles, 60)
    # print(name)

    # plt.figure()
    hist, xedges, yedges, img = plt.hist2d(dists, angles, bins=[52,60], range=[[0,520], [0, 60]])    
    # plt.savefig(name+".png")
    # plt.figure()
    # plt.imshow(hist.T)
    # # plt.show()
    # plt.close('all')
    np.savez(name, hist=hist, xedges=xedges, yedges=yedges, img=img, dists=dists, angles=angles)
    return hist
    
def resampleSpikes(gausMap, spikes):
    hist, biny, binx = np.histogram2d(spikes[:,1], spikes[:,0], bins=(52,52), range=[[-20, 499],[60, 579]])

    thresh = np.nanmax(gausMap)*.3 #changed to 0.5 to see if that helps improve field deliniation
    binaryGaus = gausMap>thresh
    fieldPlot = np.zeros_like(gausMap)
    fieldPlot[binaryGaus==0] = -1

    labelledMap, numFields = label(binaryGaus)
    labelledMapFiltered = labelledMap.copy()

    for i in range(1, numFields+1):
        # print(i, np.sum(labelledMap==i))
        if np.sum(labelledMap==i)<4: #minimum field size; keeping this small to avoid removing important spikes
            labelledMapFiltered[labelledMap==i] = 0
            labelledMapFiltered[labelledMap>i] -= 1
            numFields-=1
            
    labelledMap = labelledMapFiltered
        
    # plt.imshow(labelledMap)
    # plt.show()
       
    # f, ax = plt.subplots(1, numFields+3)
    # ax[0].imshow(gausMap)
    # ax[1].imshow(labelledMap)
    # print(gausMap.shape, labelledMap.shape, numFields)
    for i in range(numFields+1):
        # print(i)
        # print(gausMap[labelledMap==i])
        curField = gausMap.copy()
        curField[labelledMap!=i]=0
        # ax[i+2].imshow(curField.reshape((52,52)))

    # plt.show()
    
    # sys.exit()

    spikex = np.digitize(spikes[:,0], binx)
    spikey = np.digitize(spikes[:,1], biny)
    
    #if spikes are attributed to positions outside of the apparatus, remove them
    maxBinX = len(binx)-2
    maxBinY = len(biny)-2
    badSpikePos = np.concatenate((np.argwhere(spikex<0), np.argwhere(spikey<0), np.argwhere(spikex>maxBinX), np.argwhere(spikey>maxBinY))).flatten()
    spikex = np.delete(spikex, badSpikePos)
    spikey = np.delete(spikey, badSpikePos)
    
    spikeWeights = np.nan_to_num(np.array([gausMap[spikey[i],spikex[i]] for i in range(len(spikex))]))

#    print(spikeWeights)

#    print(len(spikes))

    # samples = np.random.choice(len(spikes), size=len(spikes), p=(spikeWeights/np.sum(spikeWeights))) #same number of spikes as originally present
    if np.sum(np.isnan(spikeWeights/np.sum(spikeWeights)))>0:
        print("THIS ONE WILL ERROR")
        print(np.sum(gausMap))
        print(np.sum(spikeWeights))
        return None
    samples = np.random.choice(len(spikex), size=len(spikex), p=(spikeWeights/np.sum(spikeWeights))) #same number of spikes as originally present
#    print(samples)
    sampledSpikes = spikes[samples]

#    plt.figure()
#    hist, biny, binx = np.histogram2d(sampledSpikes[:,1], sampledSpikes[:,0], bins=(52,52), range=[[-20, 499],[60, 579]])
#    plt.imshow(hist)
#    plt.show()
#    sys.exit()
#    samples = np.random.choice(len(spikes), p=(gausMap/np.sum(spikeWeights))) #same number of spikes as originally present 

    spikeSets = [[] for i in range(numFields+1)]
    sampledX = np.digitize(sampledSpikes[:,0], binx)
    sampledY = np.digitize(sampledSpikes[:,1], biny)
    sampledSpikes = sampledSpikes.tolist()

    #if spikes are attributed to positions outside of the apparatus, remove them
    badSpikePos = np.concatenate((np.argwhere(sampledX<0), np.argwhere(sampledY<0), np.argwhere(sampledX>maxBinX), np.argwhere(sampledY>maxBinY))).flatten()
    sampledX = np.delete(sampledX, badSpikePos)
    sampledY = np.delete(sampledY, badSpikePos)

    for spikeIdx in range(len(sampledX)):
        # print(spikeIdx, labelledMap[sampledY[spikeIdx], sampledX[spikeIdx]])
        spikeSets[labelledMap[sampledY[spikeIdx], sampledX[spikeIdx]]].append(sampledSpikes[spikeIdx])

    spikeSets = [np.array(x) for x in spikeSets if len(x)>0]
    # print(len(spikeSets))
    
    # print(len(spikeSets))
    # for i in spikeSets:
    #     print(len(i))
        
    # sys.exit()
    
    return np.array(spikeSets)
    

def startProcess(filename):
    # print(filename)
    mainName = filename.split("\\")[-1]
    rat = mainName[:3]
    day = mainName[4:6]
    tet = mainName.split("_")[1][1:]
    clust = mainName.split("_")[2][1:-4]

    
    #check if the file has already been processed
    unprocessedMazes = []
    for maze in mazes:

        jsonPath = outPath+rat+"-"+day+"_t"+tet+"_c"+clust+"_"+maze+"_resampledSpikeDump.json"
        if not os.path.exists(jsonPath):
            unprocessedMazes.append(maze)

        filepath = outPath+rat+"-"+day+"_t"+tet+"_c"+clust+"_"+maze+"_spikePairHist.npz"
        if not os.path.exists(filepath):
            unprocessedMazes.append(maze)
        else:
            try:
                # print('testing file ', filename)
                curFile = np.load(filepath)
                curFile['hist'] #test to see if file is okay
                curFile['angles'] #test to see if file is okay
                curFile['dists'] #test to see if file is okay
            except KeyError:
                print("Missing components in file:", filename)
                unprocessedMazes.append(maze)
            except Exception as e:
                try:
                    os.remove(filepath)
                    os.remove(jsonPath)
                except:
                    print("one of the output files is missing and the other is corrupt, will rerun")
                print("********")
                print('failed to do this file:', filename)
                print(e)
                print('deleted output file and will try to rerun')
                print("********")
                unprocessedMazes.append(maze)
                

    if len(unprocessedMazes)<=0:
        return
    # print(filename, unprocessedMazes)

    occFilename = '\\'.join(filename.split("\\")[:-1])+"\\"+rat+"-"+day+"_occLists.npz"
#    varX = curOcc[varX]
#    varY = curOcc[varY]
#    curOccList = 

    curFile = np.load(filename, allow_pickle=True)
    
    unprocessedMazes2 = []
    for maze in unprocessedMazes:
        if maze in list(curFile.keys()):
            unprocessedMazes2.append(maze)
    if len(unprocessedMazes2)<=0:
        return
    # print(unprocessedMazes2)


    curOcc = np.load(occFilename, allow_pickle=True)

    for maze in unprocessedMazes:
        # print(maze, list(curFile.keys()))
        if maze in list(curFile.keys()):
            # print("Processing:", rat, day, tet, clust, maze)
                
            curMaze = curFile[maze].flatten()[0]
            curSpikeList = curMaze['spikeList'][:,:2]
            if(len(curSpikeList)>20000):
                continue;
#            curSpikeList = curOcc[maze]['spikelist'][:,1:]
            
            try:
                curSpikeSets = resampleSpikes(curMaze['gausMap'], curSpikeList)
                if curSpikeSets is None:
                    continue
                curHist = getSpikePairsBetweenFields(curSpikeSets, outPath+rat+"-"+day+"_t"+tet+"_c"+clust+"_"+maze+"_spikePairHist.npz")
                if curHist is not None:
                    json.dump([x.tolist() for x in curSpikeSets], open(outPath+rat+"-"+day+"_t"+tet+"_c"+clust+"_"+maze+"_resampledSpikeDump.json", 'w'))
            except Exception as e:
                print("ERROR:", rat, day, tet, clust, maze)
                print(e)
                print(traceback.format_exc())
                print()
                continue

if __name__ == '__main__':
    if not os.path.exists(outPath):
        os.mkdir(outPath)
    pool = Pool(24)
    pool.map(startProcess, filenames)
    # map(startProcess, filenames)
    # for filename in filenames:
    #     startProcess(filename)