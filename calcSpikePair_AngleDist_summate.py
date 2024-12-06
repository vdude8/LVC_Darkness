# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 03:48:40 2022

@author: vyash
"""

import matplotlib.pyplot as plt
import numpy as np
import glob
import sys

import analysisParms as clp


import pandas as pd
import seaborn as sns
import analysisParms as parms
import statsmodels.api as sm
import statsmodels.formula.api as smf
import random

from multiprocessing import Pool

from tqdm import tqdm

from scipy import stats
import re

# rat = "302"
# day = sys.argv[1]
# day = "10"
# mazes = ['m1', 'm2', 'm3', 'm4']

def calcRegionRatio(img, dl,dh,al,ah):
    inSize = (dh-dl)*(ah-al)
    inside = np.sum(img[dl:dh, al:ah])
    outside = np.sum(img[:dl])+np.sum(img[dh:])+np.sum(img[dl:dh, :al])+np.sum(img[dl:dh, ah:])
    inRatio = inside/inSize
    outRatio = outside/(52*60-inSize)
    return (inRatio-outRatio)/(inRatio+outRatio)

def calcNearness(angles, dists, targAng, targDist):
    a = angles-targAng
    d = dists-targDist
    
    angles = np.deg2rad(angles*6)
    targAng = np.deg2rad(targAng*6)
    # diffY = (np.sin(angles)*dists)-(np.sin(targAng)*targDist)
    # diffX = (np.cos(angles)*dists)-(np.cos(targAng)*targDist)

    meanAng = np.arctan2(np.sum(np.sin(angles)), np.sum(np.cos(angles)))-targAng
    meanDist = np.mean(dists-targDist)
    return np.sqrt(np.square(np.sin(meanAng)*meanDist)+np.square(np.cos(meanAng)*meanDist))

    
    
    #changed into a pixel distance instead of the old deviation distance
    # a = np.deg2rad(a)
    # allDists = np.sqrt(np.square(np.sin(a)*d)+np.square(np.cos(a)*d))
    # allDists = np.sqrt(diffX**2+diffY**2)
    
    # allDists = np.sqrt(np.square(a)+np.square(d))
    return np.mean(allDists)

#%%

#20-40, 12-18

allData = []

# filenames = glob.glob("S:\\NewAnalysisOutput 10-06-2020\\*-*_t*_c*_m*_spikePairHist.npz")
# origClustFilenames = glob.glob("S:\\NewAnalysisOutput 6-27-16\\*\\*\\*-*_t*_c*.npz")

filenames = glob.glob("S:\\NewAnalysisOutput\\spikePairAnalysisOutput\\*-*_t*_c*_m*_spikePairHist.npz")
filenames = [x for x in filenames if "S:\\NewAnalysisOutput\\spikePairAnalysisOutput\\00" not in filenames]
origClustFilenames = glob.glob("S:\\NewAnalysisOutput\\*\\*\\*-*_t*_c*.npz")

origClustFilenameSplit = [x.split("\\")[-1] for x in origClustFilenames]

def runProcessing(filename):
    # print(filename)
    base = filename.split("\\")[-1]
    ratday = base[:6]
    rat = ratday[:3]
    day = ratday[-2:]


    sessOrder = parms.dayList[int(rat)].index(int(day))
    tet = base.split("_")[1][1:]
    clust = base.split("_")[2][1:]
    maze = base.split("_")[3][1:]
    try:
        curFile = np.load(filename)
        rating = np.load(origClustFilenames[origClustFilenameSplit.index(base.split("_m")[0]+".npz")])['rating'].flatten()[0]
        curFile['hist'] #test to see if file is okay
        curFile['angles'] #test to see if file is okay
        curFile['dists'] #test to see if file is okay
    except KeyError:
        print("Missing components in file:", filename)
        return
    except EOFError:
        print("Corrupt File:", filename)
        return
    except:
        print("Unknown error:", filename)
        return
    regionRatio = calcRegionRatio(curFile['hist'], 12,15,20,40)
    
    # nearnessCentral = calcNearness(curFile['angles'], curFile['dists'], 30, 13.5)
    # nearnessCentral = calcNearness(curFile['angles'], curFile['dists'], 30, 13.786966269526477)
    # nearnessPeripheral = calcNearness(curFile['angles'], curFile['dists'], 30, 36.1842024702429)
    # nearnessControl = calcNearness(curFile['angles'], curFile['dists'], 30, 24.985584369884688)
    nearnessCentral = calcNearness(curFile['angles'], curFile['dists'], 30, 119.09964008772999)
    nearnessPeripheral = calcNearness(curFile['angles'], curFile['dists'], 30, 314.7906738535524)
    nearnessControl = calcNearness(curFile['angles'], curFile['dists'], 30, 216.9451569706412)
    return [filename, ratday, tet, clust, maze, sessOrder, parms.modeList[int(rat)][sessOrder], parms.ca1pos[int(rat)][int(tet)-1], rating, regionRatio, nearnessCentral, nearnessPeripheral, nearnessControl]


allData = []

if __name__ == "__main__":
    pool = Pool(24)
    allData = pool.map(runProcessing, filenames)
    allData = [x for x in allData if x is not None]
    pool.close()


    #%%
    
    # for i in range(6):
    #     plt.hist(allRatios[i], histtype='step', label=days[i], density=True, stacked=True)
    # plt.violinplot(allRatios)
    
    # sns.scatterplot(np.arange(6), allRatios)
    df = pd.DataFrame(allData, columns = ["Filename", "RatDay", "Tetrode", "Cluster", "Maze", "SessionOrder", "IsCentralSess", "CA1Pos", "rating", "SpikePairInfieldRatio", "Nearness", "NearnessPeripheral", "NearnessControl"])
    df = df[(df["SessionOrder"]+1)%4!=0]
    df = df[df["Maze"]=="1"]
    df['RatDayInt'] = [int(x[:3]+x[-2:]) for x in df['RatDay']]
    df['Rat'] = [x[:3] for x in df['RatDay']]
    df = df[df["Rat"]!=580]
    df.sort_values(by='RatDayInt')
    df = df[df["CA1Pos"]!=-1]
    df = df[df["rating"]>0]
    df["CA1Pos"] = df["CA1Pos"].astype('category')
    df["IsCentralSess"] = df["IsCentralSess"].astype('category')
    df["ShuffledIsCentralSess"] = df["IsCentralSess"]
    
    # df.to_pickle("S:\\NewAnalysisOutput 10-06-2020\\df.pkl")
    df.to_pickle("S:\\NewAnalysisOutput\\spikePairAnalysisOutput\\df.pkl")
    
    sys.exit()
    
    
    
    #%%
    
    # df = pd.read_pickle("S:\\NewAnalysisOutput 10-06-2020\\df.pkl")
    df = pd.read_pickle("S:\\NewAnalysisOutput\\spikePairAnalysisOutput\\df.pkl")
    # df = pd.read_pickle("S:\\NewAnalysisOutput\\df_thresh0.2.pkl")
    
    #%%
    
    df2 = df.groupby(['Rat', 'Tetrode', 'RatDayInt', 'IsCentralSess']).size().reset_index()
    df2 = df2[df2[0]>0]
    df2 = df2.pivot(index=['RatDayInt', 'Rat', 'IsCentralSess'], columns='Tetrode', values=0)
    df3 = df2.groupby(['Rat', 'IsCentralSess']).idxmax().reset_index()
    df3 = pd.melt(df3, id_vars=['Rat', "IsCentralSess"], var_name="Tetrode", value_name="maxIdx")
    df3 = df3.dropna()
    df3["maxRatDay"] = [x[0] for x in df3["maxIdx"]]
    
    df = df.query(' or '.join(["(IsCentralSess == {} and Tetrode == '{}' and RatDayInt == {})".format(x[1], x[2], x[-1]) for x in df3.to_numpy()]))
    df.to_pickle("S:\\NewAnalysisOutput\\spikePairAnalysisOutput\\df_tetrode_filtered.pkl")
    

    #%%

    df = pd.read_pickle("S:\\NewAnalysisOutput\\spikePairAnalysisOutput\\df.pkl")    
    df = df.query("SessionOrder==0 or SessionOrder==4")
    df.to_pickle("S:\\NewAnalysisOutput\\spikePairAnalysisOutput\\df_session_filtered.pkl")
    
    # sys.exit()
    
    #%%
    
    
    df = pd.read_pickle("S:\\NewAnalysisOutput\\spikePairAnalysisOutput\\df_tetrode_filtered.pkl")    
    # df = pd.read_pickle("S:\\NewAnalysisOutput\\spikePairAnalysisOutput\\df_session_filtered.pkl")    
    # df = pd.read_pickle(r"\\vast.rockfish.jhu.edu\mbi-knierimlab\vyash\Darkrat\NewAnalysisOutput_SpikePairAnalyses\spikePairAnalysisOutput_thresh0.5_3regions\df_tetrode_filtered.pkl")    
    
    df["IsCentralSess"] = df["IsCentralSess"].astype('int')
    df["IsCentralSess"] = df["IsCentralSess"]*-1
    df["IsCentralSess"] = df["IsCentralSess"].astype('category')
    
    # df = df.reset_index()
    
    # df['Nearness'] = stats.zscore(df['Nearness'])
    # df['NearnessPeripheral'] = stats.zscore(df['NearnessPeripheral'])
    # df['NearnessControl'] = stats.zscore(df['NearnessControl'])
    
    meltedDf = pd.melt(df, id_vars="IsCentralSess", value_vars=['Nearness', 'NearnessPeripheral', 'NearnessControl'], var_name="NearnessType", value_name="NearnessVal")
    
    # df.melt(value_vars=['Nearness', 'NearnessPeripheral', 'NearnessControl'], var_name="NearnessType", value_name="NearnessVal")
    
    #%%
    
    sns.set(style="white", color_codes=True, font_scale=1)
    # graph = sns.stripplot(x='IsCentralSess', y='SpikePairInfieldRatio', data=df)
    graph = sns.stripplot(x='IsCentralSess', y='Nearness', data=df)
    # graph = sns.stripplot(x='IsCentralSess', y='NearnessPeripheral', data=df)
    # graph = sns.stripplot(x='IsCentralSess', y='Nearness', data=df)
    graph.axhline(0, color='k', ls='--', alpha=0.5)
    # sns.stripplot(x='RatDay', y='SpikePairInfieldRatio', data=df, hue="IsCentralSess")
    sns.despine()
    
    
    f, ax = plt.subplots()
    sns.swarmplot(x='NearnessType', y='NearnessVal', hue='IsCentralSess', data=meltedDf, ax=ax, dodge=True)
    ax.set_ylim(-5, 5)
    
    f, ax = plt.subplots()
    sns.boxplot(x='NearnessType', y='NearnessVal', hue='IsCentralSess', data=meltedDf, ax=ax, dodge=True, flierprops={"marker": "o"},
    )
    # ax.set_ylim(-4.1, 4.1)
    
    
    
    
    # sns.violinplot(x='Rat', y='SpikePairInfieldRatio', data=df, split="IsCentralSess", hue="IsCentralSess")
    # sns.violinplot(x='IsCentralSess', y='SpikePairInfieldRatio', data=df)
    f, ax = plt.subplots()
    sns.violinplot(x='NearnessType', y='NearnessVal', hue='IsCentralSess', data=meltedDf, ax=ax, split=True, inner=None)
    ax.set_ylim(0, 350)
    f, ax = plt.subplots()
    sns.violinplot(x='IsCentralSess', y='Nearness', data=df, ax=ax, inner="point")
    ax.set_ylim(0, 350)
    f, ax = plt.subplots()
    sns.violinplot(x='IsCentralSess', y='NearnessPeripheral', data=df, ax=ax, inner="point")
    ax.set_ylim(0, 350)
    f, ax = plt.subplots()
    sns.violinplot(x='IsCentralSess', y='NearnessControl', data=df, ax=ax, inner="point")
    ax.set_ylim(0, 350)
    
    print(stats.mannwhitneyu(df["Nearness"][df["IsCentralSess"]==-1], df["Nearness"][df["IsCentralSess"]==0], alternative="less"))
    print(stats.mannwhitneyu(df["NearnessPeripheral"][df["IsCentralSess"]==-1], df["NearnessPeripheral"][df["IsCentralSess"]==0], alternative="greater"))
    print(stats.mannwhitneyu(df["NearnessControl"][df["IsCentralSess"]==-1], df["NearnessControl"][df["IsCentralSess"]==0], alternative="two-sided"))
        
    print(stats.mannwhitneyu(df["Nearness"][df["IsCentralSess"]==-1][df["RatDayInt"]==0], df["Nearness"][df["IsCentralSess"]==0][df["RatDayInt"]==0], alternative="less"))
    print(stats.mannwhitneyu(df["NearnessPeripheral"][df["IsCentralSess"]==-1][df["RatDayInt"]==0], df["NearnessPeripheral"][df["IsCentralSess"]==0][df["RatDayInt"]==0], alternative="greater"))
    print(stats.mannwhitneyu(df["NearnessControl"][df["IsCentralSess"]==-1][df["RatDayInt"]==0], df["NearnessControl"][df["IsCentralSess"]==0][df["RatDayInt"]==0], alternative="two-sided"))
    
    
    print(np.median(df["Nearness"][df["IsCentralSess"]==-1]))
    print(np.median(df["Nearness"][df["IsCentralSess"]==0]))
    print(np.median(df["NearnessPeripheral"][df["IsCentralSess"]==-1]))
    print(np.median(df["NearnessPeripheral"][df["IsCentralSess"]==0]))
    print(np.median(df["NearnessControl"][df["IsCentralSess"]==-1]))
    print(np.median(df["NearnessControl"][df["IsCentralSess"]==0]))
    
    print(df["Nearness"][df["IsCentralSess"]==-1].count())
    print(df["Nearness"][df["IsCentralSess"]==-0].count())
    
    
    sns.histplot(data=df, x="SpikePairInfieldRatio", hue="IsCentralSess", element='step', fill=False, cumulative=True, stat="density", common_norm=(False), bins=10000)
    sns.histplot(data=df, x="Nearness", hue="IsCentralSess", element='step', fill=False, cumulative=True, stat="density", common_norm=(False), bins=10000)
    
    ratList = np.unique(df["Rat"])
    f, ax = plt.subplots(len(ratList))
    for ratIdx in range(len(ratList)):
        tempDF = df[df["Rat"]==ratList[ratIdx]]
        sns.histplot(ax=ax[ratIdx], data=tempDF, x="SpikePairInfieldRatio", hue="IsCentralSess", element='step', fill=False, cumulative=True, stat="density", common_norm=(False), bins=10000)
        print(stats.mannwhitneyu(tempDF["SpikePairInfieldRatio"][tempDF["IsCentralSess"]==1], tempDF["SpikePairInfieldRatio"][tempDF["IsCentralSess"]==0], alternative="greater"))
    
    
    # for day in days:
    #     subset = df[df["Day"]==day]
    #     print(day, np.sum(subset["SpikePairInfieldRatio"]>.5)/len(subset))
    
    # md = smf.mixedlm("SpikePairInfieldRatio ~ IsCentralSess + CA1Pos", df, groups=df["Rat"])
    md = smf.mixedlm("SpikePairInfieldRatio ~ IsCentralSess", df, groups=df["Maze"])
    # md = smf.mixedlm("IsCentralSess ~ SessionOrder", df, groups=df["Rat"])
    mdf = md.fit()
    print(mdf.summary())
    
    # vc = {'RatDay': '0 + C(RatDay)', }
    md = smf.mixedlm("Nearness ~ IsCentralSess", df, groups=df["Rat"])
    md = smf.mixedlm("Nearness ~ IsCentralSess", df, groups=df["Rat"])
    # md = smf.mixedlm("IsCentralSess ~ SessionOrder", df, groups=df["Rat"])
    mdf = md.fit()
    print(mdf.summary())
    
    
    for i in range(5):
        tempdf = df.copy()
        print(i+1)
        df = df[df['Maze']==str(i+1)]
        
        print(stats.mannwhitneyu(df["SpikePairInfieldRatio"][df["IsCentralSess"]==1], df["SpikePairInfieldRatio"][df["IsCentralSess"]==0], alternative="greater"))
        print(stats.mannwhitneyu(df["Nearness"][df["IsCentralSess"]==-1], df["Nearness"][df["IsCentralSess"]==0], alternative="less"))
        
        print(np.median(df["Nearness"][df["IsCentralSess"]==-1]), df["Nearness"][df["IsCentralSess"]==-1].shape)
        print(np.median(df["Nearness"][df["IsCentralSess"]==0]), df["Nearness"][df["IsCentralSess"]==0].shape)
        print('---')
        df = tempdf.copy()
    
    # sns.barplot(x="RatDay", y="SpikePairInfieldRatio", data=df, estimator=np.mean, ci="sd", capsize=.2, hue="IsCentralSess")
    
    #%%
    
    dfM1 = df[df["Maze"]=="1"]
    
    model = smf.ols('Nearness ~ C(IsCentralSess) + C(CA1Pos) + C(IsCentralSess):C(CA1Pos)', data=dfM1).fit()
    sm.stats.anova_lm(model, typ=2)
    
    dfCS = df[df["IsCentralSess"]==1]
    dfCS2 = df[df["IsCentralSess"]==0]
    dfCS2["CA1Pos"] = dfCS2["CA1Pos"].astype('int')
    dfCS2["CA1Pos"] = dfCS2["CA1Pos"]+3
    
    sns.set(style="white", color_codes=True, font_scale=1)
    # sns.histplot(data=df, x="Nearness", hue="IsCentralSess", element='step', fill=False, cumulative=True, stat="density", common_norm=(False), bins=10000)
    sns.histplot(data=dfCS, x="Nearness", hue="CA1Pos", element='step', fill=False, cumulative=True, stat="density", common_norm=(False), bins=10000)
    sns.histplot(data=dfCS2, x="Nearness", hue="CA1Pos", element='step', fill=False, cumulative=True, stat="density", common_norm=(False), bins=10000)
    # graph = sns.stripplot(x='CA1Pos', y='Nearness', data=df[df["IsCentralSess"]==1])
    # graph = sns.stripplot(x='CA1Pos', y='Nearness', data=df[df["IsCentralSess"]==0])
    sns.despine()
    
    
    #%%
    numShuffs = 1000
    
    realDiff = np.diff(df.groupby('IsCentralSess')['Nearness'].mean())[0]
    isCentralSessList = df["IsCentralSess"].tolist()
    
    shuffDiffs = np.empty(numShuffs)
    for i in range(numShuffs):
        df["ShuffledIsCentralSess"] = random.sample(isCentralSessList, k=len(df))
        shuffDiffs[i] = np.diff(df.groupby('ShuffledIsCentralSess')['Nearness'].mean())[0]
    
    f, ax = plt.subplots()
    ax.hist(shuffDiffs)
    ax.axvline(x=realDiff)
    
    #%%
    numShuffs = 1000
    
    realDiff = np.diff(df.groupby('IsCentralSess')['SpikePairInfieldRatio'].mean())[0]
    isCentralSessList = df["SpikePairInfieldRatio"].tolist()
    
    shuffDiffs = np.empty(numShuffs)
    for i in range(numShuffs):
        df["ShuffledIsCentralSess"] = random.sample(isCentralSessList, k=len(df))
        shuffDiffs[i] = np.diff(df.groupby('ShuffledIsCentralSess')['SpikePairInfieldRatio'].mean())[0]
    
    f, ax = plt.subplots()
    ax.hist(shuffDiffs)
    ax.axvline(x=realDiff)
    
    
    #%%
    
    ## for calculating the "ideal" distance between objects
    
    
    # occ1 = np.load(r"S:\NewAnalysisOutput\302\10\302-10_occLists.npz", allow_pickle=True)
    occ1 = np.load(r"S:\NewAnalysisOutput\302\06\302-06_occLists.npz", allow_pickle=True)
    # occ1 = np.load(r"S:\NewAnalysisOutput\305\12\305-12_occLists.npz", allow_pickle=True)
    
    occFilelist = glob.glob(r"S:\NewAnalysisOutput\*\*\*-*_occLists.npz")
    allDists = []
    for occFilename in occFilelist:
        occ1 = np.load(occFilename, allow_pickle=True)
        # for objSetName in ['raw', 'corrected', 'rebinned']:
        for objSetName in ['corrected']:
            objSet = np.array([x[:2] for x in occ1['stdObj'].flatten()[0][objSetName]])
            objSet = np.vstack((objSet, [objSet[0]]))
            dists = np.sqrt(np.sum(np.square(np.diff(objSet, axis=0)), axis=1))
            # print(dists)
            print(occFilename, np.mean(dists))
            allDists.append(np.mean(dists))
            
    allDists = np.array(allDists)
    plt.hist(allDists, bins=100)
    plt.show()