# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 18:18:50 2024

@author: local_user
"""

import glob
import numpy as np
import matplotlib.pyplot as plt

#%%

realPeripheral = (31, 30)
realCentral = (12, 30)

# actualPeripheral = (32, 33)
# actualCentral = (12, 27)

actualPeripheral = (31, 34)
actualCentral = (13, 32)


#%%

def calcVectorDiff(vector1, vector2):
    xVector1 = vector1[:,0]*np.cos(vector1[:,1]*np.pi/180)
    yVector1 = vector1[:,0]*np.sin(vector1[:,1]*np.pi/180)
    xVector2 = vector2[:,0]*np.cos(vector2[:,1]*np.pi/180)
    yVector2 = vector2[:,0]*np.sin(vector2[:,1]*np.pi/180)
    
    xDiff = xVector1-xVector2
    yDiff = yVector1-yVector2

    return np.sqrt(np.square(xDiff)+np.square(yDiff))

#%%

shuffs = glob.glob(r"S:\NewAnalysisOutput\spikePairAnalysisOutput\shuffle_*.*.npz")

locMin = []
locMax = []

for shuff in shuffs:
    rescaledDiff = np.load(shuff)['rescaledDiff']
    locMin.append(np.array(np.unravel_index(np.argmin(rescaledDiff), rescaledDiff.shape)))
    locMax.append(np.array(np.unravel_index(np.argmax(rescaledDiff), rescaledDiff.shape)))
    
locMax = np.array(locMax)
locMin = np.array(locMin)

#%%


peripehralDistances = np.sqrt(np.square(locMax[:,0]-realPeripheral[0])+np.square(locMax[:,1]-realPeripheral[1]))
centralDistances = np.sqrt(np.square(locMin[:,0]-realCentral[0])+np.square(locMin[:,1]-realCentral[1]))

realPeripheralDistance = np.sqrt(np.sum(np.square(np.array(actualPeripheral)-np.array(realPeripheral))))
realCentralDistance = np.sqrt(np.sum(np.square(np.array(actualCentral)-np.array(realCentral))))


#%%

# peripehralDistances = calcVectorDiff(locMax, np.array([realPeripheral]))
# centralDistances = calcVectorDiff(locMin, np.array([realCentral]))

# realPeripheralDistance = calcVectorDiff(np.array([actualPeripheral]), np.array([realPeripheral]))
# realCentralDistance = calcVectorDiff(np.array([actualCentral]), np.array([realCentral]))

    
#%%
f, ax = plt.subplots(1,3)


ax[0].hist(peripehralDistances, range=[0,40], bins=40)
ax[1].hist(centralDistances, range=[0,40], bins=40)

ax[0].axvline(x=realPeripheralDistance, color='r')
ax[1].axvline(x=realCentralDistance, color='r')

ax[2].plot(centralDistances, peripehralDistances, '.')
ax[2].axhline(realPeripheralDistance, color='r')
ax[2].axvline(realCentralDistance, color='r')



#%%

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

# Start with a square Figure.
fig = plt.figure(figsize=(6, 6))
# Add a gridspec with two rows and two columns and a ratio of 1 to 4 between
# the size of the marginal Axes and the main Axes in both directions.
# Also adjust the subplot parameters for a square plot.
gs = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
                      left=0.1, right=0.9, bottom=0.1, top=0.9,
                      wspace=0.05, hspace=0.05)
# Create the Axes.
ax = fig.add_subplot(gs[1, 0])
ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)

ax_histx.axis('off')
ax_histy.axis('off')

ax.plot(centralDistances, peripehralDistances, '.', color="#555555")
ax_histx.hist(centralDistances, range=[0,40], bins=40, color=colors[0])
ax_histy.hist(peripehralDistances, range=[0,40], bins=40, orientation='horizontal', color=colors[1])

ax.axhline(realPeripheralDistance, color='r')
ax.axvline(realCentralDistance, color='r')

ax_histy.axhline(y=realPeripheralDistance, color='r')
ax_histx.axvline(x=realCentralDistance, color='r')

ax.fill_between([0,realCentralDistance], [realPeripheralDistance,realPeripheralDistance], color='r', alpha=0.1)

ax.set_xlim([0, 40])
ax.set_ylim([0, 40])
ax_histx.set_xlim([0,40])
ax_histy.set_ylim([0,40])

ax.set_xlabel("Deviation to Central Geometry")
ax.set_ylabel("Deviation to Peripheral Geometry")

#%%

centLocDiff = locMax.copy()
centLocDiff[:,0]-=realCentral[0]
centLocDiff[:,1]-=realCentral[1]

periLocDiff = locMin.copy()
periLocDiff[:,0]-=realPeripheral[0]
periLocDiff[:,1]-=realPeripheral[1]

centRealDiff = np.abs(np.array(actualCentral)-np.array(realCentral))
periRealDiff = np.abs(np.array(actualPeripheral)-np.array(realPeripheral))

allDiff = np.zeros((len(centLocDiff), 4))
allDiff[:,:2] = centLocDiff
allDiff[:,-2:] = periLocDiff
allDiff -= list(centRealDiff)+list(periRealDiff)

diffCheck = np.sum(allDiff<0, axis=1)
sum(diffCheck==0)