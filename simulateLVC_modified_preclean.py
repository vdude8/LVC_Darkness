# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 14:07:48 2022

@author: vyash
"""

import sys
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QApplication, QWidget, QCheckBox, QSlider, QGridLayout, QLabel, QFrame, QPushButton

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

from scipy import stats
from scipy import signal

import analysisParms as parms

gausKernel = parms.makeGausKernel(5, 1)

#(y, x) notation
plotSize = (24,24)

def alterCmaps(cmapIdx, maxAlpha):
    cmap = np.zeros((256,4))
    cmap[:, cmapIdx] = np.linspace(0, 1, 256)
    cmap[:, -1] = np.linspace(0, maxAlpha, 256)
    return ListedColormap(cmap)

cmapsFull = [alterCmaps(x, 1) for x in range(3)]
cmapsOverlay = [alterCmaps(x, 0.33) for x in range(3)]

cmap = np.zeros((256,4))
cmap[:, 0] = np.linspace(1, 0, 256)
cmap[:, 1] = np.linspace(1, 0, 256)
cmap[:, 2] = np.linspace(0, 0, 256)
cmap[:, -1] = np.linspace(1, 0, 256)
cmapsOverlay.append(ListedColormap(cmap))
cmapsFull.append(ListedColormap(cmap))
del cmap


class VLine(QFrame):
  
    def __init__(self):
  
        super(VLine, self).__init__()
        self.setFrameShape(self.VLine|self.Sunken)

class HLine(QFrame):
  
    def __init__(self):
  
        super(HLine, self).__init__()
        self.setFrameShape(self.HLine|self.Sunken)


class MplMatrix(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=8, height=8, dpi=100, initialMatrix=np.zeros(plotSize), cmap=cmapsOverlay, vmin=0, vmax=1):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111)
        super(MplMatrix, self).__init__(self.fig)
        self.images = []
        for i in range(4):
            self.images.append(self.ax.imshow(np.ones(plotSize), vmin=vmin, vmax=vmax, cmap=cmap[i]))
        self.ax.axis('off')
        self.ax.set_aspect('equal')
        self.fig.tight_layout()
        
    def updateMatrix(self, i, matrix):
        self.images[i].set_data(matrix)
        self.draw()
        print('done plotting')
    
    def savePlot(self, figname):
        self.ax.set_xticks([])
        self.ax.set_yticks([])        
        self.fig.savefig(figname, transparent=True)
    
        
class MplQuiver(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=8, height=8, dpi=100, initialSize=plotSize, cmap=cmapsFull):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111)
        # self.canvas = FigureCanvasQTAgg(self.fig)
        super(MplQuiver, self).__init__(self.fig)
        self.images = []
        self.data = []
        
        ys, xs = plotSize
        self.objects, = self.ax.plot(xs/2, ys/2, 'ro', ms=15, alpha=0.5, markerfacecolor='none')
        
        x, y = np.meshgrid(np.arange(plotSize[1]), np.arange(plotSize[0]))
        
        for i in range(3):
            self.images.append(self.ax.quiver(x, y, np.ones(initialSize), np.ones(initialSize), np.ones(initialSize)*(i==0), clim=[0,1], cmap=cmap[i], headwidth=5))
            
        # self.ax.axis('off')
        self.ax.set_aspect('equal')
        self.fig.tight_layout()
        print('quiver set')
    
    def saveQuiver(self, figname):
        # self.ax.axis('off')
        # self.ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        # self.ax.tick_params(axis='y', which='both', bottom=False, top=False, labelbottom=False)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        # tempFig = self.fig
        # tempFig.axis('off')
        
        self.fig.savefig(figname, transparent=True)
        
    def showObjects(self, dist):
        dist = dist*plotSize[0]/32
        ys, xs = plotSize
        ys/=2
        xs/=2
        xs-=0.5
        ys-=0.5
        if dist>0:
            xs = [xs+dist, xs-dist]
            ys = [ys+dist, ys-dist]
            self.objects.set_color('g')
        else:
            self.objects.set_color('r')            
        self.objects.set_data(xs, ys)
        self.draw()
            
        
    def updateQuiver(self, i, u, v, c=0):
        # v = np.sin(angles)
        # u = np.cos(angles)
        u*=1.5
        v*=1.5
        if c is None:
            self.images[i].set_UVC(u, v)
        else:
            self.images[i].set_UVC(u, v, c)
        # print('quiverChanged')
        self.draw()

    def clearMatrix(self, i, initialSize = plotSize):
        self.images[i].set_UVC(np.ones(initialSize), np.ones(initialSize), np.ones(initialSize)*np.nan)

class AnotherWindow(QWidget):
    def __init__(self, data):
        super().__init__()
        self.grid_layout = QGridLayout()
        self.setLayout(self.grid_layout)
        self.outputPlot = MplMatrix(self)
        self.grid_layout.addWidget(self.outputPlot, 0, 0, 1, 1)
        for i in range(3):
            self.outputPlot.updateMatrix(i, data[i])
            
        combinedData = np.sum(data, axis=0)
        thresh = 1.8
        # print(combinedData[combinedData>thresh])
        f = combinedData*(combinedData>thresh)
        print(np.nanmax(f))
        # print(np.nanmax(combinedData))
        # print(combinedData>thresh)
        self.outputPlotFiltered = MplMatrix(self, vmax = np.nanmax(f))
        self.grid_layout.addWidget(self.outputPlotFiltered, 0, 1, 1, 1)

        for i in range(3):
            self.outputPlotFiltered.updateMatrix(i, data[i]*(combinedData>thresh))
            
        self.rateMap = MplMatrix(self, cmap=[plt.cm.get_cmap('viridis'), plt.cm.get_cmap('viridis'), plt.cm.get_cmap('viridis')], vmin=0, vmax=np.nanmax(f))
        self.rateMap.updateMatrix(2, f)
        self.grid_layout.addWidget(self.rateMap, 0, 2, 1, 1)
        self.move(0,1200)



class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.move(0,0)
        self.grid_layout = QGridLayout()
        self.setLayout(self.grid_layout)
        
        # blankMatrix = np.zeros((1024, 1024))

        plotWidth = 2
        labelRow = 0
        plotRow = labelRow+1
        configRow = plotRow+4
        totalHeight = configRow+4

        
        self.angleDelta = 10
        self.sliderRange = int(180/self.angleDelta)

        self.allocentricOverlay = np.ones(plotSize)

        
        self.grid_layout.addWidget(QLabel("Head Directions"), labelRow, plotWidth*0, 1, plotWidth, Qt.AlignBottom|Qt.AlignCenter)
        self.grid_layout.addWidget(QLabel("Distances"), labelRow, plotWidth*1+1, 1, plotWidth, Qt.AlignBottom|Qt.AlignCenter)
        self.grid_layout.addWidget(QLabel("Bearing Directions"), labelRow, plotWidth*2+2, 1, plotWidth, Qt.AlignBottom|Qt.AlignCenter)
        self.grid_layout.addWidget(QLabel("Resultant Output"), labelRow, plotWidth*3+3, 1, plotWidth, Qt.AlignBottom|Qt.AlignCenter)

        self.hdPlot = MplQuiver(self)
        self.grid_layout.addWidget(self.hdPlot, plotRow, plotWidth*0, 1, plotWidth)

        self.distancePlot = MplMatrix(self)
        self.grid_layout.addWidget(self.distancePlot, plotRow, plotWidth*1+1, 1, plotWidth)

        self.bearingPlot = MplQuiver(self)
        self.grid_layout.addWidget(self.bearingPlot, plotRow, plotWidth*2+2, 1, plotWidth)
        
        # self.lvPlot = MplMatrix(self)
        self.lvPlot = MplQuiver(self)
        self.grid_layout.addWidget(self.lvPlot, plotRow, plotWidth*3+3, 1, plotWidth)
        
        self.grid_layout.addWidget(VLine(), plotRow, plotWidth*0+2, totalHeight, 1)
        self.grid_layout.addWidget(VLine(), plotRow, plotWidth*1+1+2, totalHeight, 1)
        self.grid_layout.addWidget(VLine(), plotRow, plotWidth*2+2+2, totalHeight, 1)

        self.grid_layout.addWidget(QLabel("Head Direction Inputs"), configRow-2, plotWidth*0, 1, plotWidth, Qt.AlignBottom|Qt.AlignCenter)
        self.grid_layout.addWidget(QLabel("Distance Inputs"), configRow-2, plotWidth*1+1, 1, plotWidth, Qt.AlignBottom|Qt.AlignCenter)
        self.grid_layout.addWidget(QLabel("Bearing Direction Inputs"), configRow-2, plotWidth*2+2, 1, plotWidth, Qt.AlignBottom|Qt.AlignCenter)
        self.grid_layout.addWidget(QLabel("Difference Adjust"), configRow-2, plotWidth*3+3, 1, plotWidth, Qt.AlignBottom|Qt.AlignCenter)

        self.anchorObjectsCheckbox = QCheckBox("Anchor to Objects")
        self.anchorObjectsCheckbox.toggled.connect(self.toggleObjects)
        
        self.objectDistanceSlider = QSlider(Qt.Horizontal)
        self.objectDistanceSlider.setEnabled(False)
        self.objectDistanceSlider.setRange(0, 16)
        self.objectDistanceSlider.setTickInterval(1)
        self.objectDistanceSlider.setTickPosition(QSlider.TicksBelow)
        # self.objectDistanceSlider.setValue(5)
        self.objectDistanceSlider.valueChanged.connect(self.toggleObjects)
        self.grid_layout.addWidget(self.objectDistanceSlider, configRow+5, plotWidth*2+2, 1, plotWidth, Qt.AlignBottom|Qt.AlignCenter)

        
        self.hdInputs = []
        self.bearingInputs = []
        self.distanceInputs = []
        self.diffInputs = []

        self.initializeInputGUIElements(self.hdInputs, 0, (-180,180), configRow, 0, plotWidth)
        self.initializeInputGUIElements(self.distanceInputs, 1, (0,512), configRow, 1, plotWidth)
        self.initializeInputGUIElements(self.bearingInputs, 2, (-180,180), configRow, 2, plotWidth)
        self.initializeInputGUIElements(self.diffInputs, 3, (-1, 1), configRow, 3, plotWidth)
        
        self.regenPlotsBtn = QPushButton("Output Map")
        self.regenPlotsBtn.clicked.connect(self.regenPlotWindow)
        self.grid_layout.addWidget(self.anchorObjectsCheckbox, configRow+4, plotWidth*2+2, 1, plotWidth, Qt.AlignBottom|Qt.AlignCenter)
        self.grid_layout.addWidget(self.regenPlotsBtn, configRow+4, plotWidth*3+3, 1, plotWidth, Qt.AlignBottom|Qt.AlignCenter)
        self.lvData = [np.zeros(plotSize)]*3

        self.addAllocentricOverlaySlider = QSlider(Qt.Horizontal)
        self.addAllocentricOverlaySlider.setRange(0, 16)
        self.addAllocentricOverlaySlider.setTickInterval(1)
        self.addAllocentricOverlaySlider.setTickPosition(QSlider.TicksBelow)
        self.addAllocentricOverlaySlider.valueChanged.connect(self.genAllocentricOverlay)
        self.grid_layout.addWidget(self.addAllocentricOverlaySlider, configRow+5, plotWidth*3+3, 1, plotWidth, Qt.AlignBottom|Qt.AlignCenter)
        

        # for i in range(3):
        #     self.hdInputs[i][0].setChecked(True)
        #     self.bearingInputs[i][0].setChecked(True)
        #     self.distanceInputs[i][0].setChecked(True)
        #     self.diffInputs[i][0].setChecked(True)
        
        self.setWindowTitle('LVC Simulation')

        self.setWindowTitle("")
        
        self.regenPlotWindow()
        
    # def closeEvent(self, event):
    #     if self.window2:
    #         self.window2.close()
    
    def regenPlotWindow(self):
        # print(self.hdInputs)
        curParameters = [x[1].value() for x in self.hdInputs] + [x[1].value() for x in self.bearingInputs] + [x[1].value() for x in self.distanceInputs] + [x[1].value() for x in self.diffInputs] + [self.objectDistanceSlider.value()]
        # curParameters = [x[-1] for x in self.hdInputs] + [x[-1] for x in self.bearingInputs] + [x[-1]. for x in self.distanceInputs] + [x for x in self.diffInputs] + [self.objectDistanceSlider.value()]
        curParameters = tuple(curParameters)
        print(curParameters)
        # sys.exit()
        self.hdPlot.saveQuiver("S:\\hdPlot_HD{},{},{}_EB{},{},{}_Dist{},{},{}_Diff{},{},{}_LVDist{}.pdf".format(*curParameters))
        self.distancePlot.savePlot("S:\\distancePlot_HD{},{},{}_EB{},{},{}_Dist{},{},{}_Diff{},{},{}_LVDist{}.pdf".format(*curParameters))
        self.bearingPlot.saveQuiver("S:\\ebPlot_HD{},{},{}_EB{},{},{}_Dist{},{},{}_Diff{},{},{}_LVDist{}.pdf".format(*curParameters))
        self.lvPlot.saveQuiver('S:\\lvPlot_HD{},{},{}_EB{},{},{}_Dist{},{},{}_Diff{},{},{}_LVDist{}.pdf'.format(*curParameters))
        # self.hdPlot.saveQuiver("S:\\hdPlot.pdf")
        # self.distancePlot.savePlot("S:\\distancePlot.pdf")
        # self.bearingPlot.saveQuiver("S:\\ebPlot.pdf")
        # self.lvPlot.saveQuiver('S:\\lvPlot.pdf')

        self.recalculateLVC()
        # self.window2 = AnotherWindow(self.lvData)
        # self.window2.show()
        
    def toggleObjects(self):
        for i in range(3):
            self.changeBearing(self.bearingInputs[i][1].value(), self.bearingInputs[i][0].isChecked(), i)
        self.recalculateLVC()
        # self.distanceInputs[boxID][-1]
        if self.anchorObjectsCheckbox.isChecked():
            self.objectDistanceSlider.setEnabled(True)
        else:
            self.objectDistanceSlider.setEnabled(False)
        
    def initializeInputGUIElements(self, group, groupType, sliderRange, rowPosition, columnPosition, colPlotWidth):
        #groupType is 0 for HD, 1 for bearing, 2 for distance
        self.grid_layout.addWidget(QLabel("Enable"), rowPosition-1, columnPosition*colPlotWidth+groupType, 1, 1, Qt.AlignBottom)
        self.grid_layout.addWidget(QLabel(str(sliderRange[0])+"\t\t\t to \t\t\t"+str(sliderRange[1])), rowPosition-1, columnPosition*colPlotWidth+1+groupType, Qt.AlignBottom|Qt.AlignCenter)
        for i in range(3):
            checkbox = QCheckBox()
            checkbox.setChecked(True)
            slider = QSlider(Qt.Horizontal)
            slider.setEnabled(True)
            slider.setRange(-1*self.sliderRange, self.sliderRange)
            slider.setTickInterval(1)
            # if groupType==3:
            #     slider.setRange(-1,1)
            slider.setTickPosition(QSlider.TicksBelow)
            checkbox.stateChanged.connect(lambda state, x=group, y=groupType, z=i:self.inputCheckboxChanged(state, x, y, z))
            slider.valueChanged.connect(lambda value, x=group, y=groupType, z=i:self.inputValueChanged(value, x, y, z))
            self.grid_layout.addWidget(checkbox, rowPosition+i, columnPosition*colPlotWidth+groupType, Qt.AlignCenter)
            self.grid_layout.addWidget(slider, rowPosition+i, columnPosition*colPlotWidth+1+groupType)
            group.append([checkbox, slider, None])
            self.inputCheckboxChanged(True, group, groupType, i, True)
        # group[0][0].setChecked(True)
        # group[0][0].setChecked(True)
        # return group
        
    def inputCheckboxChanged(self, state, group, groupType, boxID, initialization=False):
        group[boxID][1].setEnabled(state)
        self.inputValueChanged(group[boxID][1].value(), group, groupType, boxID, state, initialization)
        
    def inputValueChanged(self, value, group, groupType, boxID, state=True, initialization=False):
        if groupType==0:
            group[boxID][-1] = self.changeHD(value*self.angleDelta, state, boxID)
        elif groupType==1:
            group[boxID][-1] = self.changeDistance((value+self.sliderRange)*(plotSize[0]/(self.sliderRange*4)), state, boxID, initialization)
        elif groupType==2:
            group[boxID][-1] = self.changeBearing(value*self.angleDelta, state, boxID)
        elif groupType==3:
            group[boxID][-1] = self.changeDiff((value+self.sliderRange+1)/(self.sliderRange*2+1), state, boxID, initialization)
        if not initialization:
            self.recalculateLVC()
        
        
    def changeHD(self, value, state, boxID):
        angle = value*np.pi/180
        self.hdPlot.updateQuiver(boxID, np.cos(angle), np.sin(angle), c=np.ones(plotSize)*state)
        return angle

    def multivariate_gaussian(self, pos, mu, sigma):
        """Return the multivariate Gaussian distribution on array pos."""
    
        n = mu.shape[0]
        sigma_det = np.linalg.det(sigma)
        sigma_inv = np.linalg.inv(sigma)
        N = np.sqrt((2*np.pi)**n * sigma_det)
        # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
        # way across all the input variables.
        fac = np.einsum('...k,kl,...l->...', pos-mu, sigma_inv, pos-mu)
    
        return np.exp(-fac / 2) / N

    
    def genAllocentricOverlay(self):
        seed = self.addAllocentricOverlaySlider.value()
        print('seed!', seed)
        allocentricOverlay = np.zeros(plotSize)
        if seed==0:
            self.allocentricOverlay = np.ones(plotSize)
            # return allocentricOverlay
        else:
            np.random.seed(seed)
            for i in range(seed):
                # print(i)
                X = np.linspace(0, 4, plotSize[0])
                Y = np.linspace(0, 4, plotSize[1])
                X, Y = np.meshgrid(X, Y)
                mu = np.array([np.random.random()*4, np.random.random()*4])
                little = np.random.random()*1.5-0.75
                sigma = np.array([[1 ,little], [little, 1]])
                # print(mu, sigma)
                pos = np.empty(X.shape + (2,))
                pos[:, :, 0] = X
                pos[:, :, 1] = Y
                Z = self.multivariate_gaussian(pos, mu, sigma)*2
                allocentricOverlay+=Z
            self.allocentricOverlay = allocentricOverlay/(np.nanmax(allocentricOverlay))
            
            # self.distancePlot.updateMatrix(1, self.allocentricOverlay)
            # self.distancePlot.updateMatrix(2, self.allocentricOverlay)
            # print(self.allocentricOverlay)
        self.distancePlot.updateMatrix(3, self.allocentricOverlay)
        self.toggleObjects()
        self.recalculateLVC()



        
    def changeBearing(self, value, state, boxID):
        value = self.bearingInputs[boxID][1].value()*self.angleDelta
        state = self.bearingInputs[boxID][0].isChecked()
        
        distance = self.distanceInputs[boxID][-1]
        # distance = np.ones(plotSize)

        angle = value*np.pi/180
        ys, xs = plotSize
        # ys*=2
        # xs*=2
        yc = [ys/2]
        xc = [xs/2]
        distShift = [0]
        mask = [np.ones(plotSize, dtype=bool)]
        bearingAngles = np.zeros(plotSize)
        resultantDistance = np.zeros(plotSize)
        bearingAngleSets = []
        distanceSets = []
        if self.anchorObjectsCheckbox.isChecked():
            dist = self.objectDistanceSlider.value()
            dist /= 32
            yc = [ys*(0.5-dist), ys*(0.5+dist)]
            xc = [xs*(0.5-dist), xs*(0.5+dist)]
            print(dist)
            distShift = [int(-1*ys*dist), int(ys*dist)]
            mask = [np.fliplr(np.triu(np.ones(plotSize, dtype=bool))), np.fliplr(np.tril(np.ones(plotSize, dtype=bool)))]

        for i in range(len(yc)):
            print(i)
            ys, xs = plotSize
            ys = np.arange(ys)-(yc[i])+0.5
            xs = np.arange(xs)-(xc[i])+0.5        
            xv, yv = np.meshgrid(xs, ys)
            curBearingAngles = np.arctan2(yv, xv)+angle
            bearingAngleSets.append(curBearingAngles)
            # curBearingAngles*=mask[i]
            # bearingAngles[mask[i]] = curBearingAngles[mask[i]]
            # if i==0:
            #     bearingAngles = curBearingAngles
            # else:
            #     bearingAngles = np.arctan2((np.sin(bearingAngles)+np.sin(curBearingAngles)),(np.cos(bearingAngles)+np.cos(curBearingAngles)))

            curDistance = self.shiftMatrix(distance, distShift[i], 0, 0)
            curDistance = self.shiftMatrix(curDistance, distShift[i], 1, 0)
            # curDistance*=mask[i]
#            resultantDistance[mask[i]] += curDistance[mask[i]]
            distanceSets.append(curDistance)
            distanceAdjusted = curDistance/np.max(curDistance)
            bearingAngleSets[i][distanceAdjusted<0.0001] = np.nan
        
        bearingAngleCalc = np.array([[np.sin(x), np.cos(x)] for x in bearingAngleSets])
        bearingAngles = np.arctan2(np.nansum(bearingAngleCalc[:,0], axis=0),np.nansum(bearingAngleCalc[:,1], axis=0))
        # print(bearingAngles)
        # bearingAngles = np.nan_to_num(bearingAngles)
        # bearingAngles = 0
        
        bearingAngleDistance = np.ones(plotSize)
        
        
        if len(bearingAngleSets)!=1:
            bearingAngleDistance = np.zeros(plotSize)
            isNan0 = np.isnan(bearingAngleSets[0])
            isNan1 = np.isnan(bearingAngleSets[1])
            bearingAngleDistance[isNan0] = 1
            bearingAngleDistance[isNan1] = 1
            
            dualBearingSpots = ~(isNan0 + isNan1)
            # singleBearingSpots = np.abs(np.isnan(bearingAngleSets[0])-np.isnan(bearingAngleSets[1]))==1
            bearingAngleDistance[dualBearingSpots] = np.abs(np.arctan2(np.sin(bearingAngleSets[0]-bearingAngleSets[1]),np.cos(bearingAngleSets[0]-bearingAngleSets[1])))[dualBearingSpots]
            bearingAngleDistance[(np.abs(bearingAngleDistance)>(np.pi))*dualBearingSpots] = 0
            bearingAngleDistance[dualBearingSpots] = np.abs(bearingAngleDistance[dualBearingSpots]-(np.pi))
            bearingAngleDistance[dualBearingSpots]/=(np.pi)

        resultantDistance = np.sum(distanceSets, axis=0)*(bearingAngleDistance/np.max(bearingAngleDistance))*self.allocentricOverlay

        self.bearingPlot.updateQuiver(boxID, np.cos(bearingAngles), np.sin(bearingAngles), c=(resultantDistance*state*bearingAngleDistance*self.allocentricOverlay)/np.max(resultantDistance*state*bearingAngleDistance))
        self.bearingPlot.showObjects(self.objectDistanceSlider.value())
        self.hdPlot.showObjects(self.objectDistanceSlider.value())
        self.lvPlot.showObjects(self.objectDistanceSlider.value())


        
        self.bearingInputs[boxID][-1] = [bearingAngles, resultantDistance]
        return [bearingAngles, resultantDistance]
        
    
    def changeDistance(self, value, state, boxID, initialization):
        if initialization:
            normFunc = stats.norm(0, 1)
        else:
            normFunc = stats.norm(0, 2*self.diffInputs[boxID][-1])
        ys, xs = plotSize
        ys = np.arange(ys)-(ys/2)+0.5
        xs = np.arange(xs)-(xs/2)+0.5
        xv, yv = np.meshgrid(xs, ys)
        distMatrix = np.sqrt(yv**2+xv**2)
        distMatrix-=value
        # distMatrix = np.max(distMatrix)-np.abs(distMatrix)
        distMatrix = normFunc.pdf(distMatrix)*state
        self.distancePlot.updateMatrix(boxID, distMatrix)
        
        self.distanceInputs[boxID][-1] = distMatrix
        
        if not initialization:
            if self.bearingInputs[boxID][0].isChecked():
                self.changeBearing(self.bearingInputs[boxID][1].value(), self.bearingInputs[boxID][0].isChecked(), boxID)
        
        return distMatrix

    
    def changeDiffOld(self, value, state, boxID, initialization):
        diff=0
        if not initialization:
            self.hdInputs[boxID][0].setChecked(False)
            self.bearingInputs[boxID][0].setChecked(False)

        # print(diff, value, self.diffInputs[boxID][-1])
        if self.diffInputs[boxID][-1] is None:
            self.diffInputs[boxID][-1]=0
            diff = value-self.diffInputs[boxID][-1]
            # print(diff, value, self.diffInputs[boxID][-1])
            
        self.hdInputs[boxID][1].setValue((self.hdInputs[boxID][1].value()+diff+12)%24-12)
        self.bearingInputs[boxID][1].setValue((self.bearingInputs[boxID][1].value()+diff+12)%24-12)
        self.diffInputs[boxID][-1] = value
        self.hdInputs[boxID][0].setChecked(True)
        self.bearingInputs[boxID][0].setChecked(True)
        # print(self.diffInputs[boxID][-1], value)
        self.diffInputs[boxID][1].setValue(0)
        
    def changeDiff(self, value, state, boxID, initialization):
        # self.diffInputs[boxID][-1] = value
        self.changeDistance((self.distanceInputs[boxID][1].value()+self.sliderRange)*(plotSize[0]/(self.sliderRange*4)), self.distanceInputs[boxID][0].isChecked(), boxID, initialization)
        return value
    
    def shiftMatrix(self, mat, diff, axis, minval=0):
        baseSize = mat.shape
        tempExpanded = np.ones((baseSize[0]*3, baseSize[1]*3))*minval
        tempExpanded[plotSize[0]:(baseSize[0]*2), baseSize[1]:(baseSize[1]*2)] = mat
        tempExpanded = np.roll(tempExpanded, diff, axis=axis)
        outData = tempExpanded[baseSize[0]:(baseSize[0]*2), baseSize[1]:(baseSize[1]*2)]
        return outData
    
    def recalculateLVC(self):
        sumData = np.zeros(plotSize)
        allData = [np.zeros(plotSize)]*3
        for i in range(3):
            if not (self.hdInputs[i][0].isChecked() and self.bearingInputs[i][0].isChecked() and self.distanceInputs[i][0].isChecked() and self.diffInputs[i][0].isChecked()):
                continue
            # print(len(self.bearingInputs[i]))
            # angleDiff = np.abs(self.hdInputs[i][-1]-self.bearingInputs[i][-1])/(2*np.pi)
            angleDiff = (self.hdInputs[i][-1]-self.bearingInputs[i][-1][0])*180/np.pi
            # angleDiff = self.hdInputs[i][-1]*180/np.pi
            angleDiff%360-180
            angleDiff = (angleDiff+180)%360-180
            normFunc = stats.norm(0, 0.1*self.diffInputs[i][-1])
            angleDiffMatrix = normFunc.pdf(angleDiff/180)
            angleDiffMatrix/=np.max(angleDiffMatrix)
            # distMatrix = self.distanceInputs[i][-1]/np.max(self.distanceInputs[i][-1])    
            distMatrix = self.bearingInputs[i][-1][1]/np.max(self.bearingInputs[i][-1][1])

            normFunc = stats.norm(0, 1)                
            outData = normFunc.pdf(angleDiffMatrix*distMatrix-2)*2
            

            allData[i] = outData
            # sumData=outData

        # sumData-=np.min(sumData)
        # self.lvPlot.updateMatrix(sumData/np.max(sumData))
        for i in range(len(allData)):
            if not (self.hdInputs[i][0].isChecked() and self.bearingInputs[i][0].isChecked() and self.distanceInputs[i][0].isChecked() and self.diffInputs[i][0].isChecked()):
                self.lvPlot.updateQuiver(i, 0, 0, 0)
                continue
            

            bearingShift = self.shiftMatrix(self.bearingInputs[i][-1][0], 1, 0)
            # bearingShift = self.shiftMatrix(np.ones(plotSize)*self.hdInputs[i][-1], 1, 0)
            # self.lvPlot.updateQuiver(i, np.cos(self.bearingInputs[i][-1]), np.sin(self.bearingInputs[i][-1]), (allData[i]-np.min(allData[i]))/np.max(allData[i]))
            # curQuiverValues = (allData[i]-np.min(allData[i]))/np.max(allData[i])
            # curQuiverValues = signal.convolve2d(curQuiverValues, gausKernel, mode='same')
            # curQuiverValues = curQuiverValues/np.max(curQuiverValues)
            self.lvPlot.updateQuiver(i, np.cos(bearingShift), np.sin(bearingShift), (allData[i]-np.min(allData[i]))/np.max(allData[i]))        
            # self.lvPlot.updateQuiver(i, np.cos(bearingShift), np.sin(bearingShift), curQuiverValues)        
            # self.lvPlot.updateQuiver(i, np.cos(bearingShift), np.sin(bearingShift), np.ones(plotSize))        
            self.lvData[i] = (allData[i]-np.min(allData[i]))/np.max(allData[i])

app = QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec_())