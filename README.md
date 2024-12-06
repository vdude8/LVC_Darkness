# LVC_Darkness

This is the repository for code associated with "Landmark Vector Cells in the Absence of Visual Input". Please do not share the code without permission from the lead author, Vyash Puliyadi, until the publication has been approved.

Anaconda environment can be replicated with the provided environment.yaml file, but the key dependencies are listed below. To run the simulation, you will also need QT5 installed and configured on your computer. 
- Python 3.8
- Numpy 1.24
- Scipy 1.10
- Numba 0.57
- Matplotlib 3.7
- Pandas 2.0
- Seaborn 0.12.2
- PyQT 5.15.7 (make sure that only QT 5 is used) 

Note that QT is picky. PyQT and all of its dependencies must match, so ensure that only QT 5 compatible versions are used. All packages should be available on the main Anaconda distribution channel.


Included files: 
- calcSpikePair_AngleDist3.py - Performs the main angle/distance calculations between pairs of spikes (mode can be switched to return shuffled data or real data)
- calcSpikePair_AngleDist_summate.py - Combines information from the above calculations to calculate data for Figure 2C and calculate statistics
- calcSpikePair_AngleDist_summatePlots.py - Performs the calculations to generate the images for Figure 2D
- calcSpikePair_shufflePlot.py - Used to display shuffled results in Figure 2E 
- simulateLVC_modified_preclean.py - Simulation to combine head direction and bearing direction information into a resultant place or landmark vector cell
