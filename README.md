-- The code here is a bit outdated and over time I have found out several bugs/problems. I have been slowly working in a new, updated version which will be ready "soon". In the meantime, please use with caution --

# Rfun, a GUI-equipped toolbox for the computation and analysis of receiver functions
Rfun is a graphical user interface (GUI) equipped program aimed at providing an interactive and easy to use environment in
which to perform the computation and analysis of receiver functions. The software provides a tool that automatically cuts P-wave
windows for selected events. These windows are then used to compute the receiver functions, which in turn can be used to study
the crustal thickness below the receiver (via a semblance-weighted H-k stacking algorithm) or to generate Common Conversion Point
(CCP) stacks. A tutorial is coming soon™. In the meantime, please feel free to contact me for information on how to use the program
(olivar.ac@gmail.com).

Rfun is also distributed as part of [ISP/Integrated Seismic Program](https://github.com/ProjectISP/ISP)! ISP aims to provide a comfortable GUI for performing diverse 
tasks in earthquake seismology, from array analysis to seismic moment tensor inversion. Be sure to check it out as well!

# Installation (Linux/Windows)
Rfun requires several packages, which are listed in the requeriments.txt file. I recommend using [Anaconda](https://www.anaconda.com/products/individual#Downloads)
or [Miniconda](https://docs.conda.io/en/latest/miniconda.html/) for installation as using other tools like pip can be problematic. I have not performed any tests
in macOS, but the following should work for that platform as well.

Once you have installed Anaconda/Miniconda, open up a command prompt (if working in Windows, make sure to open an Anaconda Prompt), create and activate a new Python3
environment:
```
$ conda create -n rfun python=3.9.5
$ conda activate rfun
```
Then, install Rfun's dependencies, which are listed in the requeriments.txt file:
```
(rfun) $ conda install -c conda-forge <package1>=<version> <package2>=<version> ...
```
Once the dependencies are installed, navigate to Rfun's root directory and run the setup.py file
```
(rfun) $ python setup.py install
```
All done! You should be able to run Rfun by typing in the command prompt:
```
(rfun) $ rfun
```
