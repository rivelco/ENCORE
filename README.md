<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/rivelco/ENCORE">
    <img src="gui/ENCORE_logo.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">ENCORE - Ensembles Comparison and Recognition</h3>

  <p align="center">
    GUI to identify and compare neuronal ensembles in calcium or electrical recordings using five different algorithms.
    <br />
    <a href="https://rivelco.github.io/ENCORE/"><strong>Read the documentation</strong></a>
    <br />
    <br />
    <a href="https://link.springer.com/book/10.1007/978-1-0716-4208-5">Read the book</a>
    &middot;
    <a href="https://github.com/rivelco/ENCORE/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    &middot;
    <a href="https://github.com/rivelco/ENCORE/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>

## About ENCORE

Graphic User Interface to perform a variety of neuronal ensembles identification's methods.

> This repo is under current development.

[![Product Name Screen Shot][gui-screenshot-ens-inspect]]([repo-link])

This GUI currently incorporates five different algorithms:

- SVD based method: Carrillo-Reid, et al. "Endogenous sequential cortical activity evoked by visual stimuli." Journal of Neuroscience 35.23 (2015): 8813-8828.
- PCA based method: Herzog et al. 2021 "Scalable and accurate automated method for neuronal ensemble detection in spiking neural networks. https://pubmed.ncbi.nlm.nih.gov/34329314/ Rubén Herzog Dec 2021
- ICA based method: Lopes-dos-Santos V, Ribeiro S, Tort AB (2013) Detecting cell assemblies in large neuronal populations. J Neurosci Methods 220(2):149-66. 10.1016/j.jneumeth.2013.04.010
- Xsembles2P method: Pérez-Ortega, J., Akrouh, A. & Yuste, R. 2024. Stimulus encoding by specific inactivation of cortical neurons. Nat Commun 15, 3192. doi: 10.1038/s41467-024-47515-x
- Similarity Graph Clustering method: L. Avitan et al. "Spontaneous Activity in the Zebrafish Tectum Reorganizes over Development and Is Influenced by Visual Experience". Curr. Biol. 27 (2017). DOI: 10.1016/j.cub.2017.06.056

## Watch the demo

<p align="center">
  <img src="readme_elements/vids/demo.mp4" alt="Demo of ENCORE interface" width="700"/>
</p>

**More analysis and features coming soon...**

> For detailed instructions about installation and usage, go to the [documentation website](https://rivelco.github.io/ENCORE/)

## Installation

## System requirements

The GUI is very low resource demanding. It is advisable to run it in a screen at least 1170 x 660 pixels to see al the buttons clearly. For the system requirements it is recommended a modern CPU and at least 16Gb of RAM. This for agile tuning and execution of the analysis.

This specs, while recommended, are not mandatory. If you can run MATLAB 2020A then you can run this GUI an the included analysis.

## Needed dependencies

- MATLAB (version 2020A or above)
- Python 3.10 or above

### Needed MATLAB modules

- Parallel Computing Toolbox
- Statistics and Machine Learning Toolbox
- Curve Fitting Toolbox

## Clone or download the repo

Use git to clone the repo or download it from the webpage. Then change to that directory.

```bash
git clone https://github.com/rivelco/ENCORE.git
cd ENCORE
```

## Installation using conda

The recommended installation method relies on Conda to manage your Python environments. I highly recommend using conda for this purpose. 
To install using this method simple open your terminal and type this commands.

This commands creates a conda environment called `encore` and then activates it.

```bash
conda env create -f environment.yml
conda activate encore
```

## Installation using pip

If you want to install the needed modules one by one you can use Python 3.10 and run:

```bash
pip install pyqt6
pip install numpy
pip install matplotlib
pip install h5py
pip install scikit-learn
pip install scipy
pip install pyqtdarktheme
```

## Install the MATLAB engine for Python

To run the algorithms it is necessary to install in the python environment the MATLAB engine. This can be done by looking for yor MATLAB installation path, to something like this:

```bash
cd C:\Program Files\MATLAB\R2023a\extern\engines\python
``` 

The idea is to locate the engine for Python.

Once you're there and with your correct python environment activated then simply run:

```bash
python -m pip install .
```

It is possible that you need to run that command from an elevated terminal.

## Run the GUI

To run the GUI you now just need to call main.py from your configured python environment. Make sure your in the path where you downloaded the repo.

```bash
python main.py
```

## Needed data.

This GUI requires just matrices for all of the input data that can be loaded from different files. This matrix all follow the same logic. For data that shows a timeseries like the recording of neurons of the recording of stimulation then the matrix must have a shape `[elements, timepoints]`. Every element (neuron or stimulation or even behavior) should has the recording of their activity over time.

The datasets that must provide belonging to a group must have the format of a binary matrix with shape `[groups, elements]`.

Consider that any of this matrices could be transposed in your file and you can re-transpose it using the GUI. Just make sure that the axes labels match your data.

### Variable `dFFo/FFo`

Matrix with the shape `[neurons, timepoints]` that stores continous values describing the flourescence activity of each neuron.

### Variable `Binary cell activity`

This should be a binary matrix with shape `[neurons, timepoints]`, where if the element `[neuron, timpeoint] == 1` then this indicates that that neuron was active in that timepoint.

### Variable `Coordinates`

This variable should be a matrix with shape `[neuron, dimentions]`, where for each neuron it is specified its coordinates for 2 dimentions, `x` and `y`. The current version only supports 2 dimentions.

### Variable `stimulation`

Binary matrix with shape `[stimuli, timepoints]`. The value `[stim, timpoint] == 1` indicates that the stimulation `stim` was present in the timepoint `timepoint`. 

### Variable `cells`

Binary matrix with shape `[groups, cell]`. The value `[group, cell] == 1` indicates that the group `group` includes the cell `cell`.

### Variable `behavior`

Binary matrix with shape `[behaviors, timepoints]`. The value `[behavior, timpoint] == 1` indicates that the behavior `behavior` was present in the timepoint `timepoint`. 

## Contribute to this project

The main goal of this project is to be easily used to identify neuronal ensembles using different approaches. If you have suggestions, feature requests or bug reports, don't hesitate to use the Issues section of this repo.

<!-- MARKDOWN LINKS & IMAGES -->
[repo-link]: https://github.com/rivelco/ENCORE
[gui-screenshot-ens-inspect]: readme_elements/images/Ensembles_Inspection_light.png
[demo-vid]: readme_elements/vids/demo.mp4