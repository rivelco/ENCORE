Usage
=====

System requirements
-------------------

The GUI is very low resource demanding. It is advisable to run it in a screen at least 1170 x 660 pixels to see all the buttons clearly. For the system requirements it is recommended a modern CPU and at least 16Gb of RAM. This for agile tuning and execution of the analysis.

This specs, while recommended, are not mandatory. If you can run MATLAB 2020A then you can run this GUI an the included analysis.

Needed dependencies
~~~~~~~~~~~~~~~~~~~

The general requirements are MATLAB and Python. The limiting factor here is the communication between those two. Because Python is easier to install, I recommend installing the most recent version of python that your MATLAB installation can communicate with. Check `the MATLAB site <https://www.mathworks.com/support/requirements/python-compatibility.html>`_ for the versions of Python that are compatible with MATLAB.

Here we are going to be using Python 3.10, which is compatible with MATLAB R2022b and above.

.. seealso::
    Python 3.8 is compatible with MATLAB from R2020b to R2023a.
    Python 3.9 is compatible with MATLAB from R2021b to R2024b.

.. tip::
    Use at least Python 3.8. If you are using a MATLAB version older than R2020b consider upgrading.

Needed MATLAB modules
~~~~~~~~~~~~~~~~~~~~~

- Parallel Computing Toolbox

Installation
------------

Clone or download the repo
~~~~~~~~~~~~~~~~~~~~~~~~~~

Use git to clone the repo or download it from the webpage. As stated in their webpage "Git  is a free and open source distributed version control system designed to handle everything from small to very large projects with speed and efficiency."[#]_. If you don't have it already, install git in your computer, check the available installers at `the official download site <https://git-scm.com/downloads>`_.

Once installed git, run the following command and then change to the directory of the repository.

.. code-block:: console

    git clone https://github.com/rivelco/EnsemblesGUI.git
    cd EnsemblesGUI

The recommended installation method relies on Conda to manage your Python environments. I highly recommend using conda for this purpose. Also, if you have conda you only need to install the environment file to get the necessary dependencies. If you don't want to use Conda, you have to install the dependencies manually using pip.

Installation using conda
~~~~~~~~~~~~~~~~~~~~~~~~
 
To install using this method simple open your conda terminal and type this commands.

This commands creates a conda environment called `ensgui` and then activates it.

.. code-block:: console

    conda env create -f environment.yml
    conda activate ensgui

Manual installation using pip
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you want to install the needed modules one by one you can install your preferred python version (recommended 3.8 or above) and run:

.. code-block:: console

    pip install pyqt6
    pip install numpy
    pip install matplotlib
    pip install h5py
    pip install scikit-learn
    pip install scipy

Install the MATLAB engine for Python
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To run the algorithms it is necessary to install in the python environment the MATLAB engine. This can be done by looking for yor MATLAB installation path, to something like this:

.. code-block:: console

    cd C:\Program Files\MATLAB\R2023a\extern\engines\python

The idea is to locate the engine for Python.

Once you're there and with your correct python environment activated then simply run:

.. code-block:: console

    python -m pip install .

It is possible that you need to run that command from an elevated terminal.

Run the GUI
-----------

To run the GUI you now just need to call main.py from your configured python environment. Make sure you are in the path where you downloaded the repo.

.. code-block:: console

    python main.py

.. [#] `<https://git-scm.com/>`_.