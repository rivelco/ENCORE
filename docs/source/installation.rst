Installation
============

System requirements
-------------------

The GUI is very low resource demanding. It is advisable to run it in a screen at least 1170 x 660 pixels to see all the buttons clearly. Full HD screens (1920 x 1080) or above are recommended. For the system requirements it is recommended a modern CPU and at least 16Gb of RAM. This for agile tuning and execution of the analysis.

This specs, while recommended, are not mandatory. If you can run MATLAB 2020A then you can run this GUI and the included analysis.

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
- Statistics and Machine Learning Toolbox
- Curve Fitting Toolbox

Installation of the python environment
--------------------------------------

1. Clone or download the repo
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use git to clone the repo or download it from the website. As stated in their webpage "Git is a free and open source distributed version control system designed to handle everything from small to very large projects with speed and efficiency."[#]_. If you don't have it already, install git in your computer, check the available installers at `the official download site <https://git-scm.com/downloads>`_.

Once installed git, run the following command and then change to the directory of the repository.

.. code-block:: console

    git clone https://github.com/rivelco/ENCORE.git
    cd ENCORE

If you do not want to use the git command, you can download the repository by going to the repository `<https://github.com/rivelco/ENCORE>`_ and using the Download button:

.. raw:: html

    <video width="600" controls>
        <source src="_static/Installation_download.mp4" type="video/mp4">
        Your browser does not support the video tag.
    </video>

This will download the repository as a compressed zip file. Uncompress the file to extract the folder.

The recommended installation method relies on `Conda <https://docs.conda.io/projects/conda/en/latest/index.html>`_ to manage your Python environments. I highly recommend using Conda for this purpose. Also, if you have Conda you only need to install the environment file to get the necessary dependencies. If you don't want to use Conda, you have to install the dependencies manually using pip.

2.A Installation using conda
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 
To install using this method simple open your conda terminal, go to the EnsemblesGUI folder and type this commands.

This commands creates a Conda environment called `encore` and then activates it.

.. code-block:: console

    conda env create -f environment.yml
    conda activate encore

2.B Manual installation using pip
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can install the modules needed for EnsemblesGUI manually. It's still recommended that you do this using conda or some other environment manager. If you're using conda type:

.. code-block:: console

    conda create -n encore python=3.10
    conda activate encore

The above command will create a new environment called `encore` and with python 3.10 installed. You can choose the name of the environment replacing `encore` with the name you prefer. After the environment is created it must be activated.

If you want to install the needed modules one by one you can install your preferred python version (recommended 3.8 or above) and run:

.. code-block:: console

    pip install pyqt6
    pip install numpy
    pip install matplotlib
    pip install h5py
    pip install scikit-learn
    pip install pyqtdarktheme

Installation of the MATLAB engine for Python
--------------------------------------------

To run the analysis algorithms it is necessary to install in the python environment the MATLAB engine. This can be done by looking for yor MATLAB installation path. The installation path in Windows usually looks something like this:

.. code-block:: console

    cd C:\Program Files\MATLAB\R2023a\extern\engines\python

The idea is to locate the engine for Python. Notice in the example above that the MATLAB installation that will be used is the R2023a, you can chose the version that you prefer, just consider the compatibility with the current python version.

Once you're there and with your correct python environment activated then simply run:

.. code-block:: console

    python -m pip install .

It is possible that you need to run that command from an elevated terminal.

Run the GUI
-----------

To run the GUI you now just need to call encore.py from your configured python environment. Make sure you are in the path where you downloaded the repo.

.. code-block:: console

    python encore.py

References
----------

.. [#] `<https://git-scm.com/>`_.