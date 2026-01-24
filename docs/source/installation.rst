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
    Python 3.9 is compatible with MATLAB from R2021b to R2024b.

.. tip::
    Use at least Python 3.9. If you are using a MATLAB version older than R2020b consider upgrading.

Needed MATLAB modules
~~~~~~~~~~~~~~~~~~~~~

- Parallel Computing Toolbox
- Statistics and Machine Learning Toolbox
- Curve Fitting Toolbox

Installation of the python environment
--------------------------------------

The recommended installation method relies on `Conda <https://docs.conda.io/projects/conda/en/latest/index.html>`_ to manage your Python environments. I highly recommend using Conda for this purpose.

Installation using pip
~~~~~~~~~~~~~~~~~~~~~~

You can install ENCORE using pip. It's still recommended that you do this using conda or some other environment manager. If you're using conda type:

.. code-block:: console

    conda create -n encore python=3.10
    conda activate encore

The above command will create a new environment called `encore` and with python 3.10 installed. You can choose the name of the environment replacing `encore` with the name you prefer. After the environment is created it must be activated.

To install ENCORE simply run:

.. code-block:: console

    pip install encore-toolkit


Verifying the installation
--------------------------

To verify that ENCORE is installed correctly, run:

.. code-block:: console

    python -c "import encore; print(encore.__version__)"

If the version is printed without errors, the installation was successful.


Installation of the MATLAB engine for Python
--------------------------------------------

To run the analysis algorithms that requires MATLAB, it is necessary to install in the python environment the MATLAB engine. This can be done by looking for yor MATLAB installation path. The installation path in Windows usually looks something like this:

.. code-block:: console

    cd C:\Program Files\MATLAB\R2023a\extern\engines\python

The idea is to locate the engine for Python. Notice in the example above that the MATLAB installation that will be used is the R2023a, you can chose the version that you prefer, just consider the compatibility with the current python version.

Once you're there and with your correct python environment activated then simply run:

.. code-block:: console

    python -m pip install .

It is possible that you need to run that command from an elevated terminal.

You can find more help regarding MATLAB engine install in the `official MATLAB site <https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html>`_. 

It's also possible to install the MATLAB engine from pip, check the `versions history <https://pypi.org/project/matlabengine/#history>`_ to install the correct version of the MATLAB engine.

Run the GUI
-----------

To run the GUI you now just need to call encore.py from your configured python environment. Make sure you are in the path where you downloaded the repo.

.. code-block:: console

    encore

It's also possible to launch the GUI by using:

.. code-block:: console

    python -m encore
