Python API Usage
================

The :doc:`/api/runners` module contains the Python API to analyze an algorithm using the Python API, programmatically from another program or script.

Each runner implements a complete analysis workflow for a set of databases, including:

- Input validation
- Algorithm selection
- Execution
- Result formatting

Each algorithm needs two arguments, the data needed to identify the ensembles, usually just **a raster of activity**, and the **parameters** for the algorithm.

.. seealso::

    Check the detail of the arguments for the analyses in the docs :doc:`/api/runners`.

.. seealso::
    
    Check the needed data and parameters for each algorithm in the :doc:`/extending/algorithms_config_file`

Data examples
-------------

Each algorithm needs different data inputs. This inputs should be inside a python dictionary with specific names:

.. code:: python
    :number-lines:

    import numpy as np

    # Dummy data
    neurons = 100
    timepoints = 2000
    behaviors = 2
    stimuli = 4
    neurons_sets = 5

    # The binary activity raster
    neuronal_activity = np.random.rand(neurons, timepoints)
    neuronal_activity = (neuronal_activity > 0.9).astype(np.int_)

    # The continuous fluorescence raster
    dFFo = np.random.rand(neurons, timepoints)

    # The 2D coordinates
    coordinates = np.random.rand(2, timepoints) * 100

    # The binary stimuli matrix
    stims = np.random.rand(stimuli, timepoints)
    stims = (stims > 0.25).astype(np.int_)

    # The binary groups of neurons
    cells = np.random.rand(neurons_sets, neurons)
    cells = (cells > 0.9).astype(np.int_)

    # The continuous behavior matrix
    behavior = np.random.rand(behaviors, timepoints)

    data = {
        "data_neuronal_activity":  neuronal_activity,
        "data_dFFo":  dFFo,
        "data_coordinates": coordinates,
        "data_stims": stims,
        "data_cells": cells,
        "data_behavior": behavior,
    }

Those are the available names and variables, however most algorithms need only one.

.. list-table:: Needed data for each algorithm
   :header-rows: 1
   :stub-columns: 1
   :widths: 25 70

   * - Algorithm
     - Data needed
   * - SVD
     - data_neuronal_activity
   * - PCA
     - data_neuronal_activity
   * - ICA
     - data_neuronal_activity
   * - X2P
     - data_neuronal_activity
   * - SGC
     - data_dFFo
   * - Example
     - data_dFFo and data_neuronal_activity


.. note::

    The other variables are available for possible new algorithms.


Parameters examples
-------------------

Each runner requires the parameter for the specific algorithm. You can check the parameters definitions in the `config file <https://github.com/rivelco/ENCORE/blob/main/src/encore/config/encore_runners_config.yaml>`_.

Here is an example of the definition of the parameters dictionary for each algorithm.

.. tip:: 

    The description of each parameter, the accepted values for each and other names is also in the :doc:`/extending/algorithms_config_file`.

SVD
^^^

.. code:: python
    :number-lines:

    parameters_svd = {
        "pks": 3,
        "scut": 0.24,
        "hcut": 0.24,
        "state_cut": 6,
        "csi_start": 0.01,
        "csi_step": 0.01,
        "csi_end": 0.1,
        "tf_idf_norm": True,
        "parallel_processing": False,
        "fixed_ens_cant": 0,
    }

ICA
^^^

.. code:: python
    :number-lines:

    parameters_ica = {
        "threshold_method": "MarcenkoPastur",
        "permutations_percentile": 95.0,
        "number_of_permutations": 20,
        "min_ensembles_cant": 4,
        "max_ensembles_cant": 10,
        "patterns_method": "ICA",
        "number_of_iterations": 1000,
        "threshold_for_p_value": 1.90,
    }

X2P
^^^

.. code:: python
    :number-lines:

    parameters_x2p = {
        "NetworkBin": 1,
        "NetworkIterations": 1000,
        "NetworkSignificance": 0.05,
        "CoactiveNeuronsThreshold": 2,
        "ClusteringRangeStart": 3,
        "ClusteringRangeEnd": 10,
        "ClusteringFixed": 0,
        "EnsembleIterations": 3000,
        "ParallelProcessing": False,
    }

PCA
^^^

.. code:: python
    :number-lines:

    parameters_pca = {
        "dc": 0.01,
        "npcs": 3,
        "minspk": 3,
        "nsur": 1000,
        "prct": 99.90,
        "cent_thr": 99.90,
        "inner_corr": 5.0,
        "minsize": 3,
    }

SGC
^^^

.. code:: python
    :number-lines:

    parameters_sgc = {
        "use_first_derivative": False,
        "standard_deviations_threshold": 2,
        "shuffling_rounds": 1000,
        "coactivity_significance_level": 0.05,
        "montecarlo_rounds": 5,
        "montecarlo_steps": 10000,
        "affinity_threshold": 0.2,
    }


Simple analysis script
----------------------

In the following example I define a sample raster binary with ``100`` neurons and ``2000`` timepoints and analyze it using the **SVD** algorithm.

.. code:: python
    :number-lines:

    import numpy as np
    from encore.runners.encore import run_svd

    # Dummy data for the analysis
    neurons = 100
    timepoints = 2000
    raster = np.random.rand(neurons, timepoints)
    raster = (raster > 0.9).astype(np.int_)

    data = {
        'data_neuronal_activity': raster
    }

    # Define algorithm parameters
    parameters = {
        "pks": 3,
        "scut": 0.24,
        "hcut": 0.24,
        "state_cut": 6,
        "csi_start": 0.01,
        "csi_step": 0.01,
        "csi_end": 0.1,
        "tf_idf_norm": True,
        "parallel_processing": False,
        "fixed_ens_cant": 0,
    }

    # Run analysis
    results = run_svd(data, parameters)


Analysis output
---------------

The output of every runner contains a dictionary like the following:

.. code:: python
    :number-lines:

    results = {
        "success": True, # Bool value indicating if the algorithm was executed successfully
        "algorithm_time": 95.47, # Float with the running time of the algorithm in seconds 
        "engine_time": 11.42, # Float with the loading time for the MATLAB engine if used

        "results": {
            # Probably the most important one, the minimal results of the algorithm
            "ensembles_cant": 8, # int, number of ensembles identified
            "neus_in_ens": np.ndarray,  # 2D binary numpy array showing what neurons belong to each ensemble
                                        # with shape (N, E) for N neurons and E identified ensembles
            "timecourse": np.ndarray,   # 2D binary array showing the activity of each ensemble in time
                                        # with shape (T, E) for T timepoints and E ensembles.
        },

        "update_params": {
            # Contains parameters updated by the algorithm, e.g. during automatic parameter estimation
            # these keys and values varies from each algorithm and parameter selection.     
            "pks": 3,
            "scut": 0.24, 
        },

        "answer": {
            # Contains internal variables produced by the algorithm, this depends on the algorithm
            # This variable is saved when the runner is executed with the argument include_answer=True
            "Pks_Frame": np.ndarray, #2D
            "Pools_coords": np.ndarray, #3D
            "num_state": 17,
            "state_cut": 17,
            # etc ...
        }
    }

.. important::

    The keys ``"success"``, ``"algorithm_time"``, ``"engine_time"`` and ``"results"`` are returned for every algorithm and have the same logic. The ones that change depending on the algorithm are ``"update_params"`` and ``"answer"``.


Saving results example
----------------------

You can easily save the results of that analysis using the built in function ``save_data_to_hdf5_file``, check :doc:`/api/data/save_data`.

.. code:: python
    :number-lines:

    from encore.data.save_data import save_data_to_hdf5_file
    
    result_path = folder_path / "svd_results.h5"
    save_data_to_hdf5_file(result_path, results)

.. tip::

    Use a graphical HDF5 visualizer for a quick glance on the results.


Using a logger
--------------

You can track the progress of the function by passing a logger function, for example:

.. code:: python
    :number-lines:

    def logger_adapter(message: str, level: str):
        print(f"{level.upper()} - {message}")


    results = run_svd(data, parameters, logger=logger_adapter)
