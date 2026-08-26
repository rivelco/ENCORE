Analyzing databases in parallel using the Python API
====================================================

The `parallel_runners` module contains the core analysis pipelines used to identify and compare neuronal ensembles.

These functions can be executed exclusively from Python scripts.

Each parallel runner implements a complete analysis workflow for a set of databases, including:

- Input validation
- Algorithm selection
- Execution
- Result formatting

.. seealso::

   Read the documentation for the parallel runners over databases :doc:`/api/parallel_runners`


Example parallel run
--------------------

.. code:: python
   :number-lines:
   
   import numpy as np

   databases = {}
   neurons = 100
   timepoints = 3600
   orientations = 4
   directions = 8
   behaviors = 2
   orientations_labels = ["0", "45", "90", "135"]
   directions_labels = ["0", "45", "90", "135", "180", "225", "270", "315"]
   behaviors_labels = ["speed", "pupil"]

   for database_id in ["MiceA", "MiceB", "MiceC"]:
      # The binary activity matrix for the neurons
      spikes_matrix = np.random.rand(neurons, timepoints)
      spikes_matrix = (spikes_matrix > 0.9).astype(np.int_)

      # A binary matrix for the orientations presentation
      orientations_matrix = np.random.rand(orientations, timepoints)
      orientations_matrix = (orientations_matrix > 0.75).astype(np.int_)

      # A binary matrix for the directions presentation
      directions_matrix = np.random.rand(directions, timepoints)
      directions_matrix = (directions_matrix > 0.75).astype(np.int_)

      # A continuous matrix for the behaviors
      behaviors_matrix = np.random.rand(behaviors, timepoints)

      databases[database_id] = {
            "spikes": spikes_matrix,
            "orientations": orientations_matrix,
            "directions": directions_matrix,
            "behaviors": behaviors_matrix,
            "orientations_labels": orientations_labels,
            "directions_labels": directions_labels,
            "behaviors_labels": behaviors_labels
      }

Define the parameters to use for each database and add a fallback if needed.

.. code:: python
   :number-lines:

   sessions_parameters = {
      "default": {
         "analysis": "svd",
         "data_names": {"data_neuronal_activity": "spikes"},
         "parameters": {
               "pks": 5,
               "scut": 0.4,
               "hcut": 0.4,
               "state_cut": 6,
               "csi_start": 0.01,
               "csi_step": 0.01,
               "csi_end": 0.1,
               "tf_idf_norm": True,
               "parallel_processing": False,
               "fixed_ens_cant": 0,
         },
         "evaluate_similarity": True,
         "similarity_elements": ["orientations", "directions"],
      },
      "MiceB": {
            "analysis": "ica",
            "data_names": {"data_neuronal_activity": "spikes"},
            "parameters": {
                "threshold_method": "MarcenkoPastur",
                "permutations_percentile": 95.0,
                "number_of_permutations": 20,
                "min_ensembles_cant": 4,
                "max_ensembles_cant": 10,
                "patterns_method": "ICA",
                "number_of_iterations": 1000,
                "threshold_for_p_value": 1.90,
            },
            "evaluate_similarity": True,
            "similarity_elements": ["orientations", "behaviors"],
      },
   }


These parameters indicates that the databases with ID ``MiceA`` and ``MiceC`` will be analyzed with the **SVD** algorithm, while the database ``MiceB`` will be analyzed with the algorithm **ICA**. 

For ``MiceB`` the similarity will be computed between the activity of the identified ensembles and the presentation of orientations and the behaviors. For ``MiceA`` and ``MiceC`` the similarities will be computed for the presentation of stimuli only.

.. important::

   In the example, we are using the variables ``"orientations"`` and ``"directions"`` for the similarities comparisons, 
   as stated in the variable ``sessions_parameters["default"]["similarity_elements"]``. 

   It's important for clarity to also include variables with the labels for each one. The runner will append 
   the string ``"_labels"`` at the end of the name of the variable for similarity and will look for that variable
   in the ``databases[database_id]`` dictionary. That's why here I included the variables 
   ``databases[database_id]["orientations_labels"]`` and ``databases[database_id]["directions_labels"]``.

   If no label is provided the index of each element will be used as label.


.. seealso::

   Check the :doc:`/extending/algorithms_config_file` and :doc:`/extending/usage_api` for a better understanding of the data and parameters needed for each algorithm.

.. important::

   When using parallel execution of the algorithms do not to use the parallel processing options of any algorithm, like **SVD**, **X2P** or **PCA**.


Then call the parallel runner with:

.. code:: python
   :number-lines:

   from encore.parallel_runners.sessions import run_parallel_sessions

   workers = 4 # This depends on your computer's available resources

   results = run_parallel_sessions(
      data=databases,
      parameters=sessions_parameters,
      max_workers_cant=workers,
   )

.. note::

   Each worker will create a new python instance with a new MATLAB instance each.


Results structure
-----------------

The results of this analysis will produce a python directory with the following fields:

.. code:: python
   :number-lines:

   results = {
      "info": {
         "analyzer": "ENCORE Parallel Sessions API", # str, Identifier of this analysis
         "date": "250826_143355",            # str, with the date, formatted DDMMYY_HHMMSS
         "ENCORE_version": "3.0.0"           # str, showing ENCORE version used for the analysis
      },

      "parameters": {
         # The parameters passed to the parallel analysis, a copy of the input
         # In this example it is:
         "default": {
            "analysis": "svd",
            "data_names": {"data_neuronal_activity": "spikes"},
            "parameters": {
                  "pks": 5,
                  "scut": 0.4,
                  "hcut": 0.4,
                  # ...
            },
            "evaluate_similarity": True,
            "similarity_elements": ["orientations", "directions"],
         },
         "MiceB": {
            "analysis": "ica",
            # ...
         },
      },

      "parameters_used": {
         # Each key is the name of each database analyzed, with the parameters used for that one
         # In this case MiceA and MiceB have the same parameters because those used the default parameter
         "MiceA": {
            "analysis": "svd",
            "data_names": {"data_neuronal_activity": "spikes"},
            # ...
            "similarity_elements": ["orientations", "directions"],
         },
         "MiceB": {
            "analysis": "ica",
            "data_names": {"data_neuronal_activity": "spikes"},
            # ...
            "similarity_elements": ["orientations", "behaviors"],
         },
         "MiceC": {
            "analysis": "svd",
            "data_names": {"data_neuronal_activity": "spikes"},
            # ...
            "similarity_elements": ["orientations", "directions"],
         },
      },

      "results": {
         # The most important one, each key is the name of each database used
         "MiceA": {
            "success": True, # bool, Shows if the analysis terminated successfully

            "results": {
               # The minimal results of the algorithm
               "ensembles_cant": 8, # int, number of ensembles identified
               "neus_in_ens": np.ndarray,  # 2D binary numpy array showing what neurons belong to each ensemble
                                          # with shape (N, E) for N neurons and E identified ensembles
               "timecourse": np.ndarray,   # 2D binary array showing the activity of each ensemble in time
                                          # with shape (T, E) for T timepoints and E ensembles.
            },

            "similarity": {
               # Contains a pair of similarity_metric: similarity_matrix
               # The keys are the names of the metric used
               # The matrix are square matrix, each side with the number of ensembles identified
               # plus the number of elements in the similarity variables
               # In this case (8 ensembles) + (4 orientations) + (8 directions)
               "Correlation": np.ndarray,
               "Cosine": np.ndarray,
               "Euclidean": np.ndarray, # The euclidean distance
               "Jaccard": np.ndarray, # The Jaccard index
            },

            # The labels for the elements in the similarity matrices, in this case
            # ["ens 0", "ens 1", ..., "ens 8", "ori 0", ..., "ori 135", "dir 0", ..., "dir 315"]
            "similarity_labels": np.ndarray,
         },
         "MiceB": {
            "success": True,
            "results": {
               "ensembles_cant": 6, 
               "neus_in_ens": np.ndarray,
               "timecourse": np.ndarray,
            },
            "similarity": {
               # In this case (6 ensembles) + (4 orientations) + (2 behaviors)
               "Correlation": np.ndarray,
               "Cosine": np.ndarray,
               "Euclidean": np.ndarray, # The euclidean distance
               "Jaccard": np.ndarray, # The Jaccard index
            },

            # ["ens 0", ..., "ens 6", "ori 0", ..., "ori 135", "beh speed", "beh pupil"]
            "similarity_labels": np.ndarray,
         },
         "MiceC": {
            "success": True,
            # ...
         }
      }
   }


Saving results example
----------------------

You can easily save the results of that analysis using the built in function ``save_data_to_hdf5_file``, check :doc:`/api/data/save_data`.

.. code:: python
   :number-lines:

   from encore.data.save_data import save_data_to_hdf5_file

   result_path = folder_path / "parallel_databases_results.h5"
   save_data_to_hdf5_file(result_path, results)

.. tip::

   Use a graphical HDF5 visualizer for a quick glance on the results.
