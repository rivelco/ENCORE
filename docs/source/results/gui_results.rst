GUI Results
===========

The analysis results can be saved in ``.h5``, ``.mat`` or ``.npz``. 

Results description
-------------------

The results file contain nested structures depending on the selected data to export.

.. note::

    For the following examples suppose that only ICA and X2P analyses has been performed on the current database.

.. seealso::

    Check out the available save options from the GUI in the demo section :doc:`/user_guide/usage_gui`.


Information
^^^^^^^^^^^

This is included in every results file, useful to identify the ENCORE version used for the analysis and the date of the analysis.

.. code:: yaml
    :number-lines:

    ENCORE:
        info:
            # Information about the analyzer and analysis date
            ENCORE_version: 3.0.0   # str, showing ENCORE version used for the analysis
            analyzer: ENCORE Single Database GUI    # String to identify the specific analyzer used
            date: 250826_143355     # str with the date, formatted DDMMYY_HHMMSS


.. tip::
    The specific string used in ``ENCORE["info"]["analyzer"]`` may be used to identify the specific analysis pipeline used. 
    This could be used in a better handling of results files in subsequent analyses.

.. note::

    Note the ``ENCORE`` key at the root of the file.

Input user data
^^^^^^^^^^^^^^^

This is the input data loaded by the user, useful to keep track of the data used for the analysis.

.. code:: yaml
    :number-lines:

    ENCORE:
        info:
            # ...
        input_data:
            # Input data loaded in the GUI, only the loaded data is available
            coordinates: [matrix] # Matrix with the coordinates
            neuronal_activity: [matrix] # Binary activity matrix
            stims: [matrix] # Binary matrix with stimulation


Minimal results used by ENCORE
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is the **most important** part of the results file. Contains the ensembles, neurons in ensembles and the activity of the ensembles for every algorithm used.

.. seealso::

    This is also returned for by the Python API runners. Check out :doc:`/extending/usage_api` and also :doc:`/extending/parallel_databases_api`. 

.. code:: yaml
    :number-lines:

    ENCORE:
        info:
            # ...
        results:
            # These are the minimal results
            ica:
                ensembles_cant: int     # Number of ensembles identified by this algorithm
                neus_in_ens: [matrix]   # 2D binary matrix shaped (neurons, ensembles)
                                        # A number 1 indicates that neuron belong to that ensemble
                timecourse: [matrix]    # 2D binary matrix shaped (timepoints, ensembles)
                                        # A number 1 indicates that ensemble active in that timepoint
            x2p:
                ensembles_cant: int
                neus_in_ens: [matrix]
                timecourse: [matrix]


Parameters
^^^^^^^^^^

The parameters used for every analysis on this database.

.. seealso::

    Check out the parameters structures used by every algorithm in the :doc:`/extending/algorithms_config_file` and in :doc:`/extending/usage_api`.

.. code:: yaml
    :number-lines:

    ENCORE:
        info:
            # ...
        parameters:
            # Parameters used by each algorithm
            ica:
                max_ensembles_cant: int 
                min_ensembles_cant: int
                number_of_iterations: int
                # ...
                # Each ICA parameter ...
            x2p:
                ClusteringFixed: int
                ClusteringRangeEnd: int
                # ...


Full results of every analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These are variables used internally by every algorithm, may be useful for further analysis based on their procedures. Check out the names of the variables and the references for each algorithm.

.. code:: yaml
    :number-lines:

    ENCORE:
        info:
            # ...
        algorithms_results:
            # This contains some internal variables used by the algorithms during the analysis
            # each field here is the short name of the algorithms used.
            # More documentation will be made available in future versions.
            ica:
                assembly_templates: [matrix] # 2D matrix
                binary_assembly_templates: [matrix] # 2D matrix
                # ...
            x2p:
                Count: int # integer
                Activity: [matrix] # 2D matrix


Ensembles Compare Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^

The results displayed in the "Ensembles Compare" tab of the GUI. These are similarity matrices over the members of each ensemble for every algorithm and also the similarity of ensembles activity with stimulation/behavior.

.. code:: yaml
    :number-lines:

    ENCORE:
        info:
            # ...
        ensembles_compare:
            # This is the data visualized in the "Ensembles compare" tab of the GUI.
            labels: [matrix] # 1D array with labels of the similarity matrices
            neus_in_ens:    # Matrix displayed in "Ensembles compare / Similarities in members"
                Correlation: [matrix] # 2D square matrix
                Cosine: [matrix]
                Euclidean: [matrix]
                Jaccard: [matrix]
            timecourse:     # Matrix displayed in "Ensembles compare / Similarities in timecourse"
                Correlation: [matrix] # 2D square matrix
                Cosine: [matrix]
                Euclidean: [matrix]
                Jaccard: [matrix]


Performance Comparison
^^^^^^^^^^^^^^^^^^^^^^

This is the data visualized in the "Performance Comparison" tab. The variables saved here depends on the actual variables loaded by the user, for example stimulation and behavior.


.. code:: yaml
    :number-lines:

    ENCORE:
        info:
            # ...
        ensembles_performance:
            # This is the data visualized in "Performance Comparison"
            correlation_cells:
                # From "Performance Comparison / Correlation between cells"
                ica:
                    # One matrix per ensemble identified by ica
                    Ensemble 1: [matrix] # 2D matrix of correlations between neurons in the ensemble 1
                    Ensemble 2: [matrix] # 2D matrix of correlations between neurons in the ensemble 2
                    # ... for every ensemble
                x2p:
                    # One matrix per ensemble identified by x2p
                    Ensemble 1: [matrix] # 2D matrix of correlations between neurons in the ensemble 1
                    # ... for every ensemble
            correlation_ensembles_stimuli:
                # From "Performance Comparison / Correlation with stimuli presentation"
                # One correlation matrix per algorithm.
                # This variable exists only if stimulation was provided
                ica: [matrix] # 2D matrix with shape (stimuli, ensembles)
                x2p: [matrix] # 2D matrix with shape (stimuli, ensembles)
            crosscorr_ensembles_stimuli:
                # From "Performance Comparison / Cross correlation ensembles and stimuli"
                # One matrix per algorithm.
                # This variable exists only if stimulation was provided
                ica:
                    # One matrix per ensemble identified by ica
                    Ensemble 1: [matrix] # 2D matrix of correlations between neurons in the ensemble 1
                    # ... for every ensemble
                x2p:
                    # One matrix per ensemble identified by x2p
                    Ensemble 1: [matrix] # 2D matrix of cross correlation ensembles with stimuli
                    # ... for every ensemble
            correlation_ensembles_behavior:
                # Same as for stimuli but with behavior, if provided
            crosscorr_ensembles_behavior:
                # Same as for stimuli but with behavior, if provided


Loading results files
---------------------

There are several ways to work with the results files, depending on the format used. 

The following example extracts the version of ENCORE used in the analysis, the number of ensembles identified by the ICA algorithm, the activity of the ensembles and the neurons binary activity.


Opening h5 files using python
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python
    :number-lines:

    import numpy as np
    import h5py

    file_path = "ENCORE_250826_161754_.h5"

    with h5py.File(file_path, "r") as hdf_file:
        
        # For the analyzer version
        encore_info = hdf_file["ENCORE"]["info"]
        # Read every element as a numpy array
        encore_version_d = np.array(encore_info["ENCORE_version"]).flatten()
        # Decode the text like this
        encore_version = encore_version_d[0].decode("utf-8")
        
        # For the ensembles results
        results = hdf_file["ENCORE"]["results"]
        ica_results = results["ica"]
        ensembles_cant = int(np.array(ica_results["ensembles_cant"]))   # Read it as number
        ensembles_timecourse = np.array(ica_results["timecourse"])
        
        # For the neuronal activity
        input_data = hdf_file["ENCORE"]["input_data"]
        neuronal_activity = np.array(input_data["neuronal_activity"])
        coords = np.array(input_data["coordinates"])
        
        # Just to verify that the data makes sense
        # ENCORE version
        assert encore_version == "3.0.0"
        # Number of ensembles
        assert ensembles_timecourse.shape[0] == ensembles_cant
        # Number of neurons
        assert neuronal_activity.shape[0] == coords.shape[0]
        # Number of timepoints
        assert neuronal_activity.shape[1] == ensembles_timecourse.shape[1]

.. seealso::
    Check out the `h5py documentation <https://docs.h5py.org/en/stable/quick.html>`_ for a better understanding on how to use the format.