def test_parallel():
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

    databases_ids = ["MiceA", "MiceB", "MiceC", "MiceD", "MiceE"]

    for database_id in databases_ids:
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
            "behaviors_labels": behaviors_labels,
        }

    sessions_parameters = {
        "MiceA": {
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
        "MiceC": {
            "analysis": "x2p",
            "data_names": {"data_neuronal_activity": "spikes"},
            "parameters": {
                "NetworkBin": 1,
                "NetworkIterations": 1000,
                "NetworkSignificance": 0.05,
                "CoactiveNeuronsThreshold": 2,
                "ClusteringRangeStart": 3,
                "ClusteringRangeEnd": 10,
                "ClusteringFixed": 0,
                "EnsembleIterations": 3000,
                "ParallelProcessing": False,
            },
            "evaluate_similarity": True,
            "similarity_elements": ["orientations", "directions"],
        },
        "default": {
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

    from encore.parallel_runners.sessions import run_parallel_sessions

    workers = 4  # This depends on your computer's available resources

    batch_results = run_parallel_sessions(
        data=databases,
        parameters=sessions_parameters,
        max_workers_cant=workers,
    )

    expected_keys = ["info", "parameters", "parameters_used", "results"]
    all_included = True
    for key in expected_keys:
        if not key in batch_results:
            all_included = False

    all_included_dbs = True
    for key in databases_ids:
        if (
            not key in batch_results["results"]
            or not key in batch_results["parameters_used"]
        ):
            all_included_dbs = False

    # This is paired with the order of the databases ids
    expected_analysis = ["svd", "ica", "x2p", "ica", "ica"]
    correct_algorithm_assignation = True
    for idx, key in enumerate(databases_ids):
        if batch_results["parameters_used"][key]["analysis"] != expected_analysis[idx]:
            correct_algorithm_assignation = False

    assert all_included == True
    assert all_included_dbs == True
    assert correct_algorithm_assignation == True


if __name__ == "__main__":
    test_parallel()
