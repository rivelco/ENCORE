import numpy as np
import importlib.metadata
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from encore.utils.metrics import compute_similarity_matrix


def run_single_session(
    args: tuple[str, str, dict[str, np.ndarray], dict[str, str], dict, bool, list[str]],
) -> tuple[str, dict, dict]:
    """
    Auxiliary function to identify the algorithm to run, organize the user's data and
    organize the result for this run.
    This is an internal function expected to be used only by
    :meth:`encore.parallel_runners.sessions.run_parallel_sessions`
    Every worker uses this function.

    :param args: Tuple with the arguments for the runner. This is:
        **algorithm**: str, The 3-chars name of the algorithm, e.g. 'ica', 'svd', 'pca', 'sgc', 'x2p'
        **database_name**: str, The name of the database used. This is to reference asynchronously the results
        with the correct database.
        **database_data**: dict[str, np.ndarray], The data for the analysis.
        **data_names**: dict[str, str], The names of the API variables.
        **parameters**: dict, The dictionary with the parameters for the algorithm.
        **eval_similarity**: bool, whether or not to perform the evaluation of similarities.
        **similarity_elements**: list[str], List with the names of the variables to compute similarities with.
    :type args: tuple[str, str, dict[str, np.ndarray], dict[str, str], dict, bool, list[str]]
    :raises RuntimeError: If the selected algorithm is not known.
    :return: Tuple with the results of the run. This is:
        **database_name**: str, the name of the database as it was received
        **parameters**: dict, the parameters for the algorithm
        **result**: dict, contains the fields
        "success": bool, "results": dict, "similarity": dict, "similarity_labels": list[str],
    :rtype: tuple[str, dict, dict]
    """
    algorithm = args[0]
    database_name = args[1]
    database_data = args[2]
    data_names = args[3]
    parameters = args[4]
    eval_similarity = args[5]
    similarity_elements = args[6]

    print(f"Running [{database_name}]")

    data = {
        input_key: database_data[db_key] for input_key, db_key in data_names.items()
    }

    # Run algorithm
    output = {"success": False}
    try:
        # Dynamic load of the module and function
        module_path = "encore.runners.encore"
        module = importlib.import_module(module_path)
        function_name = f"run_{algorithm}"
        func = getattr(module, function_name, None)
        
        if func is None or not callable(func):
            raise RuntimeError(f"The specified algorithm {algorithm} is not defined.")
        
        output = func(
            data,
            parameters,
            include_answer=False,
        )
    except Exception as exc:
        print(f"Error for db {database_name}: ", exc)

    # Failed run
    if not output["success"]:
        return (
            database_name,
            parameters,
            {
                "success": False,
                "results": None,
                "similarity": None,
                "similarity_labels": None,
            },
        )

    # Compute similarity
    similarities = {}
    compare_labels = []
    if eval_similarity:
        sim_elements = [
            database_data[elem] for elem in similarity_elements if elem in database_data
        ]
        if len(sim_elements) == len(similarity_elements):
            sim_elements_mat = np.vstack(sim_elements)
            compare_mat = np.vstack([output["results"]["timecourse"], sim_elements_mat])
            ensembles_cant = output["results"]["timecourse"].shape[0]

            compare_labels = [f"ens {i}" for i in range(ensembles_cant)]
            for element in similarity_elements:
                # Look for the labels, like "orientation_labels"
                label_name = f"{element}_labels"
                element_labels = database_data.get(
                    label_name, list(range(database_data[element].shape[0]))
                )
                # Append each new label for each element like "ori 90"
                compare_labels.extend(f"{label_name[0:3]} {i}" for i in element_labels)

            sim_methods = ["Cosine", "Euclidean", "Correlation", "Jaccard"]
            for method in sim_methods:
                similarities[method] = compute_similarity_matrix(
                    method,
                    compare_mat,
                )
        else:
            if len(sim_elements) == 0:
                print(f" $$ WARN: No elements for similarity found for {database_name}")
            else:
                print(
                    f" $$ WARN: Some elements missing for similarity for {database_name}"
                )

    result = {
        "success": True,
        "results": output["results"],
        "similarity": similarities,
        "similarity_labels": compare_labels,
    }

    return database_name, parameters, result


def run_parallel_sessions(
    data: dict[str, dict[str, np.ndarray]],
    parameters: dict[str, dict],
    max_workers_cant: int,
    default_key="default",
):
    """
    Run ensembles identification algorithms in parallel for different databases.
    The parameters for the analysis can be defined for each database or use a default one
    as a fallback.

    :param data: Dictionary containing the data for each database.
        The keys for the first dictionary is a string with the name or ID or the database.
        Each value at this level contains a dictionary with the data of that database.
        In this dictionary each key is a specific identifier for a numpy array containing experimental data,
        e.g. matrix of spikes, fluorescence traces, stimulation, behavior.
        You can use any name for these variables.
    :type data: dict[str, dict[str, np.ndarray]]
    :param parameters: Dictionary where each key is the ID of a database or a fallback name.
        The dictionary inside should contain the keys:
        - "analysis", with the 3-letter name of the algorithm to use.
        - "data_names", This is a dict[str, str] The keys should be one the name of the variables used by the algorithm,
        e.g "data_neuronal_activity", "data_dFFo" and the value is the name of that variable in the `data` parameter.
        - "parameters" with another dictionary with the parameters for that specific algorithm, as used by the API.
        - "evaluate_similarity", a bool variable to evaluate similarity metrics between results/data
        - "similarity_elements", a list of strings, each string is the key of variables in `data`. When `evaluate_similarity`
        is True, the batch processing will also calculate the similarity between the ensembles timecourse and this other variables, so
        these variables should be of the same length as the recording. This is useful to evaluate the performance of each
        algorithm in every database.
    :type parameters: dict[str, dict]
    :param max_workers_cant: Number of workers to use for parallel processing.
    :type max_workers_cant: int
    :param default_key: Specific key of the fallback parameter key. These set of parameters
        will be used for any database that is not explicitly in the keys of the `parameters` argument. Defaults to "default".
    :type default_key: str, optional

    :return: Dictionary with the results of the batch analyses. Contains the following keys.
            **info**: dict[str, str], the date and version information of the analyzer
            **parameters**: dict, The parameters used for the entire batch analysis.
            This is a copy of the parameters passed by the user to the function.
            **parameters_used**: dict, each key is the name of each database, contains the batch parameters used
            by this database. If it used the default parameters then it's a copy of its contents.
            **results**: dict, each key is the name of each database. Contains the minimal results of the ensembles algorithm
            The matrix of similarity and the labels of each similarity matrix and a flag of success for the algorithm.
    :rtype: dict[str, dict]

    :raises RuntimeError: If no fallback parameter is assigned and a database was not in the `parameters` keys.
    """

    print(
        f"\n{'=' * 70}\n" f" ENCORE: Running parallel session analysis\n" f"{'=' * 70}"
    )

    # Define the results info
    now = datetime.now()
    formatted_time = now.strftime("%d%m%y_%H%M%S")
    encore_version = str(importlib.metadata.version("encore-toolkit"))
    run_information = {
        "analyzer": "ENCORE Parallel Sessions API",
        "date": formatted_time,
        "ENCORE_version": encore_version,
    }

    # Define results
    results = {
        "info": run_information,
        "parameters": parameters,
        "parameters_used": {},
        "results": {},
    }

    # Process each database as a separate job
    jobs = []
    for database_name, database_data in data.items():
        parameters_set = {}
        if database_name in parameters:
            parameters_set = parameters[database_name]
        elif default_key in parameters:
            parameters_set = parameters[default_key]
        else:
            raise RuntimeError(
                f"No default parameters under key '{default_key}' set and it was needed."
            )

        results["parameters_used"][database_name] = parameters_set
        jobs.append(
            (
                parameters_set.get("analysis"),
                database_name,
                database_data,
                parameters_set.get("data_names"),
                parameters_set.get("parameters"),
                parameters_set.get("evaluate_similarity"),
                parameters_set.get("similarity_elements"),
            )
        )

    print(
        f"\n  -> After matching dbs and params {len(jobs)} databases will be added.\n"
    )

    # Parallel execution
    with ProcessPoolExecutor(max_workers=max_workers_cant) as executor:
        futures = [executor.submit(run_single_session, job) for job in jobs]

        for completed, future in enumerate(as_completed(futures), start=1):
            try:
                database_name, parameters_used, result = future.result()
                results["results"][database_name] = result

                print(f"[{completed}/{len(futures)}] " f"Finished {database_name}.")

            except Exception as exc:
                print(f"[{completed}/{len(futures)}] " f"ERROR: {exc}")

    print("     - Done with databases.")
    return results
