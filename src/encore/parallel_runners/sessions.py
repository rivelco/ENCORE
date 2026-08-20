import os
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from encore.runners.encore import (
    run_svd,
    run_ica,
    run_pca,
    run_sgc,
    run_x2p,
    run_example,
)
from encore.data.save_data import save_data_to_hdf5_file
from encore.utils.metrics import compute_similarity_matrix


def run_single_session(args):
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
        if algorithm == "svd":
            output = run_svd(
                data,
                parameters,
                include_answer=False,
            )
        elif algorithm == "ica":
            output = run_ica(
                data,
                parameters,
                include_answer=False,
            )
        elif algorithm == "x2p":
            output = run_x2p(
                data,
                parameters,
                include_answer=False,
            )
        elif algorithm == "pca":
            output = run_pca(
                data,
                parameters,
                include_answer=False,
            )
        elif algorithm == "sgc":
            output = run_sgc(
                data,
                parameters,
                include_answer=False,
            )
        elif algorithm == "example":
            output = run_example(
                data,
                parameters,
                include_answer=False,
            )
        else:
            raise RuntimeError(f"The specified algorithm {algorithm} is not defined.")
    except Exception as exc:
        print(f"Error for db {database_name}: ", exc)

    # Failed run
    if not output["success"]:
        return database_name, parameters, {
            "success": False,
            "results": None,
            "similarity": None,
            "similarity_labels": None,
        }

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
                element_labels = database_data.get(label_name, [])
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
    output_folder: Path,
    max_workers_cant: int,
    file_name="sessions_batch_results.h5",
    default_key="default",
):

    # Make sure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    print(f"\n{'=' * 70}\n" f"Running batch session analysis\n" f"{'=' * 70}")

    # Define results
    results = {
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
                f"No default parameters under key '{default_key}' set and those were needed."
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

    # Save results
    print("\n++++++++ Saving databases results file ++++++++")
    file_name = f"{file_name}.h5" if not file_name.endswith(".h5") else file_name
    result_path = output_folder / file_name
    print(f"   -> Destination: {result_path}")
    save_data_to_hdf5_file(result_path, results)

    print("     - Done with databases.")
