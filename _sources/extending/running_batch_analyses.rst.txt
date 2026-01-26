Running batch analyses using the ENCORE Python API
==================================================

In addition to the graphical interface, ENCORE provides a Python API that allows users to run ensemble identification algorithms programmatically. This is particularly useful for batch analyses, parameter sweeps, or automated pipelines where multiple datasets must be analyzed consistently.

The example below demonstrates how to organize multiple datasets and parameter sets, run the same algorithm across them, and collect the results in a structured dictionary. This approach enables reproducible analyses and facilitates downstream comparison across datasets or parameter configurations.

Simple batch analysis script
----------------------------

.. code-block:: python

    import numpy as np
    from encore.runners.encore import run_svd

    # Generate example datasets
    datasets = {}

    for dataset_id in ["session_1", "session_2", "session_3"]:
        neurons = 100
        timepoints = 2000

        raster = np.random.rand(neurons, timepoints)
        raster = (raster > 0.9).astype(np.int_)

        datasets[dataset_id] = {
            'data_neuronal_activity': raster
        }

    # Define algorithm parameters
    parameters = {
        'pks': 3,
        'scut': 0.22,
        'hcut': 0.22,
        'state_cut': 6,
        'csi_start': 0.01,
        'csi_step': 0.01,
        'csi_end': 0.1,
        'tf_idf_norm': True,
        'parallel_processing': False
    }

    # Run batch analysis
    results = {'results': {}}

    for dataset_name, data in datasets.items():
        output = run_svd(data, parameters)

        results['results'][dataset_name] = {
            'success': output['success'],
            'algorithm_time': output['algorithm_time'],
            'engine_time': output['engine_time'],
            'results': output['results'],
            'answer': output['answer']
        }

    # Example access
    # results['results']['session_1']['results']
    # results['results']['session_1']['results']['timecourse']
    # results['results']['session_1']['results']['ensembles_cant']
    # results['results']['session_1']['results']['neus_in_ens']
    # results['results']['session_1']['algorithm_time']


Notes for this example
----------------------

- Each dataset is stored as a dictionary matching the expected ENCORE input format.
- Parameters are defined once and reused across all datasets, ensuring consistent analysis.
- Parameter validation occurs inside the same algorithm.
- Results are collected in a single dictionary (results['results']), making it easy to iterate, compare, or serialize outputs.
- This workflow allows users to analyze multiple datasets sequentially, even though the GUI operates on one dataset at a time.

For advanced use cases, such as parameter sweeps, cross-algorithm comparisons, or integration with custom preprocessing pipelines, refer to the full API documentation for :doc:`../api/runners`.