def test_svd():
    import numpy as np
    from encore.runners.encore import run_svd
    
    # Dummy data
    neurons = 100
    timepoints = 2000
    
    # Dummy raster
    raster = np.random.rand(neurons, timepoints)
    raster = (raster > 0.9).astype(np.int_)
    
    # Parameters
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
    
    result = run_svd({'data_neuronal_activity': raster}, parameters)
    expected_keys = ['results', 'engine_time', 'algorithm_time', 'success', 'update_params', 'answer']
    all_included = True
    for key in expected_keys:
        if not key in result:
            all_included = False
    
    assert all_included == True
    assert len(expected_keys) == len(result.keys())

def test_pca():
    import numpy as np
    from encore.runners.encore import run_pca
    
    # Dummy data
    neurons = 100
    timepoints = 2000
    
    # Dummy raster
    raster = np.random.rand(neurons, timepoints)
    raster = (raster > 0.9).astype(np.int_)
    
    # Parameters
    parameters = {
        'dc': 0.01,
        'npcs': 3,
        'minspk': 3,
        'nsur': 1000,
        'prct': 99.90,
        'cent_thr': 99.90,
        'inner_corr': 5.0,
        'minsize': 3
    }
    
    result = run_pca({'data_neuronal_activity': raster}, parameters)
    expected_keys = ['results', 'engine_time', 'algorithm_time', 'success', 'answer']
    all_included = True
    for key in expected_keys:
        if not key in result:
            all_included = False
    
    assert all_included == True
    assert len(expected_keys) == len(result.keys())

def test_ica():
    import numpy as np
    from encore.runners.encore import run_ica
    
    # Dummy data
    neurons = 100
    timepoints = 2000
    
    # Dummy raster
    raster = np.random.rand(neurons, timepoints)
    raster = (raster > 0.9).astype(np.int_)
    
    # Parameters
    parameters = {
        'threshold_method': 'MarcenkoPastur',
        'permutations_percentile': 95.0,
        'number_of_permutations': 20,
        'patterns_method': 'ICA',
        'number_of_iterations': 500
    }
    
    result = run_ica({'data_neuronal_activity': raster}, parameters)
    expected_keys = ['results', 'engine_time', 'algorithm_time', 'success', 'original_answer', 'answer']
    all_included = True
    for key in expected_keys:
        if not key in result:
            all_included = False
    
    assert len(expected_keys) == len(result.keys())
    assert all_included == True

def test_x2p():
    import numpy as np
    from encore.runners.encore import run_x2p
    
    # Dummy data
    neurons = 100
    timepoints = 2000
    
    # Dummy raster
    raster = np.random.rand(neurons, timepoints)
    raster = (raster > 0.9).astype(np.int_)
    
    # Parameters
    parameters = {
        'NetworkBin': 1,
        'NetworkIterations': 1000,
        'NetworkSignificance': 0.05,
        'CoactiveNeuronsThreshold': 2,
        'ClusteringRangeStart': 3,
        'ClusteringRangeEnd': 10,
        'ClusteringFixed': 1,
        'EnsembleIterations': 1000,
        'ParallelProcessing': False
    }
    
    result = run_x2p({'data_neuronal_activity': raster}, parameters)
    expected_keys = ['results', 'engine_time', 'algorithm_time', 'success', 'original_answer', 'answer']
    all_included = True
    for key in expected_keys:
        if not key in result:
            all_included = False
    
    assert len(expected_keys) == len(result.keys())
    assert all_included == True
    
def test_sgc():
    import numpy as np
    from encore.runners.encore import run_sgc
    
    # Dummy data
    neurons = 100
    timepoints = 2000
    
    # Dummy dFFo
    dFFo = np.random.rand(neurons, timepoints)
    
    # Parameters
    parameters = {
        'use_first_derivative': False,
        'standard_deviations_threshold': 2,
        'shuffling_rounds': 1000,
        'coactivity_significance_level': 0.05,
        'montecarlo_rounds': 5,
        'montecarlo_steps': 10000,
        'affinity_threshold': 0.02
    }
    
    result = run_sgc({'data_dFFo': dFFo}, parameters)
    expected_keys = ['results', 'engine_time', 'algorithm_time', 'success', 'answer']
    all_included = True
    for key in expected_keys:
        if not key in result:
            all_included = False
    
    assert len(expected_keys) == len(result.keys())
    assert all_included == True

def test_example():
    import numpy as np
    from encore.runners.encore import run_example
    
    # Dummy data
    neurons = 100
    timepoints = 2000
    
    # Dummy dFFo
    dFFo = np.random.rand(neurons, timepoints)
    raster = np.random.rand(neurons, timepoints)
    raster = (raster > 0.9).astype(np.int_)
    
    # Parameters
    parameters = {
        'int_parameter_ensembles': 5,
        'int_parameter_A': 5,
        'int_parameter_B': 8,
        'float_parameter': 0.0,
        'threshold_parameter': 0.5,
        'bool_parameter': False,
        'selection_parameter': 'SUM'
    }
    data = {
        'data_neuronal_activity': raster,
        'data_dFFo': dFFo
    }
    result = run_example(data, parameters)
    expected_keys = ['results', 'answer', 'update_params', 'algorithm_time', 'success']
    all_included = True
    for key in expected_keys:
        if not key in result:
            all_included = False
    
    assert len(expected_keys) == len(result.keys())
    assert all_included == True

#def logger_adapter(message: str, level: str):
#    print(f"{level.upper()} - {message}")
    
#if __name__ == "__main__":    
#    test_example()
    
    