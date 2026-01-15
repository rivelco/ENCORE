import warnings
import numpy as np

def validate_parameters_svd(params, defaults):
    """
    Validates user-defined parameters for Singular Value Decomposition (SVD) and applies default values 
    if fields are empty.

    :param params: Dictionary with the parameters to be used by the algorithm
    :type params: dict
    :param defaults: Dictionary with the default values for the parameters.
    :type defaults: dict
    :return: Dictionary with the validated parameters.
    :rtype: dict
    """
    valid_params = {
        'pks': 0,
        'scut': 0,
        'hcut': 0,
        'statecut': 0,
        'tf_idf_norm': 0,
        'csi_start': 0,
        'csi_step': 0,
        'csi_end': 0,
        'parallel_processing': 0
    }

    if len(params['pks']) > 0:
        valid_params['pks'] = np.array([float(params['pks'])])
    else:
        valid_params['pks'] = np.array([])
        warnings.warn("pks not specified, this parameter will be calculated automatically.", UserWarning)

    if len(params['scut']) > 0:
        valid_params['scut'] = np.array([float(params['scut'])])  
    else: 
        valid_params['scut'] = np.array([])
        warnings.warn("scut not specified, this parameter will be calculated automatically.", UserWarning)

    if len(params['hcut']) > 0:
        valid_params['hcut'] = float(params['hcut']) 
    else:
        valid_params['hcut'] = defaults['hcut']
        warnings.warn(f"hcut not specified, the default value {valid_params['hcut']} will be used", UserWarning)

    if len(params['statecut']) > 0:
        valid_params['statecut'] = int(params['statecut'])
    else:
        valid_params['statecut'] = defaults['statecut']
        warnings.warn(f"state_cut not specified, the default value {valid_params['statecut']} will be used", UserWarning)

    if params['tf_idf_norm']:
        valid_params['tf_idf_norm'] = True
    else:
        valid_params['tf_idf_norm'] = False

    if len(params['csi_start']) > 0:
        valid_params['csi_start'] = float(params['csi_start'])
    else:
        valid_params['csi_start'] = defaults['csi_start']
        warnings.warn(f"csi_start not specified, the default value {valid_params['csi_start']} will be used", UserWarning)

    if len(params['csi_step']) > 0:
        valid_params['csi_step'] = float(params['csi_step'])
    else:
        valid_params['csi_step'] = defaults['csi_step']
        warnings.warn(f"csi_step not specified, the default value {valid_params['csi_step']} will be used", UserWarning)
        
    if len(params['csi_end']) > 0:
        valid_params['csi_end'] = float(params['csi_end'])
    else:
        valid_params['csi_end'] = defaults['csi_end']
        warnings.warn(f"csi_step not specified, the default value {valid_params['csi_end']} will be used", UserWarning)

    if params['parallel_processing']:
        valid_params['parallel_processing'] = True
    else:
        valid_params['parallel_processing'] = False

    return valid_params

def validate_parameters_pca(params, defaults):
    """
    Validates user-defined parameters for Principal Components Analysis (PCA) and applies default values 
    if fields are empty.

    :param params: Dictionary with the parameters to be used by the algorithm
    :type params: dict
    :param defaults: Dictionary with the default values for the parameters.
    :type defaults: dict
    :return: Dictionary with the validated parameters.
    :rtype: dict
    """
    
    valid_params = {}
    if len(params['dc']) > 0:
        valid_params['dc'] = float(params['dc'])
    else:
        valid_params['dc'] = defaults['dc']
        
    if len(params['npcs']) > 0:
        valid_params['npcs'] = int(params['npcs'])
    else:
        valid_params['npcs'] = defaults['npcs']
    
    if len(params['minspk']) > 0:
        valid_params['minspk'] = int(params['minspk'])
    else:
        valid_params['minspk'] = defaults['minspk']
    
    if len(params['nsur']) > 0:
        valid_params['nsur'] = int(params['nsur'])
    else:
        valid_params['nsur'] = defaults['nsur']
    
    if len(params['prct']) > 0:
        valid_params['prct'] = float(params['prct'])
    else:
        valid_params['prct'] = defaults['prct']
    
    if len(params['cent_thr']) > 0:
        valid_params['cent_thr'] = float(params['cent_thr'])
    else:
        valid_params['cent_thr'] = defaults['cent_thr']
    
    if len(params['inner_corr']) > 0:
        valid_params['inner_corr'] = float(params['inner_corr'])
    else:
        valid_params['inner_corr'] = defaults['inner_corr']
    
    if len(params['minsize']) > 0:
        valid_params['minsize'] = int(params['minsize'])
    else:
        valid_params['minsize'] = defaults['minsize']
    
    return valid_params

def validate_parameters_ica(params, defaults):
    """
    Validates user-defined parameters for Independent Component Analysis (ICA) and applies default values 
    if fields are empty.

    :param params: Dictionary with the parameters to be used by the algorithm
    :type params: dict
    :param defaults: Dictionary with the default values for the parameters.
    :type defaults: dict
    :return: Dictionary with the validated parameters.
    :rtype: dict
    """
    
    valid_params = {
        'threshold': {
            'method': 'MarcenkoPastur',
            'permutations_percentile': 95,
            'number_of_permutations': 1
        },
        'Patterns': {
            'method': 'ICA',
            'number_of_iterations': 1
        }
    }

    valid_params['threshold']['method'] = params['threshold']['method']

    if len(params['threshold']['permutations_percentile']) > 0:
        valid_params['threshold']['permutations_percentile'] = float(params['threshold']['permutations_percentile'])
    else:
        valid_params['threshold']['permutations_percentile'] = defaults['threshold']['permutations_percentile']

    if len(params['threshold']['number_of_permutations']) > 0:
        valid_params['threshold']['number_of_permutations'] = int(params['threshold']['number_of_permutations'])
    else:
        valid_params['threshold']['number_of_permutations'] = defaults['threshold']['number_of_permutations']

    if params['Patterns']['method'] != "ICA" and params['Patterns']['method'] != "PCA":
        valid_params['Patterns']['method'] = defaults['Patterns']['method']
    else:
        valid_params['Patterns']['method'] = params['Patterns']['method']

    if len(params['Patterns']['number_of_iterations']) > 0:
        valid_params['Patterns']['number_of_iterations'] = abs(int(params['Patterns']['number_of_iterations']))
    else:
        valid_params['Patterns']['number_of_iterations'] = defaults['Patterns']['number_of_iterations']
    return valid_params

def validate_parameters_x2p(params, defaults):
    """
    Validates user-defined parameters for xsembles2P (X2P) and applies default values 
    if fields are empty.

    :param params: Dictionary with the parameters to be used by the algorithm
    :type params: dict
    :param defaults: Dictionary with the default values for the parameters.
    :type defaults: dict
    :return: Dictionary with the validated parameters.
    :rtype: dict
    """
    
    valid_params = {}
    
    if len(params['NetworkBin']) > 0:
        valid_params['NetworkBin'] = int(params['NetworkBin'])
    else:
        valid_params['NetworkBin'] = defaults['network_bin']
    
    if len(params['NetworkIterations']) > 0:
        valid_params['NetworkIterations'] = int(params['NetworkIterations'])
    else:
        valid_params['NetworkIterations'] = defaults['network_iterations']
    
    if len(params['NetworkSignificance']) > 0:
        valid_params['NetworkSignificance'] = float(params['NetworkSignificance'])
    else:
        valid_params['NetworkSignificance'] = defaults['network_significance']  
    
    if len(params['CoactiveNeuronsThreshold']) > 0:
        valid_params['CoactiveNeuronsThreshold'] = int(params['CoactiveNeuronsThreshold'])
    else:
        valid_params['CoactiveNeuronsThreshold'] = defaults['coactive_neurons_threshold']  
    
    if len(params['ClusteringRangeStart']) > 0:
        valid_params['ClusteringRangeStart'] = int(params['ClusteringRangeStart'])
    else:
        valid_params['ClusteringRangeStart'] = defaults['clustering_range_start']  
    
    if len(params['ClusteringRangeEnd']) > 0:
        valid_params['ClusteringRangeEnd'] = int(params['ClusteringRangeEnd'])
    else:
        valid_params['ClusteringRangeEnd'] = defaults['clustering_range_end']  
    
    if valid_params['ClusteringRangeStart'] > valid_params['ClusteringRangeEnd']:
        valid_params['ClusteringRangeEnd'] = valid_params['ClusteringRangeStart'] + abs(valid_params['ClusteringRangeStart'] - valid_params['ClusteringRangeEnd'])
    
    if len(params['ClusteringFixed']) > 0:
        valid_params['ClusteringFixed'] = int(params['ClusteringFixed'])
    else:
        valid_params['ClusteringFixed'] = defaults['clustering_fixed']  
    
    if len(params['EnsembleIterations']) > 0:
        valid_params['EnsembleIterations'] = int(params['EnsembleIterations'])
    else:
        valid_params['EnsembleIterations'] = defaults['iterations_ensemble']  
    
    valid_params['ParallelProcessing'] = params['ParallelProcessing']
    valid_params['FileLog'] = params['FileLog']
    
    return valid_params

def validate_parameters_sgc(params, defaults):
    valid_params = {}
    
    valid_params['use_first_derivative'] = params['use_first_derivative']
    
    if len(params['standard_deviations_threshold']) > 0:
        valid_params['standard_deviations_threshold'] = float(params['standard_deviations_threshold'])
    else:
        valid_params['standard_deviations_threshold'] = defaults['standard_deviations_threshold']
        
    if len(params['shuffling_rounds']) > 0:
        valid_params['shuffling_rounds'] = int(params['shuffling_rounds'])
    else:
        valid_params['shuffling_rounds'] = defaults['shuffling_rounds']
    
    if len(params['coactivity_significance_level']) > 0:
        valid_params['coactivity_significance_level'] = float(params['coactivity_significance_level'])
    else:
        valid_params['coactivity_significance_level'] = defaults['coactivity_significance_level']
    
    if len(params['montecarlo_rounds']) > 0:
        valid_params['montecarlo_rounds'] = int(params['montecarlo_rounds'])
    else:
        valid_params['montecarlo_rounds'] = defaults['montecarlo_rounds']
    
    if len(params['montecarlo_steps']) > 0:
        valid_params['montecarlo_steps'] = int(params['montecarlo_steps'])
    else:
        valid_params['montecarlo_steps'] = defaults['montecarlo_steps']  
    
    if len(params['affinity_threshold']) > 0:
        valid_params['affinity_threshold'] = float(params['affinity_threshold'])
    else:
        valid_params['affinity_threshold'] = defaults['affinity_threshold']
        
    return valid_params