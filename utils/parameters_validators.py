import warnings
import numpy as np

def validate_parameters_svd(params, defaults):
    """
    Retrieves user-defined parameters for Singular Value Decomposition (SVD) applies default values 
    if fields are empty.

    :return: None
    :rtype: None
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
        valid_params['statecut'] = float(params['statecut'])
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
        valid_params['csi_end'] = defaults['statecut']
        warnings.warn(f"csi_step not specified, the default value {valid_params['csi_end']} will be used", UserWarning)

    if params['parallel_processing']:
        valid_params['parallel_processing'] = True
    else:
        valid_params['parallel_processing'] = False

    return valid_params

def validate_parameters_ica(params, defaults):
    valid_params = {
        'threshold': {
            'method': 0,
            'permutations_percentile': 0,
            'number_of_permutations': 0
        },
        'Patterns': {
            'method': 0,
            'number_of_iterations': 0
        }
    }

    valid_params['threshold']['method'] = params['threshold']['method']

    if len(params['threshold']['permutations_percentile']) > 0:
        valid_params['threshold']['permutations_percentile'] = float(params['threshold']['permutations_percentile'])
    else:
        valid_params['threshold']['permutations_percentile'] = defaults['threshold']['permutations_percentile']

    if len(params['threshold']['number_of_permutations']) > 0:
        valid_params['threshold']['number_of_permutations'] = float(params['threshold']['number_of_permutations'])
    else:
        valid_params['threshold']['number_of_permutations'] = defaults['threshold']['number_of_permutations']

    if params['Patterns']['method'] != "ICA" and params['Patterns']['method'] != "PCA":
        valid_params['Patterns']['method'] = defaults['Patterns']['method']
    else:
        valid_params['Patterns']['method'] = params['Patterns']['method']

    if len(params['Patterns']['number_of_iterations']) > 0:
        valid_params['Patterns']['number_of_iterations'] = abs(float(params['Patterns']['number_of_iterations']))
    else:
        valid_params['Patterns']['number_of_iterations'] = defaults['Patterns']['number_of_iterations']
    return valid_params