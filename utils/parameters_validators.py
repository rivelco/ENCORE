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

    return valid_params