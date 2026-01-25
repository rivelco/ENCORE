import os
import numpy as np
import yaml
try:
    from importlib.resources import files
    tmp = files('encore.gui')
except (TypeError, ImportError):  # Python < 3.10
    from importlib_resources import files
from encore.utils.parameters_validators import validate_binary_matrix

NEEDED_DATA_TEMPLATE = {
    'data_neuronal_activity': {
        'name': 'Binary Neuronal Activity',
        'type': np.ndarray,
        'expect_binary': True,
        'dimensions': 2
    },
    'data_dFFo': {
        'name': 'dFFo Activity',
        'type': np.ndarray,
        'expect_binary': False,
        'dimensions': 2
    },
    'data_coordinates': {
        'name': 'Coordinates',
        'type': np.ndarray,
        'expect_binary': False,
        'dimensions': 2
    },
    'data_stims': {
        'name': 'Stimulation data',
        'type': np.ndarray,
        'expect_binary': True,
        'dimensions': 2
    },
    'data_cells': {
        'name': 'Cells data',
        'type': np.ndarray,
        'expect_binary': True,
        'dimensions': 2
    },
    'data_behavior': {
        'name': 'Behavior data',
        'type': np.ndarray,
        'expect_binary': False,
        'dimensions': 2
    }
}

def get_algorithm_config(algorithm: str) -> dict:
    """
    Returns the config dictionary for the specified algorithm.

    :param algorithm: Key of the algorithm
    :type algorithm: str
    :raises RuntimeError: If the config file is not found.
    :return: Dictionary with the config of the algorithm
    :rtype: dict
    """
    # Read the config file
    config = {}
    config_yaml_path = str(files("encore.config").joinpath("encore_runners_config.yaml"))
    
    if os.path.exists(config_yaml_path):
        with open(config_yaml_path, 'r') as file:
            config = yaml.safe_load(file)
    else:
        raise RuntimeError(f"YAML config file not found in {config_yaml_path}")
    
    # Extract the runners from config
    runners = config.get("encore_runners", {})
    return runners.get(algorithm, {})
     
def simple_validate(algorithm: str, input_data: dict[str, np.ndarray]) -> dict:
    """
    Validates the input data given as a dict for a given algorithm, using as 
    reference the data in the ENCORE config file and the basic definition of 
    each ENCORE variable. This validation includes:
    - Checking that all the needed variables by the algorithm are present in the input dict
    - Checking that all are numpy arrays
    - Checking the number of dimensions of each variable
    - Checking that some variables are binary arrays

    :param algorithm: String with the key (short name) of the algorithm
    :type algorithm: str
    :param input_data: Dictionary mapping the name of a ENCORE variable with is value
    :type input_data: dict[str, np.ndarray]
    :raises RuntimeError: If the input data dict does not contains a needed variable
    :raises RuntimeError: If a input variable is not a numpy array
    :raises RuntimeError: If the numpy matrix has a invalid number of dimensions
    :raises RuntimeError: If a variable needs to be binary and it is not.
    :return: Same input dictionary validated
    :rtype: dict[str, np.ndarray]
    """
    
    algorithm_cfg = get_algorithm_config(algorithm)
    needed_data = algorithm_cfg.get('needed_data', {})
    
    # Sad nested ifs, will be refactored with Pydantic later
    for need in needed_data:
        if need in input_data:
            given_data = input_data[need]
            needed_type = NEEDED_DATA_TEMPLATE[need]['type']
            if isinstance(given_data, needed_type):
                needed_dims = NEEDED_DATA_TEMPLATE[need]['dimensions']
                if len(given_data.shape) == needed_dims:
                    if NEEDED_DATA_TEMPLATE[need]['expect_binary']:
                        if not validate_binary_matrix(given_data):
                            raise RuntimeError(f"The input data {need} is expected to be binary.")
                else:
                    raise RuntimeError(f"The input data {need} is expected to have {needed_dims} dimensions.")
            else:
                raise RuntimeError(f"The input data {need} is not a {needed_type}.")
        else:
            raise RuntimeError(f"The input data dict does not contains {need}.")
        
    return input_data