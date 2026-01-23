import h5py
import numpy as np
import scipy.io
import pickle

def assign_data_from_file(var_path: str, filename: str, model_type: str):
    """
    Loads a variable form a file given a particular way of finding variables
    in the file, defined by model_type. Returns None if the model type is not
    defined.

    :param var_path: String with the path to the desired variable. Nested structures
        are separated using "/"
    :type var_path: str
    :param filename: String with the path to the file containing the variable
    :type filename: str
    :param model_type: String defining the variable search type, available options
        are "hdf5", "np_flatten", "pkl", "mat", and "csv"
    :type model_type: str
    :return: Numpy matrix containing the requested variable
    :rtype: numpy.ndarray
    """
    if model_type == "hdf5": 
        with h5py.File(filename, 'r') as hdf_file:
            # Split the dataset path into individual components
            path_components = var_path.split('/')
            # Start from the root of the file
            current_group = hdf_file
            # Traverse the file hierarchy
            for component in path_components:
                # Check if the component is not empty (for cases like "//")
                if component:
                    current_group = current_group[component]
            # Read the dataset
            dataset = current_group[()]
            return dataset
    elif model_type == "np_flatten":
        numpy_file = np.load(filename)
        # Remove the trailing '//' at the beginning of the path
        return numpy_file[var_path[2:]]
    elif model_type == "pkl":
        with open(filename, 'rb') as file:
            pkl_file = pickle.load(file)
        # Split the dataset path into individual components
        path_components = var_path.split('/')
        # Start from the root of the file
        current_group = pkl_file
        # Traverse the file hierarchy
        for component in path_components:
            # Check if the component is not empty (for cases like "//")
            if component:
                current_group = current_group[component]
        # Read the dataset
        dataset = current_group
        return dataset
    elif model_type == "mat":
        mat_file = scipy.io.loadmat(filename)
        # Split the path in components avoiding the empty segments 
        path_components = [component for component in var_path.split('/') if component]

        # Only the first level elements has real names, for the rest the important thing is the index
        renamed_components = []
        for component_it in range(len(path_components)):
            if component_it == 0:
                renamed_components.append(path_components[component_it])
            else:
                component = path_components[component_it].split('_')
                component_idx = int(component[-1])
                renamed_components.append(component_idx)

        # Actual access to the variable in the file 
        current_var = mat_file
        for component in renamed_components:
            current_var = current_var[component]

        return current_var
    elif model_type == "csv":
        return np.loadtxt(filename, delimiter=',')
    else:
        return None
