import h5py
import numpy as np
import scipy.io

def assign_data_from_file(self):
    var_path = self.file_selected_var_path
    filename = self.source_filename
    model_type = self.file_model_type
    
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
