import h5py
import numpy as np

def assign_data_from_file(self):
    var_path = self.file_tree_selected
    filename = self.source_filename
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