import h5py
import numpy as np
from pathlib import Path


def save_dict_to_hdf5(group: h5py.Group, data: dict | list | int | float | str | np.ndarray):
    """
    Recursively saves data to an HDF5 file group.

    This method iterates through a dictionary and saves its contents to the provided HDF5 group. 
    If a value in the dictionary is another dictionary, it creates a subgroup and recursively saves 
    its contents. 
    If the value is a list, it attempts to create a dataset in the group, catching exceptions 
    if the data cannot be saved.
    Datasets are stored using a custom compression level 3 with gzip.
    For other data types, the method directly stores the value in the group.

    :param group: The HDF5 group to which the data will be saved.
    :type group: h5py.Group
    :param data: The data to be saved, which can be a dictionary, list, or other types.
    :type data: dict
    """
    for key, value in data.items():
        key = str(key)
        if isinstance(value, dict):
            subgroup = group.create_group(str(key))
            save_dict_to_hdf5(subgroup, value)
        elif isinstance(value, list):
            try:
                group.create_dataset(
                    key, data=value, compression="gzip", compression_opts=3
                )
                # group.create_dataset(key, data=value, **hdf5plugin.Blosc(cname='zstd', clevel=3, shuffle=hdf5plugin.Blosc.SHUFFLE))
            except:
                print(
                    f" ENCORE Saving: Could not save a variable called {key}, maybe it is not a matrix nor scalar."
                )
        elif isinstance(value, np.ndarray):
            group.create_dataset(
                key, data=value, compression="gzip", compression_opts=3
            )
        else:
            if value is None:
                value = ""
            group[key] = value


def save_data_to_hdf5_file(path: Path, data_dict: dict):
    """
    Saves a python dictionary structure to a file in the specified path.
    
    Uses the function `save_data_to_hdf5` to recursively save the contents of the
    dictionary to a path, creating the necessary groups and datasets.

    Args:
        path (Path): Path of the output file. Must be .h5 file
        data_dict (dict): Dictionary with the variables to save.
    """
    with h5py.File(path, "w") as f:
        save_dict_to_hdf5(f, data_dict)
