import matlab.engine

def dict_to_matlab_struct(self, pars_dict):
    """
    Converts a Python dictionary to a MATLAB struct.

    :param pars_dict: A dictionary where keys represent the names of fields in the MATLAB struct and the values 
                    represent the corresponding field values.
    :type pars_dict: dict

    :return: A MATLAB struct where the keys are the field names and the values are converted to the appropriate MATLAB data type.
    :rtype: dict

    This function recursively converts a Python dictionary to a MATLAB struct. It handles nested dictionaries by 
    recursively calling the conversion function. Numeric values (integers or floats) are converted into MATLAB 
    double type, while other data types are kept unchanged.
    """
    matlab_struct = {}
    for key, value in pars_dict.items():
        if isinstance(value, dict):
            matlab_struct[key] = self.dict_to_matlab_struct(value)
        elif isinstance(value, (int, float)):
            matlab_struct[key] = matlab.double([value])
        else:
            matlab_struct[key] = value
    return matlab_struct