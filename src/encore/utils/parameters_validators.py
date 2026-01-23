import warnings

def validate_params(params: dict, defaults: dict) -> dict:
    """
    Validate and coerce parameter values based on defaults.

    - Uses the type of each default value as the expected type
    - If a parameter is missing or invalid, the default is used
    - Extra keys in params are ignored

    :param params: Parameters collected from the GUI
    :param defaults: Default parameter values (type and fallback)
    :return: Validated parameter dictionary
    """
    
    validated = {}

    for key, default_value in defaults.items():
        if key not in params:
            validated[key] = default_value
            continue

        value = params[key]
        expected_type = type(default_value)

        try:
            # Special handling for booleans
            if expected_type is bool:
                if isinstance(value, bool):
                    validated[key] = value
                elif isinstance(value, (int, float)):
                    validated[key] = bool(value)
                elif isinstance(value, str):
                    validated[key] = value.strip().lower() in ("1", "true", "yes", "on")
                else:
                    validated[key] = default_value

            # Strings
            elif expected_type is str:
                validated[key] = str(value)

            # Numeric types (int, float)
            else:
                validated[key] = expected_type(value)

        except (ValueError, TypeError):
            warnings.warn(f"{key} not specified or invalid, this parameter will be loaded from defaults", UserWarning)
            validated[key] = default_value

    return validated
