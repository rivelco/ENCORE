import os
import yaml
try:
    from importlib.resources import files
    tmp = files('encore.gui')
except (TypeError, ImportError):  # Python < 3.10
    from importlib_resources import files
    
from typing import Any, Dict
from pydantic import BaseModel, create_model, field_validator

def validate_user_parameters(user_parameters: dict, defaults: dict):
    """
    Validate user parameters against defaults.

    :param user_parameters: Dictionary mapping parameter_name with value given
        by the user.
    :type user_parameters: dict
    :param defaults: Dictionary mapping the name of each parameter and the
        default value used to infer the type.
    :type defaults: dict
    :raises RuntimeError: When the user parameters failed to validate.
    :return: Validated output model.
    :rtype: Pydantic Validated Model
    """
    Model = build_parameters_model(defaults)

    try:
        return Model.model_validate(user_parameters)
    except Exception as exc:
        raise RuntimeError(f"Invalid parameters: {exc}") from exc


def build_parameters_model(defaults: dict) -> type[BaseModel]:
    """
    Dynamically create a Pydantic model that validates user parameters
    against defaults.
    
    :param defaults: Dictionary mapping the name of each parameter and the
        default value used to infer the type.
    :type defaults: dict
    :return: Pydantic model.
    :rtype: Pydantic Model
    """

    fields = {}

    for name, cfg in defaults.items():
        default_value = cfg["default_value"]
        expected_type = type(default_value)

        # Required field, no default
        fields[name] = (expected_type, ...)

    ParametersModel = create_model(
        "ParametersModel",
        __config__={"extra": "forbid"},  # No extra parameters
        **fields
    )

    # Attach enum validators where needed
    for name, cfg in defaults.items():
        possible_values = cfg.get("possible_values")

        if possible_values:

            def make_validator(param_name, allowed):
                @field_validator(param_name)
                @classmethod
                def validate_enum(cls, v):
                    if v not in allowed:
                        raise ValueError(
                            f"Invalid value '{v}'. Allowed values: {allowed}"
                        )
                    return v

                return validate_enum

            setattr(
                ParametersModel,
                f"validate_{name}_enum",
                make_validator(name, possible_values),
            )

    return ParametersModel


def collect_parameters(algorithm_cfg: dict) -> dict:
    """
    Collects the default values from an algorithm config dict as defined
    in the YAML config file.

    :param algorithm_cfg: Dictionary with the definition of an algorithm.
    :type algorithm_cfg: dict
    :return: Dictionary mapping the name of each parameter and the
        default value used to infer the type.
    :rtype: dict
    """
    parameters = algorithm_cfg.get("parameters", {})
    default_parameters = {}
    for param_name, param_cfg in parameters.items():
        # Used to infer the type of the parameter
        default_parameters[param_name] = {
            'default_value': param_cfg.get("default_value") 
        }
        possible_enum = param_cfg.get("type", "")
        
        # This is a one-of-many, multiple choice value, store available values
        if possible_enum == "enum":
            enum_values = param_cfg.get("options", [])
            possible_values = []
            for option in enum_values:
                option_value = option.get("value")
                if option_value:
                    possible_values.append(option_value)
            default_parameters[param_name]['possible_values'] = possible_values
        
    return default_parameters

def validate(algorithm: str, user_parameters: dict) -> dict:
    """
    Validates the parameters for an algorithm using as ground truth the config file 
    for the algorithm.

    :param algorithm: Name of the algorithm, is the key in the config file.
    :type algorithm: str
    :param user_parameters: Dictionary with parameters to validates. Maps 
        user_parameters['parameter_name'] = parameter_value. 
        The name of each parameter is defined in the config file.
    :type user_parameters: dict
    :return: Dictionary with the validated parameter to value mapping.
    :rtype: dict
    :raises RuntimeError: If the YAML config file can not be loaded.
    :raises RuntimeError: If the algorithm name could not be found in the YAML config file.
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
    
    if algorithm in runners:
        defaults = collect_parameters(runners[algorithm])

        validated_params = validate_user_parameters(
            user_parameters=user_parameters,
            defaults=defaults
        )
        
        return validated_params.model_dump()
    else:
        raise RuntimeError(f"The algorithm key {algorithm} is not in the config file.")