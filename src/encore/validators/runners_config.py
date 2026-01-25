import re
from typing import Any, Dict, List, Optional, Union, Literal
from pydantic import RootModel, BaseModel, field_validator, model_validator
import matplotlib.colors as mcolors

ALGO_NAME_REGEX = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

NEEDED_DATA = {
    'data_neuronal_activity': 'Binary Neuronal Activity',
    'data_dFFo': 'dFFo Activity',
    'data_coordinates': 'Coordinates',
    'data_stims': 'Stimulation data',
    'data_cells': 'Cells data',
    'data_behavior': 'Behavior data'
}

def validate_algorithm_short_name(name: str) -> str:
    """
    Validates that the name of an algorithm has no special characters.

    :param name: Short name of the algorithm
    :type name: str
    :raises ValueError: If the name contains other than letters, numbers or underscores
    :return: Same name validated
    :rtype: str
    """
    if not ALGO_NAME_REGEX.match(name):
        raise ValueError(
            f"Invalid algorithm key '{name}'. Must start with a letter or '_' "
            "and contain only letters, numbers, or underscores."
        )
    return name

class FigureConfig(BaseModel):
    name: str
    display_name: str
    
class ParameterOption(BaseModel):
    value: str
    label: str
    description: str
    
BoundValue = Union[
    int,
    float,
    Dict[str, Union[int, float, str]],
    Literal["MIN_VAL", "MAX_VAL"]
]

class ParameterConfig(BaseModel):
    object_name: str
    display_name: str
    description: str
    default_value: Union[int, float, bool, str]

    type: Optional[str] = None
    options: Optional[List[ParameterOption]] = None

    min_value: Optional[BoundValue] = None
    max_value: Optional[BoundValue] = None

    @model_validator(mode="after")
    def validate_parameter_config(self):
        # Enum validation
        if self.type == "enum":
            if not self.options or len(self.options) == 0:
                raise ValueError(
                    "Parameter with type='enum' must define a non-empty 'options' list"
                )

            allowed_values = {opt.value for opt in self.options}
            if self.default_value not in allowed_values:
                raise ValueError(
                    f"default_value '{self.default_value}' is not among enum options {allowed_values}"
                )

        # MIN MAX validation
        self._validate_bounds()

        return self

    def _validate_bounds(self):
        """
        Validate min_value and max_value logic when possible.
        """

        # If both are numeric then require order
        if isinstance(self.min_value, (int, float)) and isinstance(self.max_value, (int, float)):
            if self.min_value > self.max_value:
                raise ValueError(
                    f"min_value ({self.min_value}) cannot be greater than max_value ({self.max_value})"
                )

        # Validate default_value against numeric limits when possible
        if isinstance(self.default_value, (int, float)):
            if isinstance(self.min_value, (int, float)) and self.default_value < self.min_value:
                raise ValueError(
                    f"default_value ({self.default_value}) is less than min_value ({self.min_value})"
                )

            if isinstance(self.max_value, (int, float)) and self.default_value > self.max_value:
                raise ValueError(
                    f"default_value ({self.default_value}) is greater than max_value ({self.max_value})"
                )

class AlgorithmConfig(BaseModel):
    enabled: bool
    full_name: str
    language: Literal["MATLAB", "Python"]
    analysis_function: str
    plot_function: str
    ensemble_color: str
    needed_data: List[str]
    source: str
    figures: List[FigureConfig]
    parameters: Dict[str, ParameterConfig]

    @field_validator("ensemble_color")
    @classmethod
    def validate_color(cls, v: str) -> str:
        """
        Validates that the value for 'ensembles_color' is a valid Matplotlib color.

        :param v: Color to test
        :type v: str
        :raises ValueError: If the string given is Matplotlib incompatible
        :return: The same color validated
        :rtype: str
        """
        if not mcolors.is_color_like(v):
            raise ValueError(f"Invalid matplotlib color: '{v}'")
        return v

    @field_validator("needed_data")
    @classmethod
    def validate_needed_data(cls, v: List[str]):
        """
        Validates that there are data needed and the requested data is one
        of the available options.

        :param v: List of strings with the names of the requested variables
        :type v: List[str]
        :raises ValueError: If there's no data in the needed_data field
        :raises ValueError: If requesting a non supported variable name
        :return: The same list of names validated
        :rtype: List[str]
        """
        if not v:
            raise ValueError("'needed_data' must contain at least one entry")
        
        allowed_needed_data = set(NEEDED_DATA.keys())
        invalid = set(v) - allowed_needed_data
        if invalid:
            raise ValueError(
                f"Invalid values in 'needed_data': {sorted(invalid)}. "
                f"Allowed values are: {sorted(allowed_needed_data)}"
            )

        return v

    @field_validator("parameters")
    @classmethod
    def validate_parameter_keys(cls, params: Dict[str, ParameterConfig]):
        """
        Iterates over the parameter fields of an algorithm to validate them.

        :param params: Dictionary pairing dictionary name and config for that parameter.
        :type params: Dict[str, ParameterConfig]
        :return: Same dictionary validated
        :rtype: Dict[str, ParameterConfig]
        """
        for name in params.keys():
            validate_algorithm_short_name(name)
        return params


class EncoreConfig(RootModel):
    root: Dict[str, AlgorithmConfig]

    @field_validator("root")
    @classmethod
    def validate_algorithm_keys(cls, algorithms: Dict[str, AlgorithmConfig]):
        """
        Validates each algorithm in the config file.

        :param algorithms: Dictionary of algorithms to validate
        :type algorithms: Dict[str, AlgorithmConfig]
        :return: Same dictionary validated
        :rtype: Dict[str, AlgorithmConfig]
        """
        for algo_name in algorithms.keys():
            validate_algorithm_short_name(algo_name)
        return algorithms
    
class EncoreConfigFile(BaseModel):
    encore_runners: EncoreConfig

def validate_encore_algorithm_config(config_dict: dict) -> EncoreConfigFile:
    """
    Validates a single algorithm config.

    :param config_dict: Dictionary mapping an algorithm short name with its config.
    :type config_dict: dict
    :raises RuntimeError: If the dictionary can't be validated
    :return: Pydantic validated model
    :rtype: EncoreConfigFile
    """
    try:
        return AlgorithmConfig.model_validate(config_dict)
    except Exception as exc:
        raise RuntimeError(f"{exc}") from exc

def validate_encore_config(config_dict: dict) -> EncoreConfigFile:
    """
    Validates the entire ENCORE config file for runners

    :param config_dict: Dictionary with encore runners objects
    :type config_dict: dict
    :raises RuntimeError: If the dictionary can't be validated
    :return: Validated structure
    :rtype: EncoreConfigFile
    """
    try:
        return EncoreConfigFile.model_validate(config_dict)
    except Exception as exc:
        raise RuntimeError(f"Invalid ENCORE configuration file: {exc}") from exc

