from typing import Dict, Any, Optional
import numpy as np
from pydantic import BaseModel, field_validator, ConfigDict, model_validator

class ResultsBlock(BaseModel):
    ensembles_cant: int
    timecourse: np.ndarray
    neus_in_ens: np.ndarray

    model_config = ConfigDict(
        arbitrary_types_allowed=True
    )

    @field_validator("timecourse", "neus_in_ens")
    @classmethod
    def validate_binary_matrix(cls, v: np.ndarray):
        if not isinstance(v, np.ndarray):
            raise TypeError("Must be a numpy ndarray")

        if v.ndim != 2:
            raise ValueError("Must be a 2D matrix")

        unique_vals = np.unique(v)
        if not set(unique_vals).issubset({0, 1}):
            raise ValueError("Matrix must be binary (0/1)")

        return v

class AnalysisOutput(BaseModel):
    enabled: bool = False
    success: bool
    results: ResultsBlock
    answer: Dict[str, Any]

    engine_time: float = 0.0
    algorithm_time: float = 0.0
    update_params: Optional[Dict[str, Any]] = {}

    model_config = ConfigDict(
        extra="allow"
    )

    @model_validator(mode="after")
    def validate_shapes_with_context(self, info):
        ctx = info.context or {}

        neurons = ctx.get("neurons")
        timepoints = ctx.get("timepoints")

        if neurons is None or timepoints is None:
            raise ValueError("Validation context must include 'neurons' and 'timepoints'")

        ensembles_cant = self.results.ensembles_cant

        expected_n_shape = (ensembles_cant, neurons)
        expected_t_shape = (ensembles_cant, timepoints)

        if self.results.neus_in_ens.shape != expected_n_shape:
            raise ValueError(
                f"results.neus_in_ens has shape {self.results.neus_in_ens.shape}, expected {expected_n_shape}"
            )

        if self.results.timecourse.shape != expected_t_shape:
            raise ValueError(
                f"results.timecourse has shape {self.results.timecourse.shape}, expected {expected_t_shape}"
            )

        return self

def validate_analysis_output(output: dict, neurons: int, timepoints: int) -> AnalysisOutput:
    try:
        return AnalysisOutput.model_validate(
            output,
            context={
                "neurons": neurons,
                "timepoints": timepoints,
            }
        )
    except Exception as exc:
        raise RuntimeError(f"Invalid analysis output: {exc}") from exc
