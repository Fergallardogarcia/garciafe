"""Type definitions"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Union, Dict, List, Optional, Any

from torch import Tensor

Parameters = Tensor

Scalar = Union[bool, bytes, float, int, str]

Metrics = Dict[str, Scalar]
MetricsAggregationFn = Callable[[List[tuple[int, Metrics]]], Metrics]

class Code(Enum):
    """Client status codes."""
    OK = 0
    GET_PROPERTIES_NOT_IMPLEMENTED = 1
    GET_PARAMETERS_NOT_IMPLEMENTED = 2
    FIT_NOT_IMPLEMENTED = 3
    EVALUATE_NOT_IMPLEMENTED = 4


@dataclass
class Status:
    """Client status."""

    code: Code
    message: str


# @dataclass
# class Parameters:
#     """Model parameters."""

#     tensors: List[bytes]
#     tensor_type: str


@dataclass
class GanAttackFitPayload:
    """Additional fit payload for GAN attack clients."""
    perturbation_direction: Optional[Parameters] = None
    perturbation_magnitude: Optional[float] = None


@dataclass
class FitIns:
    """Fit instructions for a client."""

    parameters: Parameters
    config: dict[str, Scalar]
    gan_attack_payload: Optional[GanAttackFitPayload] = None


@dataclass
class FitRes:
    """Fit response from a client."""

    status: Status
    parameters: Parameters
    num_examples: int
    metrics: dict[str, Scalar]
    param_array: dict[str, Parameters] = field(default_factory=dict)


@dataclass
class EvaluateIns:
    """Evaluate instructions for a client."""

    parameters: Parameters
    config: dict[str, Scalar]
    param_array: dict[str, Parameters] = field(default_factory=dict)
    


@dataclass
class EvaluateRes:
    """Evaluate response from a client."""

    status: Status
    loss: float
    num_examples: int
    metrics: dict[str, Scalar]
