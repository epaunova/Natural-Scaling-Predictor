"""
Neural Scaling Predictor Models
"""

from .nsp_model import ScalingLawPredictor
from .scaling_laws import ExtendedScalingLaw
from .ensemble import EnsemblePredictor

__all__ = ["ScalingLawPredictor", "ExtendedScalingLaw", "EnsemblePredictor"]
