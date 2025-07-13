"""
Neural Scaling Predictor (NSP) - A framework for predicting LLM performance.
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .models import ScalingLawPredictor
from .utils import compute_metrics, visualize_results

__all__ = ["ScalingLawPredictor", "compute_metrics", "visualize_results"]
