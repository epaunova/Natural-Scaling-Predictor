import numpy as np
from scipy.optimize import curve_fit
from typing import Dict

class ExtendedScalingLaw:
    """Extended scaling law implementation."""

    def __init__(self):
        self.params = {}
        self.fitted = False

    @staticmethod
    def scaling_function(inputs: np.ndarray, A: float, alpha: float, B: float, beta: float, E: float) -> np.ndarray:
        N, D, C = inputs[:, 0], inputs[:, 1], inputs[:, 2]
        return A / (N ** alpha) + B / (D ** beta) + E

    def fit(self, params: np.ndarray, data: np.ndarray, compute: np.ndarray, losses: np.ndarray) -> Dict[str, float]:
        inputs = np.column_stack([params, data, compute])
        p0 = [1.0, 0.076, 1.0, 0.095, 0.1]
        popt, _ = curve_fit(
            lambda x, *p: self.scaling_function(x, *p), inputs, losses, p0=p0, maxfev=10000
        )
        self.params = dict(zip(['A', 'alpha', 'B', 'beta', 'E'], popt))
        self.fitted = True
        return self.params

    def predict(self, params: np.ndarray, data: np.ndarray, compute: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        inputs = np.column_stack([params, data, compute])
        return self.scaling_function(inputs, **self.params)

class TraditionalScalingLaw:
    """Traditional scaling law implementation."""

    def __init__(self):
        self.params = {}
        self.fitted = False

    @staticmethod
    def scaling_function(N: np.ndarray, A: float, alpha: float, B: float) -> np.ndarray:
        return A / (N ** alpha) + B

    def fit(self, params: np.ndarray, losses: np.ndarray) -> Dict[str, float]:
        p0 = [1.0, 0.076, 0.1]
        popt, _ = curve_fit(self.scaling_function, params, losses, p0=p0, maxfev=10000)
        self.params = dict(zip(['A', 'alpha', 'B'], popt))
        self.fitted = True
        return self.params

    def predict(self, params: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.scaling_function(params, **self.params)
