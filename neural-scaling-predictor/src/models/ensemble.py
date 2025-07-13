import torch
import torch.nn as nn
from typing import List, Dict, Optional
from .nsp_model import ScalingLawPredictor

class EnsemblePredictor(nn.Module):
    """Ensemble of multiple NSP models for improved performance."""

    def __init__(self, models: List[ScalingLawPredictor], weights: Optional[List[float]] = None):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.num_models = len(models)
        self.weights = (
            torch.ones(self.num_models) / self.num_models if weights is None else torch.tensor(weights)
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        preds, uncs, ems = [], [], []
        for m in self.models:
            out = m(x)
            preds.append(out['predictions'])
            uncs.append(out['uncertainties'])
            ems.append(out['emergence_scores'])

        preds = torch.stack(preds)
        weighted_preds = torch.sum(preds * self.weights.view(-1, 1, 1), dim=0)

        uncs = torch.stack(uncs)
        mean_unc = torch.mean(uncs, dim=0)

        ems = torch.stack(ems)
        mean_ems = torch.mean(ems, dim=0)

        return {'predictions': weighted_preds, 'uncertainties': mean_unc, 'emergence_scores': mean_ems}
