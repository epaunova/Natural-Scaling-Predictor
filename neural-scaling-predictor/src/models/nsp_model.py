import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

class ScalingLawPredictor(nn.Module):
    """
    Neural Scaling Predictor (NSP) for LLM performance prediction.
    """

    def __init__(
        self,
        input_dim: int = 127,
        hidden_dim: int = 512,
        num_layers: int = 4,
        num_tasks: int = 12,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_tasks = num_tasks

        # Feature embedding layers
        self.param_embed = nn.Linear(1, hidden_dim // 4)
        self.data_embed = nn.Linear(1, hidden_dim // 4)
        self.compute_embed = nn.Linear(1, hidden_dim // 4)
        self.arch_embed = nn.Linear(input_dim - 3, hidden_dim // 4)

        # Multi-head attention for feature interaction
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=8, dropout=dropout
        )

        # Transformer-style encoder layers
        self.encoder_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=8,
                    dim_feedforward=hidden_dim * 4,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        # Task-specific prediction heads
        self.task_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim // 2, 1),
                )
                for _ in range(num_tasks)
            ]
        )

        # Uncertainty estimation head
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_tasks),
        )

        # Emergence detection head
        self.emergence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_tasks),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size = x.size(0)

        params = x[:, 0:1]
        data = x[:, 1:2]
        compute = x[:, 2:3]
        arch_features = x[:, 3:]

        # Embed features
        param_emb = self.param_embed(torch.log10(params + 1e-8))
        data_emb = self.data_embed(torch.log10(data + 1e-8))
        compute_emb = self.compute_embed(torch.log10(compute + 1e-8))
        arch_emb = self.arch_embed(arch_features)

        embeddings = torch.cat(
            [param_emb, data_emb, compute_emb, arch_emb], dim=1
        )

        attn_output, _ = self.attention(
            embeddings.unsqueeze(1), embeddings.unsqueeze(1), embeddings.unsqueeze(1)
        )
        features = attn_output.squeeze(1)

        for layer in self.encoder_layers:
            features = layer(features.unsqueeze(1)).squeeze(1)

        predictions = torch.cat(
            [head(features) for head in self.task_heads], dim=1
        )

        uncertainties = F.softplus(self.uncertainty_head(features))
        emergence_scores = torch.sigmoid(self.emergence_head(features))

        return {
            "predictions": predictions,
            "uncertainties": uncertainties,
            "emergence_scores": emergence_scores,
        }

    @classmethod
    def from_pretrained(cls, model_name: str) -> "ScalingLawPredictor":
        model = cls()
        return model

    def save_pretrained(self, save_path: str) -> None:
        torch.save(self.state_dict(), save_path)
