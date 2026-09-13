"""Petit lecteur causal ; supervision terminale de séquence, pas d'onsets inventés."""
import numpy as np
import torch
from torch import nn


class SequenceModel(nn.Module):
    def __init__(self, hidden_size=32):
        super().__init__()
        self.lstm = nn.LSTM(10, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x, state=None):
        hidden, state = self.lstm(x, state)
        return self.head(hidden).squeeze(-1), state


def load_sequence(detector, device="cuda"):
    model = SequenceModel(detector.metadata["config"]["hidden_size"]).to(device)
    model.load_state_dict(torch.load(detector.directory / "lstm.pt", map_location=device, weights_only=True))
    model.eval()
    with np.load(detector.directory / "normalization.npz", allow_pickle=False) as values:
        mean, scale = values["mean"].copy(), values["scale"].copy()
    return model, mean, scale


def sequence_probability(model, values, mean, scale):
    x = (np.asarray(values, dtype=np.float64) - mean) / scale
    if x.shape != (30, 10) or not np.isfinite(x).all():
        raise ValueError("Contexte complet de 30×10 valeurs finies requis")
    with torch.inference_mode():
        logits, _ = model(torch.tensor(x[None], dtype=torch.float32, device=next(model.parameters()).device))
        result = float(torch.sigmoid(logits[0, -1]).item())
    if not np.isfinite(result):
        raise ValueError("Score séquentiel non fini")
    return result
