"""Faits temporels et formulations contrôlées ; aucune durée physique inventée."""
import hashlib
import json
from pathlib import Path

import numpy as np

LABELS = ("quiet", "brief", "intermittent", "persistent", "ended")
PHRASES = (
    "Aucun niveau élevé persistant n'est observé dans cet historique.",
    "Une activité élevée brève ou encore récente est observée.",
    "L'activité élevée apparaît de façon intermittente dans cet historique.",
    "Un niveau élevé persiste dans les observations récentes.",
    "Un retour durable sous le seuil est observé après une activité élevée.",
)
LETTERS = "ABCDE"
CHANNELS = ("crest factor", "kurtosis", "skewness", "absolute percentile 50", "absolute percentile 75",
            "absolute percentile 90", "absolute percentile 99", "percentile99/50 ratio", "envelope standard deviation", "C1 leak score")


def file_hash(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def validate_history(values):
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 10 or not 1 <= len(x) <= 64 or not np.isfinite(x).all():
        raise ValueError("Historique de 1 à64 fenêtres, dix valeurs finies par fenêtre requis")
    if np.any((x[:, 9] < 0) | (x[:, 9] > 1)):
        raise ValueError("Scores C1 hors [0,1]")
    return x


def facts(values):
    x = validate_history(values)
    high = x[:, 9] >= .8
    runs, current, longest = 0, 0, 0
    for flag in high:
        if flag:
            runs += current == 0
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    if current > 30:
        label = "persistent"
    elif high.any() and len(x) >= 5 and np.all(x[-5:, 9] <= .4):
        label = "ended"
    elif runs >= 2:
        label = "intermittent"
    elif high.any():
        label = "brief"
    else:
        label = "quiet"
    return {"pattern": label, "history_seconds": len(x), "high_seconds": int(high.sum()),
            "current_high_seconds": current, "longest_high_seconds": longest, "high_runs": runs}


def model_sample(values, mean, scale):
    import torch
    x = validate_history(values)
    normalized = np.clip((x - mean) / scale, -10, 10)
    normalized[:, 9] = x[:, 9] * 2 - 1
    padded = np.pad(normalized, ((64 - len(x), 0), (0, 0)))
    return {"pre_prompt": f"Describe temporal activity in {len(x)} observed one-second windows from C1. "
            f"The first {64-len(x)} values are padding, not observations. "
            "Read the numeric time series in chronological order. Do not diagnose a physical leak.",
        "time_series": torch.tensor(padded.T.copy(), dtype=torch.float32),
        "time_series_text": [name + ": " for name in CHANNELS],
        "post_prompt": "\nChoose a description: A=quiet, B=brief or recent high activity, "
            "C=intermittent high activity, D=persistent high activity for more than30 consecutive seconds, "
            "E=return below threshold for at least5seconds after high activity. Answer one letter:"}


class TemporalNarrator:
    def __init__(self, directory, expected_sha256, device="cuda"):
        import torch
        from pipe.tslm.model import AcousticQwenSP
        self.directory = Path(directory).resolve()
        if file_hash(self.directory / "metadata.json") != expected_sha256:
            raise ValueError("Métadonnées de langage différentes du gel")
        self.metadata = json.loads((self.directory / "metadata.json").read_text())
        for name, sha in self.metadata["files"].items():
            path = (self.directory / name).resolve()
            if not path.is_relative_to(self.directory) or file_hash(path) != sha:
                raise ValueError("Artefact de langage modifié")
        base = Path(self.metadata["base"])
        for name, sha in self.metadata["base_files"].items():
            path = (base / name).resolve()
            if not path.is_relative_to(base.resolve()) or file_hash(path) != sha:
                raise ValueError("Base Qwen modifiée")
        self.model = AcousticQwenSP(base, device, single_clip_acoustic_encoding=True)
        weights = torch.load(self.directory / "temporal.pt", map_location=device, weights_only=True)
        self.model.encoder.load_state_dict(weights["encoder_state"])
        self.model.projector.load_state_dict(weights["projector_state"])
        self.model.eval()
        with np.load(self.directory / "normalization.npz", allow_pickle=False) as stats:
            self.mean, self.scale = stats["mean"], stats["scale"]
        if self.mean.shape != (10,) or self.scale.shape != (10,) or not np.isfinite(self.mean).all() or not np.all(np.isfinite(self.scale) & (self.scale > 0)):
            raise ValueError("Normalisation de langage invalide")
        self.version = "opentslm-qwen-dynamics-" + expected_sha256[:12]

    def describe(self, history):
        import torch
        measured = facts(history)
        if measured["history_seconds"] < 31:
            return {"model_version":self.version,"model_pattern":None,"model_phrase":None,"candidate_scores":{},
                "facts":measured,"description":PHRASES[LABELS.index(measured["pattern"])],
                "description_source":"template_fallback","fallback_used":True,
                "fallback_reason":"fewer_than_31_recent_valid_windows","field_validated":False,
                "calibration":"none","generation":"not_run_insufficient_history"}
        with torch.inference_mode():
            logits = decision_logits(self.model, model_sample(history, self.mean, self.scale))
            scores = torch.softmax(logits.float(), -1).cpu().tolist()
        if not np.isfinite(scores).all():
            raise ValueError("Scores de langage non finis")
        index = int(np.argmax(scores))
        matched = LABELS[index] == measured["pattern"]
        return {"model_version": self.version, "model_pattern": LABELS[index],
            "model_phrase": PHRASES[index], "candidate_scores": dict(zip(LABELS, scores)),
            "facts": measured, "description": PHRASES[index] if matched else PHRASES[LABELS.index(measured["pattern"])],
            "description_source": "opentslm_qwen_constrained" if matched else "template_fallback",
            "fallback_used": not matched, "fallback_reason": None if matched else "model_disagrees_with_measured_pattern",
            "field_validated": False, "calibration": "none", "generation": "closed_vocabulary"}


def decision_logits(model, sample):
    import torch
    ids = [model.tokenizer.encode(c, add_special_tokens=False) for c in LETTERS]
    if any(len(tokens) != 1 for tokens in ids):
        raise ValueError("Une lettre doit correspondre à un seul token")
    inputs, mask = model.pad_and_apply_batch([sample])
    out = model.llm(inputs_embeds=inputs, attention_mask=mask, use_cache=False, logits_to_keep=1)
    return out.logits[0, -1, torch.tensor([tokens[0] for tokens in ids], device=model.device)].float()
