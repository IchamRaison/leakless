"""Interface Safoan predict(wav_bytes), et score brut seul pour le harness."""
import hashlib
import json
from pathlib import Path
import re
import threading
import time

import numpy as np
import torch
from opentslm.time_series_datasets.util import extend_time_series_to_match_patch_size_and_aggregate

from pipe.contracts import Observation, Prediction
from pipe.tslm.model import AcousticQwenSP
from pipe.tslm.preprocessing import CANONICAL_VERSION, VERSION, decode_wav, measured_band, model_input, preprocess_audio

OUTPUT_PATTERN = re.compile(
    r"(leak|no_leak);\s*(Greatest mean spectral energy: (?:0-1000|1000-2000|2000-3000|3000-4000) Hz\.)"
)


class PredictionError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def preprocess_for_model(waveform: np.ndarray, sample_rate: int, metadata: dict) -> np.ndarray:
    # Les anciens appels directs predict_audio n'avaient pas ce champ ; les
    # bundles le requièrent dans Predictor.__init__. Leur comportement reste V1.
    return preprocess_audio(waveform, sample_rate, version=metadata.get("preprocessing_version", VERSION))


def predict_audio(model: AcousticQwenSP, raw: bytes, metadata: dict) -> Prediction:
    started = time.monotonic()
    try:
        waveform = decode_wav(raw)
        if np.std(waveform) == 0:
            raise PredictionError("silent_audio", "Signal constant : aucune énergie acoustique exploitable")
        series = preprocess_for_model(waveform, 8000, metadata)
    except ValueError as exc:
        raise PredictionError("unsupported_audio", str(exc)) from exc
    batch = extend_time_series_to_match_patch_size_and_aggregate([model_input(series)], normalize=False)
    try:
        text = model.generate(batch, max_new_tokens=metadata["max_new_tokens"], max_time=15.0)[0].strip()
    except torch.cuda.OutOfMemoryError as exc:
        raise PredictionError("gpu_out_of_memory", "Mémoire GPU insuffisante") from exc
    matched = OUTPUT_PATTERN.fullmatch(text)
    digest = hashlib.sha256(raw).hexdigest()
    warnings = ["V0 mécanique, qualité non évaluée ; aucune confiance calibrée.",
                "Domaine expérimental ; ni localisation, ni validation terrain."]
    if not matched:
        warnings.append("Sortie générée invalide : aucune classe de remplacement inventée.")
    preprocessing_version = metadata.get("preprocessing_version", VERSION)
    return Prediction(
        sample_id=digest[:24], input_sha256=digest, model_name="tslm",
        model_version=metadata["model_version"], preprocessing_version=preprocessing_version,
        prediction=matched[1] if matched else None, score_type="none",
        abstained=matched is None, abstention_reason=None if matched else "invalid_output",
        observations=[Observation(name="greatest_mean_spectral_energy_band",
                                  value=measured_band(series), unit="Hz", method=f"DSP:{preprocessing_version}")],
        # Texte généré conservé même invalide ; les mesures DSP restent séparées.
        description=matched[2] if matched else text,
        latency_ms=(time.monotonic() - started) * 1000, warnings=warnings, execution_mode="live",
    )


class Predictor:
    """Charger une fois au démarrage. Pas de réseau ni accès aux labels à l'inférence."""
    def __init__(self, checkpoint: str | Path, device: str = "cuda"):
        root = Path(checkpoint).resolve()
        try:
            checksums = json.loads((root / "checksums.json").read_text())
            for relative, expected in checksums.items():
                path = (root / relative).resolve()
                if not path.is_relative_to(root) or sha256_file(path) != expected:
                    raise ValueError("Intégrité du checkpoint invalide")
            self.metadata = json.loads((root / "metadata.json").read_text())
            if self.metadata["preprocessing_version"] not in (VERSION, CANONICAL_VERSION):
                raise ValueError("Version de prétraitement incompatible")
            required = {"metadata.json", "temporal.pt", "base/config.json", "base/tokenizer_config.json"}
            if not required.issubset(checksums) or not any(name.endswith(".safetensors") for name in checksums):
                raise ValueError("Checkpoint incomplet")
            self.model = AcousticQwenSP(root / "base", device=device,
                single_clip_acoustic_encoding=self.metadata.get("single_clip_acoustic_encoding", False))
            temporal = torch.load(root / "temporal.pt", map_location=device, weights_only=True)
            self.model.encoder.load_state_dict(temporal["encoder_state"], strict=True)
            self.model.projector.load_state_dict(temporal["projector_state"], strict=True)
            self.model.eval()
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            raise PredictionError("model_unavailable", f"Chargement du checkpoint impossible : {exc}") from exc
        self._lock = threading.Lock()

    def predict(self, wav_bytes: bytes) -> Prediction:
        # ponytail: une requête GPU à la fois ; ajouter une file si le débit l'exige.
        if not self._lock.acquire(blocking=False):
            raise PredictionError("model_busy", "Une inférence est déjà en cours")
        try:
            return predict_audio(self.model, wav_bytes, self.metadata)
        finally:
            self._lock.release()

    def score(self, wav_bytes: bytes) -> float:
        """Score relatif non calibré, sans génération de texte ni seuil."""
        try:
            waveform = decode_wav(wav_bytes)
        except ValueError as exc:
            raise PredictionError("unsupported_audio", str(exc)) from exc
        return self.score_waveform(waveform)

    def score_waveform(self, waveform: np.ndarray, sample_rate: int = 8000) -> float:
        """Audio brut numérique : les stress passent ici sans réencoder un WAV.

        La normalisation et le preprocessing restent exactement ceux de T0.
        """
        try:
            series = preprocess_for_model(waveform, sample_rate, self.metadata)
            if np.std(waveform) == 0:
                raise PredictionError("silent_audio", "Signal constant : aucune énergie acoustique exploitable")
        except ValueError as exc:
            raise PredictionError("unsupported_audio", str(exc)) from exc
        return self.score_series(series)

    def score_series(self, series: np.ndarray) -> float:
        """Scorer une entrée déjà prétraitée (4,64), sans ID/label/métadonnée."""
        try:
            batch = extend_time_series_to_match_patch_size_and_aggregate([model_input(series)], normalize=False)
        except ValueError as exc:
            raise PredictionError("unsupported_audio", str(exc)) from exc
        if not self._lock.acquire(blocking=False):
            raise PredictionError("model_busy", "Une inférence est déjà en cours")
        try:
            probability = self.model.score_probability_leak(batch)[0]
            if not np.isfinite(probability) or not 0 <= probability <= 1:
                raise PredictionError("invalid_score", "Score non fini ou hors [0,1]")
            return probability
        except torch.cuda.OutOfMemoryError as exc:
            raise PredictionError("gpu_out_of_memory", "Mémoire GPU insuffisante") from exc
        except ValueError as exc:
            raise PredictionError("invalid_score", str(exc)) from exc
        finally:
            self._lock.release()
