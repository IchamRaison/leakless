"""Adaptateur applicatif vers la restitution fiable V2 d'Icham.

Le module ML reste optionnel pour permettre à l'API CPU de démarrer sans PyTorch.
Les chemins d'artefacts sont locaux au serveur et ne sont jamais exposés.
"""

from dataclasses import dataclass
import json
from os import environ
from pathlib import Path
from typing import Any, Callable, Mapping

from pipe.contracts import Observation, PredictResponse

ENVIRONMENT_KEYS = (
    "PIPE_TSLM_CHECKPOINT",
    "PIPE_TSLM_DECISION",
    "PIPE_TSLM_VALIDATION_EVIDENCE",
)


@dataclass
class ModelService:
    available: bool
    device: str
    version: str | None = None
    reason: str | None = None
    predictor: Any | None = None


def load_tslm(
    values: Mapping[str, str] | None = None,
    factory: Callable[..., Any] | None = None,
) -> ModelService:
    values = environ if values is None else values
    device = values.get("PIPE_TSLM_DEVICE", "cpu")
    configured = {key: values.get(key) for key in ENVIRONMENT_KEYS}
    missing = [key for key, value in configured.items() if not value]
    if missing:
        return ModelService(
            available=False,
            device=device,
            reason="Artefacts V2 non configurés : " + ", ".join(missing),
        )
    paths = {key: Path(value).expanduser().resolve() for key, value in configured.items() if value}
    absent = [key for key, path in paths.items() if not path.exists()]
    if absent:
        return ModelService(
            available=False,
            device=device,
            reason="Artefacts V2 introuvables : " + ", ".join(absent),
        )
    try:
        if factory is None:
            from pipe.tslm.coherent import CoherentPredictor

            factory = CoherentPredictor
        predictor = factory(
            checkpoint=paths["PIPE_TSLM_CHECKPOINT"],
            decision=paths["PIPE_TSLM_DECISION"],
            validation_evidence=paths["PIPE_TSLM_VALIDATION_EVIDENCE"],
            device=device,
        )
        decision = json.loads(paths["PIPE_TSLM_DECISION"].read_text())
        version = decision["model_identity"]["model_version"]
    except Exception as exc:
        return ModelService(
            available=False,
            device=device,
            # Le détail peut contenir un chemin local. Il ne doit pas sortir via /health.
            reason=f"Chargement V2 impossible ({type(exc).__name__})",
        )
    return ModelService(available=True, device=device, version=version, predictor=predictor)


def predict_tslm(service: ModelService, *, audio: Any, request_id: str) -> PredictResponse:
    if not service.available or service.predictor is None:
        raise RuntimeError(service.reason or "Modèle V2 indisponible")
    result = service.predictor.predict(audio.original)
    score = float(result["probability_leak"])
    return PredictResponse(
        request_id=request_id,
        sample_id=audio.metadata.sample_id,
        input_sha256=audio.metadata.input_sha256,
        model_input_sha256=result["input_sha256"],
        model_name="tslm",
        model_version=result["model_version"],
        preprocessing_version=result["preprocessing_version"],
        prediction=result["prediction"],
        class_scores={"leak": score, "no_leak": 1.0 - score},
        score_type="raw",
        abstained=False,
        observations=[
            Observation(
                name="greatest_mean_spectral_energy_band",
                value=result["dominant_band_hz"],
                unit="Hz",
                method=f"DSP:{result['preprocessing_version']}",
            )
        ],
        description=result["description"],
        latency_ms=result["latency_ms"],
        warnings=result["warnings"],
        execution_mode="live",
        decision_version=result["decision_version"],
        decision_artifact_sha256=result["decision_artifact_sha256"],
        threshold=result["threshold"],
        calibration="none",
        description_source=result["description_source"],
        fallback_used=result["fallback_used"],
        fallback_reasons=result["fallback_reasons"],
    )
