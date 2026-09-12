"""Restitution opt-in : décision unique score/seuil, texte contrôlé contre le DSP.

Ne remplace ni le contrat V1, ni une preuve de qualité de détection.
"""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import threading
import time

import numpy as np


def _sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _unique_fields(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Champ JSON répété : {key}")
        result[key] = value
    return result


def _read_object(path):
    path = Path(path)
    if path.stat().st_size > 1024 * 1024:
        raise ValueError("Artefact JSON limité à 1 Mio")
    result = json.loads(path.read_text(), object_pairs_hook=_unique_fields)
    if not isinstance(result, dict):
        raise ValueError("Objet JSON requis")
    return result


def _digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _source_fingerprints():
    sources = {f"src/pipe/tslm/{name}.py": Path(__file__).with_name(f"{name}.py")
               for name in ("model", "predict", "preprocessing")}
    spec = importlib.util.find_spec("leakless_acoustic.connector")
    if spec is None or spec.origin is None:
        raise ValueError("Source du connecteur acoustique introuvable")
    sources["scripts/timenet/leakless_acoustic/connector.py"] = Path(spec.origin)
    return {name: _sha256(path) for name, path in sources.items()}


def checkpoint_identity(checkpoint):
    """Identité pour le producteur du seuil ; Predictor vérifiera le bundle entier."""
    root = Path(checkpoint).resolve()
    checksums = _read_object(root / "checksums.json")
    for name in ("metadata.json", "temporal.pt", "scoring_spec.json"):
        path = (root / name).resolve()
        if (not path.is_relative_to(root) or not _digest(checksums.get(name))
                or _sha256(path) != checksums[name]):
            raise ValueError(f"Fichier de checkpoint non intègre : {name}")
    metadata = _read_object(root / "metadata.json")
    if (not _digest(metadata.get("config_hash"))
            or any(not isinstance(metadata.get(key), str) or not metadata[key]
                   for key in ("model_version", "preprocessing_version"))):
        raise ValueError("Identité/configuration du modèle incomplète")
    return {"checkpoint_checksums_sha256": _sha256(root / "checksums.json"),
            "temporal_sha256": checksums["temporal.pt"], "config_hash": metadata["config_hash"],
            "model_version": metadata["model_version"], "preprocessing_version": metadata["preprocessing_version"],
            "scoring_spec_sha256": checksums["scoring_spec.json"], "source_sha256": _source_fingerprints()}


def decision_artifact(checkpoint, validation_evidence, decision_version):
    """Construit un dict depuis un reçu de seuil déjà calculé ; aucun fit/écriture."""
    if not isinstance(decision_version, str) or re.fullmatch(r"[A-Za-z0-9._-]+", decision_version) is None:
        raise ValueError("Version de décision explicite requise")
    receipt = _read_object(validation_evidence)
    required = {"schema_version", "fit_fold", "threshold", "threshold_repr", "model_identity", "rule",
                "split_sha256", "validation_predictions_sha256", "threshold_method_sha256"}
    if (set(receipt) != required or receipt["schema_version"] != "pipe-threshold-evidence-v1"
            or receipt["fit_fold"] != "val" or not isinstance(receipt["rule"], str) or not receipt["rule"]):
        raise ValueError("Reçu de seuil validation invalide")
    value = receipt["threshold"]
    if (type(value) not in (int, float) or not math.isfinite(value)
            or receipt["threshold_repr"] != repr(float(value))):
        raise ValueError("Seuil fini en pleine précision requis, sans défaut ni conversion de chaîne")
    if any(not _digest(receipt[key]) for key in ("split_sha256", "validation_predictions_sha256", "threshold_method_sha256")):
        raise ValueError("Provenance de validation incomplète")
    identity = checkpoint_identity(checkpoint)
    if receipt["model_identity"] != identity:
        raise ValueError("Seuil ajusté pour un autre modèle, score ou preprocessing")
    return {"schema_version": "pipe-decision-v1", "decision_version": decision_version,
            "threshold": float(value), "threshold_repr": repr(float(value)), "comparison": ">=",
            "score_type": "raw", "calibration": "none", "model_identity": identity,
            "validation_evidence_sha256": _sha256(validation_evidence),
            "restoration_source_sha256": _sha256(__file__)}


def _waveform_identity(waveform, sample_rate):
    """Empreinte numérique, pas SHA d'un faux WAV ou du conteneur .npy.

    Le passage en float64 little-endian concerne uniquement le hachage. L'entrée
    fournie au modèle conserve ses valeurs/dtype dans une copie privée.
    """
    if (waveform.ndim != 1 or waveform.dtype.kind not in "fiu"
            or not np.isfinite(waveform).all()):
        raise ValueError("Waveform numérique réelle, finie et unidimensionnelle requise")
    header = {"format": "pipe-waveform-f64le-v1", "sample_rate": int(sample_rate),
              "shape": list(waveform.shape)}
    digest = hashlib.sha256(json.dumps(header, sort_keys=True, separators=(",", ":")).encode() + b"\x00")
    digest.update(np.ascontiguousarray(waveform, dtype="<f8").tobytes())
    return digest.hexdigest()


class CoherentPredictor:
    """Backend privé et verrou unique ; aucun changement implicite de Predictor V1."""
    def __init__(self, checkpoint, decision, validation_evidence, device="cuda"):
        from pipe.tslm.predict import OUTPUT_PATTERN, PredictionError, Predictor, preprocess_for_model
        from pipe.tslm.predict import extend_time_series_to_match_patch_size_and_aggregate, model_input
        from pipe.tslm.preprocessing import decode_wav, measured_band

        self._error_type, self._pattern = PredictionError, OUTPUT_PATTERN
        self._decode, self._measure, self._preprocess = decode_wav, measured_band, preprocess_for_model
        self._collate, self._model_input = extend_time_series_to_match_patch_size_and_aggregate, model_input
        try:
            artifact = _read_object(decision)
            expected = decision_artifact(checkpoint, validation_evidence, artifact.get("decision_version"))
            if artifact != expected:
                raise ValueError("Artefact de décision divergent, incomplet ou périmé")
            self._backend = Predictor(checkpoint, device=device)
            metadata = self._backend.metadata
            identity = artifact["model_identity"]
            if any(metadata[key] != identity[key] for key in ("model_version", "preprocessing_version", "config_hash")):
                raise ValueError("Backend rechargé différent de l'identité de décision")
            scoring = _read_object(Path(checkpoint) / "scoring_spec.json")
            if self._backend.model.scoring_spec() != scoring:
                raise ValueError("Méthode de score rechargée différente du checkpoint")
        except PredictionError:
            raise
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
            raise PredictionError("model_unavailable", f"Décision/modèle indisponible : {exc}") from exc
        self._artifact, self._artifact_sha = artifact, _sha256(decision)
        self._lock = threading.Lock()

    def predict(self, wav_bytes):
        """Retour distinct de Prediction v0.1 ; afficher seulement les champs hors audit."""
        return self._predict(wav_bytes, 8000, waveform_input=False)

    def predict_waveform(self, waveform, sample_rate=8000):
        """Signal numérique brut d'une seconde, sans WAV ni requantification PCM16.

        Même score, seuil, DSP et règles de restitution que predict(bytes).
        `audit.raw_api_payload` reste null : aucune API WAV n'est exécutée.
        """
        return self._predict(waveform, sample_rate, waveform_input=True)

    def _generate_waveform_text(self, series):
        """Mince adaptateur vers la génération existante, mêmes arguments que V1."""
        import torch
        batch = self._collate([self._model_input(series)], normalize=False)
        try:
            # Même validation minimale du retour que predict_audio ; conserver
            # les erreurs de type/index plutôt que les transformer en succès.
            self._backend.model.generate(batch, max_new_tokens=self._backend.metadata["max_new_tokens"],
                                         max_time=15.0)[0].strip()
        except torch.cuda.OutOfMemoryError as exc:
            raise self._error_type("gpu_out_of_memory", "Mémoire GPU insuffisante") from exc

    def _predict(self, raw_input, sample_rate, *, waveform_input):
        if not self._lock.acquire(blocking=False):
            raise self._error_type("model_busy", "Une inférence cohérente est déjà en cours")
        started = time.perf_counter()
        audit = {"raw_generation_returns": [], "raw_api_payload": None, "raw_error": None}
        try:
            # La voie officielle refuse déjà signal invalide/constant et défaut GPU.
            if waveform_input:
                try:
                    waveform = np.array(raw_input, copy=True, order="C")
                    input_sha = _waveform_identity(waveform, sample_rate)
                except (TypeError, ValueError, OverflowError) as exc:
                    raise self._error_type("unsupported_audio", str(exc)) from exc
                probability = self._backend.score_waveform(waveform, sample_rate)
                audit["input_identity"] = {"kind": "waveform", "scheme": "pipe-waveform-f64le-v1",
                    "sample_rate": int(sample_rate), "shape": list(waveform.shape),
                    "source_dtype": str(waveform.dtype), "requantized": False}
            else:
                probability = self._backend.score(raw_input)
                input_sha = hashlib.sha256(raw_input).hexdigest()
            if (type(probability) not in (int, float) or not math.isfinite(probability)
                    or not 0 <= probability <= 1):
                raise self._error_type("invalid_score", "Score non fini ou hors [0,1]")
            try:
                if not waveform_input:
                    waveform = self._decode(raw_input)
                series = self._preprocess(waveform, sample_rate, self._backend.metadata)
                band = self._measure(series)
            except ValueError as exc:
                raise self._error_type("unsupported_audio", str(exc)) from exc
            threshold = self._artifact["threshold"]
            decision = "leak" if probability >= threshold else "no_leak"
            factual = f"Greatest mean spectral energy: {band} Hz."
            original = self._backend.model.generate

            def capture(*args, **kwargs):
                value = original(*args, **kwargs)
                audit["raw_generation_returns"].append(value)
                return value

            self._backend.model.generate = capture
            try:
                if waveform_input:
                    self._generate_waveform_text(series)
                else:
                    audit["raw_api_payload"] = self._backend.predict(raw_input).model_dump(mode="json")
            except Exception as exc:
                audit["raw_error"] = {"type": type(exc).__name__, "code": getattr(exc, "code", None), "message": str(exc)}
                # Seul un échec identifié de décodage textuel est récupérable.
                # RuntimeError peut être une panne CUDA : ne pas la masquer.
                if not isinstance(exc, UnicodeError):
                    raise
            finally:
                self._backend.model.generate = original
            returns = audit["raw_generation_returns"]
            exact = returns[0][0] if (len(returns) == 1 and isinstance(returns[0], (list, tuple))
                                      and len(returns[0]) == 1 and isinstance(returns[0][0], str)) else None
            match = self._pattern.fullmatch(exact.strip()) if exact is not None else None
            audit.update({"raw_text_exact": exact, "raw_format_valid": match is not None,
                          "raw_band_matches_dsp": match[2] == factual if match else None,
                          "raw_class_matches_decision": match[1] == decision if match else None})
            reasons = []
            if audit["raw_error"] is not None:
                reasons.append("text_generation_error")
            if audit["raw_api_payload"] is not None and audit["raw_api_payload"].get("abstained") is True:
                reasons.append("raw_api_abstained")
            if match is None:
                reasons.append("invalid_generated_format")
            else:
                if match[2] != factual:
                    reasons.append("generated_band_disagrees_with_dsp")
                if match[1] != decision:
                    reasons.append("generated_class_disagrees_with_score")
            return {"schema_version": "pipe-coherent-v1", "input_sha256": input_sha,
                    "model_version": self._backend.metadata["model_version"],
                    "preprocessing_version": self._backend.metadata["preprocessing_version"],
                    "probability_leak": float(probability), "score_type": "raw", "calibration": "none",
                    "prediction": decision, "threshold": threshold,
                    "decision_version": self._artifact["decision_version"], "decision_artifact_sha256": self._artifact_sha,
                    "dominant_band_hz": band, "description": factual if reasons else match[2],
                    "description_source": "dsp_template_fallback" if reasons else "llm_checked_against_dsp",
                    "fallback_used": bool(reasons), "fallback_reasons": reasons,
                    "warnings": ["Score brut non calibré ; aucune fiabilité terrain démontrée.",
                                 "Le gabarit de secours n'est pas une réussite du texte brut du modèle."],
                    "audit": audit, "latency_ms": (time.perf_counter() - started) * 1000}
        except self._error_type as exc:
            exc.coherent_audit = audit
            raise
        except Exception as exc:
            error = self._error_type("model_unavailable", f"Restitution interrompue : {exc}")
            error.coherent_audit = audit
            raise error from exc
        finally:
            self._lock.release()
