"""Décodage et visualisation seulement : ceci n'est PAS le DSP du modèle."""

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO

import numpy as np
import soundfile as sf
from scipy.signal import stft

from pipe.contracts import Sample

MAX_BYTES = 8 * 1024 * 1024
MAX_SECONDS = 30
MAX_SAMPLES = 16
VISUALIZATION_VERSION = "display-stft-v1"


class AudioError(ValueError):
    pass


@dataclass
class StoredAudio:
    metadata: Sample
    original: bytes
    waveform: np.ndarray


def decode_audio(raw: bytes) -> StoredAudio:
    try:
        with sf.SoundFile(BytesIO(raw)) as audio:
            if audio.format not in ("WAV", "WAVEX"):
                raise AudioError("Le fichier doit être un véritable WAV.")
            if audio.subtype not in ("PCM_16", "PCM_24", "PCM_32", "FLOAT"):
                raise AudioError("WAV accepté : PCM 16/24/32 bits ou float32.")
            if audio.channels not in (1, 2) or not 8000 <= audio.samplerate <= 192000:
                raise AudioError("Utilisez un WAV mono ou stéréo entre 8 et 192 kHz.")
            if audio.frames == 0 or audio.frames > audio.samplerate * MAX_SECONDS:
                raise AudioError("La durée doit être supérieure à zéro et au plus 30 secondes.")
            rate = audio.samplerate
            frames = audio.frames
            waveform = audio.read(dtype="float32", always_2d=True)
            if len(waveform) != frames:
                raise AudioError("Le WAV est tronqué.")
    except (sf.LibsndfileError, RuntimeError) as exc:
        raise AudioError("WAV illisible ou corrompu.") from exc
    if not np.isfinite(waveform).all():
        raise AudioError("Le WAV contient des valeurs NaN ou infinies.")
    channels = waveform.shape[1]
    digest = sha256(f"{rate}:{channels}:".encode() + waveform.astype("<f4").tobytes()).hexdigest()
    warnings = []
    if not np.any(waveform):
        warnings.append("Signal silencieux : aucune énergie acoustique mesurable.")
    if np.max(np.abs(waveform)) >= 1:
        warnings.append("Amplitude à pleine échelle ou supérieure : vérifier la saturation.")
    return StoredAudio(
        metadata=Sample(
            sample_id=digest[:24], input_sha256=digest,
            duration_seconds=len(waveform) / rate, sample_rate_hz=rate,
            channels=channels, source="Import local · provenance non vérifiée",
            execution_mode="development_fixture", warnings=warnings,
        ),
        original=raw, waveform=waveform,
    )


def visualization(audio: StoredAudio) -> dict:
    # Moyenne des canaux pour l'affichage uniquement. Aucun resampling/normalisation ML.
    mono = audio.waveform.mean(axis=1)
    rate = audio.metadata.sample_rate_hz
    blocks = np.array_split(mono, min(800, len(mono)))
    edges = np.linspace(0, len(mono), len(blocks) + 1, dtype=int)
    n_fft = min(1024, len(mono))
    hop = max(1, n_fft // 4)
    frequencies, times, spectrum = stft(
        mono, fs=rate, window="hann", nperseg=n_fft,
        noverlap=n_fft - hop, boundary=None, padded=False,
    )
    # Plafonds de transport ; moyenne d'énergie par blocs avant conversion dB.
    power = np.abs(spectrum) ** 2
    freq_groups = np.array_split(np.arange(len(frequencies)), min(256, len(frequencies)))
    time_groups = np.array_split(np.arange(len(times)), min(400, len(times)))
    pooled = np.stack([power[group].mean(axis=0) for group in freq_groups])
    pooled = np.stack([pooled[:, group].mean(axis=1) for group in time_groups], axis=1)
    decibels = 10 * np.log10(np.maximum(pooled, 1e-12))
    return {
        "sample_id": audio.metadata.sample_id,
        "input_sha256": audio.metadata.input_sha256,
        "visualization_version": VISUALIZATION_VERSION,
        "duration_seconds": audio.metadata.duration_seconds,
        "waveform": {"times": (edges[:-1] / rate).tolist(),
                     "min": [float(b.min()) for b in blocks],
                     "max": [float(b.max()) for b in blocks]},
        "spectrogram": {
            "times": [float(times[g].mean()) for g in time_groups],
            "frequencies_hz": [float(frequencies[g].mean()) for g in freq_groups],
            "power_db": np.round(decibels, 2).tolist(),
            "floor_db": -120, "reference": "Amplitude numérique 1, non calibrée en pression acoustique",
        },
        "parameters": {"n_fft": n_fft, "hop_samples": hop, "window": "hann",
                       "channels": "moyenne mono pour affichage", "resampling": False,
                       "pooling": "moyenne d'énergie, au plus 256 fréquences × 400 instants"},
        "notice": "Visualisation distincte du prétraitement ML ; ce n’est pas l’attention du modèle.",
    }
