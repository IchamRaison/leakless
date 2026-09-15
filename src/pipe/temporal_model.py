"""Acoustique partagée offline/service ; pas de gabarit de score ni DSP dupliqué."""
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import sys
import wave

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/tslm"))
import v2_c1


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def decode_pcm(data, seconds=1):
    try:
        with wave.open(BytesIO(data)) as wav:
            width = wav.getsampwidth()
            if (wav.getnchannels() != 1 or wav.getframerate() != 8000
                    or wav.getnframes() != seconds * 8000 or width not in (2, 4)
                    or wav.getcomptype() != "NONE"):
                raise ValueError("WAV mono PCM16/32, 8 kHz, durée exacte requise")
            raw = wav.readframes(seconds * 8000)
        if len(raw) != seconds * 8000 * width:
            raise ValueError("WAV tronqué")
        return np.frombuffer(raw, dtype=f"<i{width}").astype(np.float64) / 2**(8 * width - 1)
    except (wave.Error, EOFError) as exc:
        raise ValueError("WAV PCM invalide") from exc


def acoustic_features(signal, *, allow_full_scale=False):
    x = np.asarray(signal, dtype=np.float64)
    if x.shape != (8000,) or not np.isfinite(x).all():
        raise ValueError("Fenêtre finie de 8000 échantillons requise")
    if np.std(x) < 1e-10:
        raise ValueError("Signal silencieux ou constant : surveillance indisponible")
    if np.max(np.abs(x)) > 1 or (not allow_full_scale and np.any((x <= -1) | (x >= 32767 / 32768))):
        raise ValueError("Pleine échelle : vérifier la saturation du signal")
    return v2_c1.features.c1_envelope(x)


class C1Detector:
    def __init__(self, directory, expected_sha256):
        self.directory = Path(directory).resolve()
        if digest(self.directory / "bundle.json") != expected_sha256:
            raise ValueError("Empreinte du bundle différente du gel")
        self.metadata = json.loads((self.directory / "bundle.json").read_text())
        for name, expected in self.metadata["files"].items():
            path = (self.directory / name).resolve()
            if not path.is_relative_to(self.directory) or digest(path) != expected:
                raise ValueError("Intégrité des artefacts incorrecte")
        self.scaler, self.classifier = v2_c1.load_checkpoint(
            self.directory / "c1.pkl", expected_sha256=self.metadata["files"]["c1.pkl"], trusted=True)
        self.version = "c1-" + self.metadata["files"]["c1.pkl"][:12]

    def score(self, signal, *, allow_full_scale=False):
        features = acoustic_features(signal, allow_full_scale=allow_full_scale)
        probability = float(v2_c1.predict_probability(self.scaler, self.classifier, features[None])[0])
        return features, probability

    def sequence_inputs(self, signal, *, allow_full_scale=False):
        x = np.asarray(signal, dtype=np.float64)
        if x.shape != (240000,):
            raise ValueError("Séquence de 30 secondes requise")
        return np.asarray([np.r_[f, p] for f, p in
            (self.score(window, allow_full_scale=allow_full_scale) for window in x.reshape(30, 8000))])
