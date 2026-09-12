"""Transformation unique des WAV/records TimeF vers les entrées numériques.

API réutilisable par la baseline. Pas de statistiques ajustées au dataset.
"""
import io
import wave

import numpy as np
from leakless_acoustic.connector import _normalise

VERSION = "rms-hann256-hop128-band4-log1p-v1"
CANONICAL_VERSION = "rms-f32-hann256-hop128-band4-log1p-v2"
SAMPLE_RATE = 8000
N_SAMPLES = 8000
FFT_SIZE = 256
HOP = 128
BANDS = ((0, 1000), (1000, 2000), (2000, 3000), (3000, 4000))
CHANNEL_NAMES = tuple(f"Relative spectral energy, {lo}-{hi} Hz" for lo, hi in BANDS)
PRE_PROMPT = (
    "Analyze four frequency-band energy time series from a one-second acoustic recording. "
    "Classify this experimental recording as leak or no_leak and name the band with "
    "the greatest mean spectral energy. No physical cause or location can be inferred."
)
POST_PROMPT = (
    "\nReturn only: CLASS; Greatest mean spectral energy: LOW-HIGH Hz.\n"
    "CLASS is leak or no_leak. LOW-HIGH is 0-1000, 1000-2000, 2000-3000, or 3000-4000.\n"
    "Answer: "
)


def decode_wav(raw: bytes) -> np.ndarray:
    """Frontière V0 stricte : PCM16 mono 8 kHz, exactement une seconde."""
    if not isinstance(raw, bytes) or not 44 <= len(raw) <= 128 * 1024:
        raise ValueError("WAV limité à 128 Kio")
    try:
        with wave.open(io.BytesIO(raw)) as wav:
            if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(),
                    wav.getnframes(), wav.getcomptype()) != (1, 2, SAMPLE_RATE, N_SAMPLES, "NONE"):
                raise ValueError("V0 : WAV PCM16 mono, 8000 Hz, 8000 échantillons requis")
            pcm = wav.readframes(N_SAMPLES)
    except (wave.Error, EOFError) as exc:
        raise ValueError("WAV illisible") from exc
    if len(pcm) != 2 * N_SAMPLES:
        raise ValueError("WAV tronqué")
    return np.frombuffer(pcm, dtype="<i2").astype(np.float64)


def band_series(normalized_waveform: np.ndarray) -> np.ndarray:
    """Quatre bandes d'énergie relative, 64 pas dont trois de padding nul.

    Hann symétrique de 256 points, fenêtres complètes uniquement, hop 128.
    Les 61 centres sont 16..976 ms. Les trois derniers pas sont du padding,
    jamais utilisés dans la propriété cible. Le collator ne renormalise pas
    chaque bande : cela effacerait leurs différences d'énergie moyenne.
    """
    x = np.asarray(normalized_waveform, dtype=np.float64)
    if x.shape != (N_SAMPLES,) or not np.isfinite(x).all():
        raise ValueError("Signal fini de forme (8000,) requis")
    frames = np.lib.stride_tricks.sliding_window_view(x, FFT_SIZE)[::HOP]
    window = np.hanning(FFT_SIZE)
    power = np.abs(np.fft.rfft(frames * window, axis=-1)) ** 2
    power[:, 1:-1] *= 2  # Spectre unilatéral, DC/Nyquist comptés une fois.
    power /= FFT_SIZE * np.square(window).sum()
    freq = np.fft.rfftfreq(FFT_SIZE, 1 / SAMPLE_RATE)
    energies = np.stack([
        power[:, (freq >= lo) & ((freq < hi) if hi < 4000 else (freq <= hi))].sum(-1)
        for lo, hi in BANDS
    ])
    # ponytail: échelle fixe log1p, sans fit ; comparer d'autres représentations en V1.
    result = np.log1p(energies).astype(np.float32)
    return np.pad(result, ((0, 0), (0, (-result.shape[1]) % 4)))


def preprocess_audio(waveform: np.ndarray, sample_rate: int, *, version: str = VERSION) -> np.ndarray:
    if version not in (VERSION, CANONICAL_VERSION):
        raise ValueError("Version de prétraitement incompatible")
    if sample_rate != SAMPLE_RATE:
        raise ValueError("V0 : fréquence autre que 8000 Hz non supportée")
    x = np.asarray(waveform)
    if x.shape != (N_SAMPLES,) or not np.isfinite(x).all():
        raise ValueError("Signal fini de forme (8000,) requis")
    normalized = _normalise(x)
    if version == CANONICAL_VERSION:
        # TimeF sérialise les valeurs normalisées en float32. Reproduire ce
        # passage AVANT la FFT, pas arrondir le score ni renormaliser TimeF.
        normalized = normalized.astype(np.float32)
    return band_series(normalized)


def measured_band(series: np.ndarray) -> str:
    if series.shape != (4, 64) or not np.isfinite(series).all() or (series < 0).any():
        raise ValueError("Séries non négatives et finies de forme (4,64) requises")
    energy = np.expm1(series[:, :61].astype(np.float64)).mean(axis=1)
    lo, hi = BANDS[int(energy.argmax())]
    return f"{lo}-{hi}"


def target_text(label: str, series: np.ndarray) -> str:
    if label not in ("leak", "no_leak"):
        raise ValueError("Classe inconnue")
    return f"{label}; Greatest mean spectral energy: {measured_band(series)} Hz."


def model_input(series: np.ndarray) -> dict:
    """Liste blanche : aucun ID, chemin, mesure cible, label ou answer."""
    if series.shape != (4, 64) or not np.isfinite(series).all():
        raise ValueError("Entrée OpenTSLM (4,64) finie requise")
    return {"pre_prompt": PRE_PROMPT, "time_series_text": list(CHANNEL_NAMES),
            "time_series": series.copy(), "post_prompt": POST_PROMPT}
