"""Transformations de stress temporel T0 à T3, et vérification de leurs invariants.

> ### Ce ne sont PAS des augmentations préservant l'étiquette.
> Rien ne garantit qu'un clip inversé dans le temps reste acoustiquement une fuite.
> Ce sont des **tests de stress** : ils servent à savoir si un modèle réagit à
> l'organisation temporelle, pas à agrandir le jeu d'entraînement. Un modèle dont
> le score ne bouge pas sous T1/T2/T3 n'utilise pas l'ordre des échantillons.

    T0  original
    T1  inversion temporelle   — distribution d'amplitude et |FFT| préservées,
                                 direction du temps renversée
    T2  permutation de blocs   — blocs de 250 échantillons (31,25 ms), graine fixe ;
                                 les morceaux locaux survivent, l'ordre long est détruit
    T3  randomisation de phase — |rFFT| préservée, phase tirée de façon déterministe,
                                 signal réel reconstruit

Piège de T3 : une phase aléatoire naïve casse la symétrie conjuguée et donne un
signal complexe. Les composantes continue et de Nyquist sont donc forcées réelles
(phase 0 ou π), ce qui laisse leur module intact.
"""

from __future__ import annotations

import hashlib

import numpy as np

# 250 échantillons = 31,25 ms à 8 kHz. Choisi parce que 8000 est divisible par 250 :
# avec 256 (32 ms) il restait 64 échantillons en fin de clip que la permutation ne
# touchait jamais, soit 0,8 % du signal laissé à sa place d'origine.
BLOCK_SAMPLES = 250           # gelé
STRESS_SEED = 20260912        # gelé


def t0_original(x: np.ndarray, _rng: np.random.Generator) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).copy()


def t1_reverse(x: np.ndarray, _rng: np.random.Generator) -> np.ndarray:
    """Inversion temporelle. Préserve exactement l'histogramme et |FFT|."""
    return np.asarray(x, dtype=np.float64)[::-1].copy()


def t2_block_permutation(x: np.ndarray, rng: np.random.Generator,
                         block_samples: int = BLOCK_SAMPLES) -> np.ndarray:
    """Permutation de blocs de taille fixe. Ordre local intact, ordre long détruit.

    Si la longueur n'est pas un multiple de la taille de bloc, le reste serait
    laissé à sa place d'origine — une portion du clip que la transformation ne
    toucherait jamais. On échoue plutôt que de le laisser passer en silence.
    """
    x = np.asarray(x, dtype=np.float64)
    if len(x) % block_samples:
        raise ValueError(
            f"longueur {len(x)} non divisible par {block_samples} : {len(x) % block_samples} "
            f"échantillons resteraient non permutés. Choisir une taille de bloc qui divise "
            f"la longueur du clip.")
    blocks = x.reshape(-1, block_samples)
    return blocks[rng.permutation(len(blocks))].reshape(-1)


def t3_phase_randomisation(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Phase randomisée, module de la rFFT préservé, sortie réelle.

    La composante continue et, pour une longueur paire, la composante de Nyquist
    doivent rester réelles pour que `irfft` rende un signal réel. On leur donne
    donc une phase de 0 ou π, ce qui ne change pas leur module.
    """
    x = np.asarray(x, dtype=np.float64)
    spectrum = np.fft.rfft(x)
    magnitude = np.abs(spectrum)
    phase = rng.uniform(-np.pi, np.pi, size=magnitude.shape)
    phase[0] = 0.0 if spectrum[0].real >= 0 else np.pi
    if len(x) % 2 == 0:
        phase[-1] = 0.0 if spectrum[-1].real >= 0 else np.pi
    return np.fft.irfft(magnitude * np.exp(1j * phase), n=len(x))


TRANSFORMS = {
    "T0": {"fn": t0_original, "description": "original, aucune transformation"},
    "T1": {"fn": t1_reverse, "description": "inversion temporelle"},
    "T2": {"fn": t2_block_permutation, "description": f"permutation de blocs de {BLOCK_SAMPLES} échantillons (31,25 ms)"},
    "T3": {"fn": t3_phase_randomisation, "description": "randomisation de phase, module préservé"},
}


# Schéma de dérivation de graine. Documenté ici parce qu'un tiers doit pouvoir
# le réimplémenter à l'identique dans un autre langage si besoin.
SEED_SCHEME = "sha256(utf-8, séparateur U+001F, 8 premiers octets, big-endian)"
SEED_SEPARATOR = "\x1f"          # UNIT SEPARATOR : ne peut apparaître dans un clip_id
SEED_DIGEST_BYTES = 8             # 64 bits


def derive_seed(name: str, clip_id: str, base: int = STRESS_SEED) -> int:
    """Graine déterministe, stable entre processus, machines et exécutions.

    **N'utilise pas `hash()`.** Le `hash()` de Python sur des chaînes est salé par
    processus (PYTHONHASHSEED aléatoire par défaut depuis 3.3) : le même tuple
    donnait quatre graines différentes sur quatre processus, donc T2 et T3
    n'étaient pas reproductibles d'une exécution à l'autre.

    Canonicalisation, explicite et figée :

      payload  = f"{base}{SEP}{name}{SEP}{clip_id}" encodé en **UTF-8**
      SEP      = U+001F (UNIT SEPARATOR), impossible dans un clip_id ou un nom de
                 transformation, donc aucune ambiguïté de concaténation
      base     = entier écrit en décimal, sans signe ni remplissage
      digest   = sha256(payload)
      graine   = int.from_bytes(digest[:8], "big")  -> entier dans [0, 2**64)

    `numpy.random.default_rng` accepte n'importe quel entier non négatif via
    SeedSequence : 64 bits passent tels quels, sans repli modulo.
    """
    payload = f"{base}{SEED_SEPARATOR}{name}{SEED_SEPARATOR}{clip_id}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:SEED_DIGEST_BYTES], "big")


def clip_rng(name: str, clip_id: str) -> np.random.Generator:
    """Générateur déterministe par (transformation, clip). Aucun état partagé."""
    return np.random.default_rng(derive_seed(name, clip_id))


def invariants(original: np.ndarray, transformed: np.ndarray) -> dict:
    """Ce qui a bougé et ce qui n'a pas bougé. Mesuré, jamais supposé."""
    o = np.asarray(original, dtype=np.float64)
    t = np.asarray(transformed, dtype=np.float64)
    mo, mt = np.abs(np.fft.rfft(o)), np.abs(np.fft.rfft(t))
    denom = float(np.linalg.norm(mo)) + 1e-12
    rms_o = float(np.sqrt(np.mean(o**2)))
    rms_t = float(np.sqrt(np.mean(t**2)))
    return {
        "n_samples_original": int(len(o)),
        "n_samples_transformed": int(len(t)),
        "n_samples_unchanged": bool(len(o) == len(t)),
        "rms_original": rms_o,
        "rms_transformed": rms_t,
        "rms_relative_deviation": abs(rms_t - rms_o) / (rms_o + 1e-12),
        "fft_magnitude_relative_deviation": float(np.linalg.norm(mt - mo) / denom),
        "waveform_identical": bool(np.array_equal(o, t)),
        "amplitude_histogram_identical": bool(
            np.array_equal(np.sort(np.round(o, 9)), np.sort(np.round(t, 9)))),
    }
