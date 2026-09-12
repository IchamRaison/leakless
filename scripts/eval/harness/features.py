"""L'échelle de contrôles C0 à C3. Définitions **gelées avant tout résultat TSLM**.

Chaque échelon isole une source d'information différente, pour qu'on puisse dire
ce qu'un modèle apporte *au-delà* de la précédente.

    C0  RMS seul            signal BRUT        raccourci de niveau absolu
    C1  enveloppe seule     signal normalisé   forme d'amplitude, aucune fréquence
    C2  spectral agrégé     signal normalisé   distribution fréquentielle, aucune phase
    C3  baseline existante  signal normalisé   mélange C1 + C2 + taux de passages par zéro
    C4  TSLM                fourni par Hicham  non implémenté ici

Choix explicite : **le taux de passages par zéro n'est PAS dans C1.** C'est un
indicateur de contenu fréquentiel déguisé en mesure temporelle ; le mettre dans
l'échelon « enveloppe seule » rendrait la comparaison C1 vs C2 illisible. Il reste
dans C3 parce que la baseline historique le contenait, et C3 sert la traçabilité.

Aucun descripteur n'utilise la phase, l'ordre des échantillons, ni une
représentation temporelle apprise. C'est voulu : ce sont les contrôles que la
modélisation temporelle doit dépasser pour prouver quelque chose.

⚠️ Ne pas ajuster ces définitions après avoir vu un score de test. Elles sont
gelées, et le commit qui les fige est antérieur à toute prédiction de TSLM.
"""

from __future__ import annotations

import numpy as np

from .split_loader import SAMPLE_RATE, normalise_rms

BANDS = ((0, 250), (250, 500), (500, 1000), (1000, 2000), (2000, 4000))


def _spectrum(x: np.ndarray):
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    freqs = np.fft.rfftfreq(len(x), 1 / SAMPLE_RATE)
    return spec, freqs, spec / (spec.sum() + 1e-12)


# --------------------------------------------------------------------------- #
# C0 — niveau absolu, signal brut
# --------------------------------------------------------------------------- #
C0_NAMES = ("rms_dbfs",)


def c0_rms_raw(raw: np.ndarray) -> np.ndarray:
    """Le seul descripteur : le niveau RMS du signal BRUT, en dBFS.

    Jamais calculé sur de l'audio normalisé : la normalisation supprimerait par
    construction l'information que ce contrôle existe pour mesurer.
    """
    x = raw - raw.mean()
    return np.array([20 * np.log10(np.sqrt(np.mean(x**2)) / 32768 + 1e-12)])


# --------------------------------------------------------------------------- #
# C1 — forme d'amplitude seule, signal normalisé, aucune fréquence
# --------------------------------------------------------------------------- #
C1_NAMES = ("facteur_crete", "kurtosis", "skewness", "abs_p50", "abs_p75",
            "abs_p90", "abs_p99", "ratio_p99_p50", "ecart_type_enveloppe")


def c1_envelope(raw: np.ndarray) -> np.ndarray:
    """Statistiques de la distribution d'amplitude, après normalisation RMS.

    Toutes invariantes à l'ordre des échantillons sauf `ecart_type_enveloppe`,
    qui mesure la dispersion des RMS de 20 blocs : c'est de la variabilité
    d'énergie, pas de l'ordre temporel (permuter les blocs ne la change pas).
    """
    x = normalise_rms(raw)
    a = np.abs(x)
    crest = float(a.max() / (np.sqrt(np.mean(x**2)) + 1e-12))
    m2 = float(np.mean(x**2)) + 1e-12
    kurt = float(np.mean(x**4) / m2**2)
    skew = float(np.mean(x**3) / m2**1.5)
    p50, p75, p90, p99 = (float(np.percentile(a, q)) for q in (50, 75, 90, 99))
    blocks = x[: (len(x) // 20) * 20].reshape(20, -1)
    env_std = float(np.std(np.sqrt(np.mean(blocks**2, axis=1))))
    return np.array([crest, kurt, skew, p50, p75, p90, p99,
                     p99 / (p50 + 1e-12), env_std])


# --------------------------------------------------------------------------- #
# C2 — spectral agrégé, invariant à l'ordre, sans phase
# --------------------------------------------------------------------------- #
C2_NAMES = ("centroide_hz", "ecart_type_spectral_hz", "platitude") + tuple(
    f"bande_{lo}_{hi}" for lo, hi in BANDS)


def c2_spectral(raw: np.ndarray) -> np.ndarray:
    """Distribution d'énergie en fréquence. La phase est jetée, l'ordre aussi."""
    x = normalise_rms(raw)
    spec, freqs, p = _spectrum(x)
    centroid = float((freqs * p).sum())
    spread = float(np.sqrt(((freqs - centroid) ** 2 * p).sum()))
    flatness = float(np.exp(np.mean(np.log(spec + 1e-12))) / (np.mean(spec) + 1e-12))
    bands = [float(p[(freqs >= lo) & (freqs < hi)].sum()) for lo, hi in BANDS]
    return np.array([centroid, spread, flatness, *bands])


# --------------------------------------------------------------------------- #
# C3 — la baseline existante, conservée à l'identique pour la traçabilité
# --------------------------------------------------------------------------- #
C3_NAMES = ("centroide_hz", "ecart_type_spectral_hz", "rolloff85_hz", "platitude") + tuple(
    f"bande_{lo}_{hi}" for lo, hi in BANDS) + ("taux_passages_zero", "facteur_crete")


def c3_legacy_baseline(raw: np.ndarray) -> np.ndarray:
    """Exactement le vecteur de `scripts/eval/baseline_logreg.py`.

    Mélange assumé : spectral (C2) + facteur de crête (C1) + taux de passages par
    zéro. C'est pour cela qu'il ne peut pas servir à attribuer une information à
    une source précise — d'où C1 et C2 séparés.
    """
    x = normalise_rms(raw)
    spec, freqs, p = _spectrum(x)
    centroid = float((freqs * p).sum())
    spread = float(np.sqrt(((freqs - centroid) ** 2 * p).sum()))
    rolloff = float(freqs[int(np.searchsorted(np.cumsum(p), 0.85))])
    flatness = float(np.exp(np.mean(np.log(spec + 1e-12))) / (np.mean(spec) + 1e-12))
    bands = [float(p[(freqs >= lo) & (freqs < hi)].sum()) for lo, hi in BANDS]
    zcr = float(np.mean(np.abs(np.diff(np.sign(x))) > 0))
    crest = float(np.max(np.abs(x)) / (np.sqrt(np.mean(x**2)) + 1e-12))
    return np.array([centroid, spread, rolloff, flatness, *bands, zcr, crest])


LADDER = {
    "C0": {"fn": c0_rms_raw, "names": C0_NAMES, "audio": "raw",
           "description": "niveau RMS absolu, signal brut — raccourci d'acquisition"},
    "C1": {"fn": c1_envelope, "names": C1_NAMES, "audio": "normalised",
           "description": "forme d'amplitude seule, aucune information fréquentielle"},
    "C2": {"fn": c2_spectral, "names": C2_NAMES, "audio": "normalised",
           "description": "spectre agrégé, invariant à l'ordre, sans phase"},
    "C3": {"fn": c3_legacy_baseline, "names": C3_NAMES, "audio": "normalised",
           "description": "baseline historique — mélange C1 + C2 + taux de passages par zéro"},
}
