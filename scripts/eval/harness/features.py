"""L'échelle de contrôles C0 à C3. Définitions **gelées avant tout résultat TSLM**.

Chaque échelon isole une source d'information différente, pour qu'on puisse dire
ce qu'un modèle apporte *au-delà* de la précédente.

    C0  RMS seul            signal BRUT        raccourci de niveau absolu
    C1  enveloppe seule     signal normalisé   forme d'amplitude, aucune fréquence
    C2  spectral agrégé     signal normalisé   distribution fréquentielle, aucune phase
    C2b structure temporelle signal normalisé   enveloppe, modulation, flux — DÉPEND DE L'ORDRE
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



# --------------------------------------------------------------------------- #
# C2b — structure temporelle peu profonde, signal normalisé
# --------------------------------------------------------------------------- #
# GELÉ le 2026-09-12, AVANT toute évaluation val/test de cet échelon.
#
# Pourquoi cet échelon existe : C0, C1, C2 et C3 sont TOUS invariants à l'ordre
# des échantillons. On passait donc directement d'un contrôle invariant à l'ordre
# au TSLM. Si le TSLM battait C1, rien ne permettrait de distinguer « la
# modélisation temporelle apporte quelque chose » de « c'est un meilleur
# extracteur de descripteurs ». C2b ferme ce trou.
#
# ─── Le piège que ce choix évite ───────────────────────────────────────────────
# L'autocorrélation de la FORME D'ONDE est la transformée de Fourier inverse de
# |FFT|² (Wiener-Khintchine). Elle ne contient donc RIEN de plus que le spectre
# de magnitude de C2 : ce n'est pas un descripteur temporel, c'est C2 déguisé.
# Seule l'ENVELOPPE — une transformation non linéaire — brise cette équivalence.
# C'est pour cela que la famille A opère sur l'enveloppe, jamais sur le signal.
#
# ─── Honnêteté sur la redondance interne ──────────────────────────────────────
# Les familles A et B sont des paires de Fourier : le spectre de modulation est
# la FFT de l'enveloppe, l'autocorrélation d'enveloppe en est le module au carré
# retransformé. A donne une statistique de maximum, B des intégrales de bande.
# Elles sont corrélées, pas redondantes. Ce n'est pas un ensemble de descripteurs
# indépendants et il ne faut pas le présenter comme tel.
#
# ─── Ce qui est délibérément absent ───────────────────────────────────────────
# Ni RMS, ni facteur de crête, ni descripteur spectral statique : C0, C1 et C2
# les couvrent déjà. Toutes les valeurs ci-dessous sont invariantes au gain, ce
# qu'un test vérifie. Six descripteurs, pas un de plus : pas de pêche aux
# descripteurs.

ENV_LAG_MIN, ENV_LAG_MAX = 32, 2000     # 250 Hz .. 4 Hz de modulation, à 8 kHz
MOD_BANDS = ((1, 4), (4, 16), (16, 64))  # grille logarithmique, facteur 4
FLUX_FRAME, FLUX_HOP = 256, 128          # 32 ms, recouvrement 50 %

C2B_NAMES = ("env_ac_peak", "mod_peak_hz_log",
             "mod_1_4", "mod_4_16", "mod_16_64", "spectral_flux")


def _envelope(x: np.ndarray) -> np.ndarray:
    """Enveloppe d'amplitude, moyenne retirée.

    `|x|` plutôt que le module du signal analytique : plus simple, sans
    dépendance supplémentaire, et le redressement suffit à briser l'équivalence
    de Wiener-Khintchine, qui est tout ce qu'on lui demande.
    """
    e = np.abs(x)
    return e - e.mean()


def c2b_shallow_temporal(raw: np.ndarray) -> np.ndarray:
    """Six descripteurs, tous dépendants de l'ordre, tous invariants au gain.

    A. autocorrélation d'enveloppe — FORCE de périodicité seulement.
       Région de recherche fixée d'avance à 4-250 Hz : sous 4 Hz un clip d'une
       seconde offre moins de quatre cycles, au-dessus de 250 Hz l'enveloppe
       redressée n'est plus une estimation fiable à 8 kHz.
    B. spectre de modulation — fréquence du pic dominant sur la même région, et
       énergie relative dans trois bandes fixes. Normalisées par leur somme :
       c'est une forme, pas un niveau.
    C. flux spectral — variation moyenne image à image du vecteur de bandes,
       chaque image étant normalisée en L1 pour retirer le niveau.
    """
    x = normalise_rms(raw)
    env = _envelope(x)

    # A — FORCE de périodicité, par autocorrélation d'enveloppe.
    #
    # Normalisation NON BIAISÉE : np.correlate donne l'estimateur biaisé, qui
    # décroît en (N - lag)/N parce que moins d'échantillons se recouvrent aux
    # grands lags. Il favorise donc mécaniquement les modulations rapides. On
    # divise par le nombre de termes réellement sommés.
    #
    # On ne lit ici qu'une FORCE, pas une fréquence. L'autocorrélation pique à
    # chaque multiple de la période : un pic à 2T reste une preuve de
    # périodicité, mais en tirer une fréquence donnerait l'erreur d'octave
    # classique. La fréquence vient du spectre de modulation, ci-dessous, où le
    # fondamental domine et où la résolution vaut 1 Hz.
    n = len(env)
    ac_raw = np.correlate(env, env, mode="full")[n - 1:]
    counts = np.arange(n, 0, -1)
    ac = (ac_raw / counts) / (ac_raw[0] / n + 1e-12)
    ac_peak = float(ac[ENV_LAG_MIN:ENV_LAG_MAX].max())

    # B — spectre de modulation, énergie relative dans trois bandes
    mspec = np.abs(np.fft.rfft(env * np.hanning(len(env)))) ** 2
    mfreq = np.fft.rfftfreq(len(env), 1 / SAMPLE_RATE)
    # Même région que la famille A : 4-250 Hz, gelée d'avance.
    search = (mfreq >= SAMPLE_RATE / ENV_LAG_MAX) & (mfreq <= SAMPLE_RATE / ENV_LAG_MIN)
    peak_hz = float(mfreq[search][int(np.argmax(mspec[search]))])
    # log : une modulation à 5 Hz et une à 200 Hz sont à distance comparable
    mod_peak_hz_log = float(np.log10(peak_hz))
    total = float(mspec[search].sum()) + 1e-12
    bands = [float(mspec[(mfreq >= lo) & (mfreq < hi)].sum() / total) for lo, hi in MOD_BANDS]

    # C — flux spectral, images normalisées pour ne mesurer que le changement
    n_frames = 1 + (len(x) - FLUX_FRAME) // FLUX_HOP
    win = np.hanning(FLUX_FRAME)
    freqs = np.fft.rfftfreq(FLUX_FRAME, 1 / SAMPLE_RATE)
    masks = [(freqs >= lo) & (freqs < hi) for lo, hi in BANDS]
    profile = np.empty((n_frames, len(BANDS)))
    for i in range(n_frames):
        seg = x[i * FLUX_HOP: i * FLUX_HOP + FLUX_FRAME] * win
        p = np.abs(np.fft.rfft(seg)) ** 2
        v = np.array([p[m].sum() for m in masks])
        profile[i] = v / (v.sum() + 1e-12)
    flux = float(np.mean(np.abs(np.diff(profile, axis=0)).sum(axis=1))) if n_frames > 1 else 0.0

    return np.array([ac_peak, mod_peak_hz_log, *bands, flux])


# Commit où chaque définition a été figée. C'est le `model_definition_commit` des
# runs : il désigne le CODE des descripteurs, pas le moment de l'ajustement
# (`training_commit`) ni celui de l'exécution (`execution_commit`). Un test
# vérifie que la fonction de chaque échelon est identique, octet pour octet, à
# sa version dans ce commit.
DEFINITION_COMMIT_C0_C3 = "2f61695072e0f16205d6a7047243c59600eb2792"
DEFINITION_COMMIT_C2B = "355f0743fb306da50897c0720e6fab5ecf947c52"

LADDER = {
    "C0": {"fn": c0_rms_raw, "names": C0_NAMES, "audio": "raw",
           "definition_commit": DEFINITION_COMMIT_C0_C3,
           "description": "niveau RMS absolu, signal brut — raccourci d'acquisition"},
    "C1": {"fn": c1_envelope, "names": C1_NAMES, "audio": "normalised",
           "definition_commit": DEFINITION_COMMIT_C0_C3,
           "description": "forme d'amplitude seule, aucune information fréquentielle"},
    "C2": {"fn": c2_spectral, "names": C2_NAMES, "audio": "normalised",
           "definition_commit": DEFINITION_COMMIT_C0_C3,
           "description": "spectre agrégé, invariant à l'ordre, sans phase"},
    "C2b": {"fn": c2b_shallow_temporal, "names": C2B_NAMES, "audio": "normalised",
            "definition_commit": DEFINITION_COMMIT_C2B,
            "description": "structure temporelle peu profonde — enveloppe, modulation, flux"},
    "C3": {"fn": c3_legacy_baseline, "names": C3_NAMES, "audio": "normalised",
           "definition_commit": DEFINITION_COMMIT_C0_C3,
           "description": "baseline historique — mélange C1 + C2 + taux de passages par zéro"},
}
