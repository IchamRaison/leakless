#!/usr/bin/env python3
"""Tests du Temporal Evidence Harness. Sans dépendance : ni pytest, ni les WAV.

Deux familles :

  SENTINELLES   dix situations qui DOIVENT faire échouer le harnais. Un contrôle
                qui passe partout ne prouve rien : chaque sentinelle vérifie que
                l'erreur est bien levée, pas qu'elle ne l'est pas.

  CORRECTION    métriques, bootstrap apparié, déterminisme, invariants temporels.

Les tests marqués `[data]` ont besoin du dataset hors dépôt et sont ignorés
proprement s'il est absent (`--data-root` non fourni ou dossier introuvable).

Usage :
  python3 tests/run_tests.py [--data-root <WAV>] [--runs-dir <runs>]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import json
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "eval"))
sys.path.insert(0, str(ROOT / "scripts" / "temporal"))

from harness import contract, features, metrics, split_loader  # noqa: E402
import stress  # noqa: E402

PASSED: list[str] = []
FAILED: list[tuple[str, str]] = []
SKIPPED: list[str] = []
OPTS = {"data_root": None, "runs_dir": None}


def test(fn):
    fn._is_test = True
    return fn


def must_raise(exc, fn, *a, **kw) -> str:
    try:
        fn(*a, **kw)
    except exc as e:
        return str(e)
    raise AssertionError(f"aucune {exc.__name__} levée — la sentinelle ne protège rien")


# --------------------------------------------------------------------------- #
# Fixture synthétique : un mini-split valide, sans audio
# --------------------------------------------------------------------------- #
def make_fixture(tmp: Path, *, n_clusters: int = 12, per_cluster: int = 3) -> Path:
    d = tmp / "manifests"
    d.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n_clusters):
        fold = "train" if i < n_clusters - 6 else ("val" if i < n_clusters - 3 else "test")
        label = "leak" if i % 2 == 0 else "no_leak"
        for j in range(per_cluster):
            rows.append({"clip_id": f"c{i:02d}{j}", "label": label, "label_3c": label,
                         "group_id": f"g{i:02d}", "fold": fold})
    with open(d / "split_v2.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["clip_id", "label", "label_3c", "group_id", "fold"],
                           lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    with open(d / "split_v2_audit.csv", "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["clip_id", "path"])
        for r in rows:
            w.writerow([r["clip_id"], f"fake/{r['clip_id']}.wav"])
    return d


def fixture_split(tmp: Path):
    return split_loader.load_split(make_fixture(tmp), expect_sha256=None)


def write_preds(dirpath: Path, split, probs: dict, *, sha=None, extra_cols=None,
                meta_overrides=None) -> Path:
    dirpath.mkdir(parents=True, exist_ok=True)
    meta = {
        "run_id": "fix", "model_name": "fixture", "checkpoint": "none",
        "training_commit": "0" * 40, "split_filename": "split_v2.csv",
        "split_sha256": sha or split.sha256, "timestamp": "2026-09-12T00:00:00+00:00",
        "threshold_rule": "val cluster macro-F1",
        "test_labels_not_used_for_tuning": True,
    }
    meta.update(meta_overrides or {})
    (dirpath / "metadata.json").write_text(json.dumps(meta))
    cols = ["clip_id", "probability_leak"] + list(extra_cols or [])
    with open(dirpath / "predictions.csv", "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(cols)
        for cid, p in probs.items():
            w.writerow([cid, p] + ["x"] * len(extra_cols or []))
    return dirpath


# --------------------------------------------------------------------------- #
# SENTINELLES 1-10
# --------------------------------------------------------------------------- #
@test
def sentinel_01_split_v1_refused():
    """[S1] fournir split_v1 doit échouer."""
    with tempfile.TemporaryDirectory() as t:
        d = make_fixture(Path(t))
        shutil.copy(d / "split_v2.csv", d / "split_v1.csv")
        msg = must_raise(split_loader.SplitIntegrityError, split_loader.load_split,
                         d, expect_sha256=None, name="split_v1.csv")
        assert "INVALIDE" in msg or "split_v2.csv" in msg


@test
def sentinel_02_sha_mismatch_refused():
    """[S2] un SHA de split qui ne correspond pas doit échouer."""
    with tempfile.TemporaryDirectory() as t:
        d = make_fixture(Path(t))
        msg = must_raise(split_loader.SplitIntegrityError, split_loader.load_split,
                         d, expect_sha256="0" * 64)
        assert "divergent" in msg


@test
def sentinel_03_missing_prediction_refused():
    """[S3] un clip de test sans prédiction doit échouer."""
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        split = fixture_split(tmp)
        probs = {c.clip_id: 0.5 for c in split.clips if c.fold in ("val", "test")}
        probs.pop(next(iter(probs)))
        run = write_preds(tmp / "run", split, probs)
        msg = must_raise(contract.ContractError, contract.load_run, run, split,
                         require_frozen_sha=False)
        assert "sans prédiction" in msg


@test
def sentinel_04_unknown_clip_refused():
    """[S4] un clip_id absent du manifeste doit échouer."""
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        split = fixture_split(tmp)
        probs = {c.clip_id: 0.5 for c in split.clips}
        probs["clip-inconnu"] = 0.5
        run = write_preds(tmp / "run", split, probs)
        msg = must_raise(contract.ContractError, contract.load_run, run, split,
                         require_frozen_sha=False)
        assert "absent du manifeste" in msg


@test
def sentinel_05_duplicate_prediction_refused():
    """[S5] un clip_id prédit deux fois doit échouer."""
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        split = fixture_split(tmp)
        run = tmp / "run"
        write_preds(run, split, {c.clip_id: 0.5 for c in split.clips})
        with open(run / "predictions.csv", "a") as fh:
            fh.write(f"{split.clips[0].clip_id},0.9\n")
        msg = must_raise(contract.ContractError, contract.load_run, run, split,
                         require_frozen_sha=False)
        assert "deux fois" in msg


@test
def sentinel_06_fold_column_refused():
    """[S6] une colonne fold fournie par le modèle doit échouer."""
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        split = fixture_split(tmp)
        run = write_preds(tmp / "run", split, {c.clip_id: 0.5 for c in split.clips},
                          extra_cols=["fold"])
        msg = must_raise(contract.ContractError, contract.load_run, run, split,
                         require_frozen_sha=False)
        assert "interdites" in msg


@test
def sentinel_07_threshold_never_fitted_on_test():
    """[S7] ajuster le seuil sur le test doit échouer, structurellement."""
    y = np.array([1, 1, 0, 0])
    s = np.array([0.9, 0.8, 0.2, 0.1])
    g = np.array(["a", "a", "b", "b"])
    must_raise(metrics.ThresholdFittingError, metrics.pick_threshold, y, s, g, fold="test")
    metrics.pick_threshold(y, s, g, fold="val")          # doit passer


@test
def sentinel_08_audit_metadata_never_exposed():
    """[S8] aucune métadonnée d'audit ne doit sortir du chargeur de split."""
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        d = make_fixture(tmp)
        # On enrichit le fichier d'audit de colonnes interdites.
        with open(d / "split_v2_audit.csv", "w", newline="") as fh:
            w = csv.writer(fh, lineterminator="\n")
            w.writerow(["clip_id", "path", "pressure_mpa", "flow_ms", "device"])
            for r in csv.DictReader(open(d / "split_v2.csv")):
                w.writerow([r["clip_id"], f"fake/{r['clip_id']}.wav", "0.3 MPa", "1.2 ms",
                            "hydrophone"])
        split = split_loader.load_split(d, expect_sha256=None)
        exposed = set()
        for c in split.clips:
            exposed |= set(vars(c))
        assert not (exposed & split_loader.FORBIDDEN_AUDIT_COLUMNS), exposed
        assert split.path_of(split.clips[0].clip_id).endswith(".wav")
        # Le seul accesseur vers l'audit renvoie une chaîne, pas une structure.
        assert isinstance(split.path_of(split.clips[0].clip_id), str)


@test
def sentinel_09_stress_preserves_mapping():
    """[S9] une transformation de stress ne doit pas toucher clip_id/fold/cluster."""
    with tempfile.TemporaryDirectory() as t:
        split = fixture_split(Path(t))
        before = {c.clip_id: (c.fold, c.label, c.group_id) for c in split.clips}
        rng = stress.clip_rng("T2", "c000")
        x = np.sin(np.linspace(0, 40, 8000))
        for name, spec in stress.TRANSFORMS.items():
            y = spec["fn"](x, stress.clip_rng(name, "c000"))
            inv = stress.invariants(x, y)
            assert inv["n_samples_unchanged"], name
        after = {c.clip_id: (c.fold, c.label, c.group_id) for c in split.clips}
        assert before == after
        assert rng is not None


@test
def sentinel_10_comparison_needs_same_population():
    """[S10] comparer deux modèles sur des populations différentes doit échouer."""
    y = np.array([1, 1, 0, 0])
    g = np.array(["a", "a", "b", "b"])
    a = np.array([0.9, 0.8, 0.2, 0.1])
    b = np.array([0.7, 0.6, 0.3])          # une longueur différente
    must_raise(ValueError, metrics.paired_bootstrap_delta, y, g, a, b, 0.5, 0.5)


# --------------------------------------------------------------------------- #
# CORRECTION
# --------------------------------------------------------------------------- #
@test
def metrics_roc_auc_known_values():
    """AUC sur des cas connus : séparation parfaite, inversée, et ex aequo."""
    y = np.array([1, 1, 0, 0])
    assert metrics.roc_auc(y, np.array([1.0, 0.9, 0.2, 0.1])) == 1.0
    assert metrics.roc_auc(y, np.array([0.1, 0.2, 0.9, 1.0])) == 0.0
    assert metrics.roc_auc(y, np.array([0.5, 0.5, 0.5, 0.5])) == 0.5


@test
def metrics_pr_auc_and_brier():
    y = np.array([1, 1, 0, 0])
    assert abs(metrics.pr_auc(y, np.array([1.0, 0.9, 0.2, 0.1])) - 1.0) < 1e-9
    assert abs(metrics.brier(y, np.array([1.0, 1.0, 0.0, 0.0]))) < 1e-12
    assert abs(metrics.brier(y, np.array([0.0, 0.0, 1.0, 1.0])) - 1.0) < 1e-12


@test
def metrics_balanced_accuracy_and_macro_f1():
    y = np.array([1, 1, 0, 0])
    pred = np.array([1, 0, 0, 0])
    assert abs(metrics.balanced_accuracy(y, pred) - 0.75) < 1e-12
    assert 0.0 < metrics.macro_f1(y, pred) < 1.0


@test
def cluster_aggregation_is_median_and_size_blind():
    """Un cluster de 100 clips pèse comme un cluster de 2 : une observation."""
    y = np.array([1] * 100 + [0] * 2)
    s = np.array([0.9] * 100 + [0.1] * 2)
    g = np.array(["big"] * 100 + ["small"] * 2)
    gids, gy, gs = metrics.aggregate_clusters(y, s, g)
    assert len(gids) == 2
    assert list(gs) == [0.9, 0.1]


@test
def cluster_aggregation_rejects_mixed_cluster():
    y, s, g = np.array([1, 0]), np.array([0.9, 0.1]), np.array(["x", "x"])
    must_raise(ValueError, metrics.aggregate_clusters, y, s, g)


@test
def paired_bootstrap_is_paired_and_zero_for_identical():
    """Comparer un modèle à lui-même donne un delta nul et un IC nul."""
    rng = np.random.default_rng(0)
    y = np.array([1] * 20 + [0] * 20)
    g = np.array([f"g{i//4}" for i in range(40)])
    s = rng.random(40)
    d = metrics.paired_bootstrap_delta(y, g, s, s, 0.5, 0.5, draws=200)
    for m in ("clip_roc_auc", "clip_macro_f1"):
        assert abs(d[m]["delta_observe"]) < 1e-12, d[m]
        assert abs(d[m]["ci95_low"]) < 1e-12 and abs(d[m]["ci95_high"]) < 1e-12
        assert d[m]["lecture"] == "inconclusive"


@test
def paired_bootstrap_detects_a_real_gap():
    """Un modèle parfait contre un modèle inversé : amélioration lisible."""
    y = np.array([1] * 20 + [0] * 20)
    g = np.array([f"g{i//2}" for i in range(40)])
    good = np.array([0.9] * 20 + [0.1] * 20)
    bad = 1.0 - good
    d = metrics.paired_bootstrap_delta(y, g, good, bad, 0.5, 0.5, draws=300)
    assert d["clip_roc_auc"]["delta_observe"] > 0.9
    assert d["clip_roc_auc"]["lecture"] == "compatible with improvement"


@test
def bootstrap_is_deterministic_under_fixed_seed():
    rng = np.random.default_rng(1)
    y = np.array([1] * 15 + [0] * 15)
    g = np.array([f"g{i//3}" for i in range(30)])
    s = rng.random(30)
    a = metrics.bootstrap_ci(y, s, g, 0.5, draws=150)
    b = metrics.bootstrap_ci(y, s, g, 0.5, draws=150)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


@test
def stress_t1_preserves_fft_magnitude_and_histogram():
    x = np.random.default_rng(3).normal(size=8000)
    inv = stress.invariants(x, stress.t1_reverse(x, np.random.default_rng(0)))
    assert inv["fft_magnitude_relative_deviation"] < 1e-12
    assert inv["amplitude_histogram_identical"]
    assert inv["n_samples_unchanged"]


@test
def stress_t2_preserves_histogram_but_breaks_spectrum():
    x = np.random.default_rng(4).normal(size=8000)
    inv = stress.invariants(x, stress.t2_block_permutation(x, np.random.default_rng(0)))
    assert inv["amplitude_histogram_identical"]
    assert inv["fft_magnitude_relative_deviation"] > 1e-3
    assert inv["n_samples_unchanged"]


@test
def stress_t3_preserves_fft_magnitude_and_is_real():
    x = np.random.default_rng(5).normal(size=8000)
    y = stress.t3_phase_randomisation(x, np.random.default_rng(0))
    assert np.isrealobj(y)
    inv = stress.invariants(x, y)
    assert inv["fft_magnitude_relative_deviation"] < 1e-10, inv
    assert not inv["waveform_identical"]


@test
def stress_t2_permutes_the_whole_clip():
    """[correctif] aucun échantillon ne doit rester à sa place d'origine par défaut."""
    x = np.arange(8000, dtype=float)
    y = stress.t2_block_permutation(x, np.random.default_rng(0))
    assert len(y) == len(x)
    n = stress.BLOCK_SAMPLES
    assert 8000 % n == 0, "la taille de bloc doit diviser la longueur du clip"
    # Une longueur non divisible doit échouer bruyamment plutôt que laisser un reste.
    must_raise(ValueError, stress.t2_block_permutation,
               np.arange(8001, dtype=float), np.random.default_rng(0))



# --------------------------------------------------------------------------- #
# C2b — validation du design. AUCUNE métrique val/test ici (commit A).
# --------------------------------------------------------------------------- #
@test
def c2b_has_exactly_six_features():
    """4 à 6 descripteurs maximum : pas de pêche aux descripteurs."""
    assert len(features.C2B_NAMES) == 6
    x = np.random.default_rng(0).normal(size=8000)
    assert len(features.c2b_shallow_temporal(x)) == 6


@test
def c2b_is_gain_invariant():
    """Aucun descripteur ne doit ré-encoder le niveau : C0 et C1 le couvrent déjà."""
    x = np.random.default_rng(1).normal(size=8000)
    a = features.c2b_shallow_temporal(x)
    b = features.c2b_shallow_temporal(x * 137.0)
    assert np.allclose(a, b, atol=1e-9), np.abs(a - b)


@test
def c2b_is_order_sensitive():
    """Permuter des blocs doit changer C2b. C'est toute sa raison d'être."""
    rng = np.random.default_rng(2)
    t = np.arange(8000) / 8000
    x = (1 + 0.6 * np.sin(2 * np.pi * 12 * t)) * rng.normal(size=8000)
    a = features.c2b_shallow_temporal(x)
    b = features.c2b_shallow_temporal(stress.t2_block_permutation(x, np.random.default_rng(0)))
    rel = np.abs(a - b) / (np.abs(a) + 1e-9)
    assert rel.max() > 0.10, f"C2b quasi inchangé par une permutation : {rel}"


@test
def c2b_sees_what_c2_structurally_cannot():
    """Sous randomisation de phase, |FFT| est préservé : C2 ne peut pas bouger, C2b doit.

    C'est la validation du design. Si C2b bougeait aussi peu que C2 sous T3, il
    ne mesurerait que du spectre et n'aurait pas lieu d'exister.
    """
    rng = np.random.default_rng(3)
    t = np.arange(8000) / 8000
    x = (1 + 0.8 * np.sin(2 * np.pi * 20 * t)) * rng.normal(size=8000)
    y = stress.t3_phase_randomisation(x, np.random.default_rng(0))

    c2_a, c2_b = features.c2_spectral(x), features.c2_spectral(y)
    c2b_a, c2b_b = features.c2b_shallow_temporal(x), features.c2b_shallow_temporal(y)
    c2_rel = float(np.median(np.abs(c2_a - c2_b) / (np.abs(c2_a) + 1e-9)))
    c2b_rel = float(np.median(np.abs(c2b_a - c2b_b) / (np.abs(c2b_a) + 1e-9)))
    # C2 n'est PAS exactement invariant : _spectrum applique une fenêtre de Hann,
    # et T3 ne préserve le |FFT| que du signal non fenêtré. Le fenêtrage est une
    # convolution en fréquence, donc une phase différente déplace un peu les
    # bandes (mesuré : jusqu'à 21 % sur une bande étroite). La bonne assertion
    # n'est donc pas « C2 ne bouge pas » mais « C2b bouge beaucoup plus ».
    # Conséquence pour les stress tests : un petit déplacement de score sous T3
    # n'est pas une preuve de sensibilité temporelle.
    assert c2b_rel > 5 * c2_rel, f"C2b ({c2b_rel:.3f}) ne bouge pas plus que C2 ({c2_rel:.3f})"
    assert c2b_rel > 0.2, f"C2b n'a quasiment pas bougé : {c2b_rel:.3f}"


@test
def c2b_recovers_a_known_modulation_frequency():
    """Vérité terrain : la fréquence de modulation doit se lire dans le descripteur.

    L'estimateur vient du spectre de modulation, pas de l'autocorrélation :
    l'autocorrélation pique à chaque multiple de la période et donnerait
    l'erreur d'octave (mesuré : 55 Hz lu comme 27,3 Hz avant correction).
    """
    rng = np.random.default_rng(4)
    t = np.arange(8000) / 8000
    for f_mod in (7.0, 20.0, 55.0, 120.0):
        x = (1 + 0.9 * np.sin(2 * np.pi * f_mod * t)) * rng.normal(size=8000)
        v = features.c2b_shallow_temporal(x)
        recovered = 10 ** v[features.C2B_NAMES.index("mod_peak_hz_log")]
        assert abs(recovered - f_mod) / f_mod < 0.05, (f_mod, recovered)


@test
def c2b_modulation_bands_are_proportions():
    x = np.random.default_rng(5).normal(size=8000)
    v = features.c2b_shallow_temporal(x)
    bands = [v[features.C2B_NAMES.index(n)] for n in ("mod_1_4", "mod_4_16", "mod_16_64")]
    assert all(0.0 <= b <= 1.0 for b in bands), bands
    assert sum(bands) <= 1.0 + 1e-9, sum(bands)


@test
def c2b_search_region_matches_declared_bounds():
    """La région de recherche est gelée : 4-250 Hz à 8 kHz."""
    assert features.ENV_LAG_MIN == 32 and features.ENV_LAG_MAX == 2000
    assert abs(8000 / features.ENV_LAG_MIN - 250.0) < 1e-9
    assert abs(8000 / features.ENV_LAG_MAX - 4.0) < 1e-9
    assert features.MOD_BANDS == ((1, 4), (4, 16), (16, 64))


@test
def c2b_periodicity_strength_is_low_on_white_noise():
    """Un bruit blanc n'est pas périodique : la force doit rester faible."""
    for seed in (0, 1, 2):
        v = features.c2b_shallow_temporal(np.random.default_rng(seed).normal(size=8000))
        assert v[features.C2B_NAMES.index("env_ac_peak")] < 0.15


@test
def c2b_is_deterministic():
    x = np.random.default_rng(6).normal(size=8000)
    assert np.array_equal(features.c2b_shallow_temporal(x),
                          features.c2b_shallow_temporal(x))


@test
def c2b_is_in_the_ladder():
    assert "C2b" in features.LADDER
    assert features.LADDER["C2b"]["audio"] == "normalised"
    assert len(features.LADDER["C2b"]["names"]) == 6



# --------------------------------------------------------------------------- #
# RNG des stress : reproductibilité INTER-PROCESSUS
# --------------------------------------------------------------------------- #
# Ces tests lancent de VRAIS sous-processus. Une assertion qui compare la
# fonction à elle-même dans le même interpréteur passerait alors que le bug
# existe : c'est exactement ce qui s'est produit avec l'ancien
# `stress_transforms_are_deterministic`, qui ne testait qu'un seul processus.

_SUBPROC_PRELUDE = (
    "import sys, hashlib, numpy as np; "
    f"sys.path.insert(0, {str(ROOT / 'scripts' / 'temporal')!r}); "
    "import stress; "
)


def _run_in_subprocess(snippet: str, hashseed: str | None = None) -> str:
    """Exécute du code dans un interpréteur neuf et renvoie sa sortie."""
    env = dict(os.environ)
    if hashseed is not None:
        env["PYTHONHASHSEED"] = hashseed
    else:
        env.pop("PYTHONHASHSEED", None)
    r = subprocess.run([sys.executable, "-c", _SUBPROC_PRELUDE + snippet],
                       capture_output=True, text=True, env=env, check=True)
    return r.stdout.strip()


@test
def seed_is_identical_within_one_process():
    a = stress.derive_seed("T2", "c003cd25f5a5f")
    b = stress.derive_seed("T2", "c003cd25f5a5f")
    assert a == b


@test
def seed_is_identical_across_two_independent_processes():
    """Le test que l'ancienne suite n'avait pas, et qui aurait attrapé le bug."""
    snippet = "print(stress.derive_seed('T2','c003cd25f5a5f'))"
    a = _run_in_subprocess(snippet)
    b = _run_in_subprocess(snippet)
    assert a == b, f"processus 1 -> {a}, processus 2 -> {b}"
    assert a == str(stress.derive_seed("T2", "c003cd25f5a5f"))


@test
def seed_is_independent_of_pythonhashseed():
    snippet = "print(stress.derive_seed('T2','c003cd25f5a5f'))"
    seeds = {h: _run_in_subprocess(snippet, hashseed=h)
             for h in ("0", "1", "12345", "4294967295", "random")}
    assert len(set(seeds.values())) == 1, seeds


@test
def seed_changes_with_clip_id():
    a = stress.derive_seed("T2", "c003cd25f5a5f")
    b = stress.derive_seed("T2", "c00c343da6afa")
    assert a != b


@test
def seed_changes_with_transform_name():
    seeds = {n: stress.derive_seed(n, "c003cd25f5a5f") for n in ("T0", "T1", "T2", "T3")}
    assert len(set(seeds.values())) == 4, seeds


@test
def seed_separator_prevents_concatenation_ambiguity():
    """(« ab », « c ») et (« a », « bc ») ne doivent pas donner la même graine."""
    assert stress.derive_seed("ab", "c") != stress.derive_seed("a", "bc")


@test
def seed_golden_value_is_frozen():
    """Valeur figée. Si le schéma de dérivation change, ce test doit échouer.

    Il ne recalcule pas la graine avec la même fonction des deux côtés : la
    valeur attendue est un littéral, et le digest est recalculé à la main
    depuis la définition documentée.
    """
    assert stress.derive_seed("T2", "c003cd25f5a5f") == 10036266156883065158
    # Re-dérivation indépendante, sans appeler derive_seed.
    payload = f"20260912\x1fT2\x1fc003cd25f5a5f".encode("utf-8")
    expected = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    assert expected == 10036266156883065158, expected
    assert stress.SEED_DIGEST_BYTES == 8 and stress.SEED_SEPARATOR == "\x1f"


@test
def no_call_to_python_hash_remains_in_stress():
    """Aucun APPEL à hash() ne doit subsister. Analyse de l'AST, pas du texte :
    la docstring cite `hash()` pour expliquer pourquoi on l'a retiré, et un
    simple grep confondrait la mention avec l'appel."""
    import ast
    tree = ast.parse((ROOT / "scripts" / "temporal" / "stress.py").read_text())
    offenders = [n.lineno for n in ast.walk(tree)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                 and n.func.id == "hash"]
    assert not offenders, f"appel à hash() aux lignes {offenders}"


@test
def t2_is_bit_identical_across_two_processes():
    snippet = ("x = np.sin(np.arange(8000)/50.0); "
               "y = stress.t2_block_permutation(x, stress.clip_rng('T2','c003cd25f5a5f')); "
               "print(hashlib.sha256(np.ascontiguousarray(y).tobytes()).hexdigest())")
    a, b = _run_in_subprocess(snippet), _run_in_subprocess(snippet, hashseed="777")
    assert a == b, f"T2 diverge entre processus : {a} vs {b}"


@test
def t3_is_bit_identical_across_two_processes():
    snippet = ("x = np.sin(np.arange(8000)/50.0); "
               "y = stress.t3_phase_randomisation(x, stress.clip_rng('T3','c003cd25f5a5f')); "
               "print(hashlib.sha256(np.ascontiguousarray(y).tobytes()).hexdigest())")
    a, b = _run_in_subprocess(snippet), _run_in_subprocess(snippet, hashseed="777")
    assert a == b, f"T3 diverge entre processus : {a} vs {b}"


@test
def t0_and_t1_are_unaffected_by_the_rng():
    """T0 et T1 ignorent le générateur : leur sortie ne doit dépendre que de l'entrée."""
    x = np.sin(np.arange(8000) / 50.0)
    for name in ("T0", "T1"):
        fn = stress.TRANSFORMS[name]["fn"]
        a = fn(x, np.random.default_rng(1))
        b = fn(x, np.random.default_rng(999999))
        assert np.array_equal(a, b), f"{name} dépend du RNG, ce qui n'était pas prévu"
    # et entre processus
    snippet = ("x = np.sin(np.arange(8000)/50.0); "
               "print(hashlib.sha256(np.ascontiguousarray("
               "stress.t1_reverse(x, stress.clip_rng('T1','c1'))).tobytes()).hexdigest())")
    assert _run_in_subprocess(snippet) == _run_in_subprocess(snippet, hashseed="42")


@test
def stress_transforms_are_deterministic():
    x = np.random.default_rng(6).normal(size=8000)
    for name, spec in stress.TRANSFORMS.items():
        a = spec["fn"](x, stress.clip_rng(name, "cid"))
        b = spec["fn"](x, stress.clip_rng(name, "cid"))
        assert np.array_equal(a, b), name


@test
def contract_rejects_non_probability():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        split = fixture_split(tmp)
        probs = {c.clip_id: 0.5 for c in split.clips}
        probs[split.clips[0].clip_id] = 1.7
        run = write_preds(tmp / "run", split, probs)
        msg = must_raise(contract.ContractError, contract.load_run, run, split,
                         require_frozen_sha=False)
        assert "hors bornes" in msg


@test
def contract_rejects_missing_no_tuning_declaration():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        split = fixture_split(tmp)
        run = write_preds(tmp / "run", split, {c.clip_id: 0.5 for c in split.clips},
                          meta_overrides={"test_labels_not_used_for_tuning": False})
        must_raise(contract.ContractError, contract.load_run, run, split,
                   require_frozen_sha=False)


@test
def contract_publishes_coverage_for_every_check():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        split = fixture_split(tmp)
        run = write_preds(tmp / "run", split, {c.clip_id: 0.5 for c in split.clips})
        loaded = contract.load_run(run, split, require_frozen_sha=False)
        assert loaded.checks, "aucun contrôle enregistré"
        for c in loaded.checks:
            assert c.coverage, f"contrôle sans couverture : {c.name}"


@test
def frozen_split_identity_matches_repo():
    """[data] le manifeste du dépôt est bien celui qui est gelé."""
    split = split_loader.load_split(ROOT / "manifests")
    assert split.sha256 == split_loader.FROZEN_SPLIT_SHA256
    assert len(split.clips) == 1000
    assert len({c.group_id for c in split.clips}) == 185


@test
def feature_vector_lengths_match_declared_names():
    """[data] chaque échelon produit autant de valeurs que de noms déclarés."""
    if not OPTS["data_root"]:
        raise SkipTest("pas de --data-root")
    split = split_loader.load_split(ROOT / "manifests")
    raw = split_loader.read_wav_raw(OPTS["data_root"], split.path_of(split.clips[0].clip_id))
    for k, spec in features.LADDER.items():
        assert len(spec["fn"](raw)) == len(spec["names"]), k


@test
def c0_reproduces_published_rms_auc():
    """[data] le harnais reproduit le chiffre publié du contrôle RMS (0.878)."""
    if not OPTS["runs_dir"]:
        raise SkipTest("pas de --runs-dir")
    m = Path(OPTS["runs_dir"]).parent / "eval-out" / "metrics.json"
    if not m.exists():
        raise SkipTest("metrics.json absent")
    r = json.load(open(m))
    auc = r["runs"]["c0"]["folds"]["test"]["clip_level"]["roc_auc"]
    assert abs(auc - 0.8779) < 0.002, f"C0 clip AUC = {auc}, publié 0.878"


# --------------------------------------------------------------------------- #
# RAPPORT FINAL ET PROVENANCE — M1 / M2 / M3
# --------------------------------------------------------------------------- #
import run_controls  # noqa: E402
import build_final_report  # noqa: E402
from evaluate_predictions import evaluate_run  # noqa: E402

LADDER_IDS = ("c0", "c1", "c2", "c2b", "c3")
FAKE_TRAINING_COMMIT = "feedfacefeed" + "0" * 28


def _md_sections(md: str) -> dict:
    """Découpe le markdown par titre `## N.` -> texte de la section."""
    out, cur = {}, None
    for line in md.splitlines():
        if line.startswith("## "):
            cur = line[3:].split(".", 1)[0].strip()
            out[cur] = []
        elif cur is not None:
            out[cur].append(line)
    return {k: "\n".join(v) for k, v in out.items()}


def _table_run_ids(section: str) -> list[str]:
    """run_id en première colonne de chaque ligne de tableau, dans l'ordre."""
    ids = []
    for line in section.splitlines():
        if line.startswith("| `"):
            ids.append(line.split("`")[1])
    return ids


def _assert_ladder_sections(md: str) -> dict:
    sec = _md_sections(md)
    for k in ("2", "3", "4", "7"):
        assert k in sec, f"section §{k} absente du rapport"
    # §2 : un tableau, §3 : test + validation, §4 : IC — les cinq contrôles, rien d'autre.
    assert _table_run_ids(sec["2"]) == list(LADDER_IDS), _table_run_ids(sec["2"])
    assert _table_run_ids(sec["3"]) == list(LADDER_IDS) * 2, _table_run_ids(sec["3"])
    assert _table_run_ids(sec["4"]) == list(LADDER_IDS), _table_run_ids(sec["4"])
    for k in ("2", "3", "4"):
        assert "-T1" not in sec[k] and "-T2" not in sec[k] and "-T3" not in sec[k], \
            f"un run de stress s'est glissé dans §{k}"
    return sec


def _synthetic_runs(root: Path, split) -> list[Path]:
    """Cinq contrôles, trois runs de stress de c2b, un contrôle négatif.

    Métadonnées produites par les MÊMES fonctions que run_controls.py. Un run de
    stress porte un nom hors convention (`c2b_phase_stress`) : le rapport doit le
    rattacher par `base_run_id`, pas par son nom.
    """
    rng = np.random.default_rng(7)
    y = {c.clip_id: c.label for c in split.clips}
    dirs = []
    fp = "ab" * 32
    for i, rid in enumerate(LADDER_IDS + ("c1-shuffled",)):
        control = {"c2b": "C2b"}.get(rid, rid[:2].upper())
        probs = {cid: float(np.clip(0.5 + (0.1 + 0.05 * i) * (2 * lab - 1)
                                    + rng.normal(0, 0.2), 0, 1)) for cid, lab in y.items()}
        prov = {"model_definition_commit": features.LADDER[control]["definition_commit"],
                "training_worktree_dirty": False, "execution_commit": FAKE_TRAINING_COMMIT,
                "execution_worktree_dirty": False, "model_fingerprint": fp if rid == "c2b" else rid}
        d = contract.write_run(root / rid, run_id=rid, model_name=f"{control} fixture",
                               checkpoint="aucun checkpoint sérialisé", split=split,
                               training_commit=FAKE_TRAINING_COMMIT,
                               threshold_rule="argmax val", probabilities=probs,
                               extra={"control_level": control, "selected_C": 0.1, **prov})
        dirs.append(d)
        if rid == "c2b":
            base_probs, base_prov = probs, prov
    for tname, rid in (("T1", "c2b-T1"), ("T2", "c2b-T2"), ("T3", "c2b_phase_stress")):
        f = run_controls.stress_run_fields("C2b", tname, 0.1, "c2b", base_prov)
        probs = {k: float(np.clip(v + rng.normal(0, 0.1), 0, 1)) for k, v in base_probs.items()}
        dirs.append(contract.write_run(root / rid, run_id=rid, model_name=f["model_name"],
                                       checkpoint=f["checkpoint"], split=split,
                                       training_commit=FAKE_TRAINING_COMMIT,
                                       threshold_rule=f["threshold_rule"],
                                       probabilities=probs, extra=f["extra"]))
    tslm_probs = {cid: float(np.clip(0.5 + 0.3 * (2 * lab - 1) + rng.normal(0, 0.2), 0, 1))
                  for cid, lab in y.items()}
    tslm_meta = {"checkpoint": "gs://fixture/step-42", "training_commit": "c0ffee" + "0" * 34}
    dirs.append(contract.write_run(root / "tslm-v1", run_id="tslm-v1", model_name="TSLM fixture",
                                   split=split, threshold_rule="aucun seuil",
                                   probabilities=tslm_probs, **tslm_meta))
    for tname in ("T1", "T2"):
        probs = {k: float(np.clip(v + rng.normal(0, 0.1), 0, 1)) for k, v in tslm_probs.items()}
        dirs.append(contract.write_run(root / f"tslm-v1-{tname}", run_id=f"tslm-v1-{tname}",
                                       model_name=f"TSLM fixture sous {tname}", split=split,
                                       threshold_rule="aucun seuil", probabilities=probs,
                                       extra={"stress_transform": tname, "retrained": False},
                                       **tslm_meta))
    return dirs


@test
def final_report_render_keeps_only_c0_to_c3_in_the_ladder():
    """M1 — le markdown GÉNÉRÉ : échelle §2-§4 par liste blanche, stress seulement en §7."""
    split = split_loader.load_split(ROOT / "manifests")
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        dirs = _synthetic_runs(tmp / "runs", split)
        out = tmp / "report"
        r = subprocess.run([sys.executable, str(ROOT / "scripts/eval/build_final_report.py"),
                            "--runs", *map(str, dirs), "--out", str(out),
                            "--tslm-run-id", "tslm-v1",
                            "--manifests", str(ROOT / "manifests"),
                            "--stress-report", str(_fixture_invariants(tmp))],
                           capture_output=True, text=True, cwd=ROOT)
        assert r.returncode == 0, r.stderr[-2000:]
        md = (out / "FINAL_EVALUATION.md").read_text()
        sec = _assert_ladder_sections(md)
        assert "c1-shuffled" not in sec["2"] + sec["3"] + sec["4"]
        assert "tslm-v1" not in sec["2"] + sec["3"] + sec["4"], "le TSLM n'est pas un contrôle"
        assert "tslm-v1" in _table_run_ids(sec["6"]), "le TSLM doit avoir ses tableaux en §6"
        # §7 : l'identité de chaque modèle stressé suit SA provenance réelle.
        ident = {l.split("`")[1]: l for l in sec["7"].splitlines() if l.startswith("- `")}
        assert set(ident) == {"c2b", "tslm-v1"}, ident
        assert "aucun checkpoint sérialisé" in ident["c2b"]
        assert "checkpoint sérialisé `gs://fixture/step-42`" in ident["tslm-v1"]
        assert "aucun checkpoint" not in ident["tslm-v1"].lower()
        assert "Aucun checkpoint n'est sérialisé" not in sec["7"]
        tslm_rows = [l.split("|")[2].strip() for l in sec["7"].splitlines()
                     if l.startswith("| `tslm-v1`")]
        assert tslm_rows == ["**T0** original", "T1", "T2"], tslm_rows
        # §7 : les trois runs de stress, y compris celui au nom hors convention.
        rows = [l for l in sec["7"].splitlines() if l.startswith("| `c2b`")]
        variants = [l.split("|")[2].strip() for l in rows]
        assert variants == ["**T0** original", "T1", "T2", "T3"], variants
        # §2 : le training_commit est celui DÉCLARÉ par le run, pas HEAD.
        assert FAKE_TRAINING_COMMIT[:12] in sec["2"]
        head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, cwd=ROOT).stdout.strip()
        assert head[:12] not in sec["2"], "le rapport a recalculé un commit d'ajustement"
        # M2 : aucune métrique dépendante du seuil dans la section stress.
        stress_table = "\n".join(l for l in sec["7"].splitlines() if l.startswith("| "))
        assert "F1" not in stress_table and "exactitude" not in stress_table.lower()
        m = json.loads((out / "metrics.json").read_text())
        assert m["control_ladder"] == list(LADDER_IDS)
        for k, v in m["stress_comparisons"].items():
            assert not any("macro_f1" in key for key in v), (k, list(v))
        assert set(m["stress_prediction_shift"]) == {"c2b-T1", "c2b-T2", "c2b_phase_stress",
                                                     "tslm-v1-T1", "tslm-v1-T2"}


def _fixture_invariants(tmp: Path) -> Path:
    inv = {"manifest_dir": "x", "n_clips": 1000, "transforms": {
        n: {"description": stress.TRANSFORMS[n]["description"], "n_records": 1000,
            "n_clusters": 185, "fft_magnitude_relative_deviation_median": 0.0,
            "amplitude_histogram_identical_all": True,
            "clip_fold_label_cluster_mapping_violations": 0} for n in stress.TRANSFORMS}}
    p = tmp / "inv.json"
    p.write_text(json.dumps(inv))
    return p


@test
def committed_final_report_keeps_stress_runs_out_of_the_ladder():
    """M1 — l'artefact COMMITÉ respecte la même règle que le rendu."""
    md = (ROOT / "artifacts/final_evaluation/FINAL_EVALUATION.md").read_text()
    sec = _assert_ladder_sections(md)
    for t in ("T1", "T2", "T3"):
        assert any(l.startswith("| `c2b`") and l.split("|")[2].strip() == t
                   for l in sec["7"].splitlines()), f"c2b sous {t} absent de §7"


@test
def stress_run_metadata_states_what_actually_happens():
    """M2/M3 — seuil, checkpoint et `retrained` décrivent le comportement réel."""
    f = run_controls.stress_run_fields("C2b", "T2", 0.1, "c2b", {"model_fingerprint": "x"})
    rule = f["threshold_rule"]
    assert "recalculé sur le fold de validation de ce run transformé" in rule
    assert "ne sont pas utilisées pour les conclusions de sensibilité temporelle" in rule
    assert "hérité" not in rule and "T0" not in rule
    ck = f["checkpoint"]
    assert ck.startswith("aucun checkpoint sérialisé")
    assert "355f074" in ck and "réajustée de façon déterministe sur T0/train" in ck
    assert "jamais ajustée sur T2" in ck
    assert "même checkpoint" not in ck and "NON réentraîné" not in ck
    e = f["extra"]
    assert e["retrained"] is False and e["retrained_meaning"] == "not retrained on stressed data"
    assert e["fit_data"] == "T0/train" and e["stress_transform"] == "T2"
    assert e["base_run_id"] == "c2b"


@test
def engine_threshold_of_a_stressed_run_comes_from_its_own_validation():
    """M2 — le moteur fait ce que la règle déclarée dit : val du run, pas celle de T0."""
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        split = fixture_split(tmp)
        rng = np.random.default_rng(3)
        base = {c.clip_id: float(rng.uniform()) for c in split.clips}
        stressed = {k: float(np.clip(v * 0.5 + 0.4 * rng.uniform(), 0, 1)) for k, v in base.items()}
        rb = contract.load_run(write_preds(tmp / "b", split, base), split, require_frozen_sha=False)
        rs = contract.load_run(write_preds(tmp / "s", split, stressed), split,
                               require_frozen_sha=False)
        from evaluate_predictions import vectors
        for run in (rb, rs):
            yv, sv, gv, _ = vectors(split, run, "val")
            expected = round(float(metrics.pick_threshold(yv, sv, gv)), 6)
            assert evaluate_run(split, run)["threshold_recomputed_on_val"] == expected
        assert (evaluate_run(split, rb)["threshold_recomputed_on_val"]
                != evaluate_run(split, rs)["threshold_recomputed_on_val"]), \
            "fixture sans pouvoir discriminant : les deux seuils coïncident"


@test
def refit_is_deterministic_and_the_fingerprint_identifies_the_model():
    """M3 — sans checkpoint, deux ajustements identiques donnent le même modèle, octet pour octet."""
    rng = np.random.default_rng(11)
    n = 240
    folds = np.array(["train"] * 140 + ["val"] * 60 + ["test"] * 40)
    g = np.array([f"g{i // 4}" for i in range(n)])
    y = np.array([int(gg[1:]) % 2 for gg in g])
    X = rng.normal(size=(n, 3)) + y[:, None] * 0.8
    ids = np.array([f"c{i}" for i in range(n)])
    runs = []
    for _ in range(2):
        probs, C, _, clf, _, scaler = run_controls.fit_control(
            "C2b", ids, X, y, g, folds, shuffle_labels=False)
        runs.append((probs, run_controls.model_fingerprint("C2b", C, scaler, clf)))
    assert runs[0] == runs[1], "réajustement non déterministe"
    X2 = X.copy()
    X2[0, 0] += 1e-9
    _, C, _, clf, _, scaler = run_controls.fit_control(
        "C2b", ids, X2, y, g, folds, shuffle_labels=False)
    assert run_controls.model_fingerprint("C2b", C, scaler, clf) != runs[0][1], \
        "l'empreinte ne distingue pas deux modèles différents"


@test
def report_refuses_a_stress_run_that_is_not_the_base_model():
    """M3 — sentinelle systématique : toute divergence d'identité entre T0 et stress est refusée.

    Oracle tiré de la spécification, pas de l'implémentation : pour chaque nature
    de modèle, chaque champ d'identité est remplacé par chaque valeur différente
    (absent, vide, autre). Seules les variations de LIBELLÉ d'un modèle non
    sérialisé, dont l'identité tient à l'empreinte, sont acceptées.
    """
    ns = contract.NO_SERIALIZED_CHECKPOINT
    bases = {
        "contrôle": {"training_commit": "a" * 40, "model_definition_commit": "d" * 40,
                     "checkpoint": f"{ns} : logreg(C=0.1)", "control_level": "C2b",
                     "model_fingerprint": "f" * 64},
        "tslm": {"training_commit": "a" * 40, "checkpoint": "gs://m/step-1"},
        "tslm+empreinte": {"training_commit": "a" * 40, "checkpoint": "gs://m/step-1",
                           "model_fingerprint": "f" * 64},
        "tslm+définition": {"training_commit": "a" * 40, "checkpoint": "gs://m/step-1",
                            "model_definition_commit": "d" * 40},
    }
    DROP = object()
    alternatives = (DROP, None, "", "   ", "\t", "zzz")
    check = build_final_report.check_stress_provenance
    n_refused = 0
    for kind, base in bases.items():
        valid = {**base, "retrained": False}
        if kind == "contrôle":
            valid["checkpoint"] = f"{ns} : jamais ajustée sur T2"   # libellé différent : accepté
        check("base", base, "stress", valid)
        serialized = kind != "contrôle"
        for key in ("retrained", "training_commit", "model_definition_commit",
                    "model_fingerprint", "checkpoint", "control_level"):
            for alt in alternatives + ((True, "false") if key == "retrained" else ()):
                m = dict(valid)
                if alt is DROP:
                    m.pop(key, None)
                else:
                    m[key] = alt
                blank = lambda v: v is DROP or v is None or (isinstance(v, str) and not v.strip())
                if m.get(key, DROP) == valid.get(key, DROP) or \
                        (key != "retrained" and blank(m.get(key, DROP)) and blank(valid.get(key, DROP))):
                    continue                                  # pas une divergence
                label_only = not serialized and key in ("checkpoint", "control_level")
                if label_only:
                    check("base", base, "stress", m)
                else:
                    must_raise(ValueError, check, "base", base, "stress", m)
                    n_refused += 1
        # Identité absente des DEUX côtés : égale, mais invérifiable.
        for key, alt in (("training_commit", None), ("training_commit", ""),
                         ("training_commit", "  "), ("training_commit", "unknown"),
                         *((("model_fingerprint", None), ("model_fingerprint", " "),
                            ("model_fingerprint", "f" * 63)) if not serialized else ()),
                         *((("checkpoint", None), ("checkpoint", ""), ("checkpoint", " "))
                           if kind == "tslm" else ()),
                         *((("model_definition_commit", "pas-un-commit"),)
                           if "model_definition_commit" in base else ())):
            nb = {**base, key: alt}
            must_raise(ValueError, check, "base", nb, "stress", {**nb, "retrained": False})
            n_refused += 1
    assert n_refused >= 80, n_refused
    # Normalisation documentée : absent et null valent « non déclaré » des deux côtés.
    t = bases["tslm"]
    check("base", t, "stress", {**t, "retrained": False, "model_definition_commit": None})
    check("base", {**t, "model_definition_commit": "  "}, "stress", {**t, "retrained": False})
    # Un commit court est une identité valide.
    short = {**t, "training_commit": "68bf202"}
    check("base", short, "stress", {**short, "retrained": False})
    # Le rendu utilise la même décision : un TSLM avec empreinte reste un checkpoint sérialisé.
    assert build_final_report.has_serialized_checkpoint(bases["tslm+empreinte"])
    assert not build_final_report.has_serialized_checkpoint(bases["contrôle"])


@test
def model_definition_commits_match_git_history():
    """M3 — chaque échelon est identique à sa version dans le commit déclaré, absente du parent."""
    import inspect
    if shutil.which("git") is None:
        raise SkipTest("git indisponible")
    assert features.LADDER["C2b"]["definition_commit"] == \
        "355f0743fb306da50897c0720e6fab5ecf947c52"
    for k, spec in features.LADDER.items():
        src = inspect.getsource(spec["fn"])
        at = subprocess.run(["git", "show", f"{spec['definition_commit']}:scripts/eval/harness/features.py"],
                            capture_output=True, text=True, cwd=ROOT)
        if at.returncode != 0:
            raise SkipTest("historique Git incomplet (clone superficiel ?)")
        assert src in at.stdout, f"{k} : la définition a changé depuis {spec['definition_commit'][:7]}"
        parent = subprocess.run(["git", "show", f"{spec['definition_commit']}^:scripts/eval/harness/features.py"],
                                capture_output=True, text=True, cwd=ROOT).stdout
        assert src not in parent, f"{k} : existait déjà avant {spec['definition_commit'][:7]}"


@test
def stress_runs_share_the_base_model():
    """[runs] M3 — mêmes empreinte, commit d'ajustement et définition que le run T0."""
    if not OPTS["runs_dir"]:
        raise SkipTest("pas de --runs-dir")
    base = Path(OPTS["runs_dir"])
    if not (base / "c2b-T2").exists():
        raise SkipTest("runs de stress absents")
    b = json.loads((base / "c2b" / "metadata.json").read_text())
    assert "stress_transform" not in b
    assert b["model_definition_commit"] == features.DEFINITION_COMMIT_C2B
    assert b["training_worktree_dirty"] is False, "run de base produit sur un worktree modifié"
    for rid in ("c2b-T1", "c2b-T2", "c2b-T3"):
        m = json.loads((base / rid / "metadata.json").read_text())
        assert m["stress_transform"] == rid.rsplit("-", 1)[1] and m["base_run_id"] == "c2b"
        assert m["model_fingerprint"] == b["model_fingerprint"], rid
        assert m["training_commit"] == b["training_commit"], rid
        assert m["model_definition_commit"] == b["model_definition_commit"], rid
        assert m["threshold_rule"] == run_controls.STRESS_THRESHOLD_RULE, rid
        assert m["retrained"] is False and m["retrained_meaning"] == "not retrained on stressed data"
        build_final_report.check_stress_provenance("c2b", b, rid, m)


@test
def report_refuses_an_orphan_stress_run():
    """M3 — un run de stress sans son run T0 est refusé, pas omis en silence."""
    split = split_loader.load_split(ROOT / "manifests")
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        probs = {c.clip_id: (0.3 if c.label == 0 else 0.7) for c in split.clips}
        d = contract.write_run(tmp / "c2b-T2", run_id="c2b-T2", model_name="orphelin",
                               checkpoint="x", training_commit="0" * 40, split=split,
                               threshold_rule="x", probabilities=probs,
                               extra={"stress_transform": "T2", "base_run_id": "c2b_typo",
                                      "retrained": False})
        r = subprocess.run([sys.executable, str(ROOT / "scripts/eval/build_final_report.py"),
                            "--runs", str(d), "--out", str(tmp / "out"),
                            "--manifests", str(ROOT / "manifests")],
                           capture_output=True, text=True, cwd=ROOT)
        assert r.returncode != 0, "un run de stress orphelin a été accepté"
        assert "c2b-T2 -> c2b_typo" in r.stderr, r.stderr[-500:]
        assert not (tmp / "out" / "FINAL_EVALUATION.md").exists()


@test
def training_commit_is_read_from_the_repo_not_the_caller_directory():
    """M3 — git_state() relève le dépôt du script, quel que soit le répertoire courant."""
    if shutil.which("git") is None:
        raise SkipTest("git indisponible")
    here = run_controls.git_state()[0]
    assert here != "unknown"
    snippet = (f"import os, sys; os.chdir('/'); sys.path.insert(0, {str(ROOT / 'scripts/eval')!r}); "
               f"sys.path.insert(0, {str(ROOT / 'scripts/temporal')!r}); "
               "import run_controls; print(run_controls.git_state()[0])")
    out = subprocess.run([sys.executable, "-c", snippet], capture_output=True, text=True, cwd="/")
    assert out.stdout.strip() == here, (out.stdout, out.stderr[-300:])


@test
def stress_base_id_normalizes_blank_declarations():
    """Un base_run_id absent, null, vide ou blanc retombe sur la convention `<base>-Tn`."""
    f = build_final_report.stress_base_id
    for meta in ({}, {"base_run_id": None}, {"base_run_id": ""}, {"base_run_id": "   "}):
        assert f("c2b-T2", meta) == "c2b", meta
    assert f("c2b_phase_stress", {"base_run_id": " c2b "}) == "c2b"


@test
def prediction_shift_never_writes_nan():
    """Une série constante rend la corrélation indéfinie : None, jamais NaN dans le JSON."""
    split = split_loader.load_split(ROOT / "manifests")
    Run = type("Run", (), {})
    base, flat = Run(), Run()
    base.probabilities = {c.clip_id: (0.2 if c.label == 0 else 0.8) for c in split.clips}
    for value in (0.5, 0.1, 0.3, 1 / 3):                 # 0.1 : écart-type flottant non nul
        flat.probabilities = {c.clip_id: value for c in split.clips}
        for a, b in ((base, flat), (flat, base)):
            sh = build_final_report.prediction_shift(split, a, b)
            assert sh["pearson_r_with_T0"] is None, (value, sh["pearson_r_with_T0"])
            json.dumps(sh, allow_nan=False)


@test
def stress_artifact_hash_refuses_missing_or_empty_sets():
    """Un jeu absent ou vide ne doit pas recevoir l'empreinte du vide."""
    import stress_provenance
    with tempfile.TemporaryDirectory() as t:
        must_raise(FileNotFoundError, stress_provenance.artifact_sha256, Path(t) / "absent")
        (Path(t) / "vide").mkdir()
        must_raise(FileNotFoundError, stress_provenance.artifact_sha256, Path(t) / "vide")


@test
def stress_artifact_hash_detects_any_byte_change():
    """Manifeste de provenance : l'empreinte d'un jeu change au moindre octet ou renommage."""
    import stress_provenance
    with tempfile.TemporaryDirectory() as t:
        d = Path(t) / "T2"
        (d / "records").mkdir(parents=True)
        (d / "records" / "a.parquet").write_bytes(b"\x00\x01\x02")
        (d / "manifest.json").write_text("{}")
        h0, n = stress_provenance.artifact_sha256(d)
        assert n == 2 and h0 == stress_provenance.artifact_sha256(d)[0]
        (d / "records" / "a.parquet").write_bytes(b"\x00\x01\x03")
        h1, _ = stress_provenance.artifact_sha256(d)
        (d / "records" / "a.parquet").rename(d / "records" / "b.parquet")
        h2, _ = stress_provenance.artifact_sha256(d)
        assert len({h0, h1, h2}) == 3


@test
def committed_stress_manifest_matches_artifacts_on_disk():
    """[data] le manifeste commité décrit bien les jeux présents sur disque."""
    man = ROOT / "artifacts/final_evaluation/stress_provenance.json"
    if not OPTS["runs_dir"] or not man.exists():
        raise SkipTest("pas de --runs-dir ou manifeste absent")
    root = Path(OPTS["runs_dir"]).parent / "timef-stress"
    if not root.exists():
        raise SkipTest("jeux de stress absents")
    import stress_provenance
    m = json.loads(man.read_text())
    assert sorted(m["transforms"]) == ["T0", "T1", "T2", "T3"]
    for name, e in m["transforms"].items():
        assert e["artifact_sha256"] == stress_provenance.artifact_sha256(root / name)[0], name
        assert e["split_sha256"] == split_loader.FROZEN_SPLIT_SHA256
        assert e["stress_seed"] == stress.STRESS_SEED and e["seed_scheme_version"] == 2
        assert e["n_records"] == e["n_records_at_generation"] == 1000
        assert e["generator_commit"] is None and e["generator_commit_note"]



class SkipTest(Exception):
    pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root")
    ap.add_argument("--runs-dir")
    args = ap.parse_args()
    OPTS["data_root"] = args.data_root if args.data_root and Path(args.data_root).exists() else None
    OPTS["runs_dir"] = args.runs_dir if args.runs_dir and Path(args.runs_dir).exists() else None

    tests = [v for v in list(globals().values()) if callable(v) and getattr(v, "_is_test", False)]
    for fn in tests:
        name = fn.__name__
        try:
            fn()
            PASSED.append(name)
            print(f"PASS  {name}")
        except SkipTest as e:
            SKIPPED.append(name)
            print(f"SKIP  {name}  ({e})")
        except Exception:
            FAILED.append((name, traceback.format_exc(limit=3)))
            print(f"FAIL  {name}")
            print("      " + traceback.format_exc(limit=3).replace("\n", "\n      "))

    print(f"\n{len(PASSED)} réussis, {len(FAILED)} échoués, {len(SKIPPED)} ignorés")
    if FAILED:
        print("échecs :", [n for n, _ in FAILED])
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
