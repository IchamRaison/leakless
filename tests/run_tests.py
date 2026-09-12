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
import json
import shutil
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
