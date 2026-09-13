"""D2 : une sonde non linéaire, trois fits CPU sur les seuls 598 train.

Préinscrire puis exécuter explicitement. Aucun réglage, scaler externe, Qwen,
cache/WAV val/test/externe ou sélection de politique. Les références publiées
sont vérifiées, jamais réentraînées. Une étape incomplète reste non reprenable.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import inspect
from pathlib import Path
import platform
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts/eval"), str(ROOT / "scripts/tslm")]
import diagnose_train as diagnostic
import run_v2_campaign as campaign
import v2_c1
from export_run import sha256_file
from harness import split_loader
from pipe.tslm.campaign import load_cache, write_json
from pipe.tslm.preprocessing import CANONICAL_VERSION

SCHEMA = "pipe-causal-d2-nonlinear-v1"
FOLDS_SHA256 = campaign.FOLDS_SHA256
PROBES_SHA256 = "aa16a7b1230ed0badbaac8db942b721ae07a3d213903645025bc2d80af456fc8"
PARAMS = dict(loss="log_loss", learning_rate=.05, max_iter=200, max_leaf_nodes=7,
              min_samples_leaf=10, l2_regularization=1, max_bins=255,
              early_stopping=False, random_state=20260913, class_weight=None,
              categorical_features=None)
COUNTS = {"n_configs_compared": 1, "n_fits": 3, "reference_fits": 0}
PRIMARY_RULE = "three_strictly_positive_heldout_group_auc_deltas_vs_TimeNet256"
LIMITS = [
    "Les folds train déjà étudiés ne constituent pas une confirmation indépendante.",
    "Trois deltas positifs décrivent une cohérence, pas une significativité statistique.",
    "L'échec de cette sonde ne prouve pas un plafond absolu d'information ; un fit faible "
    "confond capacité, régularisation et optimisation.",
    "La pondération par clip reste commune ; les groupes sont heuristiques et non une "
    "preuve d'indépendance. La classe binaire non-leak inclut du bruit.",
    "Aucun comparateur historique sur les clips vus n'est disponible ; comparaisons "
    "appariées réservées uniquement. Aucun seuil choisi ni politique adoptée."]
SOURCE_NAMES = (
    "scripts/tslm/diagnose_nonlinear.py", "scripts/tslm/diagnose_train.py",
    "scripts/tslm/run_v2_campaign.py", "scripts/tslm/v2_c1.py", "scripts/tslm/export_run.py",
    "scripts/tslm/audit_text.py", "scripts/tslm/diagnose_parity.py",
    "src/pipe/tslm/campaign.py", "src/pipe/tslm/preprocessing.py", "src/pipe/tslm/prepare.py",
    "src/pipe/tslm/model.py", "src/pipe/tslm/predict.py",
    "scripts/timenet/leakless_acoustic/connector.py", "scripts/eval/harness/features.py",
    "scripts/eval/harness/metrics.py", "scripts/eval/harness/split_loader.py")
PATH_NAMES = ("prepared", "previous_prepared", "parity_report", "reference", "manifests")


def estimator():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(**PARAMS)


def runtime_identity():
    """Construction sans fit ; paramètres complets et runtime effectivement importé."""
    from threadpoolctl import threadpool_info, threadpool_limits
    model = estimator()
    implementation = Path(inspect.getfile(type(model))).resolve()
    files = sorted(p for p in implementation.parent.rglob("*")
                   if p.is_file() and p.suffix in (".py", ".so"))
    with threadpool_limits(limits=1):
        pools = threadpool_info()
        if any(p["num_threads"] != 1 for p in pools):
            raise ValueError("Limite CPU d'un thread non effective")
    return {"python": platform.python_version(), "implementation": platform.python_implementation(),
            "executable": str(Path(sys.executable).resolve()),
            "executable_sha256": sha256_file(Path(sys.executable).resolve()),
            "packages": {name: importlib.metadata.version(name)
                         for name in ("numpy", "scipy", "scikit-learn", "threadpoolctl")},
            "estimator": f"{type(model).__module__}.{type(model).__name__}",
            "get_params": model.get_params(deep=True),
            "implementation_sha256": {str(p): sha256_file(p) for p in files},
            "threadpools_limited": sorted(pools, key=lambda p: p["filepath"])}


def matrix_from_caches(rows, prepared, previous_prepared):
    """Conserver exactement les float32 : ordre row-major et douze zéros inclus."""
    current = load_cache(prepared, "train", rows, CANONICAL_VERSION)
    previous = load_cache(previous_prepared, "train", rows, CANONICAL_VERSION)
    for cid in current:
        if (current[cid].dtype != np.float32 or previous[cid].dtype != np.float32
                or current[cid].tobytes() != previous[cid].tobytes()):
            raise ValueError("Bandes float32 différentes de la sonde publiée")
    matrix = np.stack([current[r["clip_id"]].reshape(256) for r in rows])
    validate_matrix(rows, matrix)
    return matrix


def validate_matrix(rows, matrix):
    if (matrix.dtype != np.float32 or matrix.shape != (len(rows), 256)
            or not np.isfinite(matrix).all() or np.any(matrix < 0)
            or np.any(matrix.reshape(-1, 4, 64)[:, :, 61:] != 0)):
        raise ValueError("Les 256 valeurs float32 exactes, dont douze pads, sont requises")


def report(rows, probabilities):
    result = campaign.metric_report(rows, probabilities, threshold=.5)
    scores = np.array([probabilities[r["clip_id"]] for r in rows], dtype=np.float64)
    y = np.array([r["label"] == "leak" for r in rows])
    # NLL des probabilités réellement conservées, sans clipping ou epsilon inventé.
    with np.errstate(divide="ignore"):
        nll = np.where(y, -np.log(scores), -np.log1p(-scores))
    n_infinite = int(np.isinf(nll).sum())
    result["binary_nll"] = {"n_clips": len(rows), "n_infinite": n_infinite,
        "mean": None if n_infinite else float(np.mean(nll)),
        "sum": None if n_infinite else float(np.sum(nll)),
        "policy": "unclipped_saved_probabilities;null_mean_and_sum_mean_positive_infinity"}
    return result


def reference_reports(rows, matrix, folds, probes):
    """Vérifier IDs/cibles/partitions et agrégats des scores HORS FOLD existants."""
    _, _, partitions = v2_c1._partition_indices(rows, matrix, folds)
    results = {}
    for name, n_features in (("TimeNet256", 256), ("C1_fixed", 9)):
        reference = probes[name]
        predictions = reference["predictions_by_fold"]
        if (reference["n_features"] != n_features or len(predictions) != len(rows)
                or sorted(p["clip_id"] for p in predictions) != sorted(r["clip_id"] for r in rows)
                or sorted(p["inner_fold"] for p in predictions) != sorted(
                    fold_id for fold_id, _, ho in partitions for _ in ho)
                or sorted(f["fold_id"] for f in reference["folds"]) != [0, 1, 2]):
            raise ValueError("Couverture ou dimension de la référence incorrecte")
        reports = []
        for fold_id, tr, ho in partitions:
            heldout = [rows[i] for i in ho]
            selected = [p for p in predictions if p["inner_fold"] == fold_id]
            probabilities = {p["clip_id"]: p["probability_leak"] for p in selected}
            if len(probabilities) != len(selected):
                raise ValueError("Prédiction de référence dupliquée")
            recomputed = report(heldout, probabilities)
            recomputed.update(fold_id=fold_id, n_fit_clips=len(tr))
            stored = next(f for f in reference["folds"] if f["fold_id"] == fold_id)
            if stored != {k: v for k, v in recomputed.items() if k != "binary_nll"}:
                raise ValueError(f"Agrégats/cibles de référence incompatibles : {name}, fold {fold_id}")
            reports.append(recomputed)
        for key, field in (("mean_group_roc_auc", "group_roc_auc_full"),
                           ("mean_clip_roc_auc", "clip_roc_auc_full")):
            if reference[key] != float(np.mean([r[field] for r in reports])):
                raise ValueError("Moyenne des références différente des trois folds")
        results[name] = reports
    return results


def inputs(paths):
    reference = Path(paths["reference"])
    fold_path, probe_path = reference / "folds.json", reference / "probes.json"
    if sha256_file(fold_path) != FOLDS_SHA256 or sha256_file(probe_path) != PROBES_SHA256:
        raise ValueError("Folds ou sondes publiées modifiés")
    folds, metadata = campaign.read_json(fold_path), campaign.read_json(reference / "metadata.json")
    split = split_loader.load_split(paths["manifests"])
    rows = diagnostic.train_rows(split)
    campaign.validate_folds(rows, folds)
    expected_recipe = {"C": 1., "max_iter": 5000, "solver": "lbfgs", "class_weight": None,
                       "scaler": "StandardScaler_fit_inner_train_only", "seed": 20260912}
    if (metadata.get("schema") != "pipe-train-diagnostic-v1" or metadata.get("source_fold") != "train"
            or metadata.get("folds_sha256") != FOLDS_SHA256 or metadata.get("recipe") != expected_recipe
            or metadata.get("threshold") != .5 or metadata.get("n_total_fits") != 6
            or metadata.get("n_probe_configurations") != 1 or metadata.get("n_fixed_c1_configurations") != 1
            or metadata.get("parity_report_sha256") != folds["parity_report_sha256"]
            or folds.get("preprocessing_version") != CANONICAL_VERSION
            or sha256_file(Path(paths["previous_prepared"]) / "train.npz") != folds["cache_train_sha256"]):
        raise ValueError("Provenance de référence autre que le diagnostic train fixe")
    diagnostic.verify_parity(Path(paths["parity_report"]), Path(paths["prepared"]), CANONICAL_VERSION)
    matrix = matrix_from_caches(rows, paths["prepared"], paths["previous_prepared"])
    references = reference_reports(rows, matrix, folds["folds"], campaign.read_json(probe_path))
    files = [fold_path, probe_path, reference / "metadata.json", Path(paths["parity_report"]),
             Path(paths["prepared"]) / "train.npz", Path(paths["prepared"]) / "preparation.json",
             Path(paths["previous_prepared"]) / "train.npz",
             Path(paths["manifests"]) / "split_v2.csv", Path(paths["manifests"]) / "split_v2_audit.csv"]
    identity = {"files_sha256": {str(p): sha256_file(p) for p in files},
                "source_sha256": {name: sha256_file(ROOT / name) for name in SOURCE_NAMES},
                "rows_sha256": campaign.digest(rows),
                "matrix_sha256": hashlib.sha256(matrix.tobytes()).hexdigest(),
                "matrix_shape": list(matrix.shape), "matrix_dtype": str(matrix.dtype),
                "old_and_current_bands_byte_identical": True}
    return rows, matrix, folds["folds"], references, identity


def preregister(args):
    if Path(args.output).exists():
        raise FileExistsError("Préinscription uniquement dans un dossier neuf")
    if re.fullmatch(r"[0-9a-f]{40}", args.code_revision or "") is None:
        raise ValueError("Révision Git SHA40 réelle requise")
    paths = {name: str(Path(getattr(args, name)).resolve()) for name in PATH_NAMES}
    _, _, folds, _, identity = inputs(paths)
    registration = {"schema": SCHEMA, "created_at": datetime.now(timezone.utc).isoformat(),
        "code_revision": args.code_revision, "paths": paths, "identity": identity,
        "runtime": runtime_identity(), "params": PARAMS, "counts": COUNTS, "folds": folds,
        "primary_rule": PRIMARY_RULE, "diagnostic_threshold": .5, "thread_limit": 1,
        "selection_policy": None, "pooled_auc": False, "limits": LIMITS,
        "official_validation_test_external_audio_or_cache_read": False,
        "qwen_loaded": False, "fits_during_preregistration": 0}
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "preregistration.json", registration)
    return {"preregistration_sha256": sha256_file(args.output / "preregistration.json"), "fits": 0}


def context(output):
    registration = campaign.read_json(output / "preregistration.json")
    expected = {"schema": SCHEMA, "params": PARAMS, "counts": COUNTS, "primary_rule": PRIMARY_RULE,
                "diagnostic_threshold": .5, "thread_limit": 1, "selection_policy": None,
                "pooled_auc": False, "limits": LIMITS, "fits_during_preregistration": 0,
                "official_validation_test_external_audio_or_cache_read": False, "qwen_loaded": False}
    if any(registration.get(k) != v for k, v in expected.items()):
        raise ValueError("Préinscription différente du seul contraste D2 autorisé")
    rows, matrix, folds, references, identity = inputs(registration["paths"])
    if (identity != registration["identity"] or folds != registration["folds"]
            or runtime_identity() != registration["runtime"]):
        raise ValueError("Sources, entrées, paramètres complets ou runtime modifiés depuis la préinscription")
    return registration, rows, matrix, folds, references


def fit_three(rows, matrix, folds, directory, expected_params):
    """Tous les garde-fous avant le premier fit ; trois modèles neufs, sans scaler."""
    from threadpoolctl import threadpool_limits
    validate_matrix(rows, matrix)
    y, _, partitions = v2_c1._partition_indices(rows, matrix, folds)
    reports = []
    for fold_id, tr, ho in partitions:
        classifier = estimator()
        if classifier.get_params(deep=True) != expected_params:
            raise ValueError("Paramètres HGB différents de la préinscription")
        before = hashlib.sha256(matrix.tobytes()).hexdigest()
        with threadpool_limits(limits=1):
            classifier.fit(matrix[tr], y[tr])
            if classifier.classes_.tolist() != [0, 1] or classifier.n_iter_ != PARAMS["max_iter"]:
                raise ValueError("Classes ou nombre d'itérations HGB inattendus")
            partitions_report, predictions = {}, []
            for name, indices in (("fit", tr), ("heldout", ho)):
                selected_rows = [rows[i] for i in indices]
                probability_matrix = np.asarray(classifier.predict_proba(matrix[indices]))
                if probability_matrix.shape != (len(indices), 2):
                    raise ValueError("Forme des probabilités HGB inattendue")
                probabilities = dict(zip([r["clip_id"] for r in selected_rows],
                                         probability_matrix[:, 1].tolist()))
                partitions_report[name] = report(selected_rows, probabilities)
                predictions.extend({"clip_id": cid, "inner_fold": fold_id, "partition": name,
                                    "probability_leak": p} for cid, p in probabilities.items())
        if hashlib.sha256(matrix.tobytes()).hexdigest() != before:
            raise ValueError("Matrice canonique modifiée pendant le fit")
        record = {"fold_id": fold_id, "n_iter": int(classifier.n_iter_), "partitions": partitions_report}
        write_json(directory / f"fold-{fold_id}-predictions.json", predictions)
        write_json(directory / f"fold-{fold_id}-metrics.json", record)
        reports.append(record)
        print(f"D2 : fold {fold_id} terminé ; {len(reports)}/3 fits", flush=True)
    return reports


def comparisons(reports, references):
    result = {}
    for name, baseline in references.items():
        deltas = []
        for record in reports:
            current = record["partitions"]["heldout"]
            original = next(r for r in baseline if r["fold_id"] == record["fold_id"])
            nll, reference_nll = current["binary_nll"]["mean"], original["binary_nll"]["mean"]
            deltas.append({"fold_id": record["fold_id"],
                "group_auc_delta": current["group_roc_auc_full"] - original["group_roc_auc_full"],
                "clip_auc_delta": current["clip_roc_auc_full"] - original["clip_roc_auc_full"],
                "binary_nll_delta": None if nll is None or reference_nll is None else nll-reference_nll,
                "binary_nll_delta_defined": nll is not None and reference_nll is not None})
        result[name] = {"partition": "heldout", "folds": deltas,
            "mean_group_auc_delta": float(np.mean([d["group_auc_delta"] for d in deltas])),
            "mean_clip_auc_delta": float(np.mean([d["clip_auc_delta"] for d in deltas]))}
    result["primary"] = {"rule": PRIMARY_RULE,
        "coherent_improvement_on_these_folds": all(d["group_auc_delta"] > 0
                                                  for d in result["TimeNet256"]["folds"]),
        "statistical_significance_or_independent_confirmation": False}
    return result


def run(output):
    registration, rows, matrix, folds, references = context(output)
    prereg_sha = sha256_file(output / "preregistration.json")
    directory = output / "run"
    previous = campaign.finished(directory, preregistration_sha256=prereg_sha)
    if previous is not None:
        return previous  # Vérifié, aucun nouveau fit et aucune reprise partielle.
    directory.mkdir(exist_ok=False)
    write_json(directory / "started.json", {"preregistration_sha256": prereg_sha, "counts": COUNTS})
    reports = fit_three(rows, matrix, folds, directory, registration["runtime"]["get_params"])
    context(output)  # Entrées, sources, paramètres et runtime encore intacts après les trois fits.
    if sha256_file(output / "preregistration.json") != prereg_sha:
        raise ValueError("Préinscription modifiée pendant les fits")
    mean_auc = {partition: {
        level: float(np.mean([r["partitions"][partition][level] for r in reports]))
        for level in ("group_roc_auc_full", "clip_roc_auc_full")}
        for partition in ("fit", "heldout")}
    summary = {"schema": SCHEMA, "counts": COUNTS, "folds": reports,
               "mean_auc_by_partition": mean_auc, "references_heldout": references,
               "comparisons_heldout": comparisons(reports, references), "limits": LIMITS,
               "selection_policy": None, "pooled_auc": False, "diagnostic_threshold": .5,
               "official_validation_test_external_audio_or_cache_read": False, "qwen_loaded": False}
    write_json(directory / "summary.json", summary)
    return campaign.finish(directory, {"schema": SCHEMA, "preregistration_sha256": prereg_sha,
        "counts": COUNTS, "inputs_sources_runtime_unchanged": True, "completed": True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    pre = commands.add_parser("preregister")
    for name in PATH_NAMES:
        pre.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    pre.add_argument("--output", type=Path, required=True)
    pre.add_argument("--code-revision", required=True)
    execute = commands.add_parser("run")
    execute.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = preregister(args) if args.command == "preregister" else run(args.output)
    print(campaign.digest(result), flush=True)


if __name__ == "__main__":
    main()
