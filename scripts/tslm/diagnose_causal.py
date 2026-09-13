"""D0 préinscrit : observations initiales/terminales, sans fit ni accès val/test.

preregister --campaign CAMPAGNE --output DOSSIER_NEUF --code-revision SHA40
observe --output DOSSIER_D0 --variant A|C

Les reçus CV restent immuables. Une observation interrompue est conservée et ne
peut pas être relancée dans le même dossier. Les seuils diagnostiques sont fixes.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import re
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts/eval"), str(ROOT / "scripts/timenet")]
import run_v2_campaign as campaign

SCHEMA = "pipe-causal-d0-v1"
CAMPAIGN_SHA256 = "5c5e0b4bbed6b973853e53b658350815b2d963547b68d0a08a9ade50c28b1359"
SEED = 20260912
ATOL = 1e-6
PROTOCOL = {
    "seed": SEED, "checkpoint_fold": 0, "initial_clips": 32,
    "terminal_clips": 598, "fit_clips": 500, "heldout_clips": 98,
    "initialization": "same_seed_as_campaign_no_optimizer_created",
    "subset": "debug_rows_32_fit_only_16_per_class_one_per_group",
    "donors": ["within_class_cycle", "cross_class_pairs"],
    "interventions": {"A": ["series"], "C": ["series", "amplitude", "both"]},
    "observations": {"A": 694, "C": 822}, "total_observations": 1516,
    "llm_forwards_per_observation": 2, "total_llm_forwards": 3032,
    "score_atol": ATOL, "diagnostic_threshold": 0.5,
    "controls": ["constant_half", "clip_frequency_from_500_fit_only"],
    "intervention_targets": "receiver_original_class_and_description",
    "pooled_598_performance": False, "intervention_quality_metrics": False,
    "optimizer_steps": 0, "official_validation_test_external_read": False,
}
SOURCE_NAMES = (*campaign.SOURCE_NAMES, "scripts/tslm/diagnose_causal.py",
                "scripts/tslm/causal_observations.py", "scripts/tslm/diagnose_batch_parity.py")


def source_hashes():
    return {name: campaign.sha256_file(ROOT / name) for name in SOURCE_NAMES}


def checked_campaign(directory):
    directory = Path(directory).resolve()
    if campaign.sha256_file(directory / "preregistration.json") != CAMPAIGN_SHA256:
        raise ValueError("D0 est réservé à la campagne c13fd47 préinscrite")
    ctx = campaign.context(directory)
    receipts, identity = {}, {}
    for variant in ("A", "C"):
        receipts[variant] = {}
        for fold in ctx["registration"]["folds"]:
            fid = fold["fold_id"]
            path = directory / f"cv/{variant}/fold-{fid}"
            result = campaign.finished(path, preregistration_sha256=CAMPAIGN_SHA256)
            if (result is None or result.get("variant") != variant or result.get("fold_id") != fid
                    or result.get("source_fold") != "train"
                    or result.get("training_ids") != fold["train_ids"]
                    or result.get("heldout_ids") != fold["heldout_ids"]
                    or result.get("validation_or_test_used") is not False
                    or result.get("epochs") != 4):
                raise ValueError("Six reçus CV complets, de mêmes folds et campagne, requis")
            receipts[variant][fid] = result
            identity[f"{variant}/fold-{fid}"] = {
                "complete_sha256": campaign.sha256_file(path / "complete.json"),
                "receipt_sha256": result["receipt_sha256"],
                "temporal_sha256": result["temporal_sha256"],
                "state_sha256_before": result["state_sha256_before"],
                "state_sha256_after": result["state_sha256_after"],
                "scoring_spec": result["scoring_spec"],
            }
    if set(identity) != {f"{v}/fold-{f}" for v in ("A", "C") for f in range(3)}:
        raise ValueError("Inventaire des six fits différent de la préinscription")
    return ctx, receipts, identity


def cohorts(rows, fold):
    """Sélection par IDs/groupes, jamais par scores ; 500 vus / 98 réservés."""
    from pipe.tslm.train import debug_rows
    if (len(rows) != 598 or len({r["clip_id"] for r in rows}) != 598
            or any(r["fold"] != "train" or r["label"] not in ("leak", "no_leak") for r in rows)
            or fold.get("fold_id") != 0):
        raise ValueError("Cohorte D0 : seuls les 598 train et le fold interne 0 sont autorisés")
    by_id = {r["clip_id"]: r for r in rows}
    fit_ids, heldout_ids = fold["train_ids"], fold["heldout_ids"]
    if (len(fit_ids) != 500 or len(set(fit_ids)) != 500
            or len(heldout_ids) != 98 or len(set(heldout_ids)) != 98
            or set(fit_ids) & set(heldout_ids) or set(fit_ids) | set(heldout_ids) != set(by_id)):
        raise ValueError("Partition 500/98 incomplète, répétée ou chevauchante")
    fit, heldout = [by_id[cid] for cid in fit_ids], [by_id[cid] for cid in heldout_ids]
    if {r["group_id"] for r in fit} & {r["group_id"] for r in heldout}:
        raise ValueError("Groupes partagés entre fit et réservé")
    subset = debug_rows(fit, 32)
    if (len({r["group_id"] for r in subset}) != 32 or len({r["clip_id"] for r in subset}) != 32
            or any(r["clip_id"] not in set(fit_ids) for r in subset)
            or sum(r["label"] == "leak" for r in subset) != 16):
        raise ValueError("Les 32 témoins doivent être équilibrés et issus de groupes fit distincts")
    prior = sum(r["label"] == "leak" for r in fit) / len(fit)
    if not 0 < prior < 1:
        raise ValueError("Les deux classes sont requises dans les 500 fit")
    return {"fit_ids": fit_ids, "heldout_ids": heldout_ids,
            "subset_ids": [r["clip_id"] for r in subset], "fit_clip_frequency": prior,
            "donor_maps": donor_maps(subset)}


def donor_maps(rows):
    sides = {label: sorted(r["clip_id"] for r in rows if r["label"] == label)
             for label in ("leak", "no_leak")}
    if (len(rows) != 32 or len({r["clip_id"] for r in rows}) != 32
            or any(len(ids) != 16 for ids in sides.values())):
        raise ValueError("Deux ensembles de 16 receveurs uniques requis")
    within = {cid: ids[(i + 1) % len(ids)] for ids in sides.values() for i, cid in enumerate(ids)}
    cross = {}
    for a, b in zip(sides["leak"], sides["no_leak"]):
        cross[a], cross[b] = b, a
    for mapping in (within, cross):
        if set(mapping) != set(mapping.values()) or any(cid == donor for cid, donor in mapping.items()):
            raise ValueError("Permutation de donneurs non bijective ou avec point fixe")
    return {"within_class_cycle": within, "cross_class_pairs": cross}


def input_hashes(ctx, rows):
    """Empreintes explicites : aucun cache/WAV val, test ou externe."""
    paths = ctx["registration"]["paths"]
    files = [Path(paths["prepared"]) / name for name in ("train.npz", "preparation.json")]
    files += [Path(paths["previous_prepared"]) / "train.npz"]
    files += [Path(paths[key]) for key in ("folds", "gate_a", "gate_c", "environment_lock")]
    files += [Path(paths["manifests"]) / name for name in ("split_v2.csv", "split_v2_audit.csv")]
    root = Path(paths["data_root"]).resolve()
    for row in rows:
        if row["fold"] != "train":
            raise ValueError("Empreinte audio hors train interdite")
        path = (root / ctx["split"].path_of(row["clip_id"])).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Chemin WAV hors data-root")
        files.append(path)
    return {str(path.resolve()): campaign.sha256_file(path) for path in files}


def check_files(expected):
    if any(campaign.sha256_file(Path(path)) != sha for path, sha in expected.items()):
        raise ValueError("Fichiers d'entrée D0 modifiés depuis leur vérification")


def preregister(args):
    out, parent = args.output.resolve(), args.campaign.resolve()
    if out.exists() or out.is_symlink():
        raise FileExistsError("Préinscription D0 dans un dossier neuf uniquement")
    if out.is_relative_to(parent):
        raise ValueError("Ne jamais écrire le diagnostic dans la campagne scellée")
    if not re.fullmatch(r"[0-9a-f]{40}", args.code_revision or ""):
        raise ValueError("Révision réelle du code SHA40 requise")
    ctx, _, receipt_identity = checked_campaign(parent)
    rows, _, _, _ = campaign.load_fold(ctx["registration"]["paths"], ctx["split"], "train")
    scope = cohorts(rows, ctx["registration"]["folds"][0])
    registration = {"schema": SCHEMA, "created_at": datetime.now(timezone.utc).isoformat(),
        "code_revision": args.code_revision, "campaign": str(parent),
        "campaign_preregistration_sha256": CAMPAIGN_SHA256,
        "campaign_identity": ctx["registration"]["identity"], "cv_receipts": receipt_identity,
        "protocol": PROTOCOL, "cohorts": scope, "source_sha256": source_hashes(),
        "input_sha256": input_hashes(ctx, rows), "model_loaded": False}
    out.mkdir(parents=True, exist_ok=False)
    campaign.write_json(out / "preregistration.json", registration)
    return {"preregistered": True, "preregistration_sha256": campaign.sha256_file(out / "preregistration.json")}


def context(output):
    output = Path(output).resolve()
    registration = campaign.read_json(output / "preregistration.json")
    if (registration.get("schema") != SCHEMA or registration.get("protocol") != PROTOCOL
            or registration.get("campaign_preregistration_sha256") != CAMPAIGN_SHA256
            or registration.get("model_loaded") is not False
            or registration.get("source_sha256") != source_hashes()
            or not re.fullmatch(r"[0-9a-f]{40}", registration.get("code_revision", ""))
            or output.is_relative_to(Path(registration["campaign"]).resolve())):
        raise ValueError("Préinscription D0 absente, modifiée ou incompatible")
    ctx, receipts, identity = checked_campaign(registration["campaign"])
    if (identity != registration["cv_receipts"]
            or ctx["registration"]["identity"] != registration["campaign_identity"]):
        raise ValueError("Identité de la campagne ou des six checkpoints modifiée")
    rows = campaign.diagnostic.train_rows(ctx["split"])
    if input_hashes(ctx, rows) != registration["input_sha256"]:
        raise ValueError("Fichiers d'entrée ou inventaire D0 différents de la préinscription")
    if cohorts(rows, ctx["registration"]["folds"][0]) != registration["cohorts"]:
        raise ValueError("Cohortes ou donneurs différents de la préinscription")
    return {"output": output, "registration": registration, "campaign": ctx, "receipts": receipts,
            "preregistration_sha256": campaign.sha256_file(output / "preregistration.json")}


def state_hashes(model):
    from pipe.tslm.train import tensor_state_hash
    return {name: tensor_state_hash(getattr(model, name)) for name in ("encoder", "projector", "llm")}


def initialize(base, variant):
    """Même initialisation que la campagne, sans construire d'optimiseur."""
    import torch
    from pipe.tslm.model import AcousticQwenSP
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    model = AcousticQwenSP(base, device="cuda", single_clip_acoustic_encoding=True,
                           amplitude_evidence=variant == "C")
    if any(p.requires_grad for p in model.llm.parameters()):
        raise ValueError("Qwen doit rester gelé pour D0")
    model.eval()
    return model


def load_terminal(model, path):
    import torch
    value = torch.load(path, map_location="cuda", weights_only=True)
    model.encoder.load_state_dict(value["encoder_state"], strict=True)
    model.projector.load_state_dict(value["projector_state"], strict=True)
    model.eval()


def observe_example(model, row, series, amplitudes, metadata, *, donor=None, intervention=None):
    from causal_observations import observe_clip
    from opentslm.time_series_datasets.util import extend_time_series_to_match_patch_size_and_aggregate as collate
    example = campaign.examples([row], series, amplitudes, metadata, training=True)[0]
    if donor is not None:
        if intervention not in PROTOCOL["interventions"]["C" if metadata["amplitude_evidence"] else "A"]:
            raise ValueError("Intervention non préinscrite")
        cid, did = row["clip_id"], donor["clip_id"]
        s = {cid: series[did] if intervention in ("series", "both") else series[cid]}
        a = {cid: amplitudes[did] if intervention in ("amplitude", "both") else amplitudes[cid]}
        changed = campaign.examples([row], s, a, metadata, training=False)[0]
        example = {**changed, "answer": example["answer"]}
    batch = collate([example], normalize=False)
    result = observe_clip(model, batch)
    p = result["scoring"]["probability_leak"]
    if (not math.isfinite(p) or not 0 <= p <= 1 or result.get("optimizer_steps") != 0
            or result.get("llm_forward_counts") != {"supervision": 1, "scoring": 1}
            or result.get("state_guard", {}).get("unchanged") is not True
            or result.get("alignment", {}).get("prompts_exact") is not True):
        raise ValueError("Observation D0 non finie, état modifié ou alignement causal invalide")
    return result


def read_predictions(path, expected_ids):
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["clip_id", "probability_leak"]:
            raise ValueError("Prédictions CV hors contrat")
        values = {}
        for row in reader:
            p = float(row["probability_leak"])
            if row["clip_id"] in values or not math.isfinite(p) or not 0 <= p <= 1:
                raise ValueError("Prédiction CV répétée ou invalide")
            values[row["clip_id"]] = p
    if set(values) != set(expected_ids):
        raise ValueError("Couverture CV différente des 98 réservés")
    return values


def check_scores(actual, expected, caption):
    if set(actual) != set(expected):
        raise ValueError(f"Couverture différente : {caption}")
    differences = {cid: abs(actual[cid] - expected[cid]) for cid in actual}
    report = {"n_clips": len(actual), "atol": ATOL,
              "max_abs_difference": max(differences.values(), default=0.0),
              "mismatch_ids": [cid for cid, delta in differences.items() if not math.isfinite(delta) or delta > ATOL]}
    report["passed"] = not report["mismatch_ids"]
    return report


def partition_summary(rows, observations, prior):
    """Moteur existant, jamais de performance amalgamée sur les 598 clips."""
    ids = [r["clip_id"] for r in rows]
    values = [observations[cid] for cid in ids]
    probabilities = {cid: observations[cid]["scoring"]["probability_leak"] for cid in ids}
    terms = {}
    for key in ("class", "first_discriminating_token", "description", "eos", "response"):
        count = sum(value["loss"]["terms"][key]["n_tokens"] for value in values)
        total = sum(value["loss"]["terms"][key]["nll_sum"] for value in values)
        terms[key] = {"n_tokens": count, "nll_sum": total, "nll_per_token": total / count if count else None,
                      "nll_per_clip": total / len(rows)}
    binary_sum = sum(value["scoring"]["binary_nll"] for value in values)
    controls = {}
    for name, p in (("constant_half", 0.5), ("clip_frequency_from_500_fit_only", prior)):
        scores = dict.fromkeys(ids, p)
        nll = sum(-math.log(p if row["label"] == "leak" else 1 - p) for row in rows)
        controls[name] = {"probability_leak": p, "fit_fold": "inner_fit_500" if name != "constant_half" else None,
                          "binary_nll_sum": nll, "binary_nll_mean": nll / len(rows),
                          "metrics": campaign.metric_report(rows, scores, threshold=0.5)}
    return {"n_clips": len(rows), "binary_nll_sum": binary_sum, "binary_nll_mean": binary_sum / len(rows),
            "loss_terms": terms, "metrics": campaign.metric_report(rows, probabilities, threshold=0.5),
            "controls": controls, "threshold_fitted": False}


def observe(ctx, variant):
    if variant not in ("A", "C"):
        raise ValueError("Variante A ou C requise")
    out = ctx["output"] / variant
    previous = campaign.finished(out, preregistration_sha256=ctx["preregistration_sha256"])
    if previous is not None:
        return previous
    parent, registration = ctx["campaign"], ctx["registration"]
    scope, receipt = registration["cohorts"], ctx["receipts"][variant][0]
    paths = parent["registration"]["paths"]
    rows, series, amplitudes, _ = campaign.load_fold(paths, parent["split"], "train")
    if cohorts(rows, parent["registration"]["folds"][0]) != scope:
        raise ValueError("Population D0 différente de la préinscription")
    by_id = {r["clip_id"]: r for r in rows}
    subset = [by_id[cid] for cid in scope["subset_ids"]]
    checkpoint = Path(registration["campaign"]) / f"cv/{variant}/fold-0"
    saved = read_predictions(checkpoint / "predictions.csv", scope["heldout_ids"])
    campaign.verify_runtime(parent["gates"][variant]["runtime"])
    out.mkdir(parents=True, exist_ok=False)
    campaign.write_json(out / "started.json", {"schema": SCHEMA, "variant": variant,
        "preregistration_sha256": ctx["preregistration_sha256"], "expected_observations": PROTOCOL["observations"][variant],
        "optimizer_steps": 0, "created_at": datetime.now(timezone.utc).isoformat()})
    started, count = time.monotonic(), 0
    model = initialize(paths["base"], variant)
    metadata = campaign.variant_metadata(variant)
    before = state_hashes(model)
    if before != receipt["state_sha256_before"] or model.scoring_spec() != receipt["scoring_spec"]:
        raise ValueError("État initial ou spécification différents du fit original")

    def record(stream, row, stage, **kwargs):
        nonlocal count
        value = observe_example(model, row, series, amplitudes, metadata, **kwargs)
        item = {"clip_id": row["clip_id"], "group_id": row["group_id"], "target_class": row["label"],
                "stage": stage, "source_fold": "train", "variant": variant,
                "partition": "fit" if row["clip_id"] in scope["fit_ids"] else "heldout",
                "donor_id": kwargs.get("donor", {}).get("clip_id"), "intervention": kwargs.get("intervention"),
                "observation": value}
        stream.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        count += 1
        if count % 20 == 0:
            print(json.dumps({"variant": variant, "observations": count,
                              "expected": PROTOCOL["observations"][variant], "stage": stage}), flush=True)
        return value

    initial = {}
    with (out / "initial.jsonl").open("x") as stream:
        for row in subset:
            initial[row["clip_id"]] = record(stream, row, "initial")
    after_initial = state_hashes(model)
    if after_initial != before:
        raise ValueError("Poids modifiés pendant l'observation initiale")
    load_terminal(model, checkpoint / "temporal.pt")
    terminal_before = state_hashes(model)
    if terminal_before != receipt["state_sha256_after"] or model.scoring_spec() != receipt["scoring_spec"]:
        raise ValueError("Poids terminaux ou spécification différents du reçu CV")
    terminal = {}
    with (out / "terminal.jsonl").open("x") as stream:
        for row in rows:
            terminal[row["clip_id"]] = record(stream, row, "terminal")
    reload = check_scores({cid: terminal[cid]["scoring"]["probability_leak"] for cid in saved}, saved, "reload heldout")
    campaign.write_json(out / "reload.json", reload)
    if not reload["passed"]:
        raise ValueError("Scores réservés non reproduits : arrêter l'interprétation D0")

    interventions = {}
    for name, mapping in scope["donor_maps"].items():
        for intervention in PROTOCOL["interventions"][variant]:
            key, changed = f"{name}-{intervention}", {}
            with (out / f"{key}.jsonl").open("x") as stream:
                for row in subset:
                    changed[row["clip_id"]] = record(stream, row, name,
                        donor=by_id[mapping[row["clip_id"]]], intervention=intervention)
            scores = {cid: obs["scoring"]["probability_leak"] for cid, obs in changed.items()}
            deltas = [scores[cid] - terminal[cid]["scoring"]["probability_leak"] for cid in scores]
            item = {"n_clips": len(scores), "mean_signed_score_change": float(np.mean(deltas)),
                    "mean_abs_score_change": float(np.mean(np.abs(deltas))),
                    "max_abs_score_change": float(np.max(np.abs(deltas))), "quality_metrics_calculated": False}
            if variant == "A" or intervention == "both":
                item["donor_reproduction"] = check_scores(scores,
                    {cid: terminal[mapping[cid]]["scoring"]["probability_leak"] for cid in scores}, key)
                if not item["donor_reproduction"]["passed"]:
                    campaign.write_json(out / "failed-control.json", {"intervention": key, **item})
                    raise ValueError("Entrée complète donneuse non reproduite : arrêter l'interprétation D0")
            interventions[key] = item
    after_terminal = state_hashes(model)
    if after_terminal != terminal_before or count != PROTOCOL["observations"][variant]:
        raise ValueError("État modifié ou budget d'observations non respecté")
    check_files(registration["input_sha256"])
    again = context(ctx["output"])
    if again["preregistration_sha256"] != ctx["preregistration_sha256"]:
        raise ValueError("Préinscription modifiée pendant D0")
    summary = {"schema": SCHEMA, "variant": variant, "preregistration_sha256": ctx["preregistration_sha256"],
        "observations": count, "llm_forwards": count * 2, "optimizer_steps": 0,
        "state_sha256": {"initial_before": before, "initial_after": after_initial,
                          "terminal_before": terminal_before, "terminal_after": after_terminal},
        "weights_unchanged_within_each_state": True, "reload": reload,
        "initial_subset_32_fit": partition_summary(subset, initial, scope["fit_clip_frequency"]),
        "terminal_subset_32_fit": partition_summary(subset, terminal, scope["fit_clip_frequency"]),
        "terminal_partitions": {name: partition_summary([by_id[cid] for cid in scope[f"{name}_ids"]], terminal,
            scope["fit_clip_frequency"]) for name in ("fit", "heldout")},
        "interventions": interventions, "pooled_598_performance_calculated": False,
        "validation_test_external_read": False, "elapsed_seconds": time.monotonic() - started,
        "limitations": ["Les 98 réservés sont du développement déjà consulté, pas un nouveau holdout.",
            "Les 32 témoins sont équilibrés : tous les 16 groupes fit non-fuite, 16/52 groupes fuite.",
            "Les entrées séries/texte désaccordées peuvent être hors distribution ; sensibilité n'est pas utilité.",
            "Les gradients/mises à jour ne sont pas mesurés ici ; aucune cause d'optimisation n'est prouvée."]}
    campaign.write_json(out / "summary.json", summary)
    return campaign.finish(out, summary)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    prereg = sub.add_parser("preregister")
    prereg.add_argument("--campaign", type=Path, required=True)
    prereg.add_argument("--output", type=Path, required=True)
    prereg.add_argument("--code-revision", required=True)
    run = sub.add_parser("observe")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--variant", choices=("A", "C"), required=True)
    args = parser.parse_args(argv)
    result = preregister(args) if args.stage == "preregister" else observe(context(args.output), args.variant)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
