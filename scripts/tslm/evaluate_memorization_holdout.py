"""Extension D3 autorisée par Icham : checkpoint fixe sur les 98 réservés D0.

Préinscrire avant observation ; ne pas modifier les reçus du test de mémorisation.
Aucun entraînement, réglage, génération ou accès audio/cache val/test/externe.
"""
import argparse
import json
from pathlib import Path

import diagnose_memorization as d3

d0, campaign = d3.d0, d3.campaign


def validate_holdout(rows, trained_rows, expected_ids):
    if (len(rows) != 98 or len(set(expected_ids)) != 98
            or [r["clip_id"] for r in rows] != expected_ids
            or any(r["fold"] != "train" for r in rows)
            or {r["label"] for r in rows} != {"leak", "no_leak"}
            or {r["clip_id"] for r in rows} & {r["clip_id"] for r in trained_rows}
            or {r["group_id"] for r in rows} & {r["group_id"] for r in trained_rows}):
        raise ValueError("98 IDs réservés exacts, deux classes et groupes sans recouvrement requis")


def inputs(directory):
    ctx = d3.context(directory)
    digest = ctx["preregistration_sha256"]
    trained = campaign.finished(ctx["output"] / "train", preregistration_sha256=digest)
    reload = campaign.finished(ctx["output"] / "reload", preregistration_sha256=digest)
    if trained is None or reload is None or reload.get("passed") is not True:
        raise ValueError("Checkpoint D3 terminé et reload neuf réussi requis")
    if reload["train_receipt_sha256"] != trained["receipt_sha256"]:
        raise ValueError("Le reload ne correspond pas à ce checkpoint")
    parent = ctx["d0"]["campaign"]
    all_rows = campaign.diagnostic.train_rows(parent["split"])
    ids = ctx["d0"]["registration"]["cohorts"]["heldout_ids"]
    by_id = {r["clip_id"]: r for r in all_rows}
    rows = [by_id[cid] for cid in ids]
    validate_holdout(rows, ctx["rows"], ids)
    pre = {
        "schema": "pipe-d3-heldout-98-v1", "d3_directory": str(ctx["output"]),
        "d3_preregistration_sha256": digest,
        "train_receipt_sha256": trained["receipt_sha256"],
        "reload_receipt_sha256": reload["receipt_sha256"],
        "temporal_sha256": trained["temporal_sha256"],
        "source_sha256": campaign.sha256_file(Path(__file__)),
        "rows": rows, "scoring_spec": trained["scoring_spec"],
        "optimizer_steps": 0, "n_checkpoints_evaluated": 1,
        "observations": 98, "llm_forwards": 196,
        "threshold": 0.5, "threshold_fitted": False,
        "primary_metric": "heldout_group_roc_auc",
        "official_validation_test_external_observed": False,
        "interpretation": "exploratory_development_not_independent_confirmation",
        "comparison": "A0_same_98_but_fit_on_500_not_32_no_causal_attribution",
    }
    return ctx, trained, rows, all_rows, pre


def preregister(directory, output):
    ctx, _, _, _, pre = inputs(directory)
    protected = [ctx["output"], ctx["d0"]["output"],
                 Path(ctx["d0"]["registration"]["campaign"])]
    if output.exists() or output.is_symlink() or any(output.is_relative_to(p) for p in protected):
        raise ValueError("Dossier neuf, séparé de D3/D0 et de la campagne requis")
    output.mkdir(parents=True, exist_ok=False)
    campaign.write_json(output / "preregistration.json", pre)
    return {"preregistration_sha256": campaign.sha256_file(output / "preregistration.json"),
            "model_loaded": False, "optimizer_steps": 0}


def run(output):
    pre_path = output / "preregistration.json"
    pre = campaign.read_json(pre_path)
    ctx, trained, rows, all_rows, expected = inputs(pre["d3_directory"])
    if pre != expected:
        raise ValueError("Préinscription, sources, population ou checkpoint modifiés")
    digest = campaign.sha256_file(pre_path)
    out = output / "evaluation"
    previous = campaign.finished(out, preregistration_sha256=digest)
    if previous is not None:
        return previous
    parent = ctx["d0"]["campaign"]
    paths = parent["registration"]["paths"]
    from pipe.tslm.preprocessing import CANONICAL_VERSION
    cache = d3.load_cache(paths["prepared"], "train", all_rows, CANONICAL_VERSION)
    series = {r["clip_id"]: cache[r["clip_id"]] for r in rows}
    out.mkdir(exist_ok=False)
    campaign.write_json(out / "started.json", {"preregistration_sha256": digest, "optimizer_steps": 0})
    model = d0.initialize(paths["base"], "A")
    d0.load_terminal(model, ctx["output"] / "train/temporal.pt")
    d3.validate_model(model, pre["scoring_spec"])
    before = d0.state_hashes(model)
    if before != trained["state_sha256_after"]:
        raise ValueError("Poids différents du checkpoint D3 retenu")
    observations = {}
    with (out / "observations.jsonl").open("x") as stream:
        for i, row in enumerate(rows, 1):
            value = d0.observe_example(model, row, series, {}, campaign.variant_metadata("A"))
            observations[row["clip_id"]] = value
            d3.append_json(stream, {**row, "partition": "heldout_98", "observation": value})
            if i % 20 == 0 or i == 98:
                print(json.dumps({"observations": i, "expected": 98}), flush=True)
    after = d0.state_hashes(model)
    if before != after or campaign.sha256_file(pre_path) != digest:
        raise ValueError("Poids ou préinscription modifiés pendant l'observation")
    summary = d0.partition_summary(rows, observations, 0.5)
    summary["controls"].pop("clip_frequency_from_500_fit_only")
    ids = [r["clip_id"] for r in rows]
    scores = {cid: observations[cid]["scoring"]["probability_leak"] for cid in ids}
    campaign.write_predictions(out / "predictions.csv", ids, scores)
    a0 = d0.read_predictions(Path(ctx["d0"]["registration"]["campaign"]) /
                             "cv/A/fold-0/predictions.csv", ids)
    result = {"schema": pre["schema"], "preregistration_sha256": digest,
              "train_receipt_sha256": trained["receipt_sha256"], "summary": summary,
              "a0_reference": campaign.metric_report(rows, a0, threshold=0.5),
              "comparison_caveat": pre["comparison"],
              "n_correct": sum((scores[r["clip_id"]] >= .5) == (r["label"] == "leak") for r in rows),
              "state_sha256_before": before, "state_sha256_after": after,
              "optimizer_steps": 0, "observations": 98, "llm_forwards": 196,
              "official_validation_test_external_observed": False}
    campaign.write_json(out / "summary.json", result)
    return campaign.finish(out, result)


def self_test():
    rows = [{"clip_id": str(i), "group_id": str(i), "fold": "train",
             "label": "leak" if i % 2 else "no_leak"} for i in range(98)]
    ids = [r["clip_id"] for r in rows]
    validate_holdout(rows, [{"clip_id": "fit", "group_id": "fit"}], ids)
    for invalid in (rows[:-1], list(reversed(rows)),
                    [{**rows[0], "fold": "test"}, *rows[1:]]):
        try:
            validate_holdout(invalid, [], ids)
        except ValueError:
            continue
        raise AssertionError("Population invalide acceptée")
    for overlap in ({"clip_id": "fit", "group_id": "0"},
                    {"clip_id": "0", "group_id": "fit"}):
        try:
            validate_holdout(rows, [overlap], ids)
        except ValueError:
            continue
        raise AssertionError("Recouvrement accepté")
    print("self-test PASS: population, ordre, split, IDs et groupes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("preregister", "run", "self-test"))
    parser.add_argument("--d3", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.stage == "self-test":
        self_test()
    else:
        if args.output is None or (args.stage == "preregister" and args.d3 is None):
            parser.error("--output requis ; --d3 également pour preregister")
        result = (preregister(args.d3, args.output.resolve()) if args.stage == "preregister"
                  else run(args.output.resolve()))
        print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
