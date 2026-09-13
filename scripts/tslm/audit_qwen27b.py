"""Revérifier le rush depuis ses artefacts ; aucun modèle/GPU ni nouvelle recherche."""
import argparse
import csv
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts/tslm")]
import run_v2_campaign as campaign
from pipe.tslm.campaign import epoch_batches, write_json


def audit(directory):
    read = campaign.read_json
    gate = read(directory / "gate.json")
    probe = read(directory / "probe-result.json")
    reloads = [read(directory / f"reload-node{i}.json") for i in (1, 2)]
    assert all(r["passed"] and r["reload_max_absolute_difference"] <= 1e-6 for r in reloads)
    assert reloads[0]["probabilities"] == reloads[1]["probabilities"] == probe["reload_probabilities"]
    assert all(r["runtime"] == gate["runtime"] for r in reloads)
    assert probe["state_before"]["base"] == probe["state_after"]["base"]
    assert probe["memory_gate_passed"] and probe["peak_reserved_bytes"] <= .9 * probe["gpu_total_bytes"]
    split = campaign.split_loader.load_split(ROOT / "manifests")
    rows = campaign.diagnostic.train_rows(split)
    folds = read(ROOT / "docs/evidence/tslm-v2/train-diagnostic-001/folds.json")
    campaign.validate_folds(rows, folds)
    by_id = {r["clip_id"]: r for r in rows}
    results, missing, hashes = [], [], {}
    for fold_id, fold in enumerate(folds["folds"]):
        scope = directory / f"fold-{fold_id}"
        if not (scope / "evaluation/report.json").is_file():
            missing.append(fold_id)
            continue
        pre, fit, evaluation = (read(scope / name) for name in
            ("preregistration.json", "training-result.json", "evaluation/report.json"))
        assert pre["training_ids"] == fold["train_ids"] and pre["heldout_ids"] == fold["heldout_ids"]
        assert pre["gate_sha256"] == campaign.sha256_file(directory / "gate.json")
        assert pre["epochs"] == gate["epochs"] == fit["planned_epochs"]
        assert pre["code_revision"] == gate["code_revision"]
        assert pre["config_sha256"] == gate["config_sha256"]
        assert pre["runtime"] == fit["runtime"] == evaluation["runtime"] == gate["runtime"]
        assert pre["microbatch_size"] == gate["microbatch_size"] == 4
        assert fit["state_before"] == probe["state_before"]  # même initialisation neuve, pas le probe appris
        assert fit["state_after"]["base"] == probe["state_before"]["base"]
        assert all(fit["state_after"][name] != fit["state_before"][name] for name in ("encoder", "projector"))
        assert evaluation["bundle_sha256"] == fit["bundle_sha256"] == campaign.sha256_file(scope / "bundle/checksums.json")
        assert evaluation["passed"] and evaluation["reload_max_absolute_difference"] <= 1e-6
        assert evaluation["optimizer_steps"] == 0
        assert pre["official_validation_test_external_read"] is fit["official_validation_test_external_read"] is evaluation["official_validation_test_external_read"] is False
        schedule = []
        for epoch in range(1, gate["epochs"] + 1):
            schedule.extend((epoch, [fold["train_ids"][i] for i in indices]) for indices in
                            epoch_batches(len(fold["train_ids"]), 8, pre["config"]["seed"], epoch))
        records = [json.loads(line) for line in (scope / "training.jsonl").read_text().splitlines()]
        assert len(records) == fit["optimizer_steps"]
        assert len(records) <= len(schedule)
        if fit["fit_complete"]:
            assert len(records) == len(schedule) and fit["completed_epochs"] == gate["epochs"]
        exposures = 0
        for step, (record, (epoch, expected_ids)) in enumerate(zip(records, schedule), 1):
            assert record["step"] == step and record["training_record"]["epoch"] == epoch
            assert [c["clip_id"] for c in record["clips"]] == expected_ids
            assert set(record["gradient_norms_pre_clip"]) == {"encoder", "projector", "lora"}
            assert all(math.isfinite(v) and v > 0 for v in record["gradient_norms_pre_clip"].values())
            assert sum(v*v for v in record["gradient_norms_post_clip"].values()) <= 1.00001
            assert all(v["update_l2"] > 0 for v in record["parameter_updates"].values())
            assert record["extra_forward_count"] == 0
            for clip in record["clips"]:
                assert clip["target_class_index"] == (0 if by_id[clip["clip_id"]]["label"] == "leak" else 1)
                assert clip["labels_and_attention_verified"] and clip["reconstruction_absolute_difference"] < 1e-5
            exposures += len(expected_ids)
        assert exposures == fit["presentations"]
        heldout = [by_id[cid] for cid in fold["heldout_ids"]]
        with (scope / "evaluation/predictions.csv").open(newline="") as stream:
            csv_rows = csv.DictReader(stream)
            assert csv_rows.fieldnames == ["clip_id", "probability_leak"]
            predictions = list(csv_rows)
        assert [r["clip_id"] for r in predictions] == fold["heldout_ids"]
        probabilities = {r["clip_id"]: float(r["probability_leak"]) for r in predictions}
        metrics = campaign.metric_report(heldout, probabilities)
        assert metrics == evaluation["metrics"]
        observations = [json.loads(line) for line in (scope / "evaluation/observations.jsonl").read_text().splitlines()]
        assert [r["clip_id"] for r in observations] == fold["heldout_ids"]
        nlls = []
        for item in observations:
            left, right = item["class_logprobs"]
            maximum = max(left, right)
            log_total = maximum + math.log(math.exp(left-maximum) + math.exp(right-maximum))
            p = math.exp(left-log_total)
            nll = log_total - (left if by_id[item["clip_id"]]["label"] == "leak" else right)
            assert abs(p - probabilities[item["clip_id"]]) <= 1e-12
            assert abs(nll - item["binary_nll"]) <= 1e-12
            nlls.append(nll)
        assert abs(sum(nlls)/len(nlls) - evaluation["binary_nll"]) <= 1e-12
        results.append({"fold_id": fold_id, "fit_complete": fit["fit_complete"], "metrics": metrics,
                        "binary_nll": evaluation["binary_nll"], "optimizer_steps": len(records),
                        "presentations": exposures, "peak_reserved_bytes": fit["peak_reserved_bytes"]})
        hashes.update({str(p.relative_to(directory)): campaign.sha256_file(p)
                       for p in scope.rglob("*") if p.is_file()})
    references = ROOT / "docs/evidence/tslm-v2/campaign-c13fd47"
    old_a = [read(references / f"A/fold-{i}/complete.json")["metrics"]["group_roc_auc_full"] for i in range(3)]
    c1 = read(references / "C1/comparison.json")
    selected = next(c for c in c1["candidates"] if c["C"] == c1["selected_C"])
    c1_auc = [r["group_roc_auc_full"] for r in selected["folds"]]
    complete = not missing and all(r["fit_complete"] for r in results)
    return {"status": "VERIFIED_COMPLETE" if complete else "VERIFIED_PARTIAL", "folds": results,
            "missing_folds": missing, "n_configs_compared": 1,
            "technical_optimizer_steps": probe["optimizer_steps"], "technical_presentations": probe["presentations"],
            "mean_group_auc": sum(r["metrics"]["group_roc_auc_full"] for r in results)/3 if complete else None,
            "references_group_auc": {"A_4B_4epochs": old_a, "C1_selected_C_0.01": c1_auc},
            "reference_means": {"A": sum(old_a)/3, "C1": sum(c1_auc)/3},
            "prototype_fold": 0, "official_validation_test_external_read": False,
            "artifacts_sha256": hashes,
            "limitations": ["Développement exploratoire, groupes heuristiques, pas une validation terrain.",
                            "Recettes/budgets différents : aucun effet causal isolé de la taille ou de LoRA.",
                            "C1 a été sélectionné parmi4configurations sur ces mêmes folds.",
                            "Pas de refit598 ni de nouveau test officiel/externe."]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.directory)
    write_json(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k != "artifacts_sha256"}, ensure_ascii=False, indent=2))
