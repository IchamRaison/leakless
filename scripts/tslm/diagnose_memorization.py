"""D3 : un essai A de mémorisation des 32 témoins D0, jamais de généralisation.

preregister --d0 DOSSIER_D0 --output DOSSIER_NEUF --code-revision SHA40
train --output DOSSIER_D3
reload --output DOSSIER_D3  # impérativement dans un processus neuf

Aucun refit automatique, aucune observation des 98 réservés, val/test/externe.
Les manifests/caches train et anciens reçus restent lisibles pour leur provenance.
"""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/tslm"))
import diagnose_causal as d0
from diagnose_head_precision import D0_RECEIPTS, runtime_identity
from pipe.tslm.campaign import epoch_batches, load_cache, train_epoch
from training_observations import capture_training_diagnostics, _option

campaign = d0.campaign
SCHEMA = "pipe-causal-d3-memorization-v1"
CONFIG = {**campaign.RECIPE, "epochs": 250}
PROTOCOL = {"variant": "A", "initialization": "fresh_campaign_seed_20260912",
    "n_configs_compared": 1, "n_training_attempts": 1, "n_clips": 32, "n_leak": 16,
    "max_steps": 1000, "steps_per_epoch": 4, "max_epochs": 250,
    "max_training_presentations": 8000, "observation_steps": list(range(0, 1001, 100)),
    "stop_rule": "first_positive_observation_step_with_mean_binary_nll_lt_0.1_and_32_correct",
    "binary_nll_limit_strict": .1, "diagnostic_threshold": .5,
    "max_observations_including_reload": 384, "max_observation_forwards_including_reload": 768,
    "forwards_per_observation": 2, "logger_extra_forwards": 0,
    "head": "original_frozen_bf16", "loss": CONFIG["loss"],
    "heldout_98_or_official_validation_test_external_observed": False,
    "generalization_measured": False, "additional_attempt_on_failure": False}
SOURCE_NAMES = (*d0.SOURCE_NAMES, "scripts/tslm/diagnose_head_precision.py",
                "scripts/tslm/training_observations.py", "scripts/tslm/diagnose_memorization.py")


def source_hashes():
    return {name: campaign.sha256_file(ROOT / name) for name in SOURCE_NAMES}


def checked_d0(directory):
    ctx = d0.context(Path(directory))
    receipt = campaign.finished(ctx["output"] / "A", preregistration_sha256=ctx["preregistration_sha256"])
    if (receipt is None or receipt.get("receipt_sha256") != D0_RECEIPTS["A"]
            or receipt.get("variant") != "A" or receipt.get("optimizer_steps") != 0
            or receipt.get("weights_unchanged_within_each_state") is not True
            or receipt["state_sha256"]["initial_before"] != receipt["state_sha256"]["initial_after"]):
        raise ValueError("Reçu D0 A initial intègre requis")
    campaign.verify_runtime(ctx["campaign"]["gates"]["A"]["runtime"])
    return ctx, receipt


def subset(ctx):
    """Cache train autorisé ; seuls les 32 IDs figés sortent vers le modèle."""
    from pipe.tslm.preprocessing import CANONICAL_VERSION
    parent, scope = ctx["campaign"], ctx["registration"]["cohorts"]
    rows = campaign.diagnostic.train_rows(parent["split"])
    if d0.cohorts(rows, parent["registration"]["folds"][0]) != scope:
        raise ValueError("Cohorte D0 modifiée")
    cached = load_cache(parent["registration"]["paths"]["prepared"], "train", rows, CANONICAL_VERSION)
    by_id = {r["clip_id"]: r for r in rows}
    selected = [by_id[cid] for cid in scope["subset_ids"]]
    validate_rows(selected)
    return selected, {r["clip_id"]: cached[r["clip_id"]] for r in selected}


def validate_rows(rows):
    if (len(rows) != 32 or len({r["clip_id"] for r in rows}) != 32
            or len({r["group_id"] for r in rows}) != 32
            or any(r["fold"] != "train" or r["label"] not in ("leak", "no_leak") for r in rows)
            or sum(r["label"] == "leak" for r in rows) != 16):
        raise ValueError("Exactement 32 train équilibrés de groupes distincts requis")


def optimizer_options(optimizer):
    return [{k: _option(value) for k, value in group.items() if k != "params"}
            for group in optimizer.param_groups]


def default_optimizer_options():
    """Fixer les défauts du vrai AdamW sans modèle Qwen, fit ou tirage aléatoire."""
    import torch
    optimizer = torch.optim.AdamW([
        {"params": [torch.nn.Parameter(torch.zeros(1))], "lr": CONFIG["encoder_lr"]},
        {"params": [torch.nn.Parameter(torch.zeros(1))], "lr": CONFIG["projector_lr"]}],
        weight_decay=CONFIG["weight_decay"])
    return optimizer_options(optimizer)


def preregister(args):
    output, parent = args.output.resolve(), args.d0.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError("D3 doit être préinscrit dans un dossier neuf")
    if not re.fullmatch(r"[0-9a-f]{40}", args.code_revision or ""):
        raise ValueError("Révision réelle SHA40 requise")
    ctx, receipt = checked_d0(parent)
    if output.is_relative_to(parent) or output.is_relative_to(Path(ctx["registration"]["campaign"]).resolve()):
        raise ValueError("Ne jamais écrire D3 dans les reçus D0 ou la campagne CV")
    rows, _ = subset(ctx)
    value = {"schema": SCHEMA, "created_at": datetime.now(timezone.utc).isoformat(),
        "code_revision": args.code_revision, "d0": str(parent), "protocol": PROTOCOL, "config": CONFIG,
        "optimizer_options": default_optimizer_options(), "runtime": runtime_identity(),
        "source_sha256": source_hashes(), "d0_preregistration_sha256": ctx["preregistration_sha256"],
        "d0_a_receipt_sha256": receipt["receipt_sha256"], "rows": rows,
        "input_sha256": ctx["registration"]["input_sha256"],
        "campaign_identity": ctx["registration"]["campaign_identity"],
        "initial_state_sha256": receipt["state_sha256"]["initial_before"],
        "scoring_spec": ctx["receipts"]["A"][0]["scoring_spec"],
        "model_loaded": False, "optimizer_steps": 0}
    output.mkdir(parents=True, exist_ok=False)
    campaign.write_json(output / "preregistration.json", value)
    return {"preregistration_sha256": campaign.sha256_file(output / "preregistration.json"), "optimizer_steps": 0}


def context(output):
    output = Path(output).resolve()
    value = campaign.read_json(output / "preregistration.json")
    if (value.get("schema") != SCHEMA or value.get("protocol") != PROTOCOL or value.get("config") != CONFIG
            or value.get("model_loaded") is not False or value.get("optimizer_steps") != 0
            or value.get("source_sha256") != source_hashes() or value.get("runtime") != runtime_identity()
            or value.get("optimizer_options") != default_optimizer_options()
            or not re.fullmatch(r"[0-9a-f]{40}", value.get("code_revision", ""))):
        raise ValueError("Préinscription D3, runtime ou recette modifiés")
    ctx, receipt = checked_d0(value["d0"])
    rows, series = subset(ctx)
    expected = {"d0_preregistration_sha256": ctx["preregistration_sha256"],
        "d0_a_receipt_sha256": receipt["receipt_sha256"], "rows": rows,
        "input_sha256": ctx["registration"]["input_sha256"],
        "campaign_identity": ctx["registration"]["campaign_identity"],
        "initial_state_sha256": receipt["state_sha256"]["initial_before"],
        "scoring_spec": ctx["receipts"]["A"][0]["scoring_spec"]}
    if (any(value.get(k) != v for k, v in expected.items()) or output.is_relative_to(ctx["output"])
            or output.is_relative_to(Path(ctx["registration"]["campaign"]).resolve())):
        raise ValueError("Cohorte, poids initiaux, base ou provenance D0 différents")
    return {"output": output, "registration": value, "d0": ctx, "rows": rows, "series": series,
            "preregistration_sha256": campaign.sha256_file(output / "preregistration.json")}


def validate_model(model, scoring_spec, optimizer=None, expected_options=None):
    import torch
    temporal = list(model.encoder.parameters()) + list(model.projector.parameters())
    head = model.llm.get_output_embeddings()
    if (model.scoring_spec() != scoring_spec or model.single_clip_acoustic_encoding is not True
            or model.amplitude_evidence is not False or type(head) is not torch.nn.Linear
            or getattr(model.llm, "lm_head", head) is not head or "forward" in vars(head)
            or head.weight.dtype != torch.bfloat16 or any(p.requires_grad for p in model.llm.parameters())
            or any(p.grad is not None for p in model.llm.parameters())
            or not temporal or not all(p.requires_grad for p in temporal)
            or {id(p) for p in model.parameters() if p.requires_grad} != {id(p) for p in temporal}):
        raise ValueError("Chemin A canonique, tête BF16 originale et Qwen gelé requis")
    if optimizer is not None:
        actual = [p for group in optimizer.param_groups for p in group["params"]]
        if (type(optimizer) is not torch.optim.AdamW or optimizer.state
                or any(p.grad is not None for p in temporal)
                or len(actual) != len(temporal) or {id(p) for p in actual} != {id(p) for p in temporal}
                or [list(map(id, g["params"])) for g in optimizer.param_groups] != [
                    list(map(id, model.encoder.parameters())), list(map(id, model.projector.parameters()))]
                or optimizer_options(optimizer) != expected_options):
            raise ValueError("AdamW neuf, options exactes et seuls paramètres acoustiques requis")


def append_json(stream, value):
    stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
    stream.flush()


def observation_point(model, rows, series, metadata, step, directory):
    validate_rows(rows)
    if (getattr(model.compute_loss, "_training_observer", False)
            or step not in PROTOCOL["observation_steps"]):
        raise ValueError("Observation seulement aux points préinscrits, logger TRAIN fermé")
    model.eval()
    observations = {}
    with (directory / f"observations-{step:04d}.jsonl").open("x") as stream:
        for row in rows:
            value = d0.observe_example(model, row, series, {}, metadata)
            if not math.isfinite(value["scoring"]["binary_nll"]) or value["scoring"]["binary_nll"] < 0:
                raise ValueError("NLL binaire officielle non finie ou négative")
            observations[row["clip_id"]] = value
            append_json(stream, {"step": step, "clip_id": row["clip_id"], "group_id": row["group_id"],
                                 "target_class": row["label"], "partition": "memorization_32", "observation": value})
    summary = d0.partition_summary(rows, observations, .5)
    # Le contrôle de fréquence D0 portait sur 500 clips ; ne pas lui attribuer une autre population.
    summary["controls"].pop("clip_frequency_from_500_fit_only")
    correct = sum((observations[r["clip_id"]]["scoring"]["probability_leak"] >= .5)
                  == (r["label"] == "leak") for r in rows)
    summary.update(step=step, n_correct=correct, criterion_met=correct == 32 and summary["binary_nll_mean"] < .1,
                   observations=32, llm_forwards=64, generalization_measured=False)
    scores = {cid: obs["scoring"]["probability_leak"] for cid, obs in observations.items()}
    return summary, scores


def training_loop(model, optimizer, rows, series, metadata, directory):
    validate_rows(rows)
    samples = campaign.examples(rows, series, {}, metadata, training=True)
    points, final_scores = [], None
    steps = 0
    first, final_scores = observation_point(model, rows, series, metadata, 0, directory)
    points.append(first)
    campaign.write_json(directory / "point-0000.json", first)
    print(json.dumps({"stage": "D3_observation", "step": 0, "n_correct": first["n_correct"],
                      "binary_nll_mean": first["binary_nll_mean"], "criterion_met": first["criterion_met"]}), flush=True)
    with (directory / "steps.jsonl").open("x") as step_log, (directory / "epochs.jsonl").open("x") as epoch_log:
        for epoch in range(1, CONFIG["epochs"] + 1):
            batches = epoch_batches(32, CONFIG["batch_size"], CONFIG["seed"], epoch)
            if len(batches) != 4 or any(len(batch) != 8 for batch in batches):
                raise ValueError("Quatre lots de huit clips attendus par époque")
            with capture_training_diagnostics(model, optimizer) as audit:
                n_steps = 0
                for record in train_epoch(model, optimizer, samples, CONFIG, epoch):
                    if n_steps >= 4:
                        raise ValueError("Budget d'époque dépassé")
                    ids = [rows[i]["clip_id"] for i in batches[n_steps]]
                    step = audit.pop_step(ids, training_record=record)
                    steps += 1
                    n_steps += 1
                    append_json(step_log, {"step": steps, "epoch": epoch, **step})
                if n_steps != 4:
                    raise ValueError("Époque incomplète : aucune prolongation de remplacement")
                aggregate = audit.epoch_summary(reset=True)
                if aggregate["n_steps"] != 4 or aggregate["n_clips"] != 32 or aggregate["extra_forward_count"] != 0:
                    raise ValueError("Budget ou agrégation du logger incompatible")
                append_json(epoch_log, {"epoch": epoch, "step": steps, **aggregate})
            print(json.dumps({"stage": "D3_train", "epoch": epoch, "steps": steps, "max_steps": 1000}), flush=True)
            if steps in PROTOCOL["observation_steps"]:
                point, final_scores = observation_point(model, rows, series, metadata, steps, directory)
                points.append(point)
                campaign.write_json(directory / f"point-{steps:04d}.json", point)
                print(json.dumps({"stage": "D3_observation", "step": steps, "n_correct": point["n_correct"],
                                  "binary_nll_mean": point["binary_nll_mean"], "criterion_met": point["criterion_met"]}), flush=True)
                if point["criterion_met"]:
                    break
    if (steps not in PROTOCOL["observation_steps"][1:] or steps > 1000
            or [p["step"] for p in points] != list(range(0, steps + 1, 100))):
        raise ValueError("Calendrier d'observation ou budget D3 non respecté")
    return {"steps": steps, "epochs": steps // 4, "training_presentations": steps * 8,
            "observation_points": points, "final_scores": final_scores,
            "observations": len(points) * 32, "observation_forwards": len(points) * 64,
            "stop_reason": "memorization_criterion" if points[-1]["criterion_met"] else "budget_exhausted"}


def train(ctx):
    import torch
    out, registration = ctx["output"] / "train", ctx["registration"]
    previous = campaign.finished(out, preregistration_sha256=ctx["preregistration_sha256"])
    if previous is not None:
        return previous
    out.mkdir(exist_ok=False)
    campaign.write_json(out / "started.json", {"preregistration_sha256": ctx["preregistration_sha256"],
        "pid": os.getpid(), "hostname": platform.node(), "n_training_attempts": 1})
    base = ctx["d0"]["campaign"]["registration"]["paths"]["base"]
    model, optimizer = campaign.initialize_training(base, "A")
    validate_model(model, registration["scoring_spec"], optimizer, registration["optimizer_options"])
    before = d0.state_hashes(model)
    if before != registration["initial_state_sha256"]:
        raise ValueError("Poids initiaux différents de D0 A : aucun entraînement permis")
    campaign.write_json(out / "config.json", CONFIG)
    campaign.write_json(out / "scoring_spec.json", model.scoring_spec())
    result = training_loop(model, optimizer, ctx["rows"], ctx["series"], campaign.variant_metadata("A"), out)
    validate_model(model, registration["scoring_spec"])
    after = d0.state_hashes(model)
    if before["llm"] != after["llm"]:
        raise ValueError("Qwen modifié pendant le diagnostic")
    temporal = out / "temporal.pt"
    with temporal.open("xb") as stream:
        torch.save({"encoder_state": model.encoder.state_dict(), "projector_state": model.projector.state_dict()}, stream)
    again = context(ctx["output"])
    if again["preregistration_sha256"] != ctx["preregistration_sha256"]:
        raise ValueError("Préinscription modifiée pendant D3")
    result.update(schema=SCHEMA, preregistration_sha256=ctx["preregistration_sha256"],
        state_sha256_before=before, state_sha256_after=after, temporal_sha256=campaign.sha256_file(temporal),
        scoring_spec=model.scoring_spec(), pid=os.getpid(), hostname=platform.node(),
        subset_ids=[r["clip_id"] for r in ctx["rows"]], reload_required=True, generalization_measured=False,
        heldout_98_or_official_validation_test_external_observed=False)
    campaign.write_json(out / "summary.json", result)
    return campaign.finish(out, result)


def reload(ctx):
    directory, train_dir = ctx["output"] / "reload", ctx["output"] / "train"
    previous = campaign.finished(directory, preregistration_sha256=ctx["preregistration_sha256"])
    if previous is not None:
        return previous
    trained = campaign.finished(train_dir, preregistration_sha256=ctx["preregistration_sha256"])
    if trained is None or (trained["pid"] == os.getpid() and trained["hostname"] == platform.node()):
        raise ValueError("Entraînement terminé et processus neuf requis pour reload")
    directory.mkdir(exist_ok=False)
    campaign.write_json(directory / "started.json", {"pid": os.getpid(), "hostname": platform.node(),
        "preregistration_sha256": ctx["preregistration_sha256"], "optimizer_steps": 0})
    base = ctx["d0"]["campaign"]["registration"]["paths"]["base"]
    model = d0.initialize(base, "A")
    d0.load_terminal(model, train_dir / "temporal.pt")
    validate_model(model, ctx["registration"]["scoring_spec"])
    before = d0.state_hashes(model)
    if before != trained["state_sha256_after"]:
        raise ValueError("Poids rechargés différents du checkpoint D3")
    point, scores = observation_point(model, ctx["rows"], ctx["series"], campaign.variant_metadata("A"),
                                      trained["steps"], directory)
    comparison = d0.check_scores(scores, trained["final_scores"], "D3 reload 32")
    decision_exact = all((scores[cid] >= .5) == (trained["final_scores"][cid] >= .5) for cid in scores)
    last = trained["observation_points"][-1]
    criterion_exact = point["criterion_met"] == last["criterion_met"]
    after = d0.state_hashes(model)
    result = {"schema": SCHEMA, "preregistration_sha256": ctx["preregistration_sha256"],
        "train_receipt_sha256": trained["receipt_sha256"], "scores": comparison,
        "decisions_exact": decision_exact, "criterion_exact": criterion_exact,
        "state_sha256_before": before, "state_sha256_after": after, "summary": point,
        "observations_including_training": trained["observations"] + 32,
        "observation_forwards_including_training": trained["observation_forwards"] + 64,
        "optimizer_steps": 0, "generalization_measured": False,
        "passed": comparison["passed"] and decision_exact and criterion_exact and before == after}
    campaign.write_json(directory / "comparison.json", result)
    if (not result["passed"] or result["observations_including_training"] > 384
            or result["observation_forwards_including_training"] > 768):
        raise ValueError("Rechargement, décisions, critère ou budget D3 non reproduits")
    if context(ctx["output"])["preregistration_sha256"] != ctx["preregistration_sha256"]:
        raise ValueError("Préinscription modifiée pendant le reload")
    campaign.finished(train_dir, preregistration_sha256=ctx["preregistration_sha256"])
    return campaign.finish(directory, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="stage", required=True)
    pre = commands.add_parser("preregister")
    pre.add_argument("--d0", type=Path, required=True)
    pre.add_argument("--output", type=Path, required=True)
    pre.add_argument("--code-revision", required=True)
    for name in ("train", "reload"):
        commands.add_parser(name).add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = preregister(args) if args.stage == "preregister" else globals()[args.stage](context(args.output))
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
