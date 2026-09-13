"""D1 numérique sans fit : tête BF16, vraie multiplication FP32, FP32 réarrondie.

preregister --d0 DOSSIER_D0 --output DOSSIER_NEUF --code-revision SHA40
observe --output DOSSIER_D1 --variant A|C

Les trois bras utilisent le scorer officiel inchangé. Aucune politique n'est
adoptée en produit ; aucun seuil n'est sélectionné et aucune donnée externe lue.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
from pathlib import Path
import re
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/tslm"))
import diagnose_causal as d0
from causal_observations import _official_scores, _state_versions, binary_nll

campaign = d0.campaign
SCHEMA = "pipe-head-precision-d1-v1"
D0_RECEIPTS = {
    "A": "4810270aec3c73398c4d0052393a2b4271f8497eaf485c0305ccb65ec1cc5137",
    "C": "c6d2121f1d01daeda1e0fe1aca629a99677d34cff961fac1ad1be82360e2ad04",
}
POLICIES = ("original_bf16", "fp32_linear", "fp32_linear_rounded_bf16")
PROTOCOL = {"policies": list(POLICIES), "numeric_variants_compared": 2,
    "terminal_checkpoint_fold": 0, "clips_per_model": 598, "fit_clips": 500, "heldout_clips": 98,
    "forwards_per_clip": 3, "forwards_per_model": 1794, "total_forwards": 3588,
    "score_atol_vs_d0": 1e-6, "diagnostic_threshold": 0.5, "optimizer_steps": 0,
    "head_recalculation": "F.linear(hidden.float(), detached_weight.float(), detached_bias.float())",
    "fp32_weights_copy": "once_per_model_outside_parameters", "autocast": False,
    "tf32": "unchanged_from_frozen_runtime", "inputs_and_hidden_states": "exact_byte_hash_equality",
    "rounded_vs_original_equality_required": False, "policy_selection": False,
    "pooled_598_performance": False, "validation_test_external_read": False}
SOURCE_NAMES = (*d0.SOURCE_NAMES, "scripts/tslm/diagnose_head_precision.py")


def source_hashes():
    return {name: campaign.sha256_file(ROOT / name) for name in SOURCE_NAMES}


def runtime_identity():
    """Lire le code Qwen installé sans instancier de modèle ou charger de poids."""
    import torch
    from transformers import Qwen3_5ForCausalLM
    source = inspect.getsourcefile(Qwen3_5ForCausalLM)
    if source is None or Path(source).name != "modeling_qwen3_5.py":
        raise ValueError("Source runtime officielle Qwen 3.5 introuvable")
    if torch.is_autocast_enabled("cuda") or torch.is_autocast_enabled("cpu"):
        raise ValueError("D1 exige un runtime sans autocast")
    return {"qwen_source_path": str(Path(source).resolve()),
        "qwen_source_sha256": campaign.sha256_file(Path(source)),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "autocast_cuda": False, "autocast_cpu": False}


def checked_d0(directory):
    ctx = d0.context(Path(directory))
    receipts, identities = {}, {}
    for variant in ("A", "C"):
        path = ctx["output"] / variant
        value = campaign.finished(path, preregistration_sha256=ctx["preregistration_sha256"])
        if (value is None or value.get("receipt_sha256") != D0_RECEIPTS[variant]
                or value.get("variant") != variant or value.get("optimizer_steps") != 0
                or value.get("validation_test_external_read") is not False
                or value.get("weights_unchanged_within_each_state") is not True):
            raise ValueError("Les deux reçus D0 terminés et explicitement autorisés sont requis")
        state = value["state_sha256"]
        if state["terminal_before"] != state["terminal_after"]:
            raise ValueError("État terminal D0 non immuable")
        receipts[variant] = value
        identities[variant] = {"receipt_sha256": value["receipt_sha256"],
            "complete_sha256": campaign.sha256_file(path / "complete.json"),
            "terminal_jsonl_sha256": campaign.sha256_file(path / "terminal.jsonl"),
            "terminal_state_sha256": state["terminal_after"]}
    return ctx, receipts, identities


def preregister(args):
    out, parent = args.output.resolve(), args.d0.resolve()
    if out.exists() or out.is_symlink():
        raise FileExistsError("Préinscription D1 dans un dossier neuf uniquement")
    if out.is_relative_to(parent):
        raise ValueError("Ne jamais écrire D1 dans les reçus D0")
    if not re.fullmatch(r"[0-9a-f]{40}", args.code_revision or ""):
        raise ValueError("Révision réelle du code SHA40 requise")
    ctx, _, identities = checked_d0(parent)
    if out.is_relative_to(Path(ctx["registration"]["campaign"]).resolve()):
        raise ValueError("Ne jamais écrire D1 dans la campagne CV")
    for variant in ("A", "C"):
        campaign.verify_runtime(ctx["campaign"]["gates"][variant]["runtime"])
    value = {"schema": SCHEMA, "created_at": datetime.now(timezone.utc).isoformat(),
        "code_revision": args.code_revision, "d0": str(parent), "protocol": PROTOCOL,
        "d0_preregistration_sha256": ctx["preregistration_sha256"], "d0_receipts": identities,
        "cohorts": ctx["registration"]["cohorts"], "input_sha256": ctx["registration"]["input_sha256"],
        "source_sha256": source_hashes(), "runtime": runtime_identity(), "model_loaded": False}
    out.mkdir(parents=True, exist_ok=False)
    campaign.write_json(out / "preregistration.json", value)
    return {"preregistered": True, "preregistration_sha256": campaign.sha256_file(out / "preregistration.json")}


def context(output):
    output = Path(output).resolve()
    value = campaign.read_json(output / "preregistration.json")
    if (value.get("schema") != SCHEMA or value.get("protocol") != PROTOCOL
            or value.get("model_loaded") is not False or value.get("source_sha256") != source_hashes()
            or value.get("runtime") != runtime_identity()
            or not re.fullmatch(r"[0-9a-f]{40}", value.get("code_revision", ""))):
        raise ValueError("Préinscription D1 ou runtime/source modifiés")
    ctx, receipts, identities = checked_d0(value["d0"])
    if (ctx["preregistration_sha256"] != value["d0_preregistration_sha256"]
            or identities != value["d0_receipts"] or ctx["registration"]["cohorts"] != value["cohorts"]
            or ctx["registration"]["input_sha256"] != value["input_sha256"]
            or output.is_relative_to(ctx["output"])
            or output.is_relative_to(Path(ctx["registration"]["campaign"]).resolve())):
        raise ValueError("D1 doit rester lié aux mêmes données, cohortes et reçus D0")
    return {"output": output, "registration": value, "d0": ctx, "receipts": receipts,
            "preregistration_sha256": campaign.sha256_file(output / "preregistration.json")}


def tensor_identity(tensor):
    """Empreinte des octets exacts, y compris pour BF16 (sans promotion F32)."""
    import torch
    value = tensor.detach().contiguous().cpu()
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "sha256": hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()}


def prepare_head(model):
    import torch
    head = model.llm.get_output_embeddings()
    if (type(head) is not torch.nn.Linear or getattr(model.llm, "lm_head", head) is not head
            or head.weight.dtype != torch.bfloat16 or head.weight.requires_grad
            or (head.bias is not None and (head.bias.dtype != torch.bfloat16 or head.bias.requires_grad))):
        raise ValueError("D1 exige la tête nn.Linear BF16 gelée originale")
    weight = head.weight.detach().float().clone()
    bias = head.bias.detach().float().clone() if head.bias is not None else None
    return {"module": head, "weight_fp32": weight, "bias_fp32": bias,
            "parameter_versions": tuple((id(p), p._version) for p in head.parameters()),
            "copy_versions": (weight._version, bias._version if bias is not None else None),
            "identity": {"original_weight": tensor_identity(head.weight), "fp32_weight": tensor_identity(weight),
                         "original_bias": tensor_identity(head.bias) if head.bias is not None else None,
                         "fp32_bias": tensor_identity(bias) if bias is not None else None}}


@contextmanager
def precision_head(prepared, policy, *, output_observer=None):
    """Remplacer temporairement forward, pas le module/poids ; restaurer même en erreur."""
    import torch
    import torch.nn.functional as functional
    if policy not in POLICIES:
        raise ValueError("Politique de tête non préinscrite")
    head, calls = prepared["module"], []
    sentinel = object()
    previous = vars(head).get("forward", sentinel)
    original = head.forward

    def forward(input):
        if torch.is_autocast_enabled("cuda") or torch.is_autocast_enabled("cpu"):
            raise ValueError("Autocast actif pendant D1")
        if (input.dtype != torch.bfloat16 or tuple((id(p), p._version) for p in head.parameters())
                != prepared["parameter_versions"]):
            raise ValueError("Entrée ou poids de tête modifiés")
        record = {"input": tensor_identity(input)}
        if policy == "original_bf16":
            output = original(input)
        else:
            # La multiplication est REFAITE en FP32, pas une promotion de logits BF16.
            with torch.autocast(device_type=input.device.type, enabled=False):
                output = functional.linear(input.float(), prepared["weight_fp32"], prepared["bias_fp32"])
            if output_observer is not None:
                output_observer(output, "fp32_before_round")
            if policy == "fp32_linear_rounded_bf16":
                output = output.to(head.weight.dtype)
        wanted = torch.float32 if policy == "fp32_linear" else torch.bfloat16
        if output.dtype != wanted:
            raise ValueError("La tête n'a pas retourné le dtype préinscrit")
        if output_observer is not None:
            output_observer(output, "head_return")
        record.update(output_dtype=str(output.dtype), output_shape=list(output.shape))
        calls.append(record)
        return output

    head.forward = forward
    try:
        yield calls
    finally:
        if previous is sentinel:
            delattr(head, "forward")
        else:
            head.forward = previous
        if (tuple((id(p), p._version) for p in head.parameters()) != prepared["parameter_versions"]
                or (prepared["weight_fp32"]._version, prepared["bias_fp32"]._version
                    if prepared["bias_fp32"] is not None else None) != prepared["copy_versions"]):
            raise RuntimeError("Poids de tête ou copie FP32 modifiés pendant D1")


def score_arm(model, batch, target, prepared, policy):
    """Un seul vrai forward officiel ; observations via hooks uniquement."""
    import torch
    if len(batch) != 1 or target not in (0, 1) or model.training:
        raise ValueError("Un clip en mode eval, cible extérieure au prompt, requis")
    model._validate_inference_batch(batch)
    spec, captured, handles = model.scoring_spec(), {}, []
    ids = spec["class_token_ids"]
    first = 0
    while first < min(map(len, ids)) and ids[0][first] == ids[1][first]:
        first += 1
    if first == min(map(len, ids)):
        raise ValueError("Premier token discriminant non défini")

    def before(module, args, kwargs):
        if "prompt" in captured:
            raise ValueError("Plus d'un forward officiel pour un bras D1")
        inputs, mask = kwargs["inputs_embeds"], kwargs["attention_mask"]
        prefix = inputs.shape[1] - max(map(len, ids))
        if inputs.shape[0] != 2 or prefix < 1:
            raise ValueError("Deux continuations et un préfixe causal non vide requis")
        captured.update(prompt={"full_inputs_embeds": tensor_identity(inputs),
            "full_attention_mask": tensor_identity(mask)}, prefix_length=prefix)

    def head_output(output, stage):
        # Capturer directement la sortie de nn.Linear, AVANT le retour à Qwen
        # et avant toute promotion des logits dans le scorer officiel.
        position = captured["prefix_length"] - 1 + first
        selected = output[:, position, :][:, [ids[0][first], ids[1][first]]]
        targets = selected.detach().cpu().tolist()
        captured[stage] = {"selected_logits_raw": tensor_identity(selected), "values": targets,
                           "margin": targets[0][0] - targets[0][1]}
        if stage == "head_return":
            captured.update(first_discriminating_token_index=first,
                first_decision_token_ids=[ids[0][first], ids[1][first]],
                first_decision_logits=targets, first_decision_margin=targets[0][0] - targets[0][1])

    def after(module, args, kwargs, output):
        captured["returned_logits_dtype"] = str(output.logits.dtype)

    versions = _state_versions(model)
    try:
        handles.append(model.llm.register_forward_pre_hook(before, with_kwargs=True))
        handles.append(model.llm.register_forward_hook(after, with_kwargs=True))
        with precision_head(prepared, policy, output_observer=head_output) as head_calls, _official_scores(model) as official:
            probabilities = model.score_probability_leak(batch)
        if (len(official) != 1 or official[0].shape != (1, 2) or len(probabilities) != 1
                or len(head_calls) != 1 or "first_decision_logits" not in captured):
            raise ValueError("Un forward de tête/Qwen et deux sommes officielles requis")
        p = probabilities[0]
        if type(p) not in (float, int) or not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError("Score officiel absent, non fini ou hors bornes")
        return {"policy": policy, "probability_leak": float(p), "class_logprob_sums": official[0][0].tolist(),
            "binary_nll": binary_nll(official[0][0], target), "head": head_calls[0],
            "head_weight_identity": prepared["identity"], "llm_forwards": 1, **captured}
    finally:
        for handle in handles:
            handle.remove()
        if _state_versions(model) != versions:
            raise RuntimeError("État du modèle modifié pendant un bras D1")


def score_clip(model, batch, target, prepared, expected_d0):
    arms = {}
    for policy in POLICIES:
        arm = score_arm(model, batch, target, prepared, policy)
        if policy == "original_bf16":
            if abs(arm["probability_leak"] - expected_d0) > 1e-6:
                raise ValueError("Original BF16 différent du score D0 : arrêter D1")
        else:
            baseline = arms["original_bf16"]
            if (arm["prompt"] != baseline["prompt"] or arm["head"]["input"] != baseline["head"]["input"]
                    or arm["head_weight_identity"] != baseline["head_weight_identity"]):
                raise ValueError("Inputs, hidden states ou poids différents entre les trois bras")
        arms[policy] = arm
    return {"arms": arms, "llm_forwards": 3, "input_hidden_weight_identity_exact": True,
            "original_vs_d0_abs_difference": abs(arms["original_bf16"]["probability_leak"] - expected_d0)}


def d0_scores(path, rows, variant, fit_ids):
    by_id, result = {r["clip_id"]: r for r in rows}, {}
    with Path(path).open() as stream:
        for line in stream:
            record = json.loads(line)
            cid = record["clip_id"]
            if (cid not in by_id or cid in result or record.get("stage") != "terminal"
                    or record.get("source_fold") != "train" or record.get("variant") != variant
                    or record.get("group_id") != by_id[cid]["group_id"]
                    or record.get("target_class") != by_id[cid]["label"]
                    or record.get("partition") != ("fit" if cid in fit_ids else "heldout")):
                raise ValueError("Journal terminal D0 d'une autre population/checkpoint")
            p = record["observation"]["scoring"]["probability_leak"]
            if type(p) not in (float, int) or not math.isfinite(p) or not 0 <= p <= 1:
                raise ValueError("Score D0 non fini ou hors bornes")
            result[cid] = float(p)
    if set(result) != set(by_id):
        raise ValueError("Les 598 scores terminaux D0 sont requis")
    return result


def partition_summary(rows, observations):
    summaries = {}
    for policy in POLICIES:
        values = [observations[r["clip_id"]]["arms"][policy] for r in rows]
        probabilities = {r["clip_id"]: value["probability_leak"] for r, value in zip(rows, values)}
        margins = sorted({v["first_decision_margin"] for v in values})
        summaries[policy] = {"n_clips": len(rows),
            "metrics": campaign.metric_report(rows, probabilities, threshold=0.5),
            "binary_nll_sum": sum(v["binary_nll"] for v in values),
            "binary_nll_mean": sum(v["binary_nll"] for v in values) / len(rows),
            "first_decision_margin_unique_count": len(margins), "first_decision_margin_unique_values": margins,
            "first_decision_margin_adjacent_steps": sorted(set(np.diff(margins).tolist())),
            "head_output_dtypes": sorted({v["head"]["output_dtype"] for v in values})}
    comparisons = {}
    for a, b in ((POLICIES[1], POLICIES[0]), (POLICIES[2], POLICIES[0]), (POLICIES[1], POLICIES[2])):
        left, right = summaries[a], summaries[b]
        diffs = [observations[r["clip_id"]]["arms"][a]["probability_leak"]
                 - observations[r["clip_id"]]["arms"][b]["probability_leak"] for r in rows]
        margin_diffs = [observations[r["clip_id"]]["arms"][a]["first_decision_margin"]
                       - observations[r["clip_id"]]["arms"][b]["first_decision_margin"] for r in rows]
        comparisons[f"{a}_minus_{b}"] = {
            "clip_roc_auc_delta": left["metrics"]["clip_roc_auc_full"] - right["metrics"]["clip_roc_auc_full"],
            "group_roc_auc_delta": left["metrics"]["group_roc_auc_full"] - right["metrics"]["group_roc_auc_full"],
            "binary_nll_delta": left["binary_nll_mean"] - right["binary_nll_mean"],
            "score_change_mean": float(np.mean(diffs)), "score_abs_change_mean": float(np.mean(np.abs(diffs))),
            "score_abs_change_max": float(np.max(np.abs(diffs))),
            "first_margin_abs_change_max": float(np.max(np.abs(margin_diffs))),
            "scores_differing_count": sum(delta != 0 for delta in diffs),
            "interpretation": "Descriptif sur développement ; aucun choix de politique ni causalité exclusive de l'arrondi."}
    return {"policies": summaries, "comparisons": comparisons, "threshold_fitted": False}


def observe(ctx, variant):
    if variant not in ("A", "C"):
        raise ValueError("Variante A ou C requise")
    out = ctx["output"] / variant
    previous = campaign.finished(out, preregistration_sha256=ctx["preregistration_sha256"])
    if previous is not None:
        return previous
    registration, old = ctx["registration"], ctx["d0"]
    parent, scope = old["campaign"], registration["cohorts"]
    paths = parent["registration"]["paths"]
    rows, series, amplitudes, _ = campaign.load_fold(paths, parent["split"], "train")
    if d0.cohorts(rows, parent["registration"]["folds"][0]) != scope:
        raise ValueError("Cohortes différentes de D0")
    saved = d0_scores(old["output"] / variant / "terminal.jsonl", rows, variant, set(scope["fit_ids"]))
    campaign.verify_runtime(parent["gates"][variant]["runtime"])
    if runtime_identity() != registration["runtime"]:
        raise ValueError("Runtime numérique différent de la préinscription")
    out.mkdir(parents=True, exist_ok=False)
    campaign.write_json(out / "started.json", {"schema": SCHEMA, "variant": variant,
        "preregistration_sha256": ctx["preregistration_sha256"], "expected_forwards": 1794, "optimizer_steps": 0})
    started = time.monotonic()
    model = d0.initialize(paths["base"], variant)
    checkpoint = Path(old["registration"]["campaign"]) / f"cv/{variant}/fold-0/temporal.pt"
    d0.load_terminal(model, checkpoint)
    before = d0.state_hashes(model)
    if (before != ctx["receipts"][variant]["state_sha256"]["terminal_after"]
            or before != old["receipts"][variant][0]["state_sha256_after"]
            or model.scoring_spec() != old["receipts"][variant][0]["scoring_spec"]):
        raise ValueError("Poids terminaux ou scoring différents de D0")
    prepared = prepare_head(model)
    metadata = campaign.variant_metadata(variant)
    from opentslm.time_series_datasets.util import extend_time_series_to_match_patch_size_and_aggregate as collate
    observations = {}
    try:
        with (out / "observations.jsonl").open("x") as stream:
            for row in rows:
                cid = row["clip_id"]
                batch = collate(campaign.examples([row], series, amplitudes, metadata, training=False), normalize=False)
                value = score_clip(model, batch, int(row["label"] != "leak"), prepared, saved[cid])
                record = {"clip_id": cid, "group_id": row["group_id"], "target_class": row["label"],
                    "source_fold": "train", "partition": "fit" if cid in scope["fit_ids"] else "heldout",
                    "variant": variant, "observation": value}
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                observations[cid] = value
                if len(observations) % 20 == 0:
                    print(json.dumps({"variant": variant, "clips": len(observations), "expected": 598,
                                      "llm_forwards": len(observations) * 3}), flush=True)
        after = d0.state_hashes(model)
        if after != before or len(observations) != 598:
            raise ValueError("Poids modifiés ou budget D1 incomplet")
        again = context(ctx["output"])
        if again["preregistration_sha256"] != ctx["preregistration_sha256"]:
            raise ValueError("Préinscription modifiée pendant D1")
        by_id = {r["clip_id"]: r for r in rows}
        result = {"schema": SCHEMA, "variant": variant, "preregistration_sha256": ctx["preregistration_sha256"],
            "clips": 598, "llm_forwards": sum(v["llm_forwards"] for v in observations.values()),
            "optimizer_steps": 0, "head_weight_identity": prepared["identity"],
            "state_sha256_before": before, "state_sha256_after": after, "weights_unchanged": True,
            "all_inputs_hidden_weights_exact": all(v["input_hidden_weight_identity_exact"] for v in observations.values()),
            "original_vs_d0_max_abs_difference": max(v["original_vs_d0_abs_difference"] for v in observations.values()),
            "partitions": {name: partition_summary([by_id[cid] for cid in scope[f"{name}_ids"]], observations)
                           for name in ("fit", "heldout")},
            "policy_selected": None, "pooled_598_performance_calculated": False,
            "validation_test_external_read": False, "elapsed_seconds": time.monotonic() - started,
            "limitations": ["Le témoin FP32 réarrondi peut différer du BF16 original par l'arithmétique de multiplication.",
                "Comparer FP32 à son réarrondi isole la précision de sortie de CE calcul FP32, pas toutes les erreurs numériques.",
                "Les 98 réservés sont du développement déjà consulté, pas une confirmation indépendante.",
                "Aucune politique numérique n'est adoptée et aucun seuil n'est ajusté."]}
        if result["llm_forwards"] != PROTOCOL["forwards_per_model"]:
            raise ValueError("Budget de vrais forwards différent de la préinscription")
        campaign.write_json(out / "summary.json", result)
        return campaign.finish(out, result)
    except Exception as exc:
        campaign.write_json(out / "failure.json", {"schema": SCHEMA, "variant": variant, "status": "failed",
            "completed_clips": len(observations), "error": {"type": type(exc).__name__, "message": str(exc)}})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    prereg = sub.add_parser("preregister")
    prereg.add_argument("--d0", type=Path, required=True)
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
