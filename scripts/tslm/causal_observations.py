"""Observer D0 sur un clip déjà collaté, sans entraînement ni nouveau scoring.

Deux forwards Qwen : compute_loss puis score_probability_leak. Le runner doit
figer le périmètre train et hasher les poids avant/après l'ensemble ; la garde
locale vérifie identités/versions des tenseurs, pas leur hash intégral par clip.
"""
from contextlib import contextmanager
import hashlib
import math

import numpy as np

from diagnose_train import capture_loss_forward, token_masks
from diagnose_parity import capture_stages


def answer_partition(tokenizer, spec, answer, eos):
    """Le premier token discriminant n'est pas un autre score de classification."""
    candidates = spec["class_continuations"]
    matches = [i for i, text in enumerate(candidates) if answer.startswith(text)]
    if len(candidates) != 2 or len(matches) != 1:
        raise ValueError("Réponse commençant par exactement une continuation officielle requise")
    target = matches[0]
    classes = spec["class_token_ids"]
    common = 0
    while common < min(map(len, classes)) and classes[0][common] == classes[1][common]:
        common += 1
    if common == min(map(len, classes)):
        raise ValueError("Pas de premier token discriminant communément défini dans les deux continuations")
    ids = np.asarray([tokenizer.encode(answer + eos, add_special_tokens=False)], dtype=np.int64)
    eos_ids = tokenizer.encode(eos, add_special_tokens=False)
    class_mask, rest, counts = token_masks(ids, np.ones_like(ids), [classes[target]], eos_ids)
    if counts[0]["description_tokens"] < 0:
        raise ValueError("La classe et l'EOS se chevauchent")
    eos_mask = np.zeros_like(class_mask)
    eos_mask[0, -len(eos_ids):] = True
    first = np.zeros_like(class_mask)
    first[0, common] = True
    masks = {"class": class_mask[0], "first_discriminating_token": first[0],
             "description": (rest & ~eos_mask)[0], "eos": eos_mask[0],
             "response": np.ones(ids.shape[1], dtype=bool)}
    return target, ids[0], masks, common


def nll_terms(logprobs, masks):
    values = np.asarray(logprobs, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Log-probabilités supervisées non finies ou mal dimensionnées")
    terms = {}
    for name, mask in masks.items():
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != values.shape:
            raise ValueError("Masque différent des tokens réellement supervisés")
        count = int(mask.sum())
        total = float(-values[mask].sum())
        terms[name] = {"n_tokens": count, "nll_sum": total, "nll_mean": total/count if count else None}
    return terms


def binary_nll(class_logprobs, target):
    values = np.asarray(class_logprobs, dtype=np.float64)
    if values.shape != (2,) or not np.isfinite(values).all() or target not in (0, 1):
        raise ValueError("Deux sommes officielles finies et une classe cible requises")
    # Équivalent à -logp_vrai + logsumexp(a,b), sans annulation de grandes valeurs.
    value = float(np.logaddexp(0.0, values[1-target] - values[target]))
    if not math.isfinite(value):
        raise ValueError("NLL binaire non finie")
    return value


def norm_summary(array, dtype):
    values = np.asarray(array)
    if values.ndim < 2 or not np.isfinite(values).all():
        raise ValueError("Embeddings finis à au moins deux dimensions requis")
    norms = np.linalg.norm(values.astype(np.float64).reshape(-1, values.shape[-1]), axis=-1)
    if not len(norms) or not np.isfinite(norms).all():
        raise ValueError("Normes d'embeddings absentes/non finies")
    return {"shape": list(values.shape), "source_dtype": str(dtype), "n_vectors": len(norms),
            "l2_min": float(norms.min()), "l2_mean": float(norms.mean()),
            "l2_p50": float(np.median(norms)), "l2_p95": float(np.percentile(norms, 95)),
            "l2_max": float(norms.max()),
            "float32_values_sha256": hashlib.sha256(np.ascontiguousarray(values.astype(np.float32)).tobytes()).hexdigest()}


def _state_versions(model):
    states = []
    for kind, entries in (("parameter", model.named_parameters()), ("buffer", model.named_buffers())):
        for name, tensor in entries:
            states.append((kind, name, id(tensor), tensor.data_ptr(), tuple(tensor.shape), str(tensor.dtype),
                           str(tensor.device), tensor._version, tensor.requires_grad))
            if kind == "parameter" and tensor.grad is not None:
                grad = tensor.grad
                states.append(("gradient", name, id(grad), grad.data_ptr(), grad._version))
    return tuple(states)


@contextmanager
def _official_scores(model):
    """Observer le retour officiel sans ajouter un troisième forward."""
    sentinel = object()
    previous = vars(model).get("score_class_logprobs", sentinel)
    original, calls = model.score_class_logprobs, []
    def wrapped(*args, **kwargs):
        value = original(*args, **kwargs)
        calls.append(value.detach().cpu().numpy().copy())
        return value
    model.score_class_logprobs = wrapped
    try:
        yield calls
    finally:
        if previous is sentinel:
            delattr(model, "score_class_logprobs")
        else:
            model.score_class_logprobs = previous


def observe_clip(model, collated_batch):
    """JSON par clip A/C figé ; aucun collator, seuil, gradient ou optimizer.step.

    L'appelant est responsable de la provenance et du rôle train/heldout. La
    cible answer reste celle du receveur lors d'une intervention sur ses entrées.
    """
    import torch
    if (not isinstance(collated_batch, (list, tuple)) or len(collated_batch) != 1
            or set(collated_batch[0]) != {"pre_prompt", "post_prompt", "time_series", "time_series_text", "answer"}
            or model.single_clip_acoustic_encoding is not True):
        raise ValueError("Un clip A/C déjà collaté, quatre clés d'entrée et answer sont requis")
    if any(p.requires_grad for p in model.llm.parameters()):
        raise ValueError("Qwen doit être gelé pour cette observation")
    sample = collated_batch[0]
    inference = [{key: value for key, value in sample.items() if key != "answer"}]
    model._validate_inference_batch(inference)
    spec = model.scoring_spec()
    target, answer_ids, masks, discriminant = answer_partition(model.tokenizer, spec, sample["answer"], model.get_eos_token())
    before = _state_versions(model)
    modes = [(module, module.training) for module in model.modules()]
    handles, embedding_calls = [], {"loss": [], "scoring": []}
    phase = "loss"
    def text_hook(module, args, output):
        embedding_calls[phase].append(norm_summary(output.detach().float().cpu().numpy(), output.dtype))
    try:
        model.eval()
        handles.append(model.llm.get_input_embeddings().register_forward_hook(text_hook))
        with torch.no_grad(), capture_loss_forward(model.llm) as calls:
            loss = model.compute_loss(collated_batch)
        if len(calls) != 1 or loss.numel() != 1 or not torch.isfinite(loss):
            raise ValueError("Une véritable loss finie et un seul forward supervisé sont requis")
        kwargs, output = calls[0]
        labels, attention = kwargs["labels"], kwargs["attention_mask"]
        positions = torch.nonzero(labels[0] != -100, as_tuple=False).flatten()
        prefix = int(positions[0]) if positions.numel() else 0
        if (labels.shape[0] != 1 or prefix < 1 or positions.numel() != len(answer_ids)
                or not torch.equal(positions, torch.arange(prefix, prefix+len(answer_ids), device=positions.device))
                or labels[0, positions].cpu().tolist() != answer_ids.tolist()
                or not torch.all(labels[0, :prefix] == -100)
                or not torch.all(attention[0, :prefix+len(answer_ids)] == 1)
                or not torch.all(attention[0, prefix+len(answer_ids):] == 0)):
            raise ValueError("Positions/labels/masques réels différents de la réponse attendue")
        selected_logits = output.logits[0, positions-1].float()
        token_lp = torch.log_softmax(selected_logits, dim=-1).gather(1, labels[0, positions, None]).squeeze(1).cpu().numpy()
        terms = nll_terms(token_lp, masks)
        actual_loss = float(loss)
        reconstructed = terms["response"]["nll_mean"]
        loss_prefix = kwargs["inputs_embeds"][0, :prefix].detach().float().cpu().numpy()
        loss_norms = {"prefix_mixed_text_and_acoustics": norm_summary(loss_prefix, kwargs["inputs_embeds"].dtype),
            "answer_text_valid_only": norm_summary(kwargs["inputs_embeds"][0, positions].detach().float().cpu().numpy(),
                                                   kwargs["inputs_embeds"].dtype)}
        tokens = [{"answer_index": i, "token_id": int(token), "decoded_token": model.tokenizer.decode([int(token)]),
            "target_position": prefix+i, "causal_logit_position": prefix+i-1, "attention": True,
            "logprob": float(token_lp[i]), "masks": {name: bool(mask[i]) for name, mask in masks.items()}}
            for i, token in enumerate(answer_ids)]
        # Ne pas conserver les logits vocabulaire de la réponse pendant le scoring.
        calls.clear()
        del output, kwargs, selected_logits, labels, attention, positions, loss
        phase = "scoring"
        with _official_scores(model) as official, capture_stages(model) as (arrays, dtypes):
            probabilities = model.score_probability_leak(inference)
        if (len(official) != 1 or official[0].shape != (1, 2) or len(probabilities) != 1
                or not np.isfinite(official[0]).all() or not math.isfinite(probabilities[0])
                or not 0 <= probabilities[0] <= 1):
            raise ValueError("Retour du scoring officiel absent/non fini")
        score_lp = official[0][0]
        token_scores, score_mask = arrays["class_token_logprobs"], arrays["class_token_mask"].astype(bool)
        if not np.allclose(token_scores.astype(np.float64).sum(axis=1), score_lp, atol=1e-6, rtol=0):
            raise ValueError("Les tokens capturés ne reconstruisent pas les sommes officielles")
        width = token_scores.shape[1]
        score_prefix = arrays["llm_inputs_embeds"].shape[1]-width
        score_tokens = []
        for i, ids in enumerate(spec["class_token_ids"]):
            if not np.array_equal(score_mask[i], np.arange(width) < len(ids)):
                raise ValueError("Masque réel de continuation incompatible")
            score_tokens.append([{"token_index": j, "token_id": ids[j] if j < len(ids) else model.tokenizer.pad_token_id,
                "target_position": score_prefix+j, "causal_logit_position": score_prefix+j-1,
                "included": bool(score_mask[i, j]), "logprob": float(token_scores[i, j]) if score_mask[i, j] else None,
                "masked_logprob_contribution": float(token_scores[i, j]),
                "target_logit": float(arrays["class_target_logits"][i, j]),
                "log_normalizer": float(arrays["class_log_normalizer"][i, j])} for j in range(width)])
        scored_prefixes = arrays["llm_inputs_embeds"][:, :score_prefix]
        same_shape = scored_prefixes.shape[1:] == loss_prefix.shape
        delta = float(np.max(np.abs(scored_prefixes.astype(np.float64)-loss_prefix.astype(np.float64)))) if same_shape else None
        class_delta = token_scores[target, :len(spec["class_token_ids"][target])].astype(np.float64) - token_lp[masks["class"]].astype(np.float64)
        result = {"schema": "pipe-causal-observation-v1", "batch_size": 1, "mode": "eval_no_grad",
            "variant": "C" if model.amplitude_evidence else "A", "scoring_spec": spec,
            "target_class_index": target, "target_class_continuation": spec["class_continuations"][target],
            "first_discriminating_token_index": discriminant, "loss": {"compute_loss": actual_loss,
                "reconstructed_mean": reconstructed, "absolute_difference": abs(actual_loss-reconstructed),
                "matches_atol_1e_6": abs(actual_loss-reconstructed) <= 1e-6,
                "terms": terms, "additive_partition": ["class", "description", "eos"],
                "first_discriminating_token_overlaps_class": True, "tokens": tokens,
                "prompt_length": prefix, "prompt_labels_ignored": True},
            "scoring": {"probability_leak": float(probabilities[0]), "class_logprob_sums": score_lp.tolist(),
                "binary_nll": binary_nll(score_lp, target), "binary_nll_count": 1,
                "binary_nll_formula": "logaddexp(0, logp_other - logp_true), complete official continuations",
                "tokens": score_tokens, "prompt_length": score_prefix},
            "alignment": {"same_prompt_shape": same_shape, "prompt_max_abs_difference": delta,
                "prompts_exact": same_shape and bool(np.array_equal(scored_prefixes[0], loss_prefix))
                    and bool(np.array_equal(scored_prefixes[1], loss_prefix)),
                "class_token_logprob_max_abs_difference": float(np.max(np.abs(class_delta))),
                "class_token_logprob_differences_scoring_minus_loss": class_delta.tolist()},
            "embedding_norms": {"supervision": loss_norms,
                "projector_before_llm_dtype_cast": {name: norm_summary(value, dtypes[name]) for name, value in arrays.items()
                    if name.startswith("projector_") and name.endswith("_output")},
                "scoring_prefix_mixed_text_and_acoustics": norm_summary(scored_prefixes, dtypes["llm_inputs_embeds"]),
                "qwen_embedding_lookups": embedding_calls,
                "lookup_scope": "All text embedding lookups; may include candidate padding. The valid answer norm excludes padding.",
                "projector_scope": "Projector output before possible BF16 cast; not isolated acoustics as injected into Qwen."},
            "llm_forward_counts": {"supervision": 1, "scoring": 1}, "optimizer_steps": 0,
            "state_guard": {"scope": "tensor identities/versions/gradients; runner must hash full state before/after D0",
                            "n_entries_checked": len(before), "unchanged": True}}
        return result
    finally:
        for handle in handles:
            handle.remove()
        for module, training in modes:
            module.training = training
        if _state_versions(model) != before:
            raise RuntimeError("État du modèle ou gradients modifiés pendant l'observation ; arrêter D0")
