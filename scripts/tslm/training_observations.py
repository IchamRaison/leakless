"""Observation TRAIN en ligne, sans changer la loss, les forwards ni la recette.

Usage (microbatch=1, lot effectif <=8) :
    with capture_training_diagnostics(model, optimizer) as audit:
        for record, ids in zip(train_epoch(...), batches_d_identifiants):
            journal(audit.pop_step(ids, training_record=record))
        journal(audit.epoch_summary(reset=True))

Les identifiants viennent de l'appelant, jamais des entrées du modèle. Un seul
step terminé peut attendre sa consommation. Les agrégats décrivent des poids
évoluant pendant TRAIN, pas un checkpoint fixe. Aucun score binaire n'est déduit
du seul forward supervisé ; l'observation D0/checkpoint reste distincte.
"""
from contextlib import contextmanager
import math
from types import SimpleNamespace

from causal_observations import answer_partition, nll_terms
from diagnose_train import capture_loss_forward


TERMS = ("class", "first_discriminating_token", "description", "eos", "response")
COMPONENTS = ("encoder", "projector")
_MISSING = object()


def _empty_terms():
    return {key: {"n_tokens": 0, "nll_sum": 0.0, "nll_mean": None} for key in TERMS}


def _add_terms(total, terms):
    for key in TERMS:
        count, value = terms[key]["n_tokens"], terms[key]["nll_sum"]
        if type(count) is not int or count < 0 or not math.isfinite(value) or value < 0:
            raise ValueError("Somme/effectif NLL invalide")
        total[key]["n_tokens"] += count
        total[key]["nll_sum"] += value
        item = total[key]
        item["nll_mean"] = item["nll_sum"] / item["n_tokens"] if item["n_tokens"] else None


def _relative_update(delta, before):
    # Aucun epsilon arbitraire : 0 au dénominateur est explicitement indéfini.
    return delta / before if before else None


def _restore_attribute(obj, name, previous):
    if previous is _MISSING:
        delattr(obj, name)
    else:
        setattr(obj, name, previous)


def _tensor_l2(tensors):
    value = math.sqrt(math.fsum(float(t.detach().double().square().sum()) for t in tensors))
    if not math.isfinite(value):
        raise ValueError("Norme de tenseur non finie")
    return value


def _option(value):
    """Options AdamW réelles, y compris valeurs implicites remplies par PyTorch."""
    import torch
    if value is None or type(value) in (bool, int, str):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    if isinstance(value, (tuple, list)):
        return [_option(item) for item in value]
    if isinstance(value, torch.Tensor) and value.numel() == 1:
        return {"value": _option(value.item()), "dtype": str(value.dtype)}
    raise ValueError("Option d'optimiseur non scalaire/non finie non prise en charge")


def _loss_record(model, spec, sample, loss, calls):
    """Lire les vraies cibles/logits détachés ; ne jamais remplacer la loss."""
    import torch
    if len(calls) != 1 or loss.numel() != 1 or not torch.isfinite(loss.detach()).all():
        raise ValueError("Un seul forward supervisé et une loss scalaire finie requis")
    target, answer_ids, masks, discriminant = answer_partition(
        model.tokenizer, spec, sample["answer"], model.get_eos_token())
    kwargs, output = calls[0]
    labels, attention = kwargs["labels"], kwargs["attention_mask"]
    if labels.ndim != 2 or labels.shape[0] != 1 or attention.shape != labels.shape:
        raise ValueError("Labels/attention batch=1 requis")
    positions = torch.nonzero(labels[0] != -100, as_tuple=False).flatten()
    prefix = int(positions[0]) if positions.numel() else 0
    if (prefix < 1 or positions.numel() != len(answer_ids)
            or not torch.equal(positions, torch.arange(prefix, prefix + len(answer_ids), device=positions.device))
            or labels[0, positions].tolist() != answer_ids.tolist()
            or not torch.all(labels[0, :prefix] == -100)
            or not torch.all(attention[0, :prefix + len(answer_ids)] == 1)
            or not torch.all(attention[0, prefix + len(answer_ids):] == 0)
            or output.logits.ndim != 3 or output.logits.shape[:2] != labels.shape):
        raise ValueError("Positions causales/labels/attention différents de la réponse attendue")
    selected = output.logits.detach()[0, positions - 1].float()
    logprobs = torch.log_softmax(selected, dim=-1).gather(
        1, labels[0, positions, None]).squeeze(1).cpu().numpy()
    terms = nll_terms(logprobs, masks)
    actual = float(loss.detach())
    return {"target_class_index": target, "first_discriminating_token_index": discriminant,
            "compute_loss": actual, "terms": terms,
            "reconstructed_response_mean": terms["response"]["nll_mean"],
            "reconstruction_absolute_difference": abs(actual - terms["response"]["nll_mean"]),
            "prompt_length": prefix, "first_causal_logit_position": prefix - 1,
            "last_target_position": prefix + len(answer_ids) - 1,
            "labels_and_attention_verified": True, "logits_dtype": str(output.logits.dtype),
            "nll_accumulation": "log_softmax float32; sums float64; true-answer forward only"}


class _TrainingDiagnostics:
    def __init__(self, model, optimizer, max_clips_per_step, allow_lora=False, allow_microbatches=False):
        import torch
        if type(max_clips_per_step) is not int or not 1 <= max_clips_per_step <= 8:
            raise ValueError("Tampon borné à entre 1 et 8 clips par step")
        if model.single_clip_acoustic_encoding is not True or any(
                p.requires_grad and (not allow_lora or "lora_" not in name)
                for name, p in model.llm.named_parameters()):
            raise ValueError("Encodage canonique et Qwen gelé requis")
        if not isinstance(optimizer, torch.optim.AdamW):
            raise ValueError("Cette observation minimale porte sur l'AdamW temporel existant")
        self.model, self.optimizer, self.limit = model, optimizer, max_clips_per_step
        self.parameters = {name: list(getattr(model, name).parameters()) for name in COMPONENTS}
        self.allow_microbatches = allow_microbatches
        if allow_lora:
            self.parameters["lora"] = model.get_lora_parameters()
        temporal = [p for ps in self.parameters.values() for p in ps]
        optimized = [p for group in optimizer.param_groups for p in group["params"]]
        if (not all(self.parameters.values()) or not all(p.requires_grad for p in temporal)
                or len({id(p) for p in temporal}) != len(temporal)
                or len({id(p) for p in optimized}) != len(optimized)
                or {id(p) for p in optimized} != {id(p) for p in temporal}
                or {id(p) for p in model.parameters() if p.requires_grad} != {id(p) for p in temporal}):
            raise ValueError("L'optimiseur doit contenir exactement encodeur/projecteur entraînables")
        self.spec = model.scoring_spec()
        self.pending, self.completed, self.before = [], None, None
        self.active = True
        self._reset_epoch()

    def _reset_epoch(self):
        self.totals, self.n_clips, self.n_steps = _empty_terms(), 0, 0
        self.original_loss_token_sum, self.max_reconstruction_error = 0.0, 0.0

    def _wrap_loss(self, original, batch):
        import torch
        if not self.active or self.completed is not None or len(self.pending) >= self.limit:
            raise RuntimeError("Consommer le step précédent ; tampon TRAIN plein ou contexte fermé")
        if (not isinstance(batch, (list, tuple)) or not batch
                or (not self.allow_microbatches and len(batch) != 1)
                or len(self.pending) + len(batch) > self.limit
                or any(set(sample) != {"pre_prompt", "post_prompt", "time_series", "time_series_text", "answer"}
                       for sample in batch)):
            raise ValueError("Un seul clip collaté avec answer et quatre clés d'entrée requis")
        previous = vars(self.model.llm).get("forward", _MISSING)
        calls = []
        try:
            with capture_loss_forward(self.model.llm) as calls:
                loss = original(batch)  # EXACT tenseur différentiable retourné ensuite.
            with torch.no_grad():
                if len(batch) == 1:
                    self.pending.append(_loss_record(self.model, self.spec, batch[0], loss, calls))
                else:
                    if len(calls) != 1 or calls[0][0]["labels"].shape[0] != len(batch):
                        raise ValueError("Un seul forward du microbatch entier requis")
                    kwargs, output = calls[0]
                    records = []
                    for index, sample in enumerate(batch):
                        sliced = {key: kwargs[key][index:index + 1] for key in ("labels", "attention_mask")}
                        records.append(_loss_record(self.model, self.spec, sample, loss,
                            [(sliced, SimpleNamespace(logits=output.logits[index:index + 1]))]))
                    total = sum(r["terms"]["response"]["n_tokens"] for r in records)
                    reconstructed = sum(r["terms"]["response"]["nll_sum"] for r in records) / total
                    for record in records:
                        record.update(compute_loss_scope="shared_microbatch_token_mean",
                                      microbatch_clips=len(batch),
                                      reconstruction_absolute_difference=abs(float(loss.detach()) - reconstructed))
                    self.pending.extend(records)
            return loss
        finally:
            calls.clear()  # Ne jamais garder les logits/graphe au microbatch suivant.
            _restore_attribute(self.model.llm, "forward", previous)

    def _gradient_norms(self):
        import torch
        values = {}
        for name, parameters in self.parameters.items():
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
                raise ValueError("Gradient post-clipping absent/non fini")
            # Même réduction que train_epoch, mais l'instant est POST-clipping.
            values[name] = float(torch.sqrt(sum(p.grad.detach().float().square().sum() for p in parameters)))
        return values

    def _moments(self):
        summary = {}
        for name, parameters in self.parameters.items():
            states = [self.optimizer.state.get(p, {}) for p in parameters]
            steps = [float(state["step"]) for state in states if "step" in state]
            if not all(math.isfinite(value) for value in steps):
                raise ValueError("Step AdamW non fini")
            moments = {}
            for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                values = [state[key] for state in states if key in state]
                moments[key] = {"n_tensors": len(values), "n_elements": sum(t.numel() for t in values),
                                "dtypes": sorted({str(t.dtype) for t in values}),
                                "l2": _tensor_l2(values) if values else None}
            summary[name] = {"parameters_with_state": sum(bool(state) for state in states),
                             "step_min": min(steps) if steps else None, "step_max": max(steps) if steps else None,
                             "moments": moments}
        return summary

    def _pre_step(self, optimizer, args, kwargs):
        if not self.pending or self.completed is not None or self.before is not None:
            raise RuntimeError("Step sans observation ou step précédent non consommé")
        if kwargs.get("closure") is not None or any(callable(arg) for arg in args):
            raise ValueError("Optimizer closure non prise en charge : un seul forward par clip requis")
        options = []
        for group in optimizer.param_groups:
            components = [name for name, ps in self.parameters.items()
                          if any(id(p) == id(q) for p in ps for q in group["params"])]
            options.append({"components": components, "n_parameters": len(group["params"]),
                            "options": {key: _option(value) for key, value in group.items() if key != "params"}})
        self.step_info = {"gradient_norms_post_clip": self._gradient_norms(),
            "optimizer": {"class": type(optimizer).__module__ + "." + type(optimizer).__qualname__,
                "param_groups": options, "state_before": self._moments(),
                "dtypes": {name: {"parameters": sorted({str(p.dtype) for p in ps}),
                                   "gradients": sorted({str(p.grad.dtype) for p in ps})}
                           for name, ps in self.parameters.items()}}}
        self.before = {name: [p.detach().clone() for p in ps] for name, ps in self.parameters.items()}

    def _post_step(self, optimizer, args, kwargs):
        import torch
        with torch.no_grad():
            updates = {}
            for name, parameters in self.parameters.items():
                before = self.before[name]
                norm = _tensor_l2(before)
                delta = _tensor_l2(p.detach().double() - old.double() for p, old in zip(parameters, before))
                updates[name] = {"parameter_l2_before": norm, "parameter_l2_after": _tensor_l2(parameters),
                                 "update_l2": delta, "relative_update": _relative_update(delta, norm)}
            self.step_info["parameter_updates"] = updates
            self.step_info["optimizer"]["state_after"] = self._moments()
        self.completed = {**self.step_info, "clips": self.pending}
        self.pending, self.before, self.step_info = [], None, None

    def pop_step(self, identifiers, *, training_record):
        """Raccorder le record pré-clipping émis par train_epoch, puis libérer le lot."""
        if not self.active or self.completed is None:
            raise RuntimeError("Aucun step terminé à consommer")
        ids, record, clips = list(identifiers), training_record, self.completed["clips"]
        if (len(ids) != len(clips) or not all(type(item) in (str, int) for item in ids)
                or len(set(ids)) != len(ids)
                or (not self.allow_microbatches and record.get("microbatch_size") != 1)
                or record.get("batch_samples") != len(clips)
                or record.get("supervised_tokens") != sum(c["terms"]["response"]["n_tokens"] for c in clips)
                or set(record.get("gradient_norms", {})) != set(self.parameters)
                or any(not math.isfinite(v) or v <= 0 for v in record["gradient_norms"].values())
                or not math.isfinite(record["loss"])):
            raise ValueError("Record train_epoch/identifiants incompatibles avec les clips observés")
        weighted_loss = sum(c["compute_loss"] * (c["terms"]["response"]["n_tokens"] / record["supervised_tokens"])
                            for c in clips)
        if not math.isclose(weighted_loss, record["loss"], rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("La loss du step n'est pas la moyenne observée pondérée par tokens")
        terms = _empty_terms()
        for identity, clip in zip(ids, clips):
            clip["clip_id" if type(identity) is str else "sample_index"] = identity
            clip["backward_token_weight"] = clip["terms"]["response"]["n_tokens"] / record["supervised_tokens"]
            _add_terms(terms, clip["terms"])
            _add_terms(self.totals, clip["terms"])
            self.original_loss_token_sum += clip["compute_loss"] * clip["terms"]["response"]["n_tokens"]
            self.max_reconstruction_error = max(self.max_reconstruction_error, clip["reconstruction_absolute_difference"])
        self.n_clips += len(clips)
        self.n_steps += 1
        result = {"schema": "pipe-training-observation-v1", **self.completed,
                  "training_record": {**record, "gradient_norms": dict(record["gradient_norms"])},
                  "gradient_norms_pre_clip": dict(record["gradient_norms"]), "terms": terms,
                  "nll_source": "original compute_loss true-answer forward; online evolving weights",
                  "binary_nll": None, "binary_nll_reason": "Requires separate fixed-checkpoint official scoring",
                  "extra_forward_count": 0, "additive_partition": ["class", "description", "eos"],
                  "first_discriminating_token_overlaps_class": True,
                  "update_scope": "Actual " + "/".join(self.parameters) + " AdamW step, including weight decay; no epsilon in relative-update ratio"}
        self.completed = None
        return result

    def epoch_summary(self, *, reset=False):
        if not self.active or self.pending or self.completed is not None or self.before is not None:
            raise RuntimeError("Consommer tous les steps avant l'agrégation d'époque")
        count = self.totals["response"]["n_tokens"]
        result = {"schema": "pipe-training-epoch-observation-v1", "n_clips": self.n_clips, "n_steps": self.n_steps,
                  "terms": {key: dict(value) for key, value in self.totals.items()},
                  "original_loss_token_weighted_mean": self.original_loss_token_sum / count if count else None,
                  "max_response_reconstruction_absolute_difference": self.max_reconstruction_error if count else None,
                  "nll_source": "Sum/count of consumed TRAIN exposures; weights evolve; not a fixed-checkpoint NLL",
                  "additive_partition": ["class", "description", "eos"],
                  "first_discriminating_token_overlaps_class": True, "binary_nll": None, "extra_forward_count": 0}
        if reset:
            self._reset_epoch()
        return result


@contextmanager
def capture_training_diagnostics(model, optimizer, *, max_clips_per_step=8,
                                 allow_lora=False, allow_microbatches=False):
    """Aucun changement de mode/RNG/dtype ; aucune copie des poids Qwen.

    La boucle demeure propriétaire de train()/eval(), backward, clipping et step.
    Lire pop_step après CHAQUE yield. Les hooks ne lisent que l'AdamW temporel ;
    pas de closure, AMP/GradScaler ni optimiseur externe à la recette existante.
    Le contexte nettoie ses méthodes/hooks même sur échec, sans restaurer les
    modes légitimement modifiés par la boucle et sans annuler les vrais updates.
    """
    if getattr(model.compute_loss, "_training_observer", False):
        raise RuntimeError("Observation TRAIN déjà active")
    audit = _TrainingDiagnostics(model, optimizer, max_clips_per_step, allow_lora, allow_microbatches)
    previous, original = vars(model).get("compute_loss", _MISSING), model.compute_loss
    def wrapped(batch):
        return audit._wrap_loss(original, batch)
    wrapped._training_observer = True
    handles = []
    try:
        model.compute_loss = wrapped
        handles.append(optimizer.register_step_pre_hook(audit._pre_step))
        handles.append(optimizer.register_step_post_hook(audit._post_step))
        yield audit
        if audit.pending or audit.completed is not None or audit.before is not None:
            raise RuntimeError("Observations TRAIN non consommées à la sortie du contexte")
    finally:
        for handle in handles:
            handle.remove()
        _restore_attribute(model, "compute_loss", previous)
        audit.pending, audit.completed, audit.before, audit.active = [], None, None, False
        audit.step_info = None
