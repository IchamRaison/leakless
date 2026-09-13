"""OpenTSLM-SP avec décodeur Qwen 3.5 officiel et masque de loss corrigé.

On réutilise encodeur, projecteur, entrelacement texte/séries et génération amont.
Les composants temporels sont initialisés ici, pas récupérés d'un checkpoint Llama.
"""
from pathlib import Path

import torch
from opentslm.model.encoder.TransformerCNNEncoder import TransformerCNNEncoder
from opentslm.model.llm.OpenTSLMSP import OpenTSLMSP
from opentslm.model.llm.TimeSeriesLLM import TimeSeriesLLM
from opentslm.model.projector.MLPProjector import MLPProjector
from opentslm.model_config import ENCODER_OUTPUT_DIM
from transformers import AutoTokenizer, Qwen3_5ForCausalLM

CLASS_CONTINUATIONS = ("leak;", "no_leak;")
SCORING_VERSION = "class-continuation-logprob-sum-softmax-v1"
CANONICAL_SCORING_VERSION = "class-continuation-logprob-sum-softmax-single-clip-v2"
AMPLITUDE_SCORING_VERSION = "class-continuation-logprob-sum-softmax-single-clip-c1text-v2"


class AcousticQwenSP(OpenTSLMSP):
    single_clip_acoustic_encoding = False
    amplitude_evidence = False

    def __init__(self, base_dir: str | Path, device: str = "cuda", *, single_clip_acoustic_encoding: bool = False,
                 amplitude_evidence: bool = False):
        if type(single_clip_acoustic_encoding) is not bool or type(amplitude_evidence) is not bool:
            raise ValueError("Politique d'encodage booléenne explicite requise")
        if amplitude_evidence and not single_clip_acoustic_encoding:
            raise ValueError("Les mesures d'amplitude C exigent l'encodage canonique par clip")
        TimeSeriesLLM.__init__(self, device)
        self.single_clip_acoustic_encoding = single_clip_acoustic_encoding
        self.amplitude_evidence = amplitude_evidence
        self.tokenizer = AutoTokenizer.from_pretrained(base_dir, local_files_only=True,
                                                       trust_remote_code=False, padding_side="right")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.llm, info = Qwen3_5ForCausalLM.from_pretrained(
            base_dir, local_files_only=True, trust_remote_code=False,
            dtype=torch.bfloat16, device_map={"": device},
            attn_implementation="eager", output_loading_info=True,
        )
        if info["missing_keys"] or info.get("mismatched_keys") or info.get("error_msgs"):
            raise RuntimeError(f"Poids Qwen incomplets : {info}")
        self.loading_info = {key: sorted(value) if isinstance(value, set) else value for key, value in info.items()}
        self.llm.requires_grad_(False)
        self.llm.config.use_cache = False
        self.encoder = TransformerCNNEncoder().to(device)
        self.projector = MLPProjector(ENCODER_OUTPUT_DIM, self.llm.config.hidden_size, device=device)
        self.patch_size = 4
        self.lora_enabled = False
        self.original_llm = None

    def configure_lora(self, spec):
        """Adaptation explicite ; les anciens chargements restent entièrement gelés."""
        targets = set(spec["target_modules"])
        found = {name.rsplit(".", 1)[-1] for name, module in self.llm.named_modules()
                 if isinstance(module, torch.nn.Linear)}
        if not targets or not targets <= found:
            raise ValueError(f"Cibles LoRA absentes : {targets - found}")
        self.enable_lora(lora_r=spec["r"], lora_alpha=spec["alpha"],
                         lora_dropout=spec["dropout"], target_modules=sorted(targets))
        actual = {name.rsplit(".", 1)[-1] for name, module in self.llm.named_modules()
                  if hasattr(module, "lora_A")}
        if actual != targets or any(p.requires_grad and "lora_" not in name
                                    for name, p in self.llm.named_parameters()):
            raise ValueError("Modules adaptés ou gel de la base incompatibles")
        self.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.llm.config.use_cache = False

    def pad_and_apply_batch(self, batch):
        framed = []
        for sample in batch:
            marker = "PIPE_NUMERIC_SERIES_SLOT"
            prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": sample["pre_prompt"] + "\n" + marker + sample["post_prompt"]}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
            pre, post = prompt.split(marker)
            framed.append({**sample, "pre_prompt": pre, "post_prompt": post})
        assemble = super().pad_and_apply_batch
        if self.single_clip_acoustic_encoding and len(framed) > 1:
            # ponytail: quatre canaux par appel encodeur, indépendamment du lot.
            # Batcher davantage seulement après preuve de parité numérique.
            # Pas de detach/no_grad : le même chemin sert aussi à l'apprentissage.
            parts = [assemble([sample]) for sample in framed]
            return (torch.nn.utils.rnn.pad_sequence([x[0] for x, _ in parts], batch_first=True),
                    torch.nn.utils.rnn.pad_sequence([m[0] for _, m in parts], batch_first=True))
        return assemble(framed)

    def compute_loss(self, batch):
        """Loss uniquement sur les tokens de réponse, sans BOS ajouté ni padding cible."""
        inputs, mask = self.pad_and_apply_batch(batch)
        answers = [sample["answer"] + self.get_eos_token() for sample in batch]
        targets = self.tokenizer(answers, return_tensors="pt", padding=True, add_special_tokens=False)
        ids = targets.input_ids.to(self.device)
        answer_mask = targets.attention_mask.to(self.device)
        embeddings = self.llm.get_input_embeddings()(ids)
        if self.single_clip_acoustic_encoding:
            # Prompts C de longueurs variables : pas de trou de padding entre
            # le vrai dernier token du prompt et la première cible de classe.
            # Le padding est ajouté seulement APRÈS la réponse entière.
            sequences, labels, attention = [], [], []
            for prefix, prefix_mask, answer, answer_ids, target_mask in zip(
                    inputs, mask, embeddings, ids, answer_mask):
                prefix = prefix[prefix_mask.bool()]
                answer, answer_ids = answer[target_mask.bool()], answer_ids[target_mask.bool()]
                if not len(prefix) or not len(answer_ids):
                    raise ValueError("Prompt et réponse non vides requis")
                sequences.append(torch.cat((prefix, answer), dim=0))
                labels.append(torch.cat((torch.full((len(prefix),), -100, dtype=torch.long,
                                                    device=ids.device), answer_ids)))
                attention.append(torch.ones(len(prefix) + len(answer_ids), dtype=mask.dtype, device=mask.device))
            return self.llm(
                inputs_embeds=torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True),
                attention_mask=torch.nn.utils.rnn.pad_sequence(attention, batch_first=True),
                labels=torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100),
                use_cache=False, return_dict=True,
            ).loss
        # V1 conserve son assemblage historique, y compris sa politique de lot.
        labels = ids.masked_fill(answer_mask == 0, -100)
        prefix_labels = torch.full(mask.shape, -100, device=self.device, dtype=torch.long)
        return self.llm(
            inputs_embeds=torch.cat((inputs, embeddings), dim=1),
            attention_mask=torch.cat((mask, answer_mask), dim=1),
            labels=torch.cat((prefix_labels, labels), dim=1),
            use_cache=False, return_dict=True,
        ).loss

    def _validate_inference_batch(self, batch):
        if not batch or any(set(sample) != {"pre_prompt", "post_prompt", "time_series", "time_series_text"}
                            for sample in batch):
            raise ValueError("Champs d'inférence non autorisés (answer/label/métadonnées interdits)")
        if self.amplitude_evidence:
            from pipe.tslm.preprocessing import AMPLITUDE_TEXT_PREFIX
            if any(AMPLITUDE_TEXT_PREFIX not in sample["pre_prompt"] for sample in batch):
                raise ValueError("Preuves d'amplitude absentes du prompt du modèle C")

    def scoring_spec(self) -> dict:
        """Contrat exact : tokenisation séparée, sans espace initial, BOS ni EOS.

        Le point-virgule termine chaque classe ; aucune description n'est scorée.
        Les sommes ne sont pas divisées par la longueur, même si elle diffère.
        La probabilité est relative aux deux continuations, pas calibrée terrain.
        """
        token_ids = [self.tokenizer.encode(text, add_special_tokens=False)
                     for text in CLASS_CONTINUATIONS]
        if any(not ids or self.tokenizer.decode(ids) != text
               for text, ids in zip(CLASS_CONTINUATIONS, token_ids)):
            raise ValueError("Tokenisation de classe vide ou non réversible")
        if token_ids[0] == token_ids[1]:
            raise ValueError("Les deux classes ont la même tokenisation")
        spec = {"version": CANONICAL_SCORING_VERSION if self.single_clip_acoustic_encoding else SCORING_VERSION,
                "class_continuations": list(CLASS_CONTINUATIONS),
                "class_token_ids": token_ids, "class_token_counts": list(map(len, token_ids)),
                "aggregation": "sum", "length_normalization": False,
                "class_terminator": ";", "includes_eos": False,
                "includes_description": False, "calibration": "none"}
        if self.single_clip_acoustic_encoding:
            spec["acoustic_batching"] = "one_clip_four_channels"
        if self.amplitude_evidence:
            from pipe.tslm.preprocessing import amplitude_spec
            if not self.single_clip_acoustic_encoding:
                raise ValueError("Scoring C incompatible avec la politique de lot legacy")
            spec["version"] = AMPLITUDE_SCORING_VERSION
            spec["amplitude_evidence"] = amplitude_spec()
        return spec

    @torch.inference_mode()
    def score_class_logprobs(self, batch) -> torch.Tensor:
        """Retourne [n,2] log P(continuation | même prompt, même signal).

        Les positions scorées précèdent les tokens cibles d'un cran (causal LM).
        Le padding du prompt est retiré avant d'ajouter les continuations ; seul
        le padding final des réponses subsiste et ne contribue jamais à la somme.
        """
        self._validate_inference_batch(batch)
        if self.training:
            raise ValueError("Appeler model.eval() avant le scoring")
        spec = self.scoring_spec()
        candidates = [torch.tensor(ids, device=self.device, dtype=torch.long)
                      for ids in spec["class_token_ids"]]
        ids = torch.nn.utils.rnn.pad_sequence(
            candidates, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        lengths = torch.tensor([len(candidate) for candidate in candidates], device=self.device)
        target_mask = torch.arange(ids.shape[1], device=self.device)[None, :] < lengths[:, None]
        targets = self.llm.get_input_embeddings()(ids)
        inputs, mask = self.pad_and_apply_batch(batch)
        scores = []
        # ponytail: deux candidats par clip ; batcher les clips si le débit le demande.
        for prefix, prefix_mask in zip(inputs, mask):
            prefix = prefix[prefix_mask.bool()]
            prefix_length = len(prefix)
            if not prefix_length:
                raise ValueError("Prompt vide")
            output = self.llm(
                inputs_embeds=torch.cat((prefix[None].expand(2, -1, -1), targets), dim=1),
                attention_mask=torch.cat((torch.ones((2, prefix_length), device=self.device,
                                                      dtype=torch.long), target_mask.long()), dim=1),
                use_cache=False, return_dict=True,
            )
            logits = output.logits[:, prefix_length - 1:prefix_length + ids.shape[1] - 1].float()
            token_logprobs = torch.log_softmax(logits, dim=-1).gather(-1, ids[..., None]).squeeze(-1)
            selected = token_logprobs.masked_fill(~target_mask, 0)
            if not torch.isfinite(selected).all():
                raise ValueError("Log-probabilités non finies : aucun score de remplacement")
            scores.append(selected.double().sum(dim=-1))
        return torch.stack(scores)

    def score_probability_leak(self, batch) -> list[float]:
        """Softmax stable à deux classes, scores continus bruts sans seuil."""
        return torch.softmax(self.score_class_logprobs(batch), dim=-1)[:, 0].cpu().tolist()

    def generate(self, batch, max_new_tokens=48, **kwargs):
        self._validate_inference_batch(batch)
        with torch.inference_mode():
            return super().generate(batch, max_new_tokens=max_new_tokens, do_sample=False,
                                    use_cache=True, pad_token_id=self.tokenizer.pad_token_id, **kwargs)
