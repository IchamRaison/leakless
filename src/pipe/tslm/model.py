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


class AcousticQwenSP(OpenTSLMSP):
    def __init__(self, base_dir: str | Path, device: str = "cuda"):
        TimeSeriesLLM.__init__(self, device)
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
        return super().pad_and_apply_batch(framed)

    def compute_loss(self, batch):
        """Loss uniquement sur les tokens de réponse, sans BOS ajouté ni padding cible."""
        inputs, mask = self.pad_and_apply_batch(batch)
        answers = [sample["answer"] + self.get_eos_token() for sample in batch]
        targets = self.tokenizer(answers, return_tensors="pt", padding=True, add_special_tokens=False)
        ids = targets.input_ids.to(self.device)
        answer_mask = targets.attention_mask.to(self.device)
        embeddings = self.llm.get_input_embeddings()(ids)
        labels = ids.masked_fill(answer_mask == 0, -100)
        prefix_labels = torch.full(mask.shape, -100, device=self.device, dtype=torch.long)
        return self.llm(
            inputs_embeds=torch.cat((inputs, embeddings), dim=1),
            attention_mask=torch.cat((mask, answer_mask), dim=1),
            labels=torch.cat((prefix_labels, labels), dim=1),
            use_cache=False, return_dict=True,
        ).loss

    def generate(self, batch, max_new_tokens=48, **kwargs):
        if any(set(sample) != {"pre_prompt", "post_prompt", "time_series", "time_series_text"} for sample in batch):
            raise ValueError("Champs d'inférence non autorisés (answer/label/métadonnées interdits)")
        with torch.inference_mode():
            return super().generate(batch, max_new_tokens=max_new_tokens, do_sample=False,
                                    use_cache=True, pad_token_id=self.tokenizer.pad_token_id, **kwargs)
