"""Nouvelle tâche OpenTSLM/Qwen : dynamique C1, pas nouvelle classification de fuite."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from pipe.temporal_language import LABELS, facts, model_sample, file_hash, decision_logits, TemporalNarrator
from pipe.temporal_model import C1Detector, decode_pcm
from run_campaign import split_loader, load_expected_audio_md5, verify_audio_bytes
import numpy as np


def write(path, value):
    with path.open("x") as file:
        json.dump(value, file, indent=2, allow_nan=False)
        file.write("\n")


def scenarios(bank, count, seed):
    rng = np.random.default_rng(seed)
    low, high = bank[bank[:,9] <= .4], bank[bank[:,9] >= .8]
    if not len(low) or not len(high):
        raise ValueError("Train C1 sans témoins hauts/bas pour la recette")
    values, lengths, targets = [], [], []
    for label in LABELS:
        for _ in range(count):
            n = int(rng.integers(31,65))
            flags = np.zeros(n, dtype=bool)
            if label == "brief":
                flags[-int(rng.integers(1,13)):] = True
            elif label == "persistent":
                flags[-int(rng.integers(31,n+1)):] = True
            elif label == "intermittent":
                flags[2:7] = True
                flags[-int(rng.integers(1,13)):] = True
            elif label == "ended":
                stop = int(rng.integers(10,n-4))
                flags[max(0,stop-int(rng.integers(3,10))):stop] = True
            x = np.asarray([(high if flag else low)[rng.integers(len(high if flag else low))] for flag in flags])
            assert facts(x)["pattern"] == label
            values.append(np.pad(x, ((64-n,0),(0,0))))
            lengths.append(n); targets.append(LABELS.index(label))
    return np.asarray(values), np.asarray(lengths), np.asarray(targets)


def prepare(args, config):
    args.output.mkdir(parents=True, exist_ok=False)
    detector = C1Detector(args.c1, args.c1_sha256)
    md5 = load_expected_audio_md5(ROOT / "manifests")
    split = split_loader.load_split(ROOT / "manifests")
    bank = []
    for clip in sorted(split.fold("train"), key=lambda r:r.clip_id):
        path = (args.data_root / split.path_of(clip.clip_id)).resolve()
        if not path.is_relative_to(args.data_root.resolve()):
            raise ValueError("Chemin hors racine")
        raw = path.read_bytes(); verify_audio_bytes(raw, clip.clip_id, md5)
        features, score = detector.score(decode_pcm(raw))
        bank.append(np.r_[features,score])
    bank = np.asarray(bank)
    mean, scale = bank.mean(0), bank.std(0); scale[scale == 0] = 1
    np.savez(args.output / "normalization.npz", mean=mean, scale=scale)
    for fold,count,seed in (("train",config["train_per_pattern"],config["seed"]),
                            ("development",config["development_per_pattern"],config["development_seed"])):
        x,n,y = scenarios(bank,count,seed)
        np.savez(args.output / (fold+".npz"), values=x, lengths=n, targets=y)
    write(args.output / "registration.json", {"config":config,"source_revision":args.revision,
        "c1_bundle_sha256":args.c1_sha256,"c1_version":detector.version,"zenodo_train_clips":len(bank),
        "artificial_chronology":True,"development_not_independent_acquisitions":True,
        "official_val_test_or_aghashahi_used":False,
        "base_files":{p.name:file_hash(p) for p in sorted(args.base.iterdir())
            if p.is_file() and (p.suffix in (".json",".safetensors") or p.name=="tokenizer.model")},
        "files":{name:file_hash(args.output/name) for name in ("train.npz","development.npz","normalization.npz")}})


def fit(args, config):
    import torch
    from pipe.tslm.model import AcousticQwenSP
    from pipe.tslm.train import tensor_state_hash
    registration = json.loads((args.output / "registration.json").read_text())
    assert config == registration["config"]
    for name,sha in registration["files"].items():
        assert file_hash(args.output/name) == sha
    for name,sha in registration["base_files"].items():
        assert file_hash(args.base/name) == sha
    if not torch.cuda.is_available():
        raise RuntimeError("H100 CUDA requise")
    torch.manual_seed(config["seed"]); torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    model = AcousticQwenSP(args.base, single_clip_acoustic_encoding=True)
    frozen = tensor_state_hash(model.llm)
    with np.load(args.output/"normalization.npz") as z: mean,scale=z["mean"],z["scale"]
    with np.load(args.output/"train.npz") as z: x,n,y=z["values"],z["lengths"],z["targets"]
    samples = [model_sample(v[-int(length):],mean,scale) for v,length in zip(x,n)]
    optimizer = torch.optim.AdamW([{"params":model.encoder.parameters(),"lr":config["encoder_lr"]},
                                  {"params":model.projector.parameters(),"lr":config["projector_lr"]}], weight_decay=.01)
    rng = np.random.default_rng(config["seed"])
    order = rng.permutation(len(y)); started=time.monotonic()
    model.train(); model.llm.eval()
    with (args.output/"training.jsonl").open("x") as log:
        for step in range(config["max_steps"]):
            if step % len(y) == 0: order=rng.permutation(len(y))
            index=int(order[step % len(y)])
            optimizer.zero_grad(set_to_none=True)
            logits=decision_logits(model,samples[index])
            loss=torch.nn.functional.cross_entropy(logits[None],torch.tensor([y[index]],device="cuda"))
            if not torch.isfinite(loss): raise ValueError("Loss non finie")
            loss.backward()
            params=[p for p in model.parameters() if p.requires_grad]
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in params):
                raise ValueError("Gradient invalide")
            norm=torch.nn.utils.clip_grad_norm_(params,config["gradient_clip"],error_if_nonfinite=True)
            optimizer.step()
            row={"step":step+1,"decision_nll":float(loss.detach()),"gradient_norm":float(norm),
                 "elapsed_seconds":time.monotonic()-started}
            log.write(json.dumps(row)+"\n"); log.flush()
            if step % 20 == 0: print(json.dumps(row),flush=True)
    assert not any(p.requires_grad or p.grad is not None for p in model.llm.parameters())
    assert tensor_state_hash(model.llm) == frozen
    torch.save({"encoder_state":model.encoder.state_dict(),"projector_state":model.projector.state_dict()},args.output/"temporal.pt")
    model.eval()
    with torch.inference_mode():
        references=[torch.softmax(decision_logits(model,s),-1).cpu().tolist() for s in samples[:5]]
    write(args.output/"metadata.json", {"config":config,"base":str(args.base.resolve()),
        "base_files":registration["base_files"], "registration_sha256":file_hash(args.output/"registration.json"),
        "files":{name:file_hash(args.output/name) for name in ("temporal.pt","normalization.npz")},
        "c1_bundle_sha256":registration["c1_bundle_sha256"],"frozen_llm_tensor_sha256":frozen,
        "reference_scores":references,"supervision":"rule-derived artificial chronology","field_validated":False})
    print("metadata_sha256",file_hash(args.output/"metadata.json"),flush=True)


def evaluate(args):
    narrator=TemporalNarrator(args.output,file_hash(args.output/"metadata.json"))
    with np.load(args.output/"train.npz") as z:
        x,n=z["values"][:5],z["lengths"][:5]
    actual=[list(narrator.describe(v[-int(length):])["candidate_scores"].values()) for v,length in zip(x,n)]
    delta=float(np.max(np.abs(np.asarray(actual)-narrator.metadata["reference_scores"])))
    if delta>1e-4: raise ValueError(f"Reload divergent : {delta}")
    with np.load(args.output/"development.npz") as z: x,n,y=z["values"],z["lengths"],z["targets"]
    results=[]
    for i,(v,length,target) in enumerate(zip(x,n,y)):
        history=v[-int(length):]
        result=narrator.describe(history)
        reverse=narrator.describe(history[::-1].copy())
        results.append({"id":i,"expected":LABELS[target],"actual":result,
            "reversal_expected":facts(history[::-1])["pattern"],"reversal_actual":reverse["model_pattern"]})
    report={"n":len(results),"raw_model_correct":sum(r["actual"]["model_pattern"]==r["expected"] for r in results),
        "rule_baseline_correct":len(results),"safe_output_correct":sum(r["actual"]["facts"]["pattern"]==r["expected"] for r in results),
        "fallback_count":sum(r["actual"]["fallback_used"] for r in results),"reload_max_diff":delta,
        "reversal_correct":sum(r["reversal_actual"]==r["reversal_expected"] for r in results),
        "artificial_chronology":True,"field_validated":False,"results":results}
    write(args.output/"evaluation.json",report)
    print(json.dumps({k:v for k,v in report.items() if k!="results"}),flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=("prepare","train","evaluate"))
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--base",type=Path)
    parser.add_argument("--c1",type=Path)
    parser.add_argument("--c1-sha256")
    parser.add_argument("--data-root",type=Path)
    parser.add_argument("--revision")
    parser.add_argument("--config",type=Path,default=ROOT/"configs/temporal/language.json")
    args=parser.parse_args(); config=json.loads(args.config.read_text())
    if args.phase=="prepare": prepare(args,config)
    elif args.phase=="train": fit(args,config)
    else: evaluate(args)
