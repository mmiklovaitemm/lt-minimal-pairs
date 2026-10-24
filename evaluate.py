"""Scores language models on the Lithuanian minimal pairs.

For every pair the model gives a log-probability to the good and to the bad sentence. The model
"gets it right" when the good sentence is more likely. Nothing is generated, so even small models
run fine on a laptop.

Usage: python evaluate.py Qwen/Qwen2.5-0.5B [more model names]   -> results/<model>.json
       python evaluate.py --freq                                 -> word frequency baseline
"""
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).parent
OUT = ROOT / "results"
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
BATCH = 4  # bigger batches run out of memory on a 16 GB Mac because of the large vocabularies


def load_pairs():
    return [json.loads(line) for line in open(ROOT / "data" / "pairs.jsonl", encoding="utf-8")]


@torch.no_grad()
def sentence_logprobs(model, tok, sentences):
    """Sum of log P(token | previous tokens) over the whole sentence."""
    bos = tok.bos_token or tok.eos_token
    enc = tok([bos + s for s in sentences], return_tensors="pt", padding=True, add_special_tokens=False).to(DEVICE)
    logits = model(**enc).logits[:, :-1]
    target = enc["input_ids"][:, 1:]
    # cross_entropy gives -log P(target) per token without building the full log-softmax table
    nll = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), target.reshape(-1),
                                            reduction="none").view(target.shape)
    return (-(nll * enc["attention_mask"][:, 1:])).sum(1).cpu().tolist()


def summarize(pairs, correct):
    by_kind = defaultdict(list)
    for p, c in zip(pairs, correct):
        by_kind[p["kind"]].append(c)
    out = {"all": sum(correct) / len(correct)}
    out.update({k: sum(v) / len(v) for k, v in sorted(by_kind.items())})
    return out


def run_model(name, pairs):
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).to(DEVICE).eval()
    # sentences of similar length go into the same batch, so there is little padding to compute
    order = sorted(range(len(pairs)), key=lambda i: len(pairs[i]["good"]))
    correct = [None] * len(pairs)
    for n, i in enumerate(range(0, len(order), BATCH)):
        idx = order[i:i + BATCH]
        chunk = [pairs[j] for j in idx]
        scores = sentence_logprobs(model, tok, [p["good"] for p in chunk] + [p["bad"] for p in chunk])
        for j, g, b in zip(idx, scores[:len(chunk)], scores[len(chunk):]):
            correct[j] = g > b
        if (i + BATCH) % 1000 < BATCH:
            print(f"  {name}: {min(i + BATCH, len(order))}/{len(order)} pairs, {time.time() - t0:.0f} s", flush=True)
    res = {"model": name, "params_m": round(sum(p.numel() for p in model.parameters()) / 1e6),
           "accuracy": summarize(pairs, correct), "seconds": round(time.time() - t0),
           "per_pair": dict(zip([p["id"] for p in pairs], correct))}
    return res


def run_freq(pairs):
    """Baseline with no language model: pick the sentence whose changed word is more frequent in the treebank."""
    counts = Counter()
    for line in open(ROOT / "data" / "raw" / "lt_alksnis-ud-train.conllu", encoding="utf-8"):
        c = line.split("\t")
        if len(c) == 10 and c[0].isdigit():
            counts[c[1].lower()] += 1
    correct = [counts[p["word"].lower()] > counts[p["replacement"].lower()] for p in pairs]
    return {"model": "word frequency (no LM)", "params_m": 0, "accuracy": summarize(pairs, correct),
            "per_pair": dict(zip([p["id"] for p in pairs], correct))}


def main():
    OUT.mkdir(exist_ok=True)
    pairs = load_pairs()
    jobs = sys.argv[1:]
    for job in jobs:
        res = run_freq(pairs) if job == "--freq" else run_model(job, pairs)
        name = "word_frequency" if job == "--freq" else job.replace("/", "__")
        json.dump(res, open(OUT / f"{name}.json", "w"), indent=1)
        acc = res["accuracy"]
        print(f"{res['model']}: all {acc['all']:.3f} | " + " ".join(f"{k} {v:.3f}" for k, v in acc.items() if k != "all"),
              flush=True)


if __name__ == "__main__":
    main()
