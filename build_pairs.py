"""Builds Lithuanian minimal pairs from the UD ALKSNIS treebank.

A pair is a real sentence from the treebank and a copy of it where exactly one word is swapped
for another real form of the same word, so the copy becomes ungrammatical. Three kinds:

  case       a noun gets a different case (same number), e.g. "atlyginimus" -> "atlyginimai"
  adj_agree  an adjective that describes a noun gets a different case, so it no longer agrees
  verb_agree a finite verb gets a different person or number than its subject

Every replacement form is a form the treebank itself contains. A form is skipped if it is
ambiguous, i.e. the same spelling also shows up with the original feature somewhere in the
treebank (Lithuanian has many endings that are shared between cases).

Usage: python build_pairs.py   -> data/pairs.jsonl
"""
import json
import random
import urllib.request
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent
RAW = ROOT / "data" / "raw"
UD = "https://raw.githubusercontent.com/UniversalDependencies/UD_Lithuanian-ALKSNIS/master/"
SPLITS = ["lt_alksnis-ud-train.conllu", "lt_alksnis-ud-dev.conllu", "lt_alksnis-ud-test.conllu"]
MAX_PER_SENT = 1  # at most one pair of each kind per sentence, so long sentences don't dominate


def download():
    RAW.mkdir(parents=True, exist_ok=True)
    for name in SPLITS:
        if not (RAW / name).exists():
            urllib.request.urlretrieve(UD + name, RAW / name)


def read_conllu(path):
    sent, sid = [], None
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if line.startswith("# sent_id"):
            sid = line.split("=", 1)[1].strip()
        elif not line:
            if sent:
                yield sid, sent
            sent = []
        elif not line.startswith("#"):
            c = line.split("\t")
            if "-" in c[0] or "." in c[0]:
                continue
            feats = dict(f.split("=", 1) for f in c[5].split("|")) if c[5] != "_" else {}
            sent.append({"id": int(c[0]), "form": c[1], "lemma": c[2], "upos": c[3], "feats": feats,
                         "head": int(c[6]), "deprel": c[7], "space": "SpaceAfter=No" not in c[9]})
    if sent:
        yield sid, sent


def text(tokens, replace_at=None, new_form=None):
    out = []
    for t in tokens:
        form = new_form if t["id"] == replace_at else t["form"]
        out.append(form + (" " if t["space"] else ""))
    return "".join(out).strip()


def keep_case(src, new):
    # keep a capital first letter if the original word had one
    return new[0].upper() + new[1:] if src[:1].isupper() else new


def main():
    download()
    sents = [s for name in SPLITS for s in read_conllu(RAW / name)]

    # every form of every lemma, with the features it was seen with
    forms = defaultdict(lambda: defaultdict(set))   # (lemma, upos) -> surface -> set of feature tuples
    for _, toks in sents:
        for t in toks:
            key = tuple(sorted(t["feats"].items()))
            forms[(t["lemma"].lower(), t["upos"])][t["form"].lower()].add(key)

    def candidates(t, change, keep):
        """Other forms of the same word that differ in `change` but keep the features in `keep`."""
        mine = t["feats"]
        out = []
        for surface, seen in forms[(t["lemma"].lower(), t["upos"])].items():
            if surface == t["form"].lower():
                continue
            # ambiguous: this spelling was also seen with the original value of `change`
            if any(dict(f).get(change) == mine.get(change) for f in seen):
                continue
            for f in seen:
                f = dict(f)
                if f.get(change) and f.get(change) != mine.get(change) and all(f.get(k) == mine.get(k) for k in keep):
                    out.append((surface, f[change]))
                    break
        return sorted(out)

    rng = random.Random(0)
    pairs = []
    for sid, toks in sents:
        if len(toks) < 4 or len(toks) > 40:
            continue
        by_id = {t["id"]: t for t in toks}
        options = {"case": [], "adj_agree": [], "verb_agree": []}
        for t in toks:
            f = t["feats"]
            if t["upos"] == "NOUN" and "Case" in f and "Number" in f:
                for surface, new_case in candidates(t, "Case", ["Number"]):
                    options["case"].append((t, surface, f"{f['Case']}->{new_case}"))
            if t["upos"] == "ADJ" and t["deprel"] == "amod" and by_id.get(t["head"], {}).get("upos") == "NOUN" and "Case" in f:
                for surface, new_case in candidates(t, "Case", ["Number", "Gender", "Degree", "Definite"]):
                    options["adj_agree"].append((t, surface, f"{f['Case']}->{new_case}"))
            if t["upos"] in ("VERB", "AUX") and f.get("VerbForm") == "Fin" and "Person" in f:
                if any(x["head"] == t["id"] and x["deprel"].startswith("nsubj") for x in toks):
                    for surface, new_p in candidates(t, "Person", ["Number", "Tense", "Mood"]):
                        options["verb_agree"].append((t, surface, f"Person {f['Person']}->{new_p}"))
                    if "Number" in f:
                        for surface, new_n in candidates(t, "Number", ["Person", "Tense", "Mood"]):
                            options["verb_agree"].append((t, surface, f"Number {f['Number']}->{new_n}"))
        for kind, opts in options.items():
            for t, surface, change in rng.sample(opts, min(MAX_PER_SENT, len(opts))):
                pairs.append({"id": f"{sid}:{kind}:{t['id']}", "kind": kind, "change": change,
                              "word": t["form"], "replacement": keep_case(t["form"], surface),
                              "good": text(toks), "bad": text(toks, t["id"], keep_case(t["form"], surface))})

    out = ROOT / "data" / "pairs.jsonl"
    with open(out, "w", encoding="utf-8") as fh:
        for p in pairs:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    counts = defaultdict(int)
    for p in pairs:
        counts[p["kind"]] += 1
    print(f"{len(sents)} sentences -> {len(pairs)} pairs {dict(counts)} -> {out}")


if __name__ == "__main__":
    main()
