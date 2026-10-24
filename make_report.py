"""Collects results/*.json into a table, a plot and a per-case breakdown.

Usage: python make_report.py
"""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).parent
RES = ROOT / "results"
KINDS = ["case", "adj_agree", "verb_agree"]
NICE = {"case": "noun case", "adj_agree": "adjective agreement", "verb_agree": "subject-verb agreement"}


def main():
    pairs = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / "data" / "pairs.jsonl", encoding="utf-8")}
    runs = [json.load(open(p)) for p in sorted(RES.glob("*.json"))]
    runs = [r for r in runs if "accuracy" in r]  # skip the token check file
    runs.sort(key=lambda r: r["accuracy"]["all"])

    table = pd.DataFrame([{"model": r["model"], "params (M)": r["params_m"], "all": r["accuracy"]["all"],
                           **{NICE[k]: r["accuracy"][k] for k in KINDS}} for r in runs])
    table.round(3).to_csv(RES / "summary.csv", index=False)
    print(table.round(3).to_string(index=False))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    width = 0.8 / len(KINDS)
    colors = ["#4c72b0", "#dd8452", "#55a868"]
    for i, k in enumerate(KINDS):
        xs = [j + i * width for j in range(len(table))]
        ax.bar(xs, table[NICE[k]], width, label=NICE[k], color=colors[i])
    ax.axhline(0.5, color="grey", lw=1, ls="--")
    ax.text(-0.35, 0.505, "coin flip", color="grey", fontsize=8)
    ax.set_xticks([j + width for j in range(len(table))])
    ax.set_xticklabels([m.split("/")[-1] for m in table["model"]], rotation=15, fontsize=8)
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("share of pairs where the correct sentence wins")
    ax.set_title("Which sentence is correct Lithuanian? Small language models on 4,636 minimal pairs")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(RES / "accuracy.png", dpi=130)
    plt.close(fig)

    # sanity check: does the model only win because the bad word is split into more tokens?
    # (more tokens usually means a lower probability, whatever the grammar)
    from transformers import AutoTokenizer

    best = runs[-1]
    if best["params_m"] > 0:
        tok = AutoTokenizer.from_pretrained(best["model"])
        same, more = [], []
        for pid, ok in best["per_pair"].items():
            p = pairs[pid]
            n_good, n_bad = len(tok(" " + p["word"])["input_ids"]), len(tok(" " + p["replacement"])["input_ids"])
            (same if n_good == n_bad else more if n_bad > n_good else []).append(ok)
        print(f"\n{best['model']}: same token count {sum(same) / len(same):.3f} ({len(same)} pairs), "
              f"bad word has more tokens {sum(more) / len(more):.3f} ({len(more)} pairs)")
        json.dump({"model": best["model"], "same_tokens": [sum(same) / len(same), len(same)],
                   "bad_more_tokens": [sum(more) / len(more), len(more)]},
                  open(RES / "token_count_check.json", "w"), indent=1)

    # which case changes are hard, for the best model
    by_change = defaultdict(list)
    for pid, ok in best["per_pair"].items():
        p = pairs[pid]
        if p["kind"] == "case":
            by_change[p["change"]].append(ok)
    rows = [{"change": c, "pairs": len(v), "accuracy": sum(v) / len(v)} for c, v in by_change.items() if len(v) >= 30]
    hard = pd.DataFrame(rows).sort_values("accuracy")
    hard.round(3).to_csv(RES / f"case_changes_{best['model'].split('/')[-1]}.csv", index=False)
    print(f"\ncase changes for {best['model']} (30+ pairs):")
    print(hard.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
