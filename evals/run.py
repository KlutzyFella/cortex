"""Eval runner: score backends against the golden set, write reports.

Assumes the corpus is seeded (see seed.py). Prints a markdown table, writes
report.json + report.md into --report-dir, and exits non-zero only under
--enforce enforce with a significant McNemar regression (default is warn).

Credential needs: dense needs DB_* (no LLM key anywhere in Phase 1).
"""

import argparse
import hashlib
import json
import os
import sys

from evalkit.backends import BACKENDS, BM25Backend, DenseBackend, RandomBackend
from evalkit.chunks import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_EMBEDDING_MODEL,
    chunk_corpus,
)
from evalkit.metrics import (
    hit_at_k,
    mcnemar,
    mean_ci,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

PRIMARY_K = 5


def _bootstrap(orig_cwd: str, args, *attrs: str) -> None:
    """Fix working directory and resolve relative paths.

    Bare filenames (the defaults) resolve next to this script; explicit
    relative paths resolve against the caller's cwd. Without this,
    `--compare evals/baseline.json` from the repo root silently becomes
    `evals/evals/baseline.json` — which cost one confused debugging session.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    for attr in attrs:
        val = getattr(args, attr)
        if val and not os.path.isabs(val):
            base = orig_cwd if os.path.dirname(val) else here
            setattr(args, attr, os.path.join(base, val))
    os.chdir(here)


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def verify_meta(meta: dict, corpus_path: str, chunk_size: int,
                chunk_overlap: int, model: str) -> None:
    """Refuse to score when the world drifted from the golden set's pins."""
    problems = []
    if meta.get("chunk_size") != chunk_size:
        problems.append(
            f"chunk_size: golden pins {meta.get('chunk_size')}, "
            f"run uses {chunk_size}"
        )
    if meta.get("chunk_overlap") != chunk_overlap:
        problems.append(
            f"chunk_overlap: golden pins {meta.get('chunk_overlap')}, "
            f"run uses {chunk_overlap}"
        )
    if meta.get("embedding_model") != model:
        problems.append(
            f"embedding_model: golden pins {meta.get('embedding_model')}, "
            f"run uses {model}"
        )
    if meta.get("corpus_sha256") != _sha256_file(corpus_path):
        problems.append("corpus.jsonl changed since golden.meta.json was pinned")
    if problems:
        raise SystemExit(
            "golden/meta mismatch (regenerate the golden set, do not relax "
            "the pins):\n- " + "\n- ".join(problems)
        )


def score_backend(backend, positives, negatives, ks, max_k):
    per_k = {k: {"recall": [], "precision": [], "ndcg": []} for k in ks}
    mrr, hits = [], []
    for item in positives:
        retrieved = backend.search(item["query"], max_k)
        keys = [(r.doc_id, r.chunk_index) for r in retrieved]
        gold = [(g["doc_id"], g["chunk_index"]) for g in item["gold"]]
        for k in ks:
            per_k[k]["recall"].append(recall_at_k(keys, gold, k))
            per_k[k]["precision"].append(precision_at_k(keys, gold, k))
            per_k[k]["ndcg"].append(ndcg_at_k(keys, gold, k))
        mrr.append(reciprocal_rank(keys, gold))
        hits.append(hit_at_k(keys, gold, PRIMARY_K))
    neg_scores = []
    for item in negatives:
        top = backend.search(item["query"], 1)
        neg_scores.append(top[0].score if top else None)
    return {"per_k": per_k, "mrr": mrr, "hits": hits, "neg_scores": neg_scores}


def summarize(scored, ks):
    out = {"k": {}, "mrr": {}, "n": 0}
    for k in ks:
        for m in ("recall", "precision", "ndcg"):
            mean, half = mean_ci(scored["per_k"][k][m])
            out["k"].setdefault(str(k), {})[m] = {"mean": mean, "ci95": half}
    mean, half = mean_ci(scored["mrr"])
    out["mrr"] = {"mean": mean, "ci95": half}
    out["n"] = len(scored["hits"])
    return out


def _fmt_num(scores) -> str:
    vals = [s for s in scores if s is not None]
    if not vals:
        return "n/a"
    return f"min {min(vals):.2f} / mean {sum(vals)/len(vals):.2f} / max {max(vals):.2f}"


def render_markdown(results, comparisons, meta, n_pos, n_neg) -> str:
    lines = [
        "# cortex retrieval eval",
        "",
        f"positives: {n_pos} · negatives: {n_neg} · "
        f"corpus {meta.get('corpus_sha256', '?')[:12]} · "
        f"chunk {meta.get('chunk_size')}/{meta.get('chunk_overlap')} · "
        f"embed {meta.get('embedding_model')} · gen/judge: n/a (Phase 3)",
        "",
        "| backend | recall@5 | recall@10 | recall@20 | MRR | NDCG@10 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for name, res in results.items():
        s = res["summary"]
        lines.append(
            f"| {name} | {s['k']['5']['recall']['mean']:.2f}±{s['k']['5']['recall']['ci95']:.2f}"
            f" | {s['k']['10']['recall']['mean']:.2f}±{s['k']['10']['recall']['ci95']:.2f}"
            f" | {s['k']['20']['recall']['mean']:.2f}±{s['k']['20']['recall']['ci95']:.2f}"
            f" | {s['mrr']['mean']:.2f}±{s['mrr']['ci95']:.2f}"
            f" | {s['k']['10']['ndcg']['mean']:.2f}±{s['k']['10']['ndcg']['ci95']:.2f} |"
        )
    lines += ["", "## vs baseline (McNemar on hit@5, paired)"]
    if not comparisons:
        lines.append("no --compare given: no gate computed.")
    for name, c in comparisons.items():
        lines.append(
            f"- {name}: flips to miss b={c['b']}, to hit c={c['c']}, "
            f"stat={c['stat']:.2f}, p={c['p']:.3f} → {c['verdict']}"
        )
    lines += ["",
              "## negatives as score distributions (no threshold in v1)",
              "top-1 similarity of answerable vs unanswerable queries: the gap",
              "between these two distributions is what a future threshold tunes.",
              ""]
    for name, res in results.items():
        lines.append(f"- {name} positives top-1: {_fmt_num(res['pos_top1'])}")
        lines.append(f"- {name} negatives top-1: {_fmt_num(res['neg_scores'])}")
    lines += ["",
              "_CIs are 95% normal approximations. At n≈45 a ±0.03 wiggle is",
              "noise; only McNemar p<0.05 with consistent direction gates._"]
    return "\n".join(lines) + "\n"


def main() -> int:
    orig_cwd = os.getcwd()
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="corpus.jsonl")
    ap.add_argument("--golden", default="golden.jsonl")
    ap.add_argument("--meta", default="golden.meta.json")
    ap.add_argument("--retriever", choices=(*BACKENDS, "all"), default="all")
    ap.add_argument("--k", default="5,10,20")
    ap.add_argument("--compare", default=None)
    ap.add_argument(
        "--enforce",
        choices=("warn", "enforce"),
        default=os.environ.get("EVAL_ENFORCE", "warn"),
    )
    ap.add_argument("--write-baseline", default=None)
    ap.add_argument("--report-dir", default="reports")
    args = ap.parse_args()
    if args.enforce not in ("warn", "enforce"):
        # argparse does not validate defaults against choices, so a typo'd
        # EVAL_ENFORCE would otherwise degrade silently to warn behavior.
        ap.error(f"EVAL_ENFORCE must be 'warn' or 'enforce', got {args.enforce!r}")
    _bootstrap(orig_cwd, args, "corpus", "golden", "meta", "compare",
               "write_baseline", "report_dir")

    ks = sorted({int(x) for x in args.k.split(",") if int(x) > 0})
    max_k = max(ks)
    meta = json.load(open(args.meta))
    verify_meta(meta, args.corpus, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP,
                DEFAULT_EMBEDDING_MODEL)
    corpus = [json.loads(line) for line in open(args.corpus) if line.strip()]
    golden = [json.loads(line) for line in open(args.golden) if line.strip()]
    positives = [g for g in golden if not g.get("negative")]
    negatives = [g for g in golden if g.get("negative")]

    chunks = chunk_corpus(corpus)
    index = {(d, t): i for d, i, t in chunks}
    names = BACKENDS if args.retriever == "all" else (args.retriever,)
    backends = []
    model = conn = None
    for name in names:
        if name == "dense":
            from evalkit.db import connect
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(DEFAULT_EMBEDDING_MODEL)
            conn = connect()
            backends.append(DenseBackend(model, conn, index))
        elif name == "bm25":
            backends.append(BM25Backend(chunks))
        elif name == "random":
            backends.append(RandomBackend(chunks))

    results = {}
    try:
        for backend in backends:
            scored = score_backend(backend, positives, negatives, ks, max_k)
            pos_top1 = []
            for item in positives:
                top = backend.search(item["query"], 1)
                pos_top1.append(top[0].score if top else None)
            results[backend.name] = {
                "summary": summarize(scored, ks),
                "hits": scored["hits"],
                "neg_scores": scored["neg_scores"],
                "pos_top1": pos_top1,
            }
            if getattr(backend, "unmatched", 0):
                print(f"WARN: dense left {backend.unmatched} rows unmapped "
                      f"from docs {sorted(backend.unmatched_docs)} "
                      f"(not in corpus — counts as misses)")
    finally:
        if conn is not None:
            conn.close()

    comparisons, failed = {}, False
    if args.compare:
        base = json.load(open(args.compare))
        for name, res in results.items():
            if name not in base.get("backends", {}):
                continue
            old_hits = base["backends"][name]["hits"]
            new_hits = res["hits"]
            if len(old_hits) != len(new_hits):
                print(f"WARN: {name} baseline n={len(old_hits)} != current "
                      f"n={len(new_hits)} — skipping gate")
                continue
            b = sum(1 for o, n in zip(old_hits, new_hits) if o == 1 and n == 0)
            c = sum(1 for o, n in zip(old_hits, new_hits) if o == 0 and n == 1)
            stat, p = mcnemar(b, c)
            regressing = p < 0.05 and b > c
            comparisons[name] = {
                "b": b, "c": c, "stat": stat, "p": p,
                "verdict": "REGRESSION" if regressing else "ok",
            }
            if regressing:
                msg = f"GATE: {name} regressed (b={b}, c={c}, p={p:.3f})"
                if args.enforce == "enforce":
                    print(msg)
                    failed = True
                else:
                    print(f"WARN (not enforced): {msg}")

    os.makedirs(args.report_dir, exist_ok=True)
    report = {
        "meta": meta,
        "models": {"embedding": DEFAULT_EMBEDDING_MODEL,
                   "generation": None, "judge": None},
        "n_positives": len(positives),
        "n_negatives": len(negatives),
        "backends": {
            name: {"summary": r["summary"], "hits": r["hits"]}
            for name, r in results.items()
        },
        "comparisons": comparisons,
    }
    with open(os.path.join(args.report_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    md = render_markdown(results, comparisons, meta, len(positives), len(negatives))
    with open(os.path.join(args.report_dir, "report.md"), "w") as f:
        f.write(md)
    print(md)
    if args.write_baseline:
        with open(args.write_baseline, "w") as f:
            json.dump(
                {"meta": meta,
                 "backends": {n: {"summary": r["summary"], "hits": r["hits"]}
                              for n, r in results.items()}}, f, indent=2)
        print(f"baseline written to {args.write_baseline} (commit it by hand)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
