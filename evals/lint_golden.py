"""Anti-triviality lint for the golden set.

A question that copies its answer is not retrieval, it is string matching.
Three checks, all execution (exit 1 with messages on violation):

1. No 9-word verbatim run shared between a query and any of its gold chunks
   (lowercased, punctuation stripped). Paraphrase is the requirement.
2. Every positive has a distractor: a non-gold doc sharing at least two
   content words (len > 3) with the query but a different answer.
3. Referential integrity: docs exist, chunk indices are in range, negatives
   carry empty gold.
"""

import json
import os
import sys

from evalkit.backends import tokenize
from evalkit.chunks import chunk_corpus

STOPWORDS = {
    "what", "whats", "which", "when", "where", "does", "with",
    "from", "that", "this", "have", "has", "are", "was", "were",
    "will", "would", "there", "their", "about", "into", "over",
    "after", "before", "between", "under", "while", "the", "and",
    "for", "how", "why", "you", "your", "its", "our", "can",
}

WINDOW = 9
MIN_DISTRACTOR_OVERLAP = 2


def _words(text: str) -> list[str]:
    return tokenize(text)


def _ngrams(words: list[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def _content_words(query: str) -> set[str]:
    return {w for w in _words(query) if len(w) > 3 and w not in STOPWORDS}


def lint(corpus: list[dict], golden: list[dict]) -> list[str]:
    errors: list[str] = []
    by_doc = {d["doc_id"]: d["content"] for d in corpus}
    chunk_texts: dict[tuple[str, int], str] = {}
    for doc_id, idx, text in chunk_corpus(corpus):
        chunk_texts[(doc_id, idx)] = text
    doc_words = {d: set(_words(c)) for d, c in by_doc.items()}

    for n, item in enumerate(golden):
        tag = f"golden[{n}] query={item.get('query', '')!r:.60}"
        query = item.get("query", "")
        if not query or not query.strip():
            errors.append(f"{tag}: empty query")
            continue
        if item.get("negative"):
            if item.get("gold"):
                errors.append(f"{tag}: negative must carry empty gold")
            continue
        gold = [(g["doc_id"], g["chunk_index"]) for g in item.get("gold", [])]
        if not gold:
            errors.append(f"{tag}: positive must carry non-empty gold")
            continue
        for doc_id, idx in gold:
            if doc_id not in by_doc:
                errors.append(f"{tag}: unknown doc {doc_id}")
            elif idx not in {
                i for (d, i) in chunk_texts if d == doc_id
            }:
                errors.append(f"{tag}: chunk_index {idx} out of range for {doc_id}")
        gold_docs = {d for d, _ in gold}
        q_grams = _ngrams(_words(query), WINDOW)
        for doc_id, idx in gold:
            text = chunk_texts.get((doc_id, idx), "")
            if q_grams & _ngrams(_words(text), WINDOW):
                errors.append(
                    f"{tag}: {WINDOW}-word verbatim run copied from "
                    f"{doc_id}[{idx}] — paraphrase the question"
                )
        qw = _content_words(query)
        distractors = [
            d
            for d, words in doc_words.items()
            if d not in gold_docs and len(qw & words) >= MIN_DISTRACTOR_OVERLAP
        ]
        if not distractors:
            errors.append(
                f"{tag}: no distractor doc shares "
                f"{MIN_DISTRACTOR_OVERLAP}+ content words — add one or rephrase"
            )
    return errors


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    os.chdir(here)
    corpus = [json.loads(line) for line in open("corpus.jsonl") if line.strip()]
    golden = [json.loads(line) for line in open("golden.jsonl") if line.strip()]
    errors = lint(corpus, golden)
    for e in errors:
        print(f"FAIL: {e}")
    print(f"{len(golden)} items, {len(errors)} violations")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
