"""Retriever backends behind one interface.

`Backend.search(query, top_k)` returns scored chunks; the harness never knows
which implementation served. Today's arms — dense (the product), bm25-only
and random (the controls) — become the baselines hybrid retrieval must beat
in the reranking/RRF roadmap item, with no harness change: add a class and
register it in BACKENDS.

Dense talks to Postgres with the same SQL the retriever runs (own copy, not
an import — flat-name rule). BM25 and random run off local chunks, so they
need no database at all.
"""

import math
import random
import re
from typing import NamedTuple, Optional


class Retrieved(NamedTuple):
    doc_id: str
    chunk_index: Optional[int]  # None when unresolvable (counts as a miss)
    chunk_text: str
    score: Optional[float]  # None for random (ranks carry no similarity)


class Backend:
    name: str

    def search(self, query: str, top_k: int) -> list[Retrieved]:
        raise NotImplementedError


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class DenseBackend(Backend):
    """pgvector cosine search — the production path under measurement."""

    name = "dense"

    def __init__(self, model, conn, index: dict[tuple[str, str], int]) -> None:
        self._model = model
        self._conn = conn
        # (doc_id, chunk_text) -> chunk_index, built from local chunking.
        # The retriever's SQL returns no chunk_index, so text matching is the
        # bridge. Unmatched rows keep chunk_index=None and count as misses.
        self._index = index
        self.unmatched = 0
        self.unmatched_docs: set[str] = set()

    def search(self, query: str, top_k: int) -> list[Retrieved]:
        embedding = self._model.encode(
            query, show_progress_bar=False, convert_to_numpy=True
        ).tolist()
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    id::text                          AS chunk_id,
                    doc_id,
                    chunk_text,
                    1 - (embedding <=> %s::vector)    AS score
                FROM document_chunks
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (embedding, embedding, top_k),
            )
            rows = cur.fetchall()
        out = []
        for _, doc_id, chunk_text, score in rows:
            idx = self._index.get((doc_id, chunk_text))
            if idx is None:
                self.unmatched += 1
                self.unmatched_docs.add(doc_id)
            out.append(
                Retrieved(doc_id, idx, chunk_text, float(score))
            )
        return out


class BM25Backend(Backend):
    """Pure-lexical Okapi BM25 over local chunks. No database, no vectors."""

    name = "bm25"
    k1 = 1.5
    b = 0.75

    def __init__(self, chunks: list[tuple[str, int, str]]) -> None:
        self._chunks = chunks
        self._doc_tokens = [tokenize(t) for _, _, t in chunks]
        self._doc_len = [len(t) for t in self._doc_tokens]
        self._avgdl = sum(self._doc_len) / len(self._doc_len) if self._doc_len else 0
        df: dict[str, int] = {}
        for tokens in self._doc_tokens:
            for term in set(tokens):
                df[term] = df.get(term, 0) + 1
        n = len(chunks)
        self._idf = {
            term: math.log((n - f + 0.5) / (f + 0.5) + 1) for term, f in df.items()
        }

    def search(self, query: str, top_k: int) -> list[Retrieved]:
        scored = []
        for (doc_id, idx, text), tokens, dl in zip(
            self._chunks, self._doc_tokens, self._doc_len
        ):
            tf: dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            score = 0.0
            for term in tokenize(query):
                if term not in self._idf or term not in tf:
                    continue
                f = tf[term]
                denom = f + self.k1 * (1 - self.b + self.b * dl / self._avgdl)
                score += self._idf[term] * f * (self.k1 + 1) / denom
            scored.append((score, doc_id, idx, text))
        scored.sort(key=lambda s: (-s[0], s[1], s[2]))  # ties: deterministic
        return [
            Retrieved(doc_id, idx, text, score)
            for score, doc_id, idx, text in scored[:top_k]
        ]


class RandomBackend(Backend):
    """Seeded shuffle. The control that proves the set is discriminative: if
    this scores near dense, the golden set is broken, not the retriever."""

    name = "random"

    def __init__(
        self, chunks: list[tuple[str, int, str]], seed: int = 42
    ) -> None:
        self._chunks = chunks
        self._seed = seed

    def search(self, query: str, top_k: int) -> list[Retrieved]:
        rng = random.Random(f"{self._seed}:{query}")  # deterministic per query
        order = list(self._chunks)
        rng.shuffle(order)
        return [
            Retrieved(doc_id, idx, text, None)
            for doc_id, idx, text in order[:top_k]
        ]


BACKENDS = ("dense", "bm25", "random")
