# Second-author query protocol (golden v2 hardening)

## Why you are being asked

Our retrieval eval scores a perfect recall@5, which sounds good and measures
nothing: every question was written by the same person who wrote the
documents, so the phrasing echoes the answers. Your job is to break that
mirror. You write questions; we do everything else (labeling, checking,
scoring). Budget: about 30 minutes.

## What you get (and what you deliberately don't)

You get the 18 topics below plus one-line answer hints. You do **not** get
the documents themselves — that blindness is the entire point. Do not ask to
see them; do not go looking for the repo.

## Rules

1. Write **10–15 questions**, at most one per topic (skip topics freely).
2. Phrase each as you would type it into a search engine, not as an exam.
   Short, plain, a little vague — like a real user who half-remembers.
3. Never reuse distinctive terms from the answer hint verbatim if you can
   avoid it. Say "the Python style checker", not the tool's rule codes;
   say "the container startup race", not the readiness command.
4. One question = one answer. If a topic suggests two different answers,
   pick one.
5. If you doubt a topic is really covered, ask it anyway and mark it with
   `(unsure)` — near-misses are useful data.
6. Optional bonus: 2–3 questions the docs *should* answer but probably
   don't (costs, security, production sizing). Mark these `(unanswerable)`.

## Topics and answer hints

| # | Topic | The answer is about… |
|---|---|---|
| 1 | uv workspaces | running one service instead of everything |
| 2 | uv lockfile in CI | what stops an unrecorded dependency change |
| 3 | ruff lint config | which rule sets are enforced |
| 4 | gofmt in CI | why `gofmt -l` output can't be checked naively |
| 5 | pytest markers | running only the tests that skip Postgres |
| 6 | pytest import paths | why suites can't run as one sweep |
| 7 | CI database service | which Postgres image CI uses and why |
| 8 | CI caching | what dominates the Python job's runtime |
| 9 | local startup order | what the run script waits for first |
| 10 | gateway build | why the script builds instead of `go run` |
| 11 | vector index | how distance becomes a similarity score |
| 12 | plain Postgres indexes | lookups that need no vector extension |
| 13 | Kafka consumer groups | what happens when the worker falls behind |
| 14 | Kafka topics | when ingest returns relative to embedding |
| 15 | query deadline | empty index vs dead generator statuses |
| 16 | error mapping | status codes at the gateway boundary |
| 17 | embedding model | what breaks on an uncoordinated model swap |
| 18 | chunking | why chunks overlap / what labels reference |

## How to return it

A plain numbered list is perfect:

```
1. (topic 9) What does the startup script wait on before anything else?
   Expected: database and broker readiness, then topic creation.
2. (topic 15, unsure) What happens if I search with nothing loaded?
   Expected: some kind of not-found, generator skipped.
```

Send it back as text, markdown, or on paper photographed sideways — we
convert it. What happens next (our side, for transparency): each question
gets hand-matched to the passage that answers it, checked by a
paraphrase-and-distractor lint, and scored. If your set still scores near
perfect, the verdict is our *documents* are too cozy, and that's our
problem, not yours.
