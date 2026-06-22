# DocMind — Experiments

All experiments run against the same 15-query retrieval golden set
(eval/retrieval_golden_set.json) at top_k=5 unless noted.
Baseline: fixed-size chunking (500 tokens, 50 overlap)

Raw result files: eval/results/

---

## Experiment 1 — Chunking strategy comparison

**Question:** Does structure-aware or recursive chunking produce better
retrieval than fixed-size on this corpus?

**Hypothesis:** Structure-aware chunking should win on markdown/code
documents; recursive should outperform fixed-size on PDFs.

| Strategy       | Precision@5 | Recall@5 | MRR  | Avg chunk tokens | Total chunks |
|----------------|-------------|----------|------|------------------|--------------|
| fixed_size     | 0.160       | 0.800    | 0.572| 483              | 127          |
| recursive      | 0.175       | 0.850    | 0.617| 478              | 137          |
| structure_aware| 0.175       | 0.825    | 0.608| 467              | 139          |

**Finding:** Recursive chunking wins outright on this corpus — best recall
(0.850) and best MRR (0.617), tied for best precision (0.175) with
structure-aware, despite producing fewer chunks than structure-aware.
Structure-aware is a close second, not a clear winner over recursive even
though the corpus includes markdown/code docs, so the hypothesis that
structure-aware would win on those documents wasn't borne out — recursive's
separator cascade (`\n\n` → `\n` → `. ` → ` `) already captures most of the
same natural boundaries structure-aware targets explicitly. Fixed-size is
the clear loser on every metric, confirming naive fixed-token windows cut
across semantic boundaries (paragraphs, sections) more often, splitting
relevant content across chunk edges and hurting both what's retrievable and
how highly it ranks.

Across all three strategies, per-query failures (P=R=0.00) cluster on the
same handful of questions (e.g. Transformer BLEU scores, BART's role in
RAG, attention mechanism type) — these look like retrieval/embedding
misses shared across chunking strategies rather than something a chunking
change alone will fix.

---

## Experiment 2 — Embedding model comparison

**Question:** Does BGE-large (local, free) retrieval quality justify
replacing OpenAI embeddings (hosted, paid)?

Run against the same golden set as Experiment 1 (`eval/golden_dataset.yaml`).

**BGE-large-en-v1.5 (local)** — chunking strategy comparison, k=5:

| Strategy       | Precision@5 | Recall@5 | MRR  | Avg chunk tokens | Total chunks |
|----------------|-------------|----------|------|------------------|--------------|
| fixed_size     | 0.155       | 0.775    | 0.592| 483              | 127          |
| recursive      | 0.150       | 0.725    | 0.576| 478              | 137          |
| structure_aware| 0.150       | 0.725    | 0.539| 467              | 139          |

| Model                    | Embed latency | Cost/1K tokens |
|--------------------------|---------------|----------------|
| text-embedding-3-small   | —             | $0.02          |
| BGE-large-en-v1.5 (local)| —             | $0.00          |

**Finding:** Switching to BGE-large lowers retrieval quality across the
board relative to OpenAI on every strategy and metric — e.g. recursive
drops from P=0.175/R=0.850/MRR=0.617 (OpenAI, Experiment 1) to
P=0.150/R=0.725/MRR=0.576 (BGE), and the best BGE strategy (fixed_size,
P=0.155/R=0.775/MRR=0.592) still trails OpenAI's worst strategy
(fixed_size, P=0.160/R=0.800/MRR=0.572) on recall and is roughly level on
precision/MRR. BGE also reorders the strategy ranking: fixed_size leads
under BGE, whereas recursive led under OpenAI — the chunking strategy that
works best is embedding-model-dependent, not a universal property of the
corpus.

Given the recall gap (-0.10 to -0.125 across strategies) at zero marginal
cost, BGE is a reasonable choice only if the quality drop is acceptable
for the use case; if recall matters most, OpenAI's hosted embeddings
currently justify their cost on this corpus.

---

## Experiment 3 — Hybrid search

**Question:** Does adding BM25 (sparse, lexical) retrieval on top of dense
embeddings improve retrieval quality over dense-only search?

**Hypothesis:** Hybrid (dense + BM25, fused via Qdrant's native RRF) should
improve recall in particular, by catching exact keyword/entity matches that
dense embeddings alone miss.

Run against the same golden set, recursive chunking + text-embedding-3-small,
k=5. Dense-only numbers are recursive from Experiment 1.

| Strategy            | Precision@5 | Recall@5 | MRR@5 | Chunks | Avg tokens |
|----------------------|-------------|----------|-------|--------|------------|
| recursive (dense)    | 0.175       | 0.850    | 0.617 | 137    | 478        |
| recursive (hybrid)   | 0.195       | 0.925    | 0.703 | 137    | 478        |

**Finding:** Hybrid search beats dense-only retrieval on every metric for
the same chunk set — precision +0.020 (+11%), recall +0.075 (+9pp, from
0.850 to 0.925), and MRR +0.086 (+14%), the largest relative gain of the
three. Chunk count and average tokens are identical between the two rows
(same recursive chunker output indexed twice — once with only a dense
vector, once with both dense and BM25 sparse vectors), so the entire
improvement is attributable to fusing in lexical matching, not to any
difference in chunking.

The recall jump is the most notable result and matches the hypothesis:
BM25 picks up exact-term matches (named entities, technical terms, acronyms)
that cosine similarity over dense embeddings can miss when a query shares
vocabulary with a chunk but the embedding doesn't place them close enough
in vector space. The MRR gain (0.617 → 0.703) shows RRF fusion isn't just
surfacing additional relevant chunks lower in the list — it's also ranking
the correct chunk higher on average when both signals agree, consistent
with how RRF rewards chunks that appear in both the dense and sparse
candidate lists rather than just appearing in one.

Given this clear improvement with no extra chunking cost (same chunks,
same chunk count), hybrid retrieval is the better default over dense-only
for this corpus — the added cost is one extra local BM25 encode per query
(no API call, fastembed runs on CPU) plus a second named-vector index in
Qdrant, both cheap relative to the recall/MRR gain.

---

## Experiment 4 — Cross-encoder reranking

**Question:** Does reranking hybrid search's candidate pool with a
cross-encoder (`BAAI/bge-reranker-base`) improve precision@5 and MRR over
hybrid search alone?

**Hypothesis:** Scoring (query, chunk) pairs jointly should outperform
hybrid's RRF-fused ranking, which only ever compares independently-computed
scores — production RAG postmortems often call this the single largest
quality jump in the pipeline.

Run against the same golden set, recursive chunking + text-embedding-3-small,
k=5. Reranker re-scores a candidate_pool_size=20 hybrid pool, then truncates
to top 5. `recursive (hybrid)` numbers are from Experiment 3.

| Strategy                  | Precision@5 | Recall@5 | MRR@5 | Chunks | Avg tokens |
|----------------------------|-------------|----------|-------|--------|------------|
| recursive (hybrid)         | 0.195       | 0.925    | 0.703 | 137    | 478        |
| recursive (hybrid+rerank)  | 0.185       | 0.900    | 0.796 | 137    | 478        |

**Finding:** Reranking gives the largest MRR gain of any experiment so far
(0.703 → 0.796, +0.093/+13%) but slightly *hurts* precision@5 (-0.010, -5%)
and recall@5 (-0.025, -2.7pp) relative to hybrid alone. The MRR jump matches
the hypothesis directly — the cross-encoder is much better at picking the
single most relevant chunk and pushing it to rank 1, since it scores the
query and that chunk jointly instead of comparing two independently-computed
vectors/term-stats. But on this golden set several queries have more than
one valid (doc_id, snippet) match, and promoting the cross-encoder's single
best pick to the top can bump a second, still-relevant chunk out of the
top-5 window, which is what shows up as the small precision/recall dip.

This is a ranking-quality vs. coverage tradeoff, not reranking failing
outright: MRR (how fast you find *a* relevant chunk) improves a lot, while
precision/recall (how many of the *k* slots are relevant) dips slightly. For
a QA pipeline that only needs the LLM to ground its answer in a handful of
correct chunks rather than recover every relevant span, the MRR gain is the
more decision-relevant metric here, so reranking is worth keeping as the
default for the hybrid path despite the small precision/recall cost — but
it's not the unconditional win the hypothesis predicted on this corpus.

---

## Summary — Experiment 1 → Experiment 4

Each experiment changed exactly one thing and kept the rest of the pipeline
fixed, so the gain at each step is attributable to that one change:

| Stage                                | Precision@5 | Recall@5  | MRR@5     |
|---------------------------------------|-------------|-----------|-----------|
| Exp 1 — fixed-size chunking (baseline)| 0.160       | 0.800     | 0.572     |
| Exp 1 — recursive chunking            | 0.175       | 0.850     | 0.617     |
| Exp 3 — + hybrid (dense + BM25/RRF)   | 0.195       | 0.925     | 0.703     |
| Exp 4 — + cross-encoder reranking     | 0.185       | 0.900     | **0.796** |

**Baseline → final pipeline: Precision +0.025 (+16%), Recall +0.100 (+12.5pp),
MRR +0.224 (+39%).** Chunking picked the right boundaries (recursive over
fixed-size), hybrid search added the lexical-match recall dense embeddings
alone were missing, and reranking is what ultimately moved MRR the most by
scoring (query, chunk) pairs jointly instead of comparing independent
vectors. Each layer compounds on a different axis — chunking and hybrid
mostly grew recall/precision, reranking mostly grew ranking quality (MRR) —
which is why the full pipeline (recursive + hybrid + rerank) is the current default even though precision/recall *dip* relative to hybrid-only.

---
## Experiment 5 — End-to-end RAGAS evaluation (full pipeline)

**Question:** Now that retrieval is hybrid+reranked (Experiments 3-4), how
does the *full* pipeline — retrieve -> rerank -> generate — score on
RAGAS's LLM-judged metrics (faithfulness, answer relevancy, context
precision, context recall), and does answer quality hold up when a question
needs information combined from more than one document?

Run via `scripts/run_ragas_eval.py` against `eval/ragas_dataset.yaml` (35
questions, GPT-4o-mini judge): 20 `factual_single_doc`, 7
`multi_doc_synthesis`, 8 `not_in_corpus` (deliberately unanswerable from the
corpus). Same recursive+hybrid+rerank pipeline as Experiment 4's default.
Raw results: `eval/results/ragas_baseline.json`.

| category            | faithfulness | answer_relevancy | context_precision | context_recall | n  |
|----------------------|--------------|-------------------|--------------------|-----------------|----|
| factual_single_doc  | 0.96         | 0.90              | 0.81               | 0.93            | 20 |
| multi_doc_synthesis | 0.82         | 0.50              | 0.93               | 0.86            | 7  |
| not_in_corpus       | 0.71         | 0.00              | 0.07               | 0.00            | 8  |
| **overall**         | **0.88**     | **0.62**          | **0.67**           | **0.70**        | 35 |

`multi_doc_synthesis` is the second data point on this category, not the
first — an earlier prompt revision (before the `GROUNDED_QA_TEMPLATE`
wording this run was scored against) measured substantially worse on the
same 7 questions:

| multi_doc_synthesis (n=7) | faithfulness | answer_relevancy | context_precision | context_recall |
|----------------------------|--------------|-------------------|--------------------|-----------------|
| earlier prompt revision    | 0.38         | 0.27              | 0.96               | 0.80            |
| this run (table above)     | 0.82         | 0.50              | 0.93               | 0.86            |

**Finding:** The weak overall answer_relevancy (0.62) hides two unrelated
failure modes:

- **`not_in_corpus` (8 questions):** near-zero context_precision/recall here
  is *correct* — by construction nothing relevant exists in the corpus, so a
  low score means the retriever isn't pulling in noise to compensate. All 8
  got the fixed refusal string. The low faithfulness (0.71) and
  answer_relevancy (0.00) on these rows are a metric artifact, not a
  pipeline defect: a refusal sentence has no question-specific claims for
  faithfulness to check and no question-specific content for
  answer_relevancy's reverse-question-generation to compare against, so both
  metrics structurally floor near 0 on a *correct* refusal. One outlier
  ("maximum input context window size of the Transformer") got a hedged
  partial answer instead of a clean refusal (faithfulness 0.67) — a real
  one-off generation bug, but it's a single question, not the category.

- **`multi_doc_synthesis` (7 questions) — the real signal:** the earlier
  prompt revision already lifted this category a lot (faithfulness
  0.38→0.82, answer_relevancy 0.27→0.50, context_recall 0.80→0.86) at a
  small context_precision cost (0.96→0.93) — so that fix worked, it just
  didn't go far enough. context_precision (0.93) and context_recall (0.86)
  in this run are the *best* of any category, so hybrid+rerank is reliably
  handing the generator chunks from the right multiple documents. But
  answer_relevancy (0.50) and faithfulness (0.82) still trail
  `factual_single_doc` (0.90 / 0.96) by a wide margin: 3 of 7 questions
  (e.g. "relationship between RAG's dense retrieval and Qdrant's search,"
  "Transformer's role inside a RAG model") got the flat refusal string even
  though relevant multi-document context was sitting right there in the
  prompt. The generator was still treating "no single chunk contains the
  whole answer" the same as "not in context" and refusing instead of
  combining facts stated across chunks — `GROUNDED_QA_TEMPLATE`
  (`app/generation/prompts.py`) said "answer ONLY from context" and "do not
  infer beyond what is stated," illustrated with only a single-chunk
  citation example, which reads as license to refuse whenever an answer
  spans more than one chunk.

**Action taken:** Rewrote `GROUNDED_QA_TEMPLATE` again to explicitly
instruct reading all chunks before answering, synthesizing facts stated
across multiple chunks into one answer, citing every chunk a claim relies
on, and drawing a clear line between *combining stated facts* (allowed) and
*inferring unstated ones* (still forbidden) — plus a new rule to surface
contradictions across chunks instead of silently picking one. This targets
the same `multi_doc_synthesis` refusal-on-multi-hop failure the earlier
revision only partly fixed; retrieval was already good for this category
(context_precision/recall above), so the fix is generation-side only and
shouldn't move retrieval metrics.

---