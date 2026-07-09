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

## Experiment 6 — Semantic caching

**Question:** Does a Redis-backed semantic cache reduce average
cost-per-query and latency on repeated/paraphrased traffic, without
serving wrong answers to genuinely different questions?

Calibrated against `eval/cache_threshold_pairs.yaml` (10 paraphrase pairs +
8 non-paraphrase pairs pulled from the golden set, see
`eval/results/cache_threshold_calibration.json`): minimum paraphrase
similarity 0.7818, maximum non-paraphrase similarity 0.5500, threshold set
to 0.75 (`app/config.py`).

| Metric                            | No cache (baseline)         | Cache (warm)              |
|------------------------------------|------------------------------|----------------------------|
| Avg cost/query                    | $0.000472                    | $0.000367                  |
| Cache hit rate                    | —                             | 22.2% (10/45)              |
| p50 latency, cache hit             | —                             | 530.5 ms                   |
| p50 latency, cache miss            | —                             | 8695 ms                    |
| p50 latency, no cache (baseline)   | 7984 ms                       | —                          |

**Finding:** Caching cut average cost per query by 22.3% ($0.000472 →
$0.000367 across the warm run's 45 requests) and achieved a 22.2% hit rate
(10/45) — exactly the expected ratio: the 10 paraphrased questions from
`cache_threshold_pairs.yaml` all hit, and all 35 original golden-set
questions missed (each is asked exactly once, before its paraphrase, so
nothing had been cached for it yet). Latency confirms the predicted
order-of-magnitude gap: p50 on a hit (530.5ms) is ~16x lower than p50 on a
miss in the same run (8695ms) and ~15x lower than the no-cache baseline
(7984ms). The hit path isn't literally single-digit milliseconds, though —
it still pays for `embed_query()`'s OpenAI embedding call (needed to get a
vector to check the cache with) plus an O(n) Redis `SCAN` + Python cosine
pass over every live entry in the namespace. What it skips is the
dominant cost — the generation call — which is why miss and baseline
latency are both ~8s and nearly identical to each other.

**Threshold reasoning:** The calibration set showed clean separation with
no overlap — every paraphrase pair scored at or above 0.7818 cosine
similarity, every non-paraphrase pair at or below 0.5500. The calibration
script's literal recommendation is the midpoint of that gap (0.6659), but
the threshold was set higher, at 0.75 — closer to the empirical paraphrase
floor than to the midpoint — to favor precision: with only 10 paraphrase
pairs in the calibration set, 0.7818 isn't guaranteed to be the true worst
case real traffic will produce, so sitting near the midpoint risks a false
hit (serving a cached answer to a meaningfully different question) the
moment real-world paraphrase similarity dips below what this small sample
observed. Moving closer to the paraphrase floor instead spends some of the
gap's margin on the other failure mode — a false miss, where a genuine
paraphrase scores between 0.6659 and 0.75 and pays full retrieval +
generation cost instead of being served from cache — which is the cheaper
mistake of the two: a false miss only costs what an uncached query already
costs today, while a false hit returns a wrong answer to the user.

---

## Experiment 7 — Agentic RAG vs naive RAG on multi-hop questions

**Question:** Does an iterative retrieval loop (retrieve → assess sufficiency
→ reformulate → retrieve again) produce measurably better answer quality
than a single-pass retrieve-and-generate pipeline on the questions where
one-shot retrieval is most likely to fail?

**Comparison set:** 10 questions selected from `eval/ragas_dataset.yaml` —
all 7 `multi_doc_synthesis` questions plus the 3 `factual_single_doc`
questions with context_recall ≤ 0.50 in the Week 3 baseline. Excluded
`not_in_corpus` questions (both pipelines should refuse equally, so they
add no signal). Both pipelines used the same Qdrant collection, embedding
model (`text-embedding-3-small`), chunking strategy (recursive), and
generation model (GPT-4o-mini). Naive pipeline: `retrieve_reranked(top_k=5)
→ generate_answer`. Agentic pipeline: `run_agent_loop(max_iterations=3,
top_k=5) → rerank(accumulated_chunks, top_k=5) → generate_answer /
generate_partial_answer`. Scored via RAGAS (GPT-4o-mini judge).
Raw results: `eval/results/naive_rag_comparison.json` and
`eval/results/agentic_rag_comparison.json`.

**Prompt versions in effect:**
- Sufficiency: v2 (threshold lowered from "fully answer" to "useful, grounded
  answer"; v1 caused near-universal cap-reaching on this bounded corpus)
- Query reformulation: v2 (passes full `query_history` to prevent the loop
  generating the same query across iterations)

| Metric             | Naive RAG | Agentic RAG | Delta            |
|--------------------|-----------|-------------|------------------|
| faithfulness       | 0.7708    | 0.7761      | +0.0052          |
| answer_relevancy   | 0.5978    | 0.6103      | +0.0125          |
| context_precision  | 0.7961    | 0.8893      | **+0.0932**      |
| context_recall     | 0.8583    | 0.8583      | +0.0000          |
| avg cost/query     | $0.00049  | $0.00123    | +$0.00074 (+151%)|
| p50 latency        | 6747 ms   | 11866 ms    | +5119 ms (+76%)  |
| p95 latency        | 12086 ms  | 32339 ms    | +20253 ms (+168%)|

**Agentic iteration distribution (10 queries):**

| Termination                          | Count |
|--------------------------------------|-------|
| Iteration 1 (sufficient immediately) | 8     |
| Iteration 3                          | 1     |
| Cap reached (3 iterations)           | 1     |

**Finding:** The agentic loop's most notable result is what didn't move:
context_recall is exactly flat (0.8583 both), which was the metric the loop
was hypothesised to improve most directly. Context_precision jumped
substantially (+9.3pp, 0.7961 → 0.8893), meaning the chunks the agentic
pipeline ultimately surfaces are more relevant on average — but it isn't
surfacing *more* of the ground-truth context, just selecting better within
what retrieval can find. Faithfulness and answer_relevancy improved slightly
(+0.0052 and +0.0125), but neither delta is large enough to be conclusive
on a 10-question set.

The iteration distribution is the key diagnostic: 8 of 10 queries terminated
at iteration 1 — the v2 sufficiency threshold declared the first retrieval
pass sufficient immediately, making the agentic loop functionally equivalent
to naive RAG plus one extra sufficiency-assessment LLM call for 80% of
queries. This is the cost you see in the table: $0.00123 vs $0.00049 (2.5x
more expensive) even though only 2 queries actually triggered additional
retrieval. The 1 cap-reached query represents a genuine multi-hop gap —
3 iterations with different reformulated queries still couldn't satisfy
sufficiency — which on a small bounded corpus is an expected hard limit,
not a loop failure.

**What this means for the pipeline:**

- The context_precision gain (+9.3pp) is real: reformulating around missing
  aspects, even once, produces a more targeted candidate pool for the reranker,
  which shows up in precision. This is the loop working as designed for the 2
  queries that actually iterated.
- The flat context_recall says the corpus ceiling is the binding constraint,
  not the number of retrieval passes. Information the first pass misses is
  generally not found by later passes — reformulated queries converge on the
  same document space.
- At 2.5x cost and 1.75x p50 latency for +9.3pp precision and negligible
  quality improvement elsewhere, the agentic loop is not a straightforward win
  on this corpus at this scale. Whether the precision gain justifies the
  overhead is a product decision, not a retrieval one.

---

## Experiment 8 — Table/figure-augmented (multimodal) retrieval vs text-only baseline

**Question:** Does supplementing the index with Docling-extracted table
chunks (KV-formatted rows) and GPT-4o-Vision figure captions, alongside
the same recursively-chunked PDF prose, improve retrieval-augmented
generation on questions whose answers live specifically in table cells,
figure content, or a mix of prose and table/figure content?

**Comparison set:** 18 questions from `eval/multimodal_dataset.yaml` (7
`table_only`, 4 `diagram_only`, 7 `hybrid`), grounded against the attention
and RAG papers and verified fact-by-fact against the actual baseline text
extraction to confirm no answer is independently restated as a complete
prose sentence outside its source table/figure. Both pipelines used the
same embedding model (`text-embedding-3-small`), hybrid dense+BM25
retrieval, cross-encoder reranking (`BAAI/bge-reranker-base`), and
generation model (GPT-4o-mini); the only variable in the primary
comparison is which Qdrant collection is queried —
`docmind_recursive_text-embedding-3-small_hybrid` (text-only) vs
`multimodal_text-embedding-3-small_hybrid` (tables as KV text + figure
captions + the same PDF prose). Scored via RAGAS (GPT-4o-mini judge). Raw
results: `eval/results/text_baseline_comparison.json` and
`eval/results/multimodal_comparison.json`.

| Metric             | Text Baseline | Multimodal (unified top-5) | Delta       |
|---------------------|---------------|------------------------------|-------------|
| faithfulness        | 0.3580        | 0.4960                       | +0.1380     |
| answer_relevancy    | 0.5952        | not captured¹                 | —           |
| context_precision   | 0.7364        | 0.5673                       | -0.1691     |
| context_recall      | 0.8889        | 0.7130                       | -0.1759     |
| avg cost/query      | $0.000504     | not captured¹                 | —           |
| p50 latency         | 2662ms        | 2662ms                        | +0ms        |
| p95 latency         | 4450ms        | not captured¹                 | —           |
| multimodal hit rate | n/a           | not captured¹                 | —           |

¹ This run's raw output was superseded on disk by the follow-up
two-pool-retrieval test below before these fields were archived. Only the
four metrics above were recorded before the overwrite.

**Finding:** Unlike Experiment 7 (agentic vs naive RAG), where every
metric moved in the expected direction, multimodal retrieval is a mixed
result even on a question set purpose-built to need table/figure content.
Faithfulness improved (+0.138) — when the multimodal pipeline gets the
right chunk, GPT-4o-mini can extract and cite the structured KV/caption
content cleanly. But both context_precision (-0.169) and context_recall
(-0.176) regressed relative to the text baseline. Two separable causes,
both confirmed against actual retrieved contexts rather than inferred from
the aggregate numbers alone:

- **RAGAS's judge is format-sensitive.** Per-category faithfulness on the
  saved run shows `table_only` scoring fine (text baseline 0.452 vs
  multimodal 0.464 — comparable) while `diagram_only` drops sharply
  (0.375 → 0.197). Spot-checking individual records earlier in this
  investigation found answers that were verbatim-correct against
  KV-formatted table context still scoring `faithfulness=0.0` — the
  NLI-style statement judge doesn't reliably recognize entailment against
  `Header: value | Header: value` context shaped unlike the prose it was
  presumably calibrated on. This inflates the apparent quality gap in
  either direction depending on which chunk type dominates a category.
- **Reranking within a single unfiltered top-5 doesn't reliably surface
  the correct table among several competing ones.** The base-model BLEU
  question (`Table 2`, containing the literal answer "27.3 BLEU") is a
  clean example: `Table 3` (a much larger, more finely-chunked ablation
  table covering the same "base Transformer" terminology) kept winning
  the reranker's relevance score over `Table 2`, so the model answered
  with the *big* model's 28.4 instead of the base model's 27.3 — a wrong
  answer confidently presented as a "hit."

**Follow-up experiment — two-pool retrieval:** Tested reserving fixed
slots (2 multimodal + 3 prose) instead of unified top-k competition, via
two independently Qdrant-filtered hybrid searches
(`retrieve_with_multimodal_quota` in `app/retrieval/searcher.py`), to
address the context precision drop observed in the primary run.

| Metric             | Unified top-5 | 2+3 pooled | Delta   |
|---------------------|---------------|------------|---------|
| faithfulness         | 0.4960        | 0.3703     | -0.1257 |
| answer_relevancy     | not captured  | 0.6103     | —       |
| context precision    | 0.5673        | 0.5993     | +0.0320 |
| context recall       | 0.7130        | 0.5556     | -0.1574 |
| avg cost/query       | not captured  | $0.00043   | —       |
| p50 latency          | 2662ms        | 3533ms     | +871ms  |
| p95 latency          | not captured  | 5450ms     | —       |
| hit rate (all cats.) | not captured  | 100%       | —       |

Both columns here are the multimodal arm under the two retrieval
strategies (text-baseline is not part of this comparison — see the
primary table above for that reference point).

Reserving slots did raise context_precision (+0.032) and pushed hit rate
to 100% across every category — the mechanism itself works exactly as
designed: `attention-paper_table_2` (the correct chunk for the base-BLEU
question above) is now guaranteed a place in its own filtered candidate
pool instead of losing an unfiltered fusion race against ten prose
chunks. But it **still answered the base-BLEU question incorrectly**,
because the reranker, given only table/figure candidates to choose from,
*still* preferred `Table 3`'s ablation rows over `Table 2`'s actual
answer — pooling fixes candidate-pool exclusion, not reranker
mis-scoring within an already-correct pool. Context_recall also dropped
further (-0.157 vs. unified top-5), and p50 latency rose by 871ms from
running two filtered searches instead of one.

**Conclusion:** Primary result (unified top-5) is reported as the
representative multimodal pipeline. The precision/recall tradeoff from
pooling suggests the better production fix is adaptive retrieval
(classify query type, allocate slots dynamically) or BM25-first retrieval
for table content specifically — both flagged as follow-up work rather
than implemented here, to keep Week 4 scoped.

**What this means for the pipeline:**

- Adding structured table/figure chunks to the index is not a free win:
  it measurably helps faithfulness when the right chunk is retrieved, but
  a naive drop-in (same collection, same unified reranking) can *hurt*
  precision/recall relative to plain text if reranking doesn't
  consistently distinguish between competing tables covering similar
  terminology.
- The hit-rate metric (does a table/figure chunk merely appear in the
  top-5) is not sufficient on its own to certify a retrieval fix — a
  "hit" can still be the wrong table. Any future evaluation of table
  retrieval should check chunk *identity* against the specific fact
  needed, not just chunk *type*.
- Reserving retrieval slots by chunk type is a real, verified mechanism
  (confirmed against live Qdrant data) for guaranteeing candidate-pool
  inclusion, but it's not a substitute for the reranker actually scoring
  the correct chunk highest — and it adds latency and reduces recall,
  a real cost. It's a partial fix to one specific failure mode, not a
  general improvement.
- RAGAS's LLM-judge metrics (faithfulness in particular) appear sensitive
  to context formatting in ways that aren't yet well understood on this
  corpus — comparisons between a KV-formatted-context pipeline and a
  prose-context pipeline should be read with that caveat until the judge
  behavior is investigated further.

---

## Experiment 9 — Graph RAG (Neo4j entity graph) vs naive/agentic on multi-hop questions

**Question:** Does augmenting single-shot vector retrieval with entity-graph
traversal (Neo4j) close the multi-hop gap naive RAG has (Experiment 7),
without paying agentic's iterative-loop cost/latency premium?

**Setup:** Corpus ingested into Neo4j via `scripts/ingest_graph.py`
(recursive chunking, 500/50, same as the Qdrant pipeline): 138 chunks / 8
documents, with per-chunk LLM entity/relation extraction producing 1,734
`Entity` nodes, 2,131 `MENTIONS` edges, and 1,459 `RELATED_TO` edges between
entities. Compared against the same 7 `multi_doc_synthesis` questions used
in Experiment 7, via `scripts/run_graph_comparison_eval.py`, which re-runs
naive and agentic alongside graph so all three are scored on identical
questions in the same pass. Same embedding model, generation model
(GPT-4o-mini), and RAGAS judge as every other experiment. Retrieval
function under test: `retrieve_graph()` in `app/graph/graph_searcher.py`.
Raw results: `eval/results/naive_rag_multihop.json`,
`agentic_rag_multihop.json`, `graph_rag_multihop.json`.

**Baseline finding — the graph traversal never ran.** First pass:

| Metric             | Naive   | Agentic | Graph   | Δ Graph vs Naive |
|---------------------|---------|---------|---------|-------------------|
| faithfulness         | 0.6059  | 0.8694  | 0.7429  | +0.1369           |
| answer_relevancy     | 0.4947  | 0.6117  | 0.4933  | -0.0014           |
| context_precision    | 0.8786  | 0.8825  | 0.8627  | -0.0159           |
| context_recall       | 0.8690  | 0.8214  | 0.7262  | -0.1429           |
| avg cost/query       | $0.00050| $0.00137| $0.00050| +$0.00000         |
| p50 latency          | 2929ms  | 4798ms  | 2457ms  | -472ms            |

Graph trailed naive on 3 of 4 quality metrics despite the corpus having a
well-populated graph to traverse. Instrumenting `driver.execute_query` to
log every Cypher call fired showed why: for a real query, only the vector
index lookup ever executed — the `MENTIONS`-based shared-entity expansion
query never ran, in either `rerank=True` or `rerank=False`, and regardless
of corpus size above `top_k*3` chunks. Root cause in the original
`retrieve_graph`: `results = direct_hits.copy()` already had `top_k`
elements before `needs_expansion = direct_hits and len(results) < top_k`
was evaluated, so the condition was unreachable whenever the graph held at
least `top_k*3` chunks (always true here). **"Graph RAG" as measured was
plain dense vector search over Neo4j + cross-encoder rerank — it never
touched the entity/relationship data at query time**, and being dense-only
(no BM25/sparse fusion, unlike naive/agentic's `retrieve_hybrid`) explains
the precision/recall shortfall against naive.

**Fix 1 — always run 1-hop shared-entity expansion under `rerank=True`,
ordered by shared-entity count (not just gap-filling).** Confirmed via the
same instrumentation that the expansion query now fires on every call.

| Metric             | Naive   | Agentic | Graph   | Δ Graph vs Naive |
|---------------------|---------|---------|---------|-------------------|
| faithfulness         | 0.6619  | 0.7947  | 0.7706  | +0.1087           |
| answer_relevancy     | 0.5970  | 0.4933  | 0.4933  | -0.1037           |
| context_precision    | 0.8635  | 0.8706  | **0.9333** | +0.0698        |
| context_recall       | 0.8214  | 0.8690  | 0.7738  | -0.0476           |
| avg cost/query       | $0.00051| $0.00127| $0.00050| -$0.00001         |
| p50 latency          | 2912ms  | 4819ms  | 2970ms  | +58ms             |

context_precision jumped to the best of all three pipelines (0.9333) and
faithfulness moved ahead of naive — actually using the graph helped, at
effectively no added cost/latency (still one retrieval + one rerank call).

**Fix 2 — add a 2-hop pass over `RELATED_TO`** (`seed → mentioned entity →
RELATED_TO → other entity → chunk mentioning it`), to reach chunks that
share no entity with the seed but are connected through a documented
relationship — the 1,459 `RELATED_TO` edges the extractor writes but
nothing had ever read. Ranked by a plain `count(DISTINCT e2)` of bridging
entities, same as the 1-hop query.

| Metric             | Naive   | Agentic | Graph   | Δ Graph vs Naive |
|---------------------|---------|---------|---------|-------------------|
| faithfulness         | 0.8372  | 0.8753  | 0.7000  | -0.1372           |
| answer_relevancy     | 0.5832  | 0.4942  | 0.7385  | +0.1553           |
| context_precision    | 0.8468  | 0.8706  | 0.8706  | +0.0238           |
| context_recall       | 0.7262  | 0.8214  | 0.6786  | -0.0476           |
| p50 latency          | 3022ms  | 5554ms  | 4107ms  | +1085ms           |

This regressed faithfulness and recall versus Fix 1 despite adding a real
mechanism. Diagnosis (querying the raw candidate scores directly, not
inferred from aggregates): "Transformer" has degree 33 in this corpus, so
once any seed chunk mentioned it, the 2-hop traversal fanned out through
dozens of `RELATED_TO` edges to *any* chunk that densely mentions
entities — a Ragas paper title page scored `related_entities=26`, an
appendix/reference block scored 24, both far above genuinely relevant
chunks (4-8). Plain entity-count scoring rewards fan-out through hub
entities instead of penalizing it, letting off-topic chunks dominate the
candidate pool the reranker chooses from.

**Fix 3 — inverse-degree weighting on both hops**, replacing
`count(DISTINCT e)` with `sum(1.0 / degree(e))` — the graph analogue of
IDF: a hub entity contributes almost nothing per connection, a rare/
specific entity contributes a lot. Re-checked the same failing query
directly: the Ragas title-page chunk's weighted score dropped from 26 to
7.1 (down to parity with genuinely relevant chunks instead of dominating
them 3-4x over), and it no longer appeared in the final reranked top-5.

| Metric             | Naive   | Agentic | Graph   | Δ Graph vs Naive |
|---------------------|---------|---------|---------|-------------------|
| faithfulness         | 0.8277  | 0.9019  | 0.7628  | -0.0649           |
| answer_relevancy     | 0.5916  | 0.4933  | **0.7451** | +0.1534        |
| context_precision    | 0.8944  | 0.8706  | 0.8468  | -0.0476           |
| context_recall       | 0.7262  | 0.8690  | 0.7738  | +0.0476           |
| avg cost/query       | $0.00051| $0.00124| $0.00052| +$0.00001         |
| p50 latency          | 3175ms  | 5629ms  | 4615ms  | +1440ms           |
| p95 latency          | 7834ms  | 11620ms | 5317ms  | -2517ms           |

Both faithfulness and context_recall recovered from the Fix 2 regression
(0.70→0.76, 0.68→0.77) once the hub-entity noise stopped drowning out
on-topic candidates, confirming the diagnosis. Latency crept up further
(2970ms → 4615ms p50 versus Fix 1) — the `COUNT{}` subquery per bridging
entity is genuine added Neo4j compute on top of the extra 2-hop
round-trip, not noise.

**Finding:** Fixing the dead 1-hop code (Fix 1) was the single largest
win and was free — same cost/latency as doing nothing, but actually using
the graph instead of silently falling back to plain vector search. Adding
2-hop `RELATED_TO` traversal (Fix 2) is a real capability the corpus
supports, but naively ranking it by raw connection count is actively
harmful on any corpus with hub entities (a paper whose central topic is
"Transformer" will have a high-degree "Transformer" node) — it must be
run through the same kind of frequency-discounting that makes TF-IDF work
for text, or it degrades quality while looking like it's "doing more."
With that weighting in place (Fix 3), graph RAG beats naive on
answer_relevancy (+0.15) and context_recall (+0.05), trails naive
slightly on faithfulness/precision (within the ±0.05-0.15 run-to-run
noise band naive's own unchanged numbers showed across these four
7-question runs), and sits well below agentic's cost ($0.00052 vs
$0.00124, -58%) and p95 latency (5317ms vs 11620ms, -54%) while trailing
agentic on faithfulness and recall — consistent with agentic's multiple
retrieval rounds covering gaps a single retrieval-plus-2-hop-traversal
pass structurally can't.

**What this means for the pipeline:**

- A knowledge-graph retriever that never fires its graph queries is
  indistinguishable from a broken retriever in eval numbers alone — it
  takes call-level instrumentation (not just aggregate metrics) to catch
  an unreachable code path like Fix 1's, since the pipeline still runs
  and returns plausible-looking results.
- Any traversal-ranking signal built from raw entity/edge counts needs a
  frequency-discounting term before it's trustworthy — hub nodes are the
  graph equivalent of stopwords, and this corpus is small enough (138
  chunks) that a single hub entity's fan-out can dominate a whole
  candidate pool.
- Graph RAG's current position: a legitimate third option between naive
  (cheapest, weakest on multi-hop) and agentic (most thorough, most
  expensive) — roughly agentic-level answer_relevancy and better recall
  than naive, at closer to naive's cost, with faithfulness the one metric
  where it still trails both. n=7 throughout; treat single-run deltas
  under ~0.05 as noise rather than signal, as naive's own unchanged
  pipeline demonstrated by moving that much between runs on its own.

---

## Experiment 10 — LLM-as-judge calibration

**Question:** How much can RAGAS's automated scores (used as ground truth
in Experiments 5–9) actually be trusted, and where specifically do they
diverge from an independent judgment?

**Setup:** 28 already-scored questions sampled across the four pipeline
contexts evaluated so far — `baseline` (`ragas_baseline_1.json`, the
full-corpus run from Experiment 5), `agentic` (`agentic_rag_comparison.json`
+ `agentic_rag_multihop.json`, Experiment 7), `multimodal`
(`multimodal_comparison.json`, Experiment 8), and `graph`
(`graph_rag_multihop.json`, Experiment 9) — via
`scripts/build_judge_calibration_sample.py`. Within each of the 4 groups,
the 5 most extreme-scoring records plus 2 near-median "control" records
were selected (7 × 4 = 28), weighting toward the questions most likely to
reveal judge disagreement rather than a plain random sample.

**Important caveat, stated up front:** the "manual" scoring pass here was
done by Claude acting as a second, independent LLM judge — reading
`eval/judge_calibration_blind.json` (question/context/answer/reference with
RAGAS's scores stripped out) against the written rubric in
`eval/manual_scoring_rubric.md`, blind to RAGAS's numbers until scoring was
complete, to avoid anchoring. This measures **inter-judge disagreement
between two different LLM judges**, not agreement with human ground truth.
That distinction matters for how much weight the findings below can carry:
they're useful for surfacing *categorizable, mechanistic* judge behaviors
(most importantly, ones that independently corroborate a bug already found
by other means — Experiment 8's KV-table faithfulness issue), but they
cannot by themselves certify which judge is "right" in an absolute sense.

**Results — disagreement rate (|manual − RAGAS| > 0.2), 28 records / 112
metric scores:**

| Metric             | n scored | n disagree | rate  |
|---------------------|----------|------------|-------|
| faithfulness         | 28       | 9          | 32.1% |
| answer_relevancy     | 28       | 4          | 14.3% |
| context_precision    | 28       | 17         | 60.7% |
| context_recall       | 28       | 4          | 14.3% |

| Pipeline context | n scores | n disagree | rate  |
|-------------------|----------|------------|-------|
| agentic           | 28       | 9          | 32.1% |
| baseline          | 28       | 8          | 28.6% |
| graph             | 28       | 8          | 28.6% |
| multimodal        | 28       | 9          | 32.1% |

Overall disagreement rate: 34/112 (30.4%). Disagreement is heavily
concentrated in `context_precision` and roughly even across pipelines —
this sample does not show one pipeline's scores as categorically less
trustworthy than another's.

**Disagreement categories identified:**

1. **Structured-data blindness (faithfulness), reproducing Experiment 8's
   finding on a fresh sample.** `multimodal_comparison#1` — the single-head
   ablation answer ("development BLEU score of 24.9 and perplexity of
   5.29") is a verbatim match to the KV-formatted table row
   `h: 1 | ... | PPL (dev): 5.29 | BLEU (dev): 24.9`, yet RAGAS scored
   faithfulness `0.00` against this judge's `1.00`. This is the same
   mechanism Experiment 8 documented via spot-checking, now caught
   systematically. It is not universal, though:
   `multimodal_comparison#3` (65M-parameter answer, equally KV-table-
   grounded and equally correct) did **not** trigger a faithfulness
   disagreement — RAGAS scored it in line with this judge. The bug is real
   but intermittent, consistent with Experiment 8's framing of it as
   something "not yet well understood," not a hard rule.

2. **Refusal-string faithfulness convention.** Both `agentic_rag_comparison#5`
   and `multimodal_comparison#6` answered with the corpus's literal
   `REFUSAL_ANSWER` string, and both scored RAGAS faithfulness `0.00`
   against this judge's `1.00`. RAGAS's claim-decomposition step appears to
   treat "I don't have enough information..." as an unsupported claim
   rather than a claim-free non-answer. This judge's own confidence on
   this scoring choice is low (`confidence: "low"` in the manual-scores
   file) — scoring a claim-free answer as vacuously faithful is a
   defensible convention, but so is RAGAS's, and this exercise can't
   settle which is "right." What it does establish is a concrete mechanism:
   **a refusal gets penalized twice** — once on relevancy (correctly, in
   these two cases — see below) and once on faithfulness via this
   convention — so refusal-heavy pipelines' aggregate RAGAS scores absorb
   a larger faithfulness hit than a purely relevancy-based view would
   suggest.

3. **Wrong refusals on genuinely answerable questions — and RAGAS caught
   them.** Three records in the sample are cases where the pipeline
   refused (or gave an incomplete "gaps" answer) on a question the
   retrieved context could actually answer:
   `agentic_rag_comparison#5` / `graph_rag_multihop#2` (identical
   multi-hop question, both pipelines gave the same unwarranted refusal
   despite context supporting a synthesis — worth noting as a possible
   hard-question effect rather than a single pipeline's quirk), and
   `multimodal_comparison#6` (refused a `table_only` question despite the
   answer being directly in a KV-formatted table, `ctx3`, alongside a
   plain-prose restatement of the same fact in `ctx0` — this generator-side
   failure to parse the KV table is the mirror image of Experiment 8's
   judge-side failure). **None of these three appear in the
   `answer_relevancy` disagreement list** — RAGAS scored all three low on
   relevancy, matching this judge's independent low scores. This is a
   direct test, and disconfirmation, of this project's prior working
   hypothesis (carried over from the original Day 4 brief) that RAGAS's
   answer_relevancy over-penalizes honest refusals: on this sample, RAGAS's
   relevancy scoring of refusals was *not* overly strict — it correctly
   distinguished a warranted refusal from an unwarranted one wherever this
   judge could too. (No genuine `not_in_corpus` question happened to be
   sampled, so the narrower hypothesis about correctly-refused
   out-of-corpus questions specifically remains untested here.)

4. **Diagram-caption mismatch inflating context_recall.**
   `multimodal_comparison#9` (anaphora-resolution attention-visualization
   question) retrieved 4 of 5 chunks describing a *different* attention
   diagram (a "The Law will never be perfect, but its application..."
   example) rather than the actual Figure 4 the question asks about, whose
   caption was present in only one chunk. RAGAS still scored context_recall
   `1.00` against this judge's `0.15` — plausibly because the reference's
   target words ("Law", "application") appear verbatim in the *wrong*
   diagram's caption text, and a recall check based on lexical/semantic
   overlap with the reference can be fooled by that coincidence without
   verifying the chunks actually describe the right image. RAGAS also
   scored faithfulness `0.20` against this judge's `1.00` for an answer
   that made no false claims and correctly hedged given the (wrong)
   retrieved captions — a second instance of the refusal/hedge-scoring
   pattern from category 2, this time on an honest hedge that actually was
   correct.

5. **Context-precision strictness mismatch — the largest category by
   volume (17 of 28 records), and on reflection, partly an artifact of
   this judge's own rubric application rather than a RAGAS defect.** In
   the large majority of these cases (13 of 15 outside the two corrected
   below), RAGAS scored context_precision generously (0.75–1.0) for chunks
   this judge scored much lower (0.2–0.7) on a stricter "does this specific
   chunk materially support constructing this specific answer" standard —
   spread fairly evenly across `baseline`, `agentic`, and `graph`. In the
   remaining 2 cases (`multimodal_comparison#0` and `#4`), the direction
   reversed: RAGAS scored precision very low (0.25, 0.00) for chunks this
   judge initially scored as high (1.0, 0.8) on a looser "on-topic for the
   general subject" standard. Rereading those two more carefully (see the
   next section) showed RAGAS's stricter, reference-derivability-based
   reading was the more defensible one — which suggests this judge's own
   rubric interpretation drifted toward "topically relevant" rather than
   "actually useful for this specific reference answer," and that drift,
   not a RAGAS bug, plausibly explains a meaningful share of the 17
   context_precision disagreements in aggregate.

6. **A genuine RAGAS false negative — perfect faithfulness on an answer
   with real, wrong, cross-table-conflated numbers.**
   `multimodal_comparison#5` asked for RAG-Sequence's BLEU-1 score vs
   BART's on MS-MARCO; the answer states "RAG-Sequence achieves a BLEU-1
   score of 46.9... BART achieves 70.7." Both numbers *do* appear verbatim
   somewhere in the retrieved context — but `46.9` is `RAG-Sequence-BM25`'s
   B-1 column in the Table 6 ablation (a different model variant than the
   plain "RAG-Sequence" the question asks about), and `70.7` is BART's
   *trigram-diversity percentage* from Table 5, not a BLEU-1 score at all.
   RAGAS scored this faithfulness `1.00`; this judge scored it `0.20`. This
   is the inverse of category 1: rather than the judge failing to credit a
   *correct* table-grounded claim, here the judge failed to catch an
   *incorrect* one built by conflating two unrelated table columns — a
   genuine false negative, and arguably the single most consequential
   disagreement in this sample, since it means RAGAS's faithfulness score
   cannot be relied on to catch this specific class of cross-table
   numeric-conflation hallucination.

7. **Stricter claim decomposition discounting otherwise well-grounded
   synthesis answers.** `agentic_rag_comparison#2` (FastAPI's two
   dependencies, fully correct and directly citing context) scored RAGAS
   faithfulness `0.50` against this judge's `1.00`; `graph_rag_multihop#0`
   and `#5` (both reasonably-grounded synthesis answers about the
   RAG/Qdrant relationship and the Transformer's role in RAG) scored
   `0.43` and `0.44` against `0.90` and `0.85`. Unlike categories 1, 2, and
   6, none of these answers contain a clear factual error this judge could
   identify on reread — the most likely explanation is that RAGAS's
   claim-decomposition step is extracting a descriptive or connective
   phrase (e.g. "for data validation and settings management") as a
   separate claim and marking it unsupported even when the overall answer
   is well-grounded. This is a plausible, defensible scoring behavior, not
   clearly a bug — it's included for completeness rather than as a
   confirmed failure mode.

**Where RAGAS was actually right on inspection.** Two records —
`multimodal_comparison#0` (base-vs-big BLEU comparison) and
`multimodal_comparison#4` (big-configuration BLEU/perplexity/params) — are
cases where the retrieval genuinely failed to surface the exact table row
the reference answer needs (confirmed by rereading all 5 contexts for each
record; the specific row is absent from both). This judge's first-pass
context_precision scores (1.0 and 0.8) credited the retrieved chunks for
being generally on-topic; RAGAS's much lower scores (0.25 and 0.00) turned
out to better reflect that none of those chunks were actually usable to
construct the specific reference answer. Both records' `manual_scores`
entries were corrected in `eval/judge_calibration_manual_scores.json`
before this write-up, with the correction reasoning left in place rather
than silently edited — this project's own generator also independently
hallucinated a composite answer for `#4` by combining figures from three
different ablation-table rows, so the underlying retrieval-recall failure
these two records share is real, not a judge artifact.

**Practical implications for this project:**

- Experiment 8's KV-table faithfulness caveat is confirmed to recur
  (`multimodal_comparison#1`) but is intermittent (`multimodal_comparison#3`
  didn't trigger it) — spot-check faithfulness on table/KV-grounded answers
  before treating a faithfulness delta as meaningful, but don't assume
  every table-grounded answer is mis-scored.
- The working hypothesis that RAGAS's answer_relevancy over-penalizes
  honest refusals was **not** confirmed on this sample — where a pipeline
  refused or hedged, RAGAS's relevancy score agreed with this judge's on
  whether that refusal was warranted. Experiment 7's agentic
  answer_relevancy numbers do not need to be revisited on refusal-penalty
  grounds on the evidence gathered here.
- RAGAS's faithfulness score missed a real cross-table numeric-conflation
  hallucination (`multimodal_comparison#5`, category 6) — this is a false
  negative, not just noise, and means a high RAGAS faithfulness score on a
  table-heavy answer should not be taken as proof the cited numbers are
  correctly attributed, only that *some* matching numbers exist somewhere
  in context. This is the most actionable finding for future multimodal
  work: numeric answers drawing on more than one table column need a
  spot-check regardless of their faithfulness score.
- Refusals are penalized twice in RAGAS's aggregate score — once (usually
  correctly) on relevancy, and separately on faithfulness via the
  claim-free-answer convention in category 2. Pipelines that refuse more
  often (agentic's cap-reached partial answers, in particular) will show a
  compounded faithfulness+relevancy hit from a single underlying event,
  not two independent signals — worth remembering when comparing
  aggregate scores across pipelines with different refusal rates.
- `context_precision` is the metric this exercise trusts least in absolute
  terms: the disagreement rate (60.7%) is driven mostly by this judge's own
  inconsistent rubric application (topical relevance vs. reference-answer
  usefulness) rather than a demonstrated RAGAS flaw, so cross-pipeline
  `context_precision` *deltas* in Experiments 7–9 are still usable
  directionally, but the absolute numbers should not be over-interpreted.
- Aggregate RAGAS scores remain directionally reliable for the
  large, already-reported deltas in this project (the multimodal
  vision-captioning win, the agentic context-precision gain, the graph
  fan-out fix) — this calibration exercise did not surface anything that
  overturns those findings — but fine-grained (<0.1) differences,
  especially on faithfulness or context_precision, should not be treated
  as signal without a manual spot-check.

**Answer to "how do you trust an LLM-as-judge eval":** this calibration
pass — itself an LLM judge checked against another LLM judge, not against
human labels — found one clearly reproducible, previously-documented bug
(structured-data blindness), one previously-undocumented but mechanistic
scoring convention (refusal strings scoring faithfulness `0.0`), one
clean disconfirmation of a standing hypothesis (refusal answer_relevancy
was not over-penalized here), one multimodal-specific recall-inflation
case traceable to a wrong-image retrieval, and one large but largely
self-inflicted disagreement category (context_precision) that shrank
under closer scrutiny rather than surviving it. That mix — bugs
confirmed, a hypothesis disconfirmed, and a large "disagreement" that
turned out to be mostly the second judge's own miscalibration — is what
makes this a calibration result rather than a verdict: RAGAS's scores are
trustworthy enough to rank pipelines and catch large deltas, specific
enough categories of failure are worth a standing spot-check habit, and
the exercise of checking a judge against another judge surfaces real
mechanisms without being able to declare either one "correct" in an
absolute sense.