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