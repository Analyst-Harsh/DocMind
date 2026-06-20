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

| Model                    | Precision@5 | Recall@5 | MRR  | Embed latency | Cost/1K tokens |
|--------------------------|-------------|----------|------|---------------|----------------|
| text-embedding-3-small   | —           | —        | —    | —             | $0.02          |
| BGE-large-en-v1.5 (local)| —           | —        | —    | —             | $0.00          |

**Finding:** _Fill in after Day 4_

---

## Experiment 3 — RAG vs long-context

_Week 4_

---

## Experiment 4 — GraphRAG vs flat retrieval

_Week 4_