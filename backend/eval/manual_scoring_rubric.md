# Manual scoring rubric — LLM-judge calibration (Experiment 10)

## Framing

This rubric is applied by Claude, acting as an **independent second LLM judge**, scoring the
records in `eval/judge_calibration_blind.json` — not by a human rater. The point of this
exercise is to measure *inter-judge disagreement* between this second judge and RAGAS's judge
model, and to categorize the disagreement's likely causes (e.g. does judge #2 also stumble on
KV-table context, or is that specific to RAGAS's own claim-decomposition step?). It is **not**
an attempt to establish human ground truth. Every downstream artifact (the manual scores file,
the disagreement analysis, the `EXPERIMENTS.md` write-up) must carry this caveat forward rather
than imply "manual" means "human."

Score from `eval/judge_calibration_blind.json` only. Do not open
`eval/results/judge_calibration_sample.json` (which has RAGAS's scores) until every record below
has been scored — anchoring on RAGAS's number before forming an independent judgment is the
specific failure mode this guards against.

## Process

For each record: read `question`, all of `contexts`, `answer`, and `reference` in full before
scoring anything. Score the 4 metrics independently — don't let a low faithfulness score bias
the recall judgment, or vice versa. Record a one-sentence justification per metric. Mark
`confidence: "low"` on any record where the scoring judgment itself feels genuinely ambiguous
(not just where the score is extreme).

## Metric definitions

Match RAGAS's own metric semantics as closely as possible — the goal is comparing two judgments
of the *same construct*, not substituting a different personal notion of "good answer."

**Faithfulness (0.0–1.0).** What fraction of factual claims in `answer` are directly supported
by `contexts`? Score claim-by-claim, then average. 1.0 = every claim traceable to context, 0.0 =
answer is entirely unsupported/hallucinated relative to context (note: this is about
context-groundedness, not real-world truth — an answer can be faithful to a wrong context).

Claims grounded in `Header: value | Header: value`-style table rows **count as supported** if
the row's data matches the claim, even though the surrounding text isn't prose. This is exactly
the behavior Experiment 8 found RAGAS's own judge to miss (`faithfulness=0.0` on a
verbatim-correct table-grounded answer) — this rubric must not repeat that mistake by only
recognizing prose-shaped support.

**Answer relevancy (0.0–1.0).** Does `answer` actually address what `question` asked? Penalize
answers that are faithful but evasive or tangential.

If `category == "not_in_corpus"` and `answer` matches or closely paraphrases the corpus's fixed
refusal string (`"I don't have enough information in the provided documents to answer this."`),
score relevancy **high (≥0.8)** — a correct refusal to an unanswerable question is relevant, not
off-topic. This directly tests the "over-strict refusal penalization" hypothesis: does RAGAS's
own answer_relevancy score these lower than a second judge would?

**Context precision (0.0–1.0).** Of the chunks in `contexts`, what fraction are actually useful
for answering *this specific* question? Judge each chunk independently, then compute the
fraction useful.

**Context recall (0.0–1.0).** Does `contexts` contain enough information to fully construct
`reference`? If `reference` expresses N distinct claims and `contexts` supports M of them,
recall ≈ M/N.

## Output

Scores and reasoning go into `eval/judge_calibration_manual_scores.json`, keyed by `sample_id`,
per the schema in the Day 4 plan (`manual_<metric>`, `manual_<metric>_reasoning`, and a
per-record `confidence`).
