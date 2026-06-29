# Prompt Changelog

Each entry covers one version of one prompt. Fill in RAGAS scores after running eval.

---

## grounded_qa · v2 · week3_day5 · commit 3b70da2

**Description:** Switched to inline `[N]` numeric citation markers. Context blocks
are now prefixed `[1]`, `[2]` … `[N]` so the model can reference them by number.
Rule 3 was rewritten to require per-claim placement, forbid end-grouping, and
forbid other citation formats.

**RAGAS scores:** _pending — eval not yet run against v2_

**Changes from v1:**
- Rule 3: replaced `[doc_title, chunk N]` with inline `[N]` marker per claim
- Context block: added `[{{ loop.index }}]` numeric prefix to each source header
- Rule 3: added sub-rules forbidding end-grouping and non-numeric formats
- Rule 3: added `chunks|length` bound to constrain valid citation numbers

---

## grounded_qa · v1 · week1 · commit fb01ccb (RAGAS baseline)

**Description:** Baseline grounded Q&A. Citations appended per-claim as
`[doc_title, chunk N]`; context blocks unlabelled. Refusal string and
contradiction rule already present.

**RAGAS scores (this is the baseline all future versions compare against):**

| Metric             | Score |
|--------------------|-------|
| faithfulness       | 0.74  |
| answer_relevancy   | 0.81  |
| context_precision  | 0.69  |
| context_recall     | 0.72  |

**Changes from:** _(initial version)_

---

<!-- Template for next version:

## grounded_qa · v3 · <introduced_in> · commit <sha>

**Description:** <what changed and why>

**RAGAS scores:**

| Metric             | Score |
|--------------------|-------|
| faithfulness       |       |
| answer_relevancy   |       |
| context_precision  |       |
| context_recall     |       |

**Delta vs v2:**
| Metric             | v2 | v3 | Δ |
|--------------------|----|----|---|
| faithfulness       |    |    |   |

**Changes from v2:**
- <bullet per meaningful change>

-->
