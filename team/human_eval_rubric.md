# Human Evaluation Rubric — Biomedical Text Simplification

Used alongside `results/human_eval_sample.csv`. Each row is one (source
sentence, baseline system output) pair. Rate each of the 5 dimensions below
on a 1-5 scale. Leave no blank cells — if a dimension genuinely doesn't
apply (e.g., "actionability" for a sentence with no clinical action implied),
use 3 (neutral) and note why in a comments column if you add one.

**Do not compare across baselines while rating a single row.** Rate each
system output on its own merits against the source sentence. Comparison
happens later, in aggregate analysis — not during annotation, to avoid
anchoring one baseline's score against another's.

---

## 1. Correctness
*Does the output accurately reflect the factual/clinical content of the source?*

| Score | Meaning |
|---|---|
| 5 | Fully accurate — no factual, clinical, or numerical errors introduced |
| 4 | Accurate with a very minor imprecision that doesn't change clinical meaning |
| 3 | Mostly accurate, but contains at least one imprecision a clinician would flag |
| 2 | Contains a factual or clinical error that could mislead a reader |
| 1 | Contains a serious factual/clinical error (e.g., wrong drug, wrong dosage, reversed meaning) |

## 2. Completeness
*Does the output preserve all clinically relevant information from the source (not just safety warnings — see separate Safety dimension)?*

| Score | Meaning |
|---|---|
| 5 | Nothing clinically relevant is missing |
| 4 | Minor supporting detail omitted, doesn't affect understanding |
| 3 | Some relevant detail missing, but core message intact |
| 2 | Significant relevant information dropped |
| 1 | Core clinical content is missing or the output is substantially incomplete |

## 3. Readability
*Would a layperson without medical training understand this output?*

| Score | Meaning |
|---|---|
| 5 | Clear, plain language throughout; no unexplained jargon |
| 4 | Mostly plain language; one term might need a dictionary lookup |
| 3 | Understandable with effort; several unfamiliar terms |
| 2 | Difficult for a layperson; jargon-heavy or awkward phrasing |
| 1 | Unreadable for the intended audience; effectively still clinical language |

## 4. Actionability
*If the sentence implies something the patient should do (take a medication a certain way, avoid something, seek care), is that action clear and unambiguous?*

| Score | Meaning |
|---|---|
| 5 | Any implied action is stated clearly and unambiguously |
| 4 | Action is clear with very minor ambiguity |
| 3 | Action is present but requires some inference from the reader |
| 2 | Action is unclear or could be misinterpreted |
| 1 | A necessary action is missing entirely or dangerously unclear |
| N/A → use 3 | Sentence implies no patient action (e.g., general background info) |

## 5. Safety
*Specifically: are contraindications, warnings, or adverse-effect information preserved? This is the dimension most directly tied to our warning-preservation metric — rate independently of your own metric score, based on your judgment reading the text.*

| Score | Meaning |
|---|---|
| 5 | All safety-critical information from the source is present and clear |
| 4 | Safety information present, slightly weakened in emphasis |
| 3 | Safety information present but easy to miss or de-emphasized |
| 2 | Safety information is vague, softened in a way that changes risk perception |
| 1 | Safety-critical information (contraindication, warning, adverse effect) is dropped entirely |
| N/A → use 5 | Source sentence has no safety-critical content (this is the majority-safe default, not a penalty) |

---

## Notes on stratified sampling

This sample deliberately **oversamples warning-bearing sentences** (~50% of
the sample, vs. their ~33% natural base rate in the data). This is
intentional: purely random sampling under-detects rare safety failures (see
prelim's "Critical methodological finding"). When analyzing results, report
warning-bearing and non-warning-bearing subsets **separately**, not
blended into one aggregate score — blending would understate the sample's
deliberate oversampling and misrepresent the true population base rate if
misread as a natural sample.

## Annotator process

1. Read the `source` column first, in full, before looking at `system_output`
2. Rate all 5 dimensions for that one row
3. Move to the next row — do not revisit earlier ratings after seeing later rows, to avoid drift
4. If uncertain between two adjacent scores, round down (be conservative on safety-adjacent dimensions especially)