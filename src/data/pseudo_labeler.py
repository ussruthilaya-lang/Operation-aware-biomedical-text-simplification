
import os
import json

# --- Heuristic thresholds -----------------------------------------------------
EXPANSION_THRESHOLD = 1.15      # target >= 15% longer  -> Explanation
COMPRESSION_THRESHOLD = 0.85    # target <= 15% shorter -> Generalization

# For similar-length pairs, we default to Substitution after ruling out
# Explanation/Generalization by length ratio.


def _levenshtein(a, b):
    """
    Character-level edit distance (insertions + deletions + substitutions).

    Pure-Python DP so the labeler has zero external dependencies — it must be
    runnable by every teammate without installing anything. Sentences are short,
    so O(len(a) * len(b)) is fine.
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            insert = current[j - 1] + 1
            delete = previous[j] + 1
            replace = previous[j - 1] + (ca != cb)
            current.append(min(insert, delete, replace))
        previous = current
    return previous[-1]


def _normalized_edit_distance(source, target):
    """Edit distance scaled to [0, 1] by the longer string — 0 = identical."""
    if not source and not target:
        return 0.0
    distance = _levenshtein(source, target)
    return distance / max(len(source), len(target))

    # following instuction function 
def label_span(source_span, target_span):
    """
    Substitution: similar length, high character edit distance
    Explanation: target significantly longer than source
    Generalization: target significantly shorter than source
    """
    source_words = source_span.split()
    target_words = target_span.split()

    # Degenerate guards. An empty source can't have been shortened, and an
    # empty target means everything was dropped — treat as generalization.
    if len(source_words) == 0:
        return "Explanation"
    if len(target_words) == 0:
        return "Generalization"

    length_ratio = len(target_words) / len(source_words)

    if length_ratio >= EXPANSION_THRESHOLD:
        return "Explanation"
    if length_ratio <= COMPRESSION_THRESHOLD:
        return "Generalization"

    # Similar length: treat as Substitution (length-based heuristics already ruled
    # out Explanation/Generalization).
    return "Substitution"

    #Following main instruction
def label_sentence_pair(source, target):
    return label_span(source.strip(), target.strip())


def label_dataset(df, source_column="source", target_column="target"):
    """
    Run labeling over a full DataFrame of aligned pairs.
    """
    df = df.copy()
    df["operation"] = [
        label_sentence_pair(src, tgt)
        for src, tgt in zip(df[source_column], df[target_column])
    ]
    return df


def build_pairs_from_plaba_json(json_path, exclude_pmids=None):
    """
    Walk the PLABA data.json tree and produce sentence-aligned source->target
    pairs.

    Structure of data.json:
        { question_id: { pmid: {
              "Title": str,
              "abstract":    { "1": src_sent, "2": src_sent, ... },
              "adaptations": { "adaptation2": { "1": tgt_sent, "2": tgt_sent, ... }, ... }
        } } }

    Source and adaptation sentences share the same numeric key, which gives us
    free sentence-level alignment (source sentence N <-> adaptation sentence N).

    We iterate EVERY adaptation available for a PMID (some abstracts have more
    than one plain-language version) to maximize training volume, and skip any
    sentence number that is missing on either side — alignment is mostly but
    not perfectly 1:1 because writers occasionally merge or split sentences.

    Returns a pandas DataFrame with columns:
        question, pmid, adaptation, sent_id, source, target
    """
    import pandas as pd

    exclude_pmids = set(str(p) for p in (exclude_pmids or set()))

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for question_id, pmids in data.items():
        for pmid, record in pmids.items():
            # Each question block also carries string metadata keys
            # ("question", "question_type") alongside the PMID records.
            # Only real records are dicts with an "abstract" — skip the rest.
            if not isinstance(record, dict) or "abstract" not in record:
                continue

            # data.json is the FULL PLABA corpus, so it also contains the
            # abstracts held out in val.csv / test.csv. Training on those and
            # evaluating on them would be data leakage — exclude them so the
            # train/val/test split stays honest.
            if str(pmid) in exclude_pmids:
                continue

            abstract = record.get("abstract", {})
            adaptations = record.get("adaptations", {})

            for adaptation_name, adaptation_sents in adaptations.items():
                # Align on sentence numbers present in BOTH source and target.
                shared_ids = set(abstract.keys()) & set(adaptation_sents.keys())
                for sent_id in sorted(shared_ids, key=lambda x: int(x)):
                    source = abstract[sent_id].strip()
                    target = adaptation_sents[sent_id].strip()
                    if not source or not target:
                        continue
                    rows.append({
                        "question": question_id,
                        "pmid": pmid,
                        "adaptation": adaptation_name,
                        "sent_id": sent_id,
                        "source": source,
                        "target": target,
                    })

    return pd.DataFrame(rows)


# Repo-root-relative paths so the script runs from anywhere.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DEFAULT_JSON = os.path.join(_REPO_ROOT, "data", "plaba", "data.json")
_DEFAULT_OUT = os.path.join(_REPO_ROOT, "results", "pseudo_labeled_train.csv")
_VAL_CSV = os.path.join(_REPO_ROOT, "data", "plaba", "val.csv")
_TEST_CSV = os.path.join(_REPO_ROOT, "data", "plaba", "test.csv")


def _held_out_pmids(*csv_paths):
    """Collect PMIDs from val/test CSVs so we never train on them."""
    import pandas as pd

    pmids = set()
    for path in csv_paths:
        if os.path.exists(path):
            pmids.update(str(p) for p in pd.read_csv(path)["pmid"].unique())
    return pmids


def build_labeled_training_set(json_path=_DEFAULT_JSON, out_path=_DEFAULT_OUT):
    """
    End-to-end: load data.json -> drop val/test PMIDs -> align pairs
    -> label -> write CSV. Returns the labeled DataFrame.
    """
    exclude = _held_out_pmids(_VAL_CSV, _TEST_CSV)
    df = build_pairs_from_plaba_json(json_path, exclude_pmids=exclude)
    labeled = label_dataset(df)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    labeled.to_csv(out_path, index=False)
    print(f"Excluded {len(exclude)} held-out PMIDs (val/test) to prevent leakage.")
    return labeled


if __name__ == "__main__":
    # Quick sanity check on canonical examples before touching the corpus.
    examples = [
        # Substitution: similar length, jargon swapped for plain English.
        ("The patient suffered a myocardial infarction.",
         "The patient suffered a heart attack."),
        # Explanation: target adds a parenthetical unpacking the term.
        ("Botulinum toxin injections were administered.",
         "Botulinum toxin (used as a muscle relaxant) injections were given to the patient."),
        # Generalization: dense clinical detail compressed away.
        ("A dose-dependent improvement of bladder capacity (5-fold) and "
         "periurethral EMG activity (8-fold) of the striated sphincter muscles was found.",
         "The drug improved bladder capacity and control."),
    ]
    print("=== label_span sanity check ===")
    for src, tgt in examples:
        print(f"[{label_span(src, tgt):14s}] {src[:55]}... -> {tgt[:55]}...")

    # Full corpus run.
    print("\n=== Building labeled training set from PLABA data.json ===")
    labeled = build_labeled_training_set()
    print(f"Aligned sentence pairs: {len(labeled)}")
    print("\nOperation label distribution:")
    counts = labeled["operation"].value_counts()
    for label, count in counts.items():
        print(f"  {label:15s} {count:5d}  ({count / len(labeled):.1%})")
    print(f"\nWrote: {_DEFAULT_OUT}")
