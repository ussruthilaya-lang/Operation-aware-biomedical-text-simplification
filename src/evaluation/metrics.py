# Core evaluation metrics for the operation-aware simplification pipeline.
#
# What lives here (PRELIM scope):
#   - SARI  (readability-oriented n-gram edit metric, primary score)
#   - FKGL  (Flesch-Kincaid Grade Level, readability grade)
#   - Compression ratio (length change, for context)
#
# Deliberately NOT here:
#   - Warning preservation  -> src/evaluation/warning_preservation.py (Sruthilaya)
#   - BERTScore             -> Rishabh
#   - Entity / numerical preservation, SBERT  -> Final paper scope
#
# Why SARI + FKGL together:
# SARI rewards good simplification edits vs references. FKGL scores raw
# reading level. Neither one, alone, tells you whether critical info
# survived — that's what the safety metrics (warning / entity / number
# preservation) are for. The whole point of our paper is showing this gap.

from easse.sari import corpus_sari
import textstat


def compute_sari(source, hypothesis, references):
    """
    Corpus-level SARI over a single (source, hypothesis) pair.

    Args:
        source:     original sentence (str)
        hypothesis: system output (str)
        references: list of reference simplifications (list[str])

    Returns:
        float: SARI score in [0, 100]
    """
    # easse expects lists of strings for sources / hypotheses,
    # and a list-of-lists for references where refs_sents[i] is the
    # list of ALL i-th references across the corpus. For a single pair
    # with N references, that's N inner lists each of length 1.
    refs_sents = [[ref] for ref in references]
    return corpus_sari(
        orig_sents=[source],
        sys_sents=[hypothesis],
        refs_sents=refs_sents,
    )


def compute_corpus_sari(sources, hypotheses, references_list):
    """
    Corpus-level SARI over a batch.

    Args:
        sources:         list[str]
        hypotheses:      list[str]
        references_list: list[list[str]]  -- references_list[i] is the
                         list of references for the i-th example.
                         Different examples may have different numbers
                         of references (PLABA has 1-4 per source).

    Returns:
        float: corpus SARI in [0, 100]
    """
    # Transpose to easse's expected shape: refs_sents[k][i] = k-th ref
    # for example i. Pad shorter reference lists with the first reference
    # so every column has the same length (easse requires this).
    max_refs = max(len(r) for r in references_list)
    refs_sents = []
    for k in range(max_refs):
        col = []
        for refs in references_list:
            col.append(refs[k] if k < len(refs) else refs[0])
        refs_sents.append(col)

    return corpus_sari(
        orig_sents=sources,
        sys_sents=hypotheses,
        refs_sents=refs_sents,
    )


def compute_fkgl(text):
    """
    Flesch-Kincaid Grade Level.
    Lower = easier. Target for patient-facing text is roughly grade 8.
    """
    return textstat.flesch_kincaid_grade(text)


def compute_compression_ratio(source, hypothesis):
    """
    Word-count ratio of hypothesis to source.
    <1.0 means shorter; >1.0 means the "simplification" grew.
    """
    src_len = len(source.split())
    if src_len == 0:
        return 0.0
    return len(hypothesis.split()) / src_len


def compute_all_metrics(source, hypothesis, references):
    """
    Per-example convenience wrapper. Note that SARI computed here is a
    single-example score; for reporting numbers in the paper always use
    the corpus-level function.
    """
    return {
        'sari': compute_sari(source, hypothesis, references),
        'fkgl': compute_fkgl(hypothesis),
        'compression_ratio': compute_compression_ratio(source, hypothesis),
    }


if __name__ == "__main__":
    src = "The patient was diagnosed with myocardial infarction."
    hyp = "The patient was diagnosed with a heart attack."
    refs = [
        "The patient had a heart attack.",
        "The patient was diagnosed with a heart attack.",
    ]

    print("=== metrics.py smoke test ===")
    print(f"Source:     {src}")
    print(f"Hypothesis: {hyp}")
    print(f"References: {refs}")
    print()
    m = compute_all_metrics(src, hyp, refs)
    for k, v in m.items():
        print(f"  {k}: {v}")