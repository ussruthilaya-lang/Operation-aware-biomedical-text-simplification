
import os

import pandas as pd


from src.evaluation.warning_preservation import compute_corpus_warning_preservation

MODEL_NAME = "google/flan-t5-base"
PROMPT_TEMPLATE = "Simplify this biomedical sentence for a patient: {text}"

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_TRAIN_CSV = os.path.join(_REPO_ROOT, "results", "pseudo_labeled_train.csv")
_OUT_CSV = os.path.join(_REPO_ROOT, "results", "baseline3_direct_llm.csv")


# The model is heavy to load, so we load it once and cache it. We load the
# seq2seq model directly (rather than via a pipeline) because newer transformers
# versions dropped the "text2text-generation" pipeline task — loading the model
# and tokenizer explicitly is version-proof.
_model = None
_tokenizer = None
_device = "cpu"


def _load(model=MODEL_NAME):
    """Lazily load the flan-t5 seq2seq model + tokenizer (GPU if available)."""
    global _model, _tokenizer, _device
    if _model is None:
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        _device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading {model} on {_device.upper()} ...")
        _tokenizer = AutoTokenizer.from_pretrained(model)
        _model = AutoModelForSeq2SeqLM.from_pretrained(model).to(_device)
    return _model, _tokenizer


def simplify(text, model=MODEL_NAME):
    """
    Simplify one biomedical sentence via direct LLM prompting — no operation
    guidance. This is the deliberately-naive baseline.
    """
    import torch

    mdl, tok = _load(model)
    prompt = PROMPT_TEMPLATE.format(text=text)
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(_device)
    with torch.no_grad():
        output_ids = mdl.generate(**inputs, max_new_tokens=256, num_beams=4)
    return tok.decode(output_ids[0], skip_special_tokens=True).strip()


def run_on_sample(df, n=30, text_column="source"):
    """
    Run the direct-LLM baseline on an n-sentence sample and measure warning
    preservation (the same safety metric used for Baselines 1 and 2).

    Returns (pairs, result) where pairs is a list of (source, simplified) and
    result is the corpus warning-preservation summary.
    """
    sample = df.head(n)
    pairs = []
    for i, source in enumerate(sample[text_column], start=1):
        simplified = simplify(source)
        pairs.append((source, simplified))
        print(f"[{i}/{len(sample)}] done")

    result = compute_corpus_warning_preservation(pairs)
    return pairs, result


if __name__ == "__main__":
    print("=== Baseline 3: Direct LLM prompting ===\n")

    # Quick qualitative look at a few simplifications.
    demo_sentences = [
        "This anticoagulant is contraindicated in patients with hepatic failure.",
        "Myocardial infarction risk increased 2.3-fold in the treatment group.",
        "Prophylaxis with subcutaneous heparin reduced the incidence of thrombosis.",
    ]
    for s in demo_sentences:
        print(f"Source:     {s}")
        print(f"Simplified: {simplify(s)}\n")

    # Quantitative run on a sample from the labeled corpus.
    # IMPORTANT: sample only sentences that actually CONTAIN a warning phrase —
    # otherwise warning preservation is trivially 1.0 (nothing to measure) and
    # the baseline proves nothing. This is exactly the safety-critical subset
    # our operation-aware system must protect.
    from src.complexity.warning_lexicon import get_warning_phrases

    df = pd.read_csv(_TRAIN_CSV)
    has_warning = df["source"].apply(lambda s: len(get_warning_phrases(str(s))) > 0)
    warning_df = df[has_warning].reset_index(drop=True)
    print(f"Warning-bearing sentences available: {len(warning_df)}")
    pairs, result = run_on_sample(warning_df, n=30)

    print("\n=== Warning preservation (Baseline 3) ===")
    print(f"Mean warning preservation: {result['mean_score']}")
    print(f"Warning sentences: {result['warning_sentences']}/{result['total_sentences']}")

    # Persist for the paper's baselines section.
    os.makedirs(os.path.dirname(_OUT_CSV), exist_ok=True)
    pd.DataFrame(pairs, columns=["source", "simplified"]).to_csv(_OUT_CSV, index=False)
    print(f"\nWrote: {_OUT_CSV}")
