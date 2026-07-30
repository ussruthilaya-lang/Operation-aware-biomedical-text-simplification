# End-to-End Operation-Aware Simplification Pipeline
#
# Orchestrates: Classifier → Router → Method (CHV or constrained LLM)
#
# Pipeline flow:
# 1. BioBERT classifier predicts operation type
# 2. Operation router routes to appropriate method
# 3. Execute method:
#    - Substitution → CHV lookup (Rishabh's code)
#    - Explanation → Constrained LLM with operation prompt
#    - Generalization → Constrained LLM with operation prompt
# 4. Return simplified sentence + metrics

import os
import sys
import torch
import pandas as pd
from pathlib import Path

# Add repo root to path
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO_ROOT)

from transformers import (
    BertTokenizer, BertForSequenceClassification,
    AutoTokenizer, T5ForConditionalGeneration
)
import re

from src.pipeline.operation_router import route_operation
from src.pipeline.constrained_prompts import get_prompt
from src.evaluation.warning_preservation import compute_warning_preservation_rate
from src.baselines.baseline2_rule_based_chv import chv_substitute
from src.data.chv_lookup import CHV_DICTIONARY
from src.complexity.numerical_extractor import extract_numerical_expressions
from src.complexity.warning_lexicon import detect_warnings

from src.classifier.biobert_classifier import CHECKPOINT_DIR as BIOBERT_CHECKPOINT_DIR

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSIFIER_MODEL = "dmis-lab/biobert-base-cased-v1.2"
LLM_MODEL = "google/flan-t5-base"
LABELS = ["Substitution", "Explanation", "Generalization"]


def _get_chv_replacements_applied(original_text):
    """
    Return the plain-English replacement terms chv_substitute() actually
    inserted for this sentence (reuses the same matching logic as
    chv_substitute itself, so the list is exactly what got substituted).
    """
    applied = []
    sorted_terms = sorted(CHV_DICTIONARY.keys(), key=len, reverse=True)
    for term in sorted_terms:
        pattern = re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE)
        if pattern.search(original_text):
            applied.append(CHV_DICTIONARY[term])
    return applied


class OperationAwarePipeline:
    """End-to-end operation-aware simplification pipeline."""

    def __init__(self, llm_model=None, chv_lookup_fn=None, load_llm=True):
        """
        Initialize the pipeline.

        Args:
            llm_model: LLM for Explanation/Generalization (e.g., FLAN-T5).
                      If None and load_llm=True, loads FLAN-T5-base.
            chv_lookup_fn: Function for CHV lookup (Substitution).
                          If None, uses placeholder.
            load_llm: Whether to load FLAN-T5 by default (True).
        """
        if not os.path.isdir(BIOBERT_CHECKPOINT_DIR):
            raise FileNotFoundError(
                f"No fine-tuned BioBERT checkpoint found at {BIOBERT_CHECKPOINT_DIR}. "
                f"Loading raw '{CLASSIFIER_MODEL}' here would silently attach a "
                f"randomly initialized (untrained) classification head and produce "
                f"meaningless operation predictions. Run "
                f"`python -m src.classifier.biobert_classifier` first to train and "
                f"save the checkpoint."
            )
        print(f"Loading fine-tuned BioBERT classifier from {BIOBERT_CHECKPOINT_DIR}...")
        self.tokenizer = BertTokenizer.from_pretrained(BIOBERT_CHECKPOINT_DIR)
        self.classifier = BertForSequenceClassification.from_pretrained(
            BIOBERT_CHECKPOINT_DIR
        ).to(DEVICE)
        self.classifier.eval()

        print("Loading FLAN-T5 LLM...")
        if llm_model is None and load_llm:
            self.llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
            self.llm_model = T5ForConditionalGeneration.from_pretrained(LLM_MODEL).to(DEVICE)
            self.llm_model.eval()
        else:
            self.llm_model = llm_model
            self.llm_tokenizer = None

        # Default to the real sentence-level CHV substitution (Rishabh's
        # chv_lookup.py, applied via baseline2's whole-sentence substitution
        # function) rather than a no-op placeholder, so Substitution-routed
        # sentences are actually simplified instead of silently passed through.
        self.chv_lookup_fn = chv_lookup_fn or chv_substitute

    def predict_operation(self, sentence: str) -> str:
        """Predict operation type for a sentence."""
        inputs = self.tokenizer(
            sentence,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        self.classifier.eval()  # Ensure eval mode
        with torch.no_grad():
            outputs = self.classifier(**inputs)
            logits = outputs.logits
            pred_idx = torch.argmax(logits, dim=1).item()

        operation = LABELS[pred_idx]
        return operation

    def chv_lookup_placeholder(self, sentence: str) -> str:
        """
        Placeholder for CHV lookup (Rishabh's responsibility).
        Replace this with actual CHV lookup when available.
        """
        if self.chv_lookup_fn:
            return self.chv_lookup_fn(sentence)
        else:
            # Placeholder: return original sentence
            return sentence

    def chv_substitute_and_polish(self, sentence: str) -> str:
        """
        Substitution pathway: CHV lookup first (safe, deterministic), then an
        optional fluency-polish LLM pass to recover readability that pure
        dictionary word-swapping can't provide.

        The polish step is constrained using the same complexity-detection
        features built earlier in the project: numerical expressions, warning
        phrases, and the CHV replacement terms just inserted are extracted
        explicitly and injected into the prompt as spans the LLM must not
        alter, rather than a vague "don't change important terms"
        instruction. The polish output is only accepted if every protected
        span actually survived — verified post-hoc, not just requested —
        otherwise we fall back to the CHV-only (guaranteed-safe) output.

        Warning phrases were added after an initial version (numeric +
        CHV-term protection only) measurably dropped warning preservation
        from 1.000 to 0.986 on the val set — a Substitution-routed sentence
        that also contained a warning phrase had its jargon/numbers protected
        but not the warning language itself, so the polish pass could reword
        it. Protecting warning phrases explicitly closes that gap.
        """
        substituted = self.chv_lookup_fn(sentence)

        if not self.llm_model or not self.llm_tokenizer:
            return substituted

        numeric_spans = [m['match'] for m in extract_numerical_expressions(sentence)]
        warning_spans = [w['match'] for w in detect_warnings(sentence)]
        replaced_terms = _get_chv_replacements_applied(sentence)
        protected = sorted(set(numeric_spans + warning_spans + replaced_terms))

        if not protected:
            # Nothing safety-critical to protect and nothing was substituted —
            # not worth risking an LLM rewrite for no expected benefit.
            return substituted

        protected_str = "; ".join(protected)
        prompt = (
            f"Rewrite this sentence to sound more natural, but you must keep "
            f"these exact words and numbers unchanged: {protected_str}. "
            f"Sentence: {substituted}"
        )
        inputs = self.llm_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(DEVICE)

        self.llm_model.eval()
        with torch.no_grad():
            outputs = self.llm_model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=256,
                num_beams=4,
            )
        polished = self.llm_tokenizer.decode(outputs[0], skip_special_tokens=True)

        polished_lower = polished.lower()
        if all(p.lower() in polished_lower for p in protected):
            return polished
        # Polish dropped or altered a protected span — reject it and keep
        # the CHV-only output, which is guaranteed safe by construction.
        return substituted

    def llm_constrained_simplify(self, sentence: str, prompt_type: str) -> str:
        """
        Simplify using constrained LLM prompt (Explanation/Generalization).
        Uses FLAN-T5 with operation-specific prompts.
        """
        if self.llm_model and self.llm_tokenizer:
            prompt = get_prompt(prompt_type, sentence)

            # Tokenize and generate
            inputs = self.llm_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(DEVICE)

            self.llm_model.eval()  # Ensure eval mode
            with torch.no_grad():
                # Matched to baseline3_direct_llm.py's decoding config
                # (num_beams=4, max_new_tokens=256, deterministic beam search,
                # no sampling) so the operation-aware-vs-baselines comparison
                # isolates the effect of classify-then-route, not a
                # confounded difference in decoding strategy.
                outputs = self.llm_model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=256,
                    num_beams=4,
                )

            # Decode output
            simplified = self.llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
            return simplified
        else:
            # Placeholder: return original with annotation
            prompt_type_upper = prompt_type.upper()
            return f"[{prompt_type_upper}] {sentence}"

    def simplify(self, sentence: str) -> dict:
        """
        Simplify a sentence using operation-aware pipeline.

        Args:
            sentence: Medical sentence to simplify

        Returns:
            dict: {
                'input': str,
                'operation': str,
                'method': str,
                'output': str,
                'warning_preservation': float
            }
        """
        # Step 1: Predict operation
        operation = self.predict_operation(sentence)

        # Step 2: Route to appropriate method
        routing = route_operation(operation, sentence)

        # Step 3: Execute method
        if routing["method"] == "chv_lookup":
            simplified = self.chv_substitute_and_polish(sentence)
        else:  # llm_constrained
            prompt_type = routing["prompt_type"]
            simplified = self.llm_constrained_simplify(sentence, prompt_type)

        # Step 4: Evaluate warning preservation
        warning_result = compute_warning_preservation_rate(sentence, simplified)

        return {
            "input": sentence,
            "operation": operation,
            "method": routing["method"],
            "output": simplified,
            "warning_preservation": warning_result["score"],
            "warnings_preserved": warning_result["preserved"],
            "warnings_dropped": warning_result["dropped"],
        }

    def simplify_batch(self, sentences: list) -> list:
        """Simplify a batch of sentences."""
        return [self.simplify(sent) for sent in sentences]


def evaluate_pipeline(pipeline, test_df: pd.DataFrame) -> dict:
    """
    Evaluate pipeline on a dataset.

    Args:
        pipeline: OperationAwarePipeline instance
        test_df: DataFrame with 'source' column

    Returns:
        dict: Evaluation metrics
    """
    results = pipeline.simplify_batch(test_df["source"].tolist())

    warning_scores = [r["warning_preservation"] for r in results]
    mean_warning_preservation = sum(warning_scores) / len(warning_scores)

    warning_sentences = sum(1 for r in results if r["warnings_preserved"] or r["warnings_dropped"])
    total_sentences = len(results)

    operation_counts = {}
    for r in results:
        op = r["operation"]
        operation_counts[op] = operation_counts.get(op, 0) + 1

    return {
        "mean_warning_preservation": round(mean_warning_preservation, 3),
        "warning_sentences": warning_sentences,
        "total_sentences": total_sentences,
        "operation_distribution": operation_counts,
        "per_sample_results": results,
    }


if __name__ == "__main__":
    print("=" * 80)
    print("OPERATION-AWARE PIPELINE (Baseline 4)")
    print("=" * 80)

    # Initialize pipeline (loads FLAN-T5 by default)
    pipeline = OperationAwarePipeline(load_llm=True, chv_lookup_fn=None)

    # Test examples - diverse operations
    examples = [
        "The patient has hypertension and hyperlipidemia.",  # Simple jargon → Substitution
        "Botulinum toxin injections were administered intramuscularly.",  # Should explain → Explanation
        "A dose-dependent improvement of bladder capacity (5-fold) and periurethral EMG activity (8-fold) was observed.",  # Remove details → Generalization
    ]

    print("\n=== Testing on examples ===\n")
    for sentence in examples:
        result = pipeline.simplify(sentence)
        print(f"Input: {result['input']}")
        print(f"Operation: {result['operation']}")
        print(f"Method: {result['method']}")
        print(f"Output: {result['output']}")
        print(f"Warning Preservation: {result['warning_preservation']}")
        print()