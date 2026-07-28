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
from src.pipeline.operation_router import route_operation
from src.pipeline.constrained_prompts import get_prompt
from src.evaluation.warning_preservation import compute_warning_preservation_rate

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSIFIER_MODEL = "dmis-lab/biobert-base-cased-v1.2"
LLM_MODEL = "google/flan-t5-base"
LABELS = ["Substitution", "Explanation", "Generalization"]


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
        print("Loading BioBERT classifier...")
        self.tokenizer = BertTokenizer.from_pretrained(CLASSIFIER_MODEL)
        self.classifier = BertForSequenceClassification.from_pretrained(
            CLASSIFIER_MODEL, num_labels=3
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

        self.chv_lookup_fn = chv_lookup_fn

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
                outputs = self.llm_model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=100,
                    min_length=5,
                    do_sample=True,
                    top_p=0.9,
                    temperature=0.8,
                    num_return_sequences=1
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
            simplified = self.chv_lookup_placeholder(sentence)
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