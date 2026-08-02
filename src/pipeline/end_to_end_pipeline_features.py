# Operation-aware pipeline variant using the BioBERT+UMLS-features classifier
# (models/biobert_umls_only/) instead of plain BioBERT.
#
# A NEW variant, not a modification of end_to_end_pipeline.py, so the
# already-validated plain-BioBERT pipeline (headline val/test results) stays
# untouched. Subclasses OperationAwarePipeline and overrides only what
# differs: __init__ (loads BioBERTWithFeatures instead of
# BertForSequenceClassification) and predict_operation (looks up the 3 UMLS
# features for the sentence and feeds them through the model alongside the
# tokenized text). CHV substitution, the protected-span polish step, and
# constrained FLAN-T5 generation are all inherited unchanged.
#
# Feature lookup uses results/classifier_features.csv directly (already
# covers train+val+test source sentences -- see
# src/evaluation/extract_classifier_features.py) rather than recomputing
# UMLS features live via QuickUMLS at inference time. This guarantees the
# features match what the classifier was actually trained on, and avoids
# requiring a real UMLS/QuickUMLS installation just to run the pipeline.
#
# The StandardScaler used at training time was never persisted to disk (a
# real gap in src/classifier/biobert_features_classifier.py) -- refit here
# from the same training data, which is deterministic and reproduces the
# same scaling exactly, rather than adding new pickle-persistence machinery
# under time pressure.

import os
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.pipeline.end_to_end_pipeline import OperationAwarePipeline, DEVICE
from src.classifier.biobert_features_classifier import (
    BioBERTWithFeatures, UMLS_COLS, attach_features,
    load_training_set, FEATURES_CSV,
)
from src.data.pseudo_labeler import _DEFAULT_OUT

LABELS = ["Substitution", "Explanation", "Generalization"]
FEATURES_CHECKPOINT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "models", "biobert_umls_only"
)


class OperationAwarePipelineWithFeatures(OperationAwarePipeline):
    """Same pipeline, BioBERT+UMLS-features classifier instead of plain BioBERT."""

    def __init__(self, llm_model=None, chv_lookup_fn=None, load_llm=True):
        if not os.path.isdir(FEATURES_CHECKPOINT_DIR):
            raise FileNotFoundError(
                f"No BioBERT+features checkpoint found at {FEATURES_CHECKPOINT_DIR}. "
                f"Run `python -m src.classifier.biobert_features_classifier` first."
            )

        print(f"Loading BioBERT+UMLS-features classifier from {FEATURES_CHECKPOINT_DIR}...")
        from transformers import BertTokenizer
        self.tokenizer = BertTokenizer.from_pretrained(FEATURES_CHECKPOINT_DIR)
        self.classifier = BioBERTWithFeatures(
            FEATURES_CHECKPOINT_DIR, num_features=len(UMLS_COLS)
        )
        self.classifier.classifier.load_state_dict(
            torch.load(os.path.join(FEATURES_CHECKPOINT_DIR, "classifier_head.pt"),
                       map_location=DEVICE)
        )
        self.classifier.to(DEVICE)
        self.classifier.eval()

        print("Refitting feature scaler from training data (not persisted at train time)...")
        train_df = load_training_set(_DEFAULT_OUT)
        features_df = pd.read_csv(FEATURES_CSV)
        train_df = attach_features(train_df, features_df)
        self.scaler = StandardScaler()
        self.scaler.fit(train_df[UMLS_COLS].fillna(0))

        # Lookup table: source sentence -> UMLS feature vector, precomputed
        # for train+val+test (src/evaluation/extract_classifier_features.py).
        # A given source sentence can repeat across multiple adaptation rows
        # (one row per source-target pair) with identical feature values
        # since features are computed on the source alone -- dedup keeping
        # the first occurrence so the index is unique.
        features_df = features_df.copy()
        features_df["source"] = features_df["source"].astype(str)
        features_df = features_df.drop_duplicates(subset="source", keep="first")
        self._feature_lookup = features_df.set_index("source")[UMLS_COLS].to_dict("index")

        print("Loading FLAN-T5 LLM...")
        if llm_model is None and load_llm:
            from transformers import AutoTokenizer, T5ForConditionalGeneration
            self.llm_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
            self.llm_model = T5ForConditionalGeneration.from_pretrained(
                "google/flan-t5-base"
            ).to(DEVICE)
            self.llm_model.eval()
        else:
            self.llm_model = llm_model
            self.llm_tokenizer = None

        from src.baselines.baseline2_rule_based_chv import chv_substitute
        self.chv_lookup_fn = chv_lookup_fn or chv_substitute

    def _lookup_features(self, sentence: str):
        row = self._feature_lookup.get(str(sentence))
        if row is None:
            # Sentence not in the precomputed table (shouldn't happen for
            # train/val/test PLABA sentences) -- fall back to zeros rather
            # than crashing, and flag it loudly since it means the feature
            # lookup and the eval set have silently diverged.
            print(f"WARNING: no precomputed UMLS features for sentence, "
                  f"using zeros: {sentence[:80]}...")
            return [0.0, 0.0, 0.0]
        return [row[c] for c in UMLS_COLS]

    def predict_operation(self, sentence: str) -> str:
        inputs = self.tokenizer(
            sentence, max_length=128, padding="max_length",
            truncation=True, return_tensors="pt"
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        raw_features = self._lookup_features(sentence)
        scaled = self.scaler.transform([raw_features])
        features_tensor = torch.tensor(scaled, dtype=torch.float).to(DEVICE)

        self.classifier.eval()
        with torch.no_grad():
            logits = self.classifier(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                features=features_tensor,
            )
            pred_idx = torch.argmax(logits, dim=1).item()

        return LABELS[pred_idx]
