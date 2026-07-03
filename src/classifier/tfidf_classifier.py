# TF-IDF + Logistic Regression Operation Classifier (Deliverable 2)
#
# First rung of the three-model progression: TF-IDF+LR -> BERT-base -> BioBERT.
# Question it answers: can surface lexical features alone predict the operation
# (Substitution / Explanation / Generalization) a source sentence needs?
#
# Input  = SOURCE sentence only (the target is unavailable at inference — we
#          decide the operation before generating the simplified text).
# Output = operation label.
#
# For prelim: train on the pseudo-labeled spans, evaluate on the val set.

import os

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from src.data.pseudo_labeler import (
    build_pairs_from_plaba_json,
    label_dataset,
    _DEFAULT_JSON,
    _DEFAULT_OUT,
    _VAL_CSV,
)

# Fixed label order so the confusion matrix rows/columns are always readable.
LABELS = ["Substitution", "Explanation", "Generalization"]

# ngram_range=(1,2): unigrams catch jargon tokens, bigrams catch multi-word
# concepts. class_weight='balanced': Generalization is the minority class.
VECTORIZER = TfidfVectorizer(ngram_range=(1, 2), max_features=50000, lowercase=True)
MODEL = LogisticRegression(class_weight="balanced", solver="lbfgs", max_iter=1000)


def train(X_train, y_train):
    """Fit TF-IDF on the training text, then train Logistic Regression."""
    X_vec = VECTORIZER.fit_transform(X_train)
    MODEL.fit(X_vec, y_train)


def predict(X_test):
    """Predict operation labels for new source sentences."""
    return MODEL.predict(VECTORIZER.transform(X_test))


def evaluate(X_test, y_test):
    """Return accuracy, macro-F1, per-class F1, and confusion matrix."""
    y_pred = predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "macro_f1": f1_score(y_test, y_pred, average="macro", labels=LABELS),
        "per_class_f1": dict(
            zip(LABELS, f1_score(y_test, y_pred, average=None, labels=LABELS))
        ),
        "report": classification_report(y_test, y_pred, labels=LABELS, digits=3),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=LABELS),
    }


def load_training_set(csv_path=_DEFAULT_OUT):
    """Load the pseudo-labeled training pairs from the pseudo-labeler output."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"{csv_path} not found. Run `python -m src.data.pseudo_labeler` first."
        )
    return pd.read_csv(csv_path)


def build_val_set(json_path=_DEFAULT_JSON, val_csv=_VAL_CSV):
    """
    Build the validation set: pull the val.csv PMIDs' sentence pairs from
    data.json and pseudo-label them exactly like the training set. These PMIDs
    were held out of training, so there is no leakage.
    """
    val_pmids = set(str(p) for p in pd.read_csv(val_csv)["pmid"].unique())
    df = build_pairs_from_plaba_json(json_path)
    df = df[df["pmid"].astype(str).isin(val_pmids)].reset_index(drop=True)
    return label_dataset(df)


if __name__ == "__main__":
    print("=== Loading data ===")
    train_df = load_training_set()
    val_df = build_val_set()
    print(f"Train pairs: {len(train_df)}   Val pairs: {len(val_df)}")

    print("\n=== Training TF-IDF + Logistic Regression ===")
    train(train_df["source"], train_df["operation"])

    print("\n=== Evaluation on validation set ===")
    results = evaluate(val_df["source"], val_df["operation"])
    print(f"Accuracy : {results['accuracy']:.3f}")
    print(f"Macro-F1 : {results['macro_f1']:.3f}")
    print("\nPer-class F1:")
    for label, score in results["per_class_f1"].items():
        print(f"  {label:15s} {score:.3f}")
    print("\nClassification report:")
    print(results["report"])
    print("Confusion matrix (rows=true, cols=pred):")
    print("labels:", LABELS)
    print(results["confusion_matrix"])
