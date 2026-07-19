"""
Standalone NER flagging script — runs inside ner_env (Python 3.9, scispaCy).

Isolated from umls_env because en_ner_bc5cdr_md requires spaCy <3.5,
which conflicts with QuickUMLS's runtime dependency on spaCy 3.8.x
in the same process. See bug log: NER/QuickUMLS spaCy version conflict.

Usage (called via subprocess from the main orchestrator):
    <ner_env_python> ner_runner_subprocess.py <input.json> <output.json>

input.json:  {"sentences": ["sent1", "sent2", ...]}
output.json: {"flags": [true, false, ...]}  (same order as input)
"""

import sys
import json


def main():
    if len(sys.argv) != 3:
        print("Usage: ner_runner_subprocess.py <input.json> <output.json>", file=sys.stderr)
        sys.exit(1)

    input_path, output_path = sys.argv[1], sys.argv[2]

    with open(input_path) as f:
        payload = json.load(f)
    sentences = payload["sentences"]

    # Import here, not at module level — this script only runs inside ner_env,
    # so scispacy/spacy are guaranteed available in that interpreter.
    import spacy
    nlp = spacy.load("en_ner_bc5cdr_md")

    flags = []
    for sent in sentences:
        doc = nlp(sent)
        flags.append(len(doc.ents) > 0)

    with open(output_path, "w") as f:
        json.dump({"flags": flags}, f)

    print(f"Processed {len(sentences)} sentences, wrote {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()