# Why SciSpaCy en_ner_bc5cdr_md?
# Trained on BC5CDR corpus — 1500 PubMed abstracts annotated for
# chemicals and diseases. Recognizes exactly the entity types that
# get dropped or mistranslated during biomedical simplification.
# General spaCy models miss "nephrotoxicity", "hepatocellular" etc.
# Biomedical pretraining is what makes this detector meaningful.

import spacy

nlp = None

def load_model():
    """Lazy load — only load once, reuse across calls."""
    global nlp
    if nlp is None:
        import en_ner_bc5cdr_md
        nlp = en_ner_bc5cdr_md.load()
    return nlp

def detect_biomedical_entities(text):
    """
    Detects biomedical named entities: diseases and chemicals.
    
    Why this matters: entity terms are the primary targets for 
    substitution operations. Detecting them tells the pipeline 
    exactly which spans need CHV lookup or LLM explanation.
    A span containing 'nephrotoxicity' needs different treatment 
    than a span containing 'kidney damage' — this detector tells 
    us which is which.

    Returns:
        list of dicts: [{text, label, start, end}]
    """
    model = load_model()
    doc = model(text)
    results = []
    for ent in doc.ents:
        results.append({
            'text': ent.text,
            'label': ent.label_,  # 'DISEASE' or 'CHEMICAL'
            'start': ent.start_char,
            'end': ent.end_char
        })
    return results

def flag_sentence(text):
    """
    Returns True if sentence contains biomedical named entities.
    This is the interface the pipeline uses.
    """
    return len(detect_biomedical_entities(text)) > 0

def get_entity_spans(text):
    """
    Returns just entity text spans — used by entity preservation metric.
    """
    return [(e['text'], e['label']) for e in detect_biomedical_entities(text)]


if __name__ == "__main__":
    test_sentences = [
        "Patients with hepatocellular carcinoma received sorafenib 400mg daily.",
        "The drug is contraindicated in patients with renal failure.",
        "Muscle cramps are a common problem.",
        "Nephrotoxicity and hepatotoxicity were the primary adverse effects.",
        "Metformin reduces blood glucose levels in type 2 diabetes patients.",
    ]
    for s in test_sentences:
        entities = detect_biomedical_entities(s)
        print(f"\nSentence: {s}")
        print(f"Flagged: {flag_sentence(s)}")
        print(f"Entities: {[(e['text'], e['label']) for e in entities]}")