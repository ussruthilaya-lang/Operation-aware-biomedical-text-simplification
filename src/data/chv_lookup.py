"""
Consumer Health Vocabulary (CHV) lookup.

Standalone dictionary of biomedical jargon mapped to plain English. The core of
this used to live inside baseline2_rule_based_chv.py. It is pulled out here so
Sophakotra's operation router and any other module can import it directly
instead of reaching into a baseline script. baseline2 now imports from here too,
so there is a single source of truth.

The dictionary keeps every mapping baseline2 already used (so the rule-based
baseline behaves the same), plus additional terms for broader coverage.

Reference: NLM Consumer Health Vocabulary
https://www.nlm.nih.gov/research/umls/sourcereleasedocs/current/CHV/

Usage:
    from src.data.chv_lookup import lookup, batch_lookup, get_all_terms

    lookup("myocardial infarction")   # -> "heart attack"
    lookup("aspirin")                 # -> None
    batch_lookup(["dyspnea", "edema"])# -> {"dyspnea": "shortness of breath", ...}
    get_all_terms()                   # -> full dict (a copy)

Lookup is case-insensitive and ignores surrounding whitespace.
"""


# term (lowercase) -> plain English equivalent
CHV_DICTIONARY = {
    # --- Original baseline2 terms (do not change these mappings) ---
    # Diseases / conditions
    "myocardial infarction": "heart attack",
    "cerebrovascular accident": "stroke",
    "hypertension": "high blood pressure",
    "hypotension": "low blood pressure",
    "tachycardia": "fast heart rate",
    "bradycardia": "slow heart rate",
    "dyspnea": "shortness of breath",
    "edema": "swelling",
    "pyrexia": "fever",
    "epistaxis": "nosebleed",
    "syncope": "fainting",
    "vertigo": "dizziness",
    "pruritus": "itching",
    "erythema": "redness",
    "alopecia": "hair loss",
    "fracture": "broken bone",
    "laceration": "cut",
    "contusion": "bruise",
    "neoplasm": "tumor",
    "carcinoma": "cancer",
    "renal failure": "kidney failure",
    "hepatic failure": "liver failure",
    "pneumonia": "lung infection",
    "sepsis": "blood infection",
    "anemia": "low blood count",
    # Procedures
    "angioplasty": "artery-opening procedure",
    "appendectomy": "appendix removal surgery",
    "cholecystectomy": "gallbladder removal surgery",
    "colonoscopy": "colon examination",
    "mammography": "breast X-ray",
    # Drugs / administration terms
    "analgesic": "painkiller",
    "antibiotic": "infection-fighting drug",
    "anticoagulant": "blood thinner",
    "contraindicated": "should not be used",
    "prophylaxis": "prevention treatment",
    "subcutaneous": "under the skin",
    "intravenous": "into the vein",
    "oral": "by mouth",
    "intravenously": "through a vein",
    "subcutaneously": "under the skin",
    "orally": "by mouth",

    # --- Added terms (broader coverage for the operation router) ---
    "hyperglycemia": "high blood sugar",
    "hypoglycemia": "low blood sugar",
    "hyperlipidemia": "high cholesterol",
    "renal": "kidney",
    "hepatic": "liver",
    "hepatotoxicity": "liver damage",
    "nephrotoxicity": "kidney damage",
    "cardiac": "heart",
    "pulmonary": "lung",
    "gastrointestinal": "stomach and gut",
    "cutaneous": "skin",
    "antipyretic": "fever reducer",
    "antiemetic": "anti-nausea medicine",
    "antihypertensive": "blood pressure medicine",
    "adverse event": "harmful side effect",
    "adverse reaction": "bad reaction",
    "efficacy": "how well it works",
    "prognosis": "likely outcome",
    "etiology": "cause",
    "pathogenesis": "how the disease develops",
    "benign": "not cancer",
    "malignant": "cancerous",
    "metastasis": "cancer spread",
    "lesion": "area of damage",
    "inflammation": "swelling and redness",
    "chronic": "long-lasting",
    "acute": "sudden and severe",
    "asymptomatic": "having no symptoms",
    "febrile": "having a fever",
    "lethargy": "tiredness",
    "dysphagia": "trouble swallowing",
    "dysuria": "painful urination",
    "hematuria": "blood in urine",
    "administer": "give",
    "comorbidity": "other health condition at the same time",
    "mortality": "death rate",
    "morbidity": "illness rate",
    "incidence": "number of new cases",
    "prevalence": "how common it is",
}


def lookup(term):
    """Return the plain English equivalent for a term, or None if not found."""
    if term is None:
        return None
    return CHV_DICTIONARY.get(term.strip().lower())


def batch_lookup(terms):
    """
    Look up many terms at once.

    Returns a dict mapping each input term to its plain English equivalent,
    or None where there is no match.
    """
    return {term: lookup(term) for term in terms}


def get_all_terms():
    """Return a copy of the full CHV dictionary."""
    return dict(CHV_DICTIONARY)


if __name__ == "__main__":
    print("Total terms:", len(CHV_DICTIONARY))
    for t in ["myocardial infarction", "Dyspnea", "  renal failure ", "aspirin"]:
        print(f"  {t!r:26s} -> {lookup(t)}")
