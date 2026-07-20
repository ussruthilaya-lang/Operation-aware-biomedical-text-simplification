cat > ~/setup_ner_env.sh << 'SCRIPT_END'
#!/bin/bash
set -e  # stop immediately on any failure

echo "=== Step 1: Install Miniconda (if not already present) ==="
if [ ! -d "$HOME/miniconda3" ]; then
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
fi

echo "=== Step 2: Init conda for this shell ==="
"$HOME/miniconda3/bin/conda" init bash
source "$HOME/.bashrc"

echo "=== Step 3: Accept Anaconda ToS (required as of 2026) ==="
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

echo "=== Step 4: Create ner_env (Python 3.9 — needed for prebuilt blis/spaCy wheels) ==="
conda create -n ner_env python=3.9 -y

echo "=== Step 5: Install scispaCy + BC5CDR model inside ner_env ==="
source "$HOME/miniconda3/bin/activate" ner_env
pip install scispacy
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz

echo "=== Step 6: Verify ==="
python -c "
import spacy
nlp = spacy.load('en_ner_bc5cdr_md')
doc = nlp('Patients with hepatocellular carcinoma received sorafenib.')
assert len(doc.ents) > 0, 'No entities detected — install may be broken'
print('NER environment OK:', [(e.text, e.label_) for e in doc.ents])
"

echo "=== Done. Use via: source ~/miniconda3/bin/activate ner_env ==="
SCRIPT_END

chmod +x ~/setup_ner_env.sh