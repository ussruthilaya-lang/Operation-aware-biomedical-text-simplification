cat > ~/setup_quickumls.sh << 'SCRIPT_END'
#!/bin/bash
set -e  # exit immediately if any command fails, so we don't silently continue on a broken step

echo "=== Step 1: System dependencies ==="
sudo apt update
sudo apt install -y python3-pip python3-venv build-essential libleveldb-dev libsnappy-dev

echo "=== Step 2: Create venv ==="
python3 -m venv umls_env
source umls_env/bin/activate
pip install --upgrade pip

echo "=== Step 3: Install QuickUMLS without its broken leveldb dependency ==="
pip install quickumls --no-deps
pip install unidecode six spacy unqlite nltk quickumls-simstring

echo "=== Step 4: Stub leveldb (Python 3.12+ incompatible, unused since we use unqlite backend) ==="
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
touch "$SITE_PACKAGES/leveldb.py"

echo "=== Step 5: Patch quickumls_simstring's legacy SWIG loader (imp module removed in 3.12+) ==="
python3 << 'PYEOF'
path_glob = None
import glob
matches = glob.glob("umls_env/lib/python3.*/site-packages/quickumls_simstring/simstring.py")
path = matches[0]

with open(path) as f:
    content = f.read()

old_imp = '''    def swig_import_helper():
        from os.path import dirname
        import imp
        fp = None
        try:
            fp, pathname, description = imp.find_module('_simstring', [dirname(__file__)])
        except ImportError:
            import _simstring
            return _simstring
        if fp is not None:
            try:
                _mod = imp.load_module('_simstring', fp, pathname, description)
            finally:
                fp.close()
            return _mod'''
new_direct = '''    def swig_import_helper():
        from . import _simstring
        return _simstring'''

if old_imp in content:
    content = content.replace(old_imp, new_direct)
    with open(path, 'w') as f:
        f.write(content)
    print("Patched successfully:", path)
else:
    print("Pattern not found — file may already be patched, or quickumls-simstring version differs.")
PYEOF

echo "=== Step 6: Verify ==="
python -c "import quickumls; print('QuickUMLS import OK')"

echo "=== Done ==="
SCRIPT_END

chmod +x ~/setup_quickumls.sh