"""
Wspólne miejsce na ścieżki i ustawienia aplikacji.
"""

from pathlib import Path

# Katalog główny repo (ScikitFastApi/)
# app/settings.py -> app -> repo root
REPO_ROOT = Path(__file__).resolve().parents[1]

# artefakty modelu (po treningu)
MODEL_DIR = REPO_ROOT / "model"

# Plik modelu (joblib)
MODEL_FILE = MODEL_DIR / "model.joblib"

# Metadane
METADATA_FILE = MODEL_DIR / "metadata.json"
