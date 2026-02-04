"""
settings.py

Jedno miejsce na ścieżki i ustawienia aplikacji.
Dzięki temu nie "rozsypują się" po kodzie i łatwiej je zmienić.
"""

from pathlib import Path

# Katalog główny repo (ScikitFastApi/)
# app/settings.py -> app -> repo root
REPO_ROOT = Path(__file__).resolve().parents[1]

# Katalog na artefakty modelu (po treningu)
MODEL_DIR = REPO_ROOT / "model"

# Plik modelu zapisany przez notebook (joblib)
MODEL_FILE = MODEL_DIR / "model.joblib"

# Metadane (opcjonalnie)
METADATA_FILE = MODEL_DIR / "metadata.json"
