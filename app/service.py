"""
service.py

Warstwa "logiki" ML oddzielona od routingu FastAPI.

Dlaczego to jest dobre?
- main.py jest wtedy czytelny: tylko endpointy i HTTP
- service.py robi całą "robotę": load model, predict, predict_proba
- łatwiej testować i rozwijać
"""

from __future__ import annotations

import json
import pandas as pd

from dataclasses import dataclass
from typing import Any

import joblib

from .settings import MODEL_FILE, METADATA_FILE


@dataclass
class ModelAssets:
    """
    Trzymamy razem to, co ładuje serwer:
    - model: obiekt scikit-learn (np. Pipeline)
    - metadata: słownik z opisem cech/klas, data treningu itd.
    """
    model: Any
    metadata: dict[str, Any]


def load_assets() -> ModelAssets:
    """
    Ładowanie modelu i metadanych z dysku.

    Wymóg zadania:
    - model jest trenowany offline
    - serwer tylko ładuje zapisany plik (joblib)
    """
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Brak pliku modelu: {MODEL_FILE}. "
            "Uruchom notebooks/train_model.ipynb aby go wygenerować."
        )

    model = joblib.load(MODEL_FILE)

    metadata: dict[str, Any] = {}
    if METADATA_FILE.exists():
        metadata = json.loads(METADATA_FILE.read_text(encoding="utf-8"))
    else:
        metadata = {"warning": "metadata.json not found"}

    return ModelAssets(model=model, metadata=metadata)

def predict_class(assets: ModelAssets, features: list[float]) -> int:
    feature_names = assets.metadata.get("feature_names")
    if feature_names:
        X = pd.DataFrame([features], columns=feature_names)
    else:
        X = [features]
    pred = assets.model.predict(X)[0]
    return int(pred)


# def predict_class(assets: ModelAssets, features: list[float]) -> int:
#     """
#     Predykcja klasy.
#     scikit-learn oczekuje 2D: (n_samples, n_features),
#     więc podajemy [features] jako jedną próbkę.
#     """
#     pred = assets.model.predict([features])[0]
#     return int(pred)


def predict_proba(assets: ModelAssets, features: list[float]) -> tuple[int, list[float]]:
    if not hasattr(assets.model, "predict_proba"):
        raise AttributeError("Model does not support predict_proba")

    feature_names = assets.metadata.get("feature_names")
    if feature_names:
        X = pd.DataFrame([features], columns=feature_names)
    else:
        X = [features]

    proba = assets.model.predict_proba(X)[0]
    probs = [float(x) for x in proba]
    pred = int(max(range(len(probs)), key=lambda i: probs[i]))
    return pred, probs


# def predict_proba(assets: ModelAssets, features: list[float]) -> tuple[int, list[float]]:
#     """
#     Predykcja prawdopodobieństw.
#     Działa tylko, jeśli model ma predict_proba.
#
#     Zwracamy:
#     - prediction: klasa o największym prawdopodobieństwie
#     - probabilities: lista floatów
#     """
#     if not hasattr(assets.model, "predict_proba"):
#         raise AttributeError("Model does not support predict_proba")
#
#     proba = assets.model.predict_proba([features])[0]
#     # proba to np. array([0.98, 0.01, 0.01])
#     probs = [float(x) for x in proba]
#     pred = int(max(range(len(probs)), key=lambda i: probs[i]))
#     return pred, probs
