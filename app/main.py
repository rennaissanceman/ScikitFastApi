"""
main.py

Routing FastAPI + integracja z service.py.

Tu NIE trenujemy modelu.
Tu tylko:
- ładujemy model przy starcie (startup)
- wystawiamy endpointy
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, HTTPException

from .schemas import PredictRequest, PredictResponse, PredictProbaResponse
from .service import ModelAssets, load_assets, predict_class, predict_proba

logger = logging.getLogger("scikitfastapi")
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="ScikitFastApi - Iris Classifier",
    version="1.0.0",
    description="A simple scikit-learn model served via FastAPI (trained offline in a notebook).",
)

assets: ModelAssets | None = None


@app.on_event("startup")
def startup() -> None:
    """
    Uruchamia się raz przy starcie serwera.
    Ładujemy model z pliku joblib (wygenerowanego przez notebook).
    """
    global assets
    assets = load_assets()
    logger.info("Model loaded. Metadata keys: %s", list(assets.metadata.keys()))


@app.get("/health")
def health() -> dict[str, str]:
    """Prosty endpoint sprawdzający czy serwer działa."""
    return {"status": "ok"}

@app.get("/")
def root():
    return {
        "message": "ScikitFastApi is running. Go to /docs for Swagger UI.",
        "docs": "/docs",
        "health": "/health",
        "predict": "/v1/predict",
    }


@app.get("/v1/model-info")
def model_info() -> dict[str, Any]:
    """
    Zwraca metadane modelu + informację czy obsługuje predict_proba.
    """
    if assets is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    info = dict(assets.metadata)
    info["has_predict_proba"] = hasattr(assets.model, "predict_proba")
    return info


@app.post("/v1/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    """
    Endpoint predykcji klasy.
    """
    if assets is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        pred = predict_class(assets, req.features)
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")

    return PredictResponse(prediction=pred)


@app.post("/v1/predict_proba", response_model=PredictProbaResponse)
def predict_probabilities(req: PredictRequest) -> PredictProbaResponse:
    """
    Endpoint predykcji prawdopodobieństw klas.
    """
    if assets is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        pred, probs = predict_proba(assets, req.features)
    except AttributeError as e:
        # Model nie wspiera predict_proba
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Predict_proba failed")
        raise HTTPException(status_code=400, detail=f"Predict_proba failed: {e}")

    class_names = assets.metadata.get("class_names", [str(i) for i in range(len(probs))])
    return PredictProbaResponse(
        prediction=pred,
        probabilities=probs,
        class_names=list(class_names),
    )
