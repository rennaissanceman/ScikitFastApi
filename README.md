# ScikitFastApi — Iris Classifier (scikit-learn + FastAPI)

## Cel
Projekt pokazuje prosty system AI jako usługę:
- model ML trenowany lokalnie (offline) w notebooku Jupyter
- zapis modelu do pliku `model/model.joblib`
- FastAPI ładuje model przy starcie (bez trenowania) i udostępnia endpointy predykcyjne

## Wymagania
- Python 3.10+
- uv

## Instalacja środowiska (uv)
```bash
uv sync
