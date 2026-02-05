# ScikitFastApi – Iris Classifier (scikit-learn + FastAPI)

## Opis systemu / usługi
Projekt prezentuje prosty system AI jako usługę webową:
- model uczenia maszynowego trenowany lokalnie (offline) w notebooku Jupyter,
- zapis wytrenowanego modelu do pliku `model/model.joblib`,
- aplikacja FastAPI ładuje model przy starcie (bez trenowania) i udostępnia endpointy predykcyjne.

Rozwiązywany problem: klasyfikacja gatunku kwiatu na podstawie 4 cech Iris.

---

## Wymagania
- Python 3.10+
- `uv` – narzędzie do zarządzania środowiskiem i zależnościami

---

## Przygotowanie środowiska (uv)

```bash
uv venv
uv sync
```