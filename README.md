# ScikitFastApi – Prosty model ML jako usługa FastAPI

## 1. Opis projektu

Projekt przedstawia prosty system sztucznej inteligencji udostępniony jako usługa webowa (API).
System wykorzystuje wcześniej wytrenowany model uczenia maszynowego z biblioteki scikit-learn
i udostępnia go poprzez framework FastAPI.

Celem projektu jest połączenie wiedzy z zakresu:
- uczenia maszynowego (trenowanie modelu, predykcja),
- programowania aplikacji webowych (API, FastAPI),
- zarządzania środowiskiem i pracy zespołowej z użyciem systemu kontroli wersji Git.

---

## 1.1. Wykorzystane technologie

- Python 3.x
- scikit-learn
- FastAPI
- Pydantic
- uv
- Git

---

## 1.2. Model uczenia maszynowego

W projekcie wykorzystano prosty model klasyfikacyjny z biblioteki scikit-learn,
wytrenowany na klasycznym zbiorze danych Iris Dataset.

Cechy wejściowe modelu:
- sepal length
- sepal width
- petal length
- petal width

Model:
- jest trenowany lokalnie (offline) w środowisku Jupyter Notebook,
- zapisywany do pliku przy użyciu biblioteki `joblib`,
- ładowany przez aplikację FastAPI podczas uruchamiania serwera,
- nie jest trenowany przy każdym uruchomieniu aplikacji.

---

## 1.3. Struktura projektu

```text
ScikitFastApi/
├── app/
│   ├── main.py
│   ├── service.py
│   ├── schemas.py
│   └── settings.py
├── model/
├── notebooks/
├── tests/
├── pyproject.toml
├── uv.lock
└── README.md
```
---

## 2. Instrukcja instalacji i uruchomienia

### 2.1. Wymagania
- Python 3.x
- narzędzie `uv` do zarządzania środowiskiem i zależnościami
- system kontroli wersji Git (do pobrania repozytorium)

### 2.2. Sposób przygotowania środowiska (uv)

Po sklonowaniu repozytorium należy przejść do katalogu głównego projektu
i przygotować środowisko przy użyciu narzędzia `uv`:

```bash
uv sync
```
### 2.3. Sposób uruchomienia serwera FastAPI

Serwer FastAPI uruchamiany jest lokalnie przy użyciu polecenia:

```bash
uvicorn app.main:app --reload
```
Po uruchomieniu aplikacja dostępna jest pod adresem:
- http://127.0.0.1:8000

Automatyczna dokumentacja API (Swagger UI) dostępna jest pod adresem:
- http://127.0.0.1:8000/docs

---

## 3. Instrukcja użycia

### 3.1. Opis endpointów

- `POST /v1/predict`  
  Endpoint predykcyjny. Przyjmuje dane wejściowe w formacie JSON i zwraca
  przewidywaną klasę na podstawie wytrenowanego modelu.

- `POST /v1/predict_proba`  
  Endpoint predykcyjny. Przyjmuje dane wejściowe w formacie JSON i zwraca
  prawdopodobieństwa przynależności do poszczególnych klas.

### 3.2. Przykład zapytania (request)

```json
{
  "features": [5.1, 3.5, 1.4, 0.2]
}
```

### Przykład odpowiedzi (response) dla `/v1/predict`

```json
{
  "prediction": 0
}
```
### Przykład odpowiedzi (response) dla `/v1/predict_proba`

```json
{
  "prediction": 1,
  "probabilities": [0.02, 0.93, 0.05],
  "class_names": ["setosa", "versicolor", "virginica"]
}
```
---

## 4. Informacje o modelu i danych

### 4.1. Użyty model

W projekcie wykorzystano prosty model klasyfikacyjny z biblioteki scikit-learn,
wytrenowany na klasycznym zbiorze danych Iris Dataset.

Model:
- jest trenowany lokalnie (offline) w środowisku Jupyter Notebook,
- zapisywany do pliku przy użyciu biblioteki `joblib`,
- ładowany przez aplikację FastAPI podczas uruchamiania serwera,
- nie jest trenowany ponownie w trakcie działania aplikacji.

### 4.2. Dane wejściowe

Dane wejściowe przekazywane do API mają postać JSON i zawierają cztery cechy
liczbowe typu `float`:

- sepal length  
- sepal width  
- petal length  
- petal width  

### 4.3. Dane wyjściowe

W odpowiedzi API zwraca:
- przewidywaną klasę obiektu (np. `setosa`),
- opcjonalnie prawdopodobieństwa przynależności do poszczególnych klas
  (w zależności od użytego endpointu).
