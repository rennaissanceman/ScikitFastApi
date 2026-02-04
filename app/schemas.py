"""
schemas.py

Modele Pydantic: walidacja wejścia i opis odpowiedzi.
FastAPI używa ich do:
- walidacji JSON request,
- generowania Swagger UI (/docs),
- automatycznych kodów błędów 422 przy złym input.
"""

from pydantic import BaseModel, Field, conlist


class PredictRequest(BaseModel):
    """
    Dla Iris oczekujemy 4 cech (float), w stałej kolejności:
    [sepal length, sepal width, petal length, petal width]

    conlist(...) wymusza:
    - typ float
    - dokładnie 4 elementy
    """

    features: conlist(float, min_length=4, max_length=4) = Field(
        ...,
        description="Iris features: 4 floats in correct order.",
        examples=[[5.1, 3.5, 1.4, 0.2]],
    )


class PredictResponse(BaseModel):
    """
    Odpowiedź dla /v1/predict.
    prediction to ID klasy (0/1/2).
    """

    prediction: int = Field(..., description="Predicted class id (0/1/2).")


class PredictProbaResponse(BaseModel):
    """
    Odpowiedź dla /v1/predict_proba.
    probabilities: prawdopodobieństwo dla każdej klasy
    class_names: nazwy klas (np. setosa, versicolor, virginica)
    """

    prediction: int
    probabilities: list[float]
    class_names: list[str]
