from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Схема запроса для предсказания одного текста."""

    text: str = Field(..., description="Текст для классификации на токсичность")


class PredictBatchRequest(BaseModel):
    """Схема запроса для предсказания списка текстов."""

    texts: list[str] = Field(..., description="Список текстов для классификации")


class PredictResponse(BaseModel):
    """Схема ответа для предсказания."""

    text: str
    class_label: str
    confidence: float


class PredictBatchResponse(BaseModel):
    """Схема ответа для batch предсказания."""

    results: list[PredictResponse]


class ModelInfoResponse(BaseModel):
    """Схема ответа с информацией о модели."""

    model_name: str
    device: str
    num_parameters: int
    max_length: int


class HealthResponse(BaseModel):
    """Схема ответа для health check."""

    status: str
    message: str
