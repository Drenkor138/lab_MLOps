from fastapi import FastAPI

from app.model import ToxicityClassifier
from app.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictBatchRequest,
    PredictBatchResponse,
    PredictRequest,
    PredictResponse,
)

app = FastAPI(
    title="Russian Toxicity Classifier API",
    description="API для классификации токсичности русских текстов",
    version="1.0.0",
)
classifier = ToxicityClassifier()


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Проверка работоспособности сервиса.

    Returns:
        Статус работы сервиса
    """
    return HealthResponse(status="ok", message="Service is running")


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Предсказание класса токсичности для одного текста.

    Args:
        request: Запрос с текстом для классификации

    Returns:
        Предсказание с классом и уверенностью
    """
    result = classifier.predict(request.text)
    return PredictResponse(**result)


@app.post("/predict_batch", response_model=PredictBatchResponse)
async def predict_batch(request: PredictBatchRequest):
    """
    Предсказание классов токсичности для списка текстов.

    Args:
        request: Запрос со списком текстов для классификации

    Returns:
        Список предсказаний с классами и уверенностью
    """
    results = classifier.predict_batch(request.texts)
    return PredictBatchResponse(
        results=[PredictResponse(**result) for result in results]
    )


@app.get("/model_info", response_model=ModelInfoResponse)
async def model_info():
    """
    Получение информации о модели.

    Returns:
        Информация о модели (название, устройство, количество параметров и т.д.)
    """
    info = classifier.get_model_info()
    return ModelInfoResponse(**info)
