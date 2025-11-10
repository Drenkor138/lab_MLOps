from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_check():
    """Тест проверки работоспособности сервиса."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "message" in data


def test_model_info():
    """Тест получения информации о модели."""
    response = client.get("/model_info")
    assert response.status_code == 200
    data = response.json()
    assert "model_name" in data
    assert "device" in data
    assert "num_parameters" in data
    assert "max_length" in data
    assert data["model_name"] == "s-nlp/russian_toxicity_classifier"


def test_predict_single_text():
    """Тест предсказания для одного текста."""
    response = client.post(
        "/predict", json={"text": "Это какой-то текст для тестирования"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "text" in data
    assert "class_label" in data
    assert "confidence" in data
    assert data["class_label"] in ["toxic", "non-toxic"]
    assert 0 <= data["confidence"] <= 1


def test_predict_batch():
    """Тест предсказания для нескольких текстов."""
    texts = [
        "Это обычный текст",
        "Это еще один текст для проверки",
    ]
    response = client.post("/predict_batch", json={"texts": texts})
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert len(data["results"]) == len(texts)
    for result in data["results"]:
        assert "text" in result
        assert "class_label" in result
        assert "confidence" in result


def test_predict_empty_text():
    """Тест обработки пустого текста."""
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 200
    data = response.json()
    assert "class_label" in data
    assert "confidence" in data
