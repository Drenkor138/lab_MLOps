# lab_MLOps

**Лабораторная работа 2: Разработка и контейнеризация ML-сервиса в Triton Inference Server**

## Цель работы

Научиться:
- разворачивать Triton Inference Server локально в Docker;
- использовать Python backend Triton для выполнения модели машинного обучения;
- упаковывать модель и код в модельный репозиторий Triton;
- реализовывать обработку текстов и инференс внутри Python-модуля модели;
- писать клиентский код для обращения к Triton API;
- тестировать работоспособность ML-сервиса.

## Структура проекта

```
lab_MLOps/
├── model_repository/
│   └── toxicity_classifier/
│       ├── 1/
│       │   └── model.py
│       └── config.pbtxt
├── client/
│   └── client.py
├── requirements.txt
├── Dockerfile
└── README.md
```

## Модель

Используется модель классификации токсичности текста:
- **s-nlp/russian_toxicity_classifier**
- https://huggingface.co/s-nlp/russian_toxicity_classifier

Модель загружается напрямую из HuggingFace в Python backend Triton, конвертация в ONNX не требуется.

## Подготовка виртуального окружения

```bash
python3 -m venv venv
source venv/bin/activate  # На Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Запуск Triton Inference Server в Docker

1. Убедитесь, что Docker запущен (Docker Desktop должен быть запущен).

2. Соберите Docker образ:
```bash
docker build -t triton-toxicity-classifier .
```

3. Запустите контейнер:

**С GPU (рекомендуется для ускорения):**
```bash
docker run --gpus all -p 8000:8000 -p 8001:8001 -p 8002:8002 triton-toxicity-classifier
```

**Без GPU (работает на CPU):**
```bash
docker run -p 8000:8000 -p 8001:8001 -p 8002:8002 triton-toxicity-classifier
```

Модель автоматически определяет доступность GPU и использует его для ускорения инференса. Если GPU недоступен, модель работает на CPU.

4. Проверьте, что контейнер запущен:
```bash
docker ps
```

Вы должны увидеть запущенный контейнер с именем образа `triton-toxicity-classifier`.

5. Проверьте логи контейнера (если есть проблемы):
```bash
docker logs <container_id>
```

6. Проверьте доступность сервера:
```bash
curl http://localhost:8000/v2/health/ready
```

Должен вернуться статус `200 OK`.

## Использование клиента

Запустите клиент для тестирования:
```bash
python client/client.py
```

Или используйте в своем коде:
```python
from client.client import predict_toxicity

results = predict_toxicity([
    "Это обычный текст",
    "Пример токсичного текста"
])

for result in results:
    print(f"{result['text']}: {result['class_label']} ({result['confidence']:.2f})")
```

## API Triton

После запуска сервера доступны следующие эндпоинты:
- **HTTP**: http://localhost:8000
- **gRPC**: localhost:8001
- **Metrics**: http://localhost:8002/metrics

## Тестирование

Проверить работоспособность сервера можно через клиент или напрямую через HTTP API:

```bash
curl -X POST http://localhost:8000/v2/models/toxicity_classifier/infer \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [{
      "name": "TEXT",
      "shape": [1, 2],
      "datatype": "BYTES",
      "data": [["Текст 1", "Текст 2"]]
    }],
    "outputs": [{"name": "LOGITS"}]
  }'
```

## Поддержка GPU

Модель автоматически определяет и использует GPU для ускорения инференса:
- При наличии GPU с поддержкой CUDA модель будет использовать его автоматически
- Если GPU недоступен, модель работает на CPU
- Для использования GPU необходим Docker с поддержкой NVIDIA Container Toolkit
- Проверить доступность GPU в контейнере: `docker exec <container_id> nvidia-smi`

## Диагностика проблем

### Ошибка "error: not found" или "Connection refused" на порту 8000

**Проблема:** Triton сервер не запущен или недоступен.

**Решение:**
1. Проверьте, запущен ли Docker Desktop
2. Проверьте запущенные контейнеры:
   ```bash
   docker ps
   ```
3. Если контейнер не запущен, запустите его (см. раздел "Запуск Triton Inference Server в Docker")
4. Проверьте, что порты не заняты другими приложениями:
   ```bash
   netstat -ano | findstr :8000
   ```
5. Проверьте логи контейнера на наличие ошибок:
   ```bash
   docker logs <container_id>
   ```

### Ошибка при создании виртуального окружения

См. раздел "Подготовка виртуального окружения" для инструкций по Windows.

### Модель не загружается

1. Проверьте логи контейнера:
   ```bash
   docker logs <container_id>
   ```
2. Убедитесь, что есть доступ к интернету (модель загружается из HuggingFace)
3. Проверьте, что модель правильно указана в `config.pbtxt`

## Примечания

- Модель загружается из HuggingFace при инициализации, поэтому первый запрос может занять больше времени.
- Модель автоматически использует GPU если доступен (настройки в config.pbtxt: `KIND_AUTO`).
- Для продакшн использования рекомендуется настроить кэширование модели.
- При использовании GPU убедитесь, что установлен NVIDIA Container Toolkit и драйверы CUDA.
