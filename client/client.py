import numpy as np
import tritonclient.http as httpclient
import sys


def predict_toxicity(
texts, server_url="localhost:8000", model_name="toxicity_classifier"
):
    """
    Предсказание токсичности для батча текстов

    Args:
        texts: Список текстов для классификации
        server_url: URL Triton сервера
        model_name: Имя модели в Triton

    Returns:
        Список предсказаний с метками и вероятностями
    """

    try:
        client = httpclient.InferenceServerClient(server_url)
    except Exception as e:
        print(f"Ошибка подключения к Triton-серверу на {server_url}")
        print(f"Ошибка: {e}")
        sys.exit(1)

    try:
        if not client.is_server_live():
            print(f"Ошибка: Triton сервер на {server_url} не отвечает")
            sys.exit(1)
        
        if not client.is_model_ready(model_name):
            print(f"Ошибка: Модель '{model_name}' не загружена на сервере")
            sys.exit(1)
    except Exception as e:
        print(f"Ошибка при проверке сервера")
        print(f"Ошибка: {e}")
        sys.exit(1)

    batch_size = len(texts)
    texts_array = np.array([text.encode("utf-8") for text in texts], dtype=np.object_)
    texts_array = texts_array.reshape([1, batch_size])

    inputs = [httpclient.InferInput("TEXT", texts_array.shape, "BYTES")]
    inputs[0].set_data_from_numpy(texts_array)
    outputs = [httpclient.InferRequestedOutput("LOGITS")]

    try:
        response = client.infer(model_name, inputs=inputs, outputs=outputs)
        logits = response.as_numpy("LOGITS")
    except Exception as e:
        print(f"Ошибка при выполнении инференса: {e}")
        print("Модель некорректно загружена или сервер не работает")
        raise

    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    predicted_classes = np.argmax(probabilities, axis=1)
    confidences = np.max(probabilities, axis=1)

    class_labels = ["non-toxic", "toxic"]
    results = []
    for i, text in enumerate(texts):
        class_idx = predicted_classes[i]
        results.append(
            {
                "text": text,
                "class_label": class_labels[class_idx],
                "confidence": float(confidences[i]),
                "logits": logits[i].tolist(),
                "probabilities": probabilities[i].tolist(),
            }
        )

    return results


if __name__ == "__main__":
    test_texts = [
        "Просто какой-то текст для угнетения",
        "Я ненавижу работать, хочу в отпуск, бесконечные релизы задолбали...",
        "Блять, этот сранный текст обязан быть токсичным!",
    ]

    print("Отправка запроса...")
    results = predict_toxicity(test_texts)

    print("\nРезультат предсказания:")
    for result in results:
        print(f"\nТекст: {result['text']}")
        print(f"Класс: {result['class_label']}")
        print(f"Уверенность: {result['confidence']:.4f}")
        print(
            f"Вероятности: non-toxic={result['probabilities'][0]:.4f}, \
            toxic={result['probabilities'][1]:.4f}"
        )
    
    
