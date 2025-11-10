import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class ToxicityClassifier:
    """Класс для работы с моделью классификации токсичности."""

    def __init__(self, model_name: str = "s-nlp/russian_toxicity_classifier"):
        """
        Инициализация модели.

        Args:
            model_name: Название модели
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model()

    def _load_model(self):
        """Загрузка модели и токенизатора."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str) -> dict:
        """
        Предсказание класса токсичности для одного текста.

        Args:
            text: Текст для классификации

        Returns:
            Предсказание (класс и вероятность)
        """
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512, padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)

        predicted_class = torch.argmax(probabilities, dim=-1).item()
        confidence = probabilities[0][predicted_class].item()

        class_label = "toxic" if predicted_class == 1 else "non-toxic"

        return {
            "text": text,
            "class_label": class_label,
            "confidence": round(confidence, 4),
        }

    def predict_batch(self, texts: list[str]) -> list[dict]:
        """
        Предсказание классов токсичности для нескольких текстов.

        Args:
            texts: Массив текстов для классификации

        Returns:
            Массив предсказаний
        """
        results = []
        for text in texts:
            result = self.predict(text)
            results.append(result)
        return results

    def get_model_info(self) -> dict:
        """
        Получение информации о модели.

        Returns:
            Информация о модели (название, устройство и т.д.)
        """
        return {
            "model_name": self.model_name,
            "device": str(self.device),
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
            "max_length": 512,
        }
