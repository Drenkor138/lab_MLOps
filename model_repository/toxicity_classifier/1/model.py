import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class TritonPythonModel:

    def initialize(self, args):
        model_name = "s-nlp/russian_toxicity_classifier"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        print(f"Модель загружена на устройство: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"CUDA версия: {torch.version.cuda}")

        self.class_labels = ["non-toxic", "toxic"]

    def execute(self, requests):
        """
        Обработка batch запросов.

        Args:
            requests: Список запросов от Triton

        Returns:
            Список ответов с logits
        """
        responses = []

        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "TEXT")

            texts_array = input_tensor.as_numpy()[0]
            texts = [text.decode("utf-8") for text in texts_array]

            encodings = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**encodings)
                logits = outputs.logits.cpu().numpy()

            output_tensor = pb_utils.Tensor("LOGITS", logits.astype(np.float32))
            response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
            responses.append(response)

        return responses

    def finalize(self):
        """
        Финализация модели и освобождение ресурсов.
        """
        del self.model
        del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
