FROM nvcr.io/nvidia/tritonserver:24.11-py3

RUN pip install --no-cache-dir \
    transformers \
    torch \
    numpy

COPY model_repository /models

EXPOSE 8000 8001 8002

CMD ["tritonserver", \
     "--model-repository=/models", \
     "--strict-model-config=false", \
     "--log-verbose=0", \
     "--model-control-mode=explicit", \
     "--load-model=toxicity_classifier"]
