FROM python:3.9-slim

RUN pip install --no-cache-dir torch pandas transformers fastapi onnxruntime uvicorn

RUN mkdir /app
COPY . app/
WORKDIR /app



