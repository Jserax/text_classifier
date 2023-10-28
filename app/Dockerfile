FROM python:3.9-slim

RUN mkdir /app
COPY . app/
COPY ../model /app
WORKDIR /app

RUN pip install --no-cache-dir torch pandas transformers fastapi onnxruntime uvicorn

