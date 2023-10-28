import onnxruntime as ort
import uvicorn
from fastapi import FastAPI, status
from pandas import DataFrame
from pydantic import BaseModel
from transformers import AutoTokenizer

app = FastAPI()


class HealthCheck(BaseModel):
    status: str = "OK"


class Input(BaseModel):
    text: str


class Response(BaseModel):
    toxic: float


@app.on_event("startup")
def startup_event():
    global session, tokenizer, max_len
    max_len = 128
    providers = ["CPUExecutionProvider"]
    session = ort.InferenceSession("model.onnx", providers=providers)
    tokenizer = AutoTokenizer.from_pretrained(
        "DeepPavlov/distilrubert-tiny-cased-conversational-v1"
    )


@app.get(
    "/health",
    tags=["healthcheck"],
    summary="Perform a Health Check",
    response_description="Return HTTP Status Code 200 (OK)",
    status_code=status.HTTP_200_OK,
    response_model=HealthCheck,
)
def get_health() -> HealthCheck:
    return HealthCheck(status="OK")


@app.post(
    "/predict",
    tags=["predict"],
    summary="Predict toxic cooment",
    response_description="Return toxic probability (float value)",
    status_code=status.HTTP_200_OK,
    response_model=Response,
)
def predict(input_data: Input) -> dict[str, str]:
    data = DataFrame(input_data.dict(), index=[0])
    data = tokenizer.batch_encode_plus(
        data.text.tolist(),
        add_special_tokens=True,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="np",
    )
    out = session.run(
        ["output"],
        {
            "input1": data.input_ids,
            "input2": data.attention_mask,
        },
    )
    return {"toxic": out[0]}
