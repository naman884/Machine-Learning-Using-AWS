# Importing packages

from fastapi import FastAPI
from pydantic import BaseModel
from app.model.model_inference import predict_pipeline

#----------------------------------------------------------------------#

app = FastAPI()


class TextIn(BaseModel):
    text: str

class PredictionOut(BaseModel):
    language: str


@app.get("/")
def home():
    return {"health_check": "OK"}


@app.post("/predict", response_model=PredictionOut)
def predict(payload: TextIn):
    language = predict_pipeline(payload.text)
    return {"language": language}

