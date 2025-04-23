from fastapi import FastAPI
from pydantic import BaseModel
from src.inference import classify_text


app = FastAPI()

class Inquiry(BaseModel):
    text: str

@app.post("/predict")
def predict(inquiry: Inquiry):
    prediction = classify_text(inquiry.text)
    return {"category": prediction}
