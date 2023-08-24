import json
import sys

from fastapi import FastAPI, Request, Response

sys.path.insert(0, "../src")
from src.inference import inference, load_model

app = FastAPI()


@app.get("/")
def root():
    return {"status": "healthy"}


# Configuration
with open("../config.json", "r") as f:
    cfg = json.load(f)

# Load inference model
model = load_model(f"../models/{cfg['inference_model_name']}.pt")


@app.post("/predict")
async def predict(imgs: Request):
    preds = inference(model, imgs)
    return Response(content=preds)
