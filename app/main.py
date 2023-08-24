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

# Classes
classes = [
    "0 T-shirt/top",
    "1 Trouser",
    "2 Pullover",
    "3 Dress",
    "4 Coat",
    "5 Sandal",
    "6 Shirt",
    "7 Sneaker",
    "8 Bag",
    "9 Ankle boot",
]


@app.post("/predict")
async def predict(imgs: Request):
    preds = inference(model, imgs)
    class_preds = [classes[i] for i in preds]
    return Response(content=class_preds)
