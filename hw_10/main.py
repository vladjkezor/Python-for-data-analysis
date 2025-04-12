import uvicorn
import os

import pandas as pd
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.utils import prepare_data, train_model, read_model

if not os.path.exists('model/my_model.pkl'):
    train_data = prepare_data()
    train_model(train_data)

model = read_model('model/my_model.pkl')

app = FastAPI(title='MyFirstApiModel')


class ModelRequestData(BaseModel):
    total_square: int
    rooms: int
    floor: int


class Result(BaseModel):
    price: float


@app.get('/health/live')
def liveness_probe():
    return JSONResponse(content={"status": "alive"}, status_code=200)


@app.get('/health/ready')
def readiness_probe():
    if os.path.exists("model/my_model.pkl"):
        return JSONResponse(content={"status": "ready"}, status_code=200)
    else:
        return JSONResponse(content={"status": "not ready"}, status_code=503)


@app.get('/predict_get', response_model=Result)
def predict_get(
        square: int = Query(default=42, ge=8, le=2200),
        rooms: int = Query(default=2, ge=0, le=20),
        floor: int = Query(default=4, ge=1, le=100),
):
    df = pd.DataFrame(
        {
            "total_square": square,
            "rooms": rooms,
            "floor": floor
        },
        index=[0]
    )

    result = model.predict(df)[0]

    return Result(price=result)


@app.post("/predict_post", response_model=Result)
def predict_post(data: ModelRequestData):
    input_data = data.model_dump()
    input_df = pd.DataFrame(input_data, index=[0])
    result = model.predict(input_df)[0]
    return Result(price=result)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
