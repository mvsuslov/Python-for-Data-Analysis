from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

# Загрузка модели
model = joblib.load('realty_model.pkl')

# Создание приложения FastAPI
app = FastAPI()

# Определение модели запроса
class RealtyFeatures(BaseModel):
    area: float
    rooms: int

# Health-check endpoint
@app.get("/health")
def health_check():
    return {"status": "ok"}

# Предсказание через GET-запрос
@app.get("/predict_get")
def predict_get(area: float, rooms: int):
    try:
        features = np.array([[area, rooms]])
        prediction = model.predict(features)[0]
        return {"predicted_price": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Предсказание через POST-запрос
@app.post("/predict_post")
def predict_post(features: RealtyFeatures):
    try:
        input_data = np.array([[features.area, features.rooms]])
        prediction = model.predict(input_data)[0]
        return {"predicted_price": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
