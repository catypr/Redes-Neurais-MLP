
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(
    title="API de Previsão de Cancelamento de Reservas",
    description="API que utiliza uma rede neural MLP para prever cancelamentos de reservas de hotel.",
    version="1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

CAMINHO_MODELO = Path(__file__).resolve().parents[1] / "models" / "modelo_mlp.joblib"
modelo = joblib.load(CAMINHO_MODELO)


class Reserva(BaseModel):
    no_of_adults: int
    no_of_children: int
    no_of_weekend_nights: int
    no_of_week_nights: int
    type_of_meal_plan: str
    required_car_parking_space: int
    room_type_reserved: str
    lead_time: int
    arrival_year: int
    arrival_month: int
    arrival_date: int
    market_segment_type: str
    repeated_guest: int
    no_of_previous_cancellations: int
    no_of_previous_bookings_not_canceled: int
    avg_price_per_room: float
    no_of_special_requests: int


@app.get("/")
def inicio():
    return {"mensagem": "API de previsão de cancelamento de reservas está funcionando."}


@app.get("/info")
def informacoes():
    return {
        "modelo": "MLPClassifier",
        "tipo": "Classificação binária",
        "objetivo": "Prever se uma reserva será cancelada",
        "variavel_alvo": "booking_status"
    }


@app.post("/predict")
def prever(reserva: Reserva):
    dados = pd.DataFrame([reserva.model_dump()])
    previsao = modelo.predict(dados)[0]

    return {
        "previsao": previsao,
        "cancelamento": previsao == "Canceled"
    }
