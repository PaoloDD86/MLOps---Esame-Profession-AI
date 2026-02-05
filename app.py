"""
app.py
--------
API REST che espone il modello di sentiment analysis.
Uso FastAPI perché è veloce, semplice e genera documentazione automatica.
"""

from fastapi import FastAPI
from transformers import pipeline

# Creo l'app FastAPI
app = FastAPI(title="Sentiment Analysis API")

# Carico il modello UNA SOLA VOLTA all'avvio
# Questo evita di ricaricarlo a ogni richiesta (molto costoso)
sentiment_model = pipeline(
    task="sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest"
)

@app.get("/")
def health_check():
    """
    Endpoint di controllo.
    Serve per verificare che il servizio sia attivo.
    """
    return {"status": "API running"}

@app.post("/predict")
def predict(text: str):
    """
    Endpoint principale.

    Parametri:
        text (str): testo da analizzare

    Ritorna:
        dizionario con predizione del sentiment
    """
    prediction = sentiment_model(text)

    return {
        "input_text": text,
        "prediction": prediction
    }