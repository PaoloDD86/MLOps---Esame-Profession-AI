"""
Test automatico per verificare che il modello funzioni correttamente.
Uso pytest.
"""

from transformers import pipeline

def test_sentiment_pipeline():
    model = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest"
    )

    result = model("I love this product!")

    # Verifica che il risultato non sia vuoto
    assert len(result) > 0

    # Verifica che esista l'etichetta
    assert "label" in result[0]