"""
monitor.py
------------
Simula il monitoraggio delle predizioni e il rilevamento del drift.
"""

import pandas as pd
from transformers import pipeline

# Carico il modello
model = pipeline(
    "sentiment-analysis",
    model="sshleifer/tiny-distilroberta-base"
)

# Simulo dati in arrivo
texts = [
    "I love this product!",
    "This is terrible.",
    "Not bad, could be better.",
    "Amazing experience!",
    "Worst service ever!"
]

logs = []

print("Raccolta predizioni...")

for text in texts:
    prediction = model(text)[0]
    logs.append({
        "text": text,
        "label": prediction["label"],
        "score": prediction["score"]
    })

df = pd.DataFrame(logs)
print(df)

# =========================
# Drift Detection semplice
# =========================

distribution = df["label"].value_counts(normalize=True)
print("\nDistribuzione classi:")
print(distribution)

# Se una classe supera 80%, segnalo possibile drift
if distribution.max() > 0.8:
    print("\n Possibile DRIFT rilevato!")
else:
    print("\n Distribuzione bilanciata.")