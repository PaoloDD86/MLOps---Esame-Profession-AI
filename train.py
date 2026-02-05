"""
Script di training del modello.
Simula una pipeline MLOps con:
- caricamento dataset
- preprocessing
- training
- logging con MLflow
"""

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
import mlflow
try:
    import torch
    _dataloader_pin_memory = torch.cuda.is_available()
except Exception:
    _dataloader_pin_memory = False

# =========================
# 1. Caricamento Dataset
# =========================

print("📥 Caricamento dataset...")
dataset = load_dataset("tweet_eval", "sentiment")

# =========================
# 2. Tokenizzazione
# =========================

print("🔤 Tokenizzazione...")
tokenizer = AutoTokenizer.from_pretrained(
    "sshleifer/tiny-distilroberta-base"
)

def tokenize(batch):
    """
    Trasforma il testo in token numerici.
    """
    # Uso padding/truncation a lunghezza fissa per evitare mismatch 
    # nelle dimensioni delle sequenze durante il batching.
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

tokenized_dataset = dataset.map(tokenize, batched=True)

# =========================
# 3. Modello
# =========================

print("🤖 Caricamento modello...")
model = AutoModelForSequenceClassification.from_pretrained(
    "sshleifer/tiny-distilroberta-base",
    num_labels=3
)

# =========================
# 4. Parametri di Training
# =========================

# Costruisce TrainingArguments in modo compatibile con diverse versioni
# di `transformers`. Alcune versioni più vecchie non accettano
# l'argomento `evaluation_strategy` e solleveranno TypeError.
try:
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        per_device_train_batch_size=4,
        num_train_epochs=1,
        logging_dir="./logs",
        report_to="none",  # disabilita wandb
        dataloader_pin_memory=_dataloader_pin_memory,
        dataloader_num_workers=0,
    )
except TypeError:
    # Fallback per versioni più vecchie: provo con l'argomento
    # `evaluate_during_training` (se disponibile), altrimenti senza
    # argomenti di valutazione espliciti.
    try:
        training_args = TrainingArguments(
            output_dir="./results",
            evaluate_during_training=True,
            per_device_train_batch_size=4,
            num_train_epochs=1,
            logging_dir="./logs",
            report_to="none",
            dataloader_pin_memory=_dataloader_pin_memory,
            dataloader_num_workers=0,
        )
    except TypeError:
        training_args = TrainingArguments(
            output_dir="./results",
            per_device_train_batch_size=4,
            num_train_epochs=1,
            logging_dir="./logs",
            report_to="none",
            dataloader_pin_memory=_dataloader_pin_memory,
            dataloader_num_workers=0,
        )

# Dataset ridotto per velocità su Codespace
train_dataset = tokenized_dataset["train"].shuffle().select(range(500))
eval_dataset = tokenized_dataset["test"].shuffle().select(range(100))

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

# =========================
# 5. Tracking con MLflow
# =========================

mlflow.set_experiment("Sentiment-MLOps")

with mlflow.start_run():
    print("Avvio training...")
    trainer.train()

    metrics = trainer.evaluate()
    accuracy = metrics.get("eval_accuracy", 0)

    # Log parametri e metriche
    mlflow.log_param("model_name", "roberta-twitter")
    mlflow.log_metric("accuracy", accuracy)

    print("✅ Training completato")
    print("Metriche:", metrics)