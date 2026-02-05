"""
Test per il modello e il training pipeline.
Valida il caricamento del modello e la tokenizzazione.
"""

import pytest
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def test_model_loading():
    """Test che il modello si carica correttamente."""
    model_name = "sshleifer/tiny-distilroberta-base"
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3
    )
    assert model is not None
    assert model.config.num_labels == 3


def test_tokenizer_loading():
    """Test che il tokenizer si carica correttamente."""
    model_name = "sshleifer/tiny-distilroberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    assert tokenizer is not None


def test_tokenization():
    """Test che la tokenizzazione funziona con padding fisso."""
    model_name = "sshleifer/tiny-distilroberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    text = "I love this product!"
    encoded = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=128
    )
    
    assert "input_ids" in encoded
    assert len(encoded["input_ids"]) == 128
    assert "attention_mask" in encoded


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
