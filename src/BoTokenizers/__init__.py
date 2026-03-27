"""
BoTokenizers - Tibetan tokenizer training and usage

Includes Tibetan-optimized preprocessing and training.
"""

__version__ = "0.1.0"

# Original exports
from BoTokenizers.config import BO_BPE_MODEL_ID, BO_SENTENCEPIECE_MODEL_ID
from BoTokenizers.tokenize import (
    BpeTokenizer,
    SentencePieceTokenizer,
    TibetanSentencePieceTokenizer,  # Now in tokenize.py
)

# Tibetan preprocessing exports
from BoTokenizers.tibetan_preprocessor import (
    normalize_tibetan_text,
    pretokenize_on_tsheg,
    extract_tibetan_only,
    preprocess_for_training,
    preprocess_for_inference,
)

__all__ = [
    # Original
    "BO_BPE_MODEL_ID",
    "BO_SENTENCEPIECE_MODEL_ID",
    "BpeTokenizer",
    "SentencePieceTokenizer",
    # Tibetan-optimized
    "TibetanSentencePieceTokenizer",
    "normalize_tibetan_text",
    "pretokenize_on_tsheg",
    "extract_tibetan_only",
    "preprocess_for_training",
    "preprocess_for_inference",
]
