"""
Tibetan-specific text preprocessing for tokenizer training and inference.

This module provides utilities for:
1. Unicode and graphical normalization using botok
2. Tsheg-aware pre-tokenization
3. Text cleaning for OCR applications
"""

import re
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)

# Lazy load botok to avoid import errors if not installed
_botok_loaded = False
_normalize_unicode = None
_normalize_graphical = None


def _load_botok():
    """Lazy load botok normalization functions"""
    global _botok_loaded, _normalize_unicode, _normalize_graphical
    
    if _botok_loaded:
        return True
    
    try:
        from botok import normalize_unicode
        from botok.utils.lenient_normalization import normalize_graphical
        _normalize_unicode = normalize_unicode
        _normalize_graphical = normalize_graphical
        _botok_loaded = True
        logger.info("Botok normalization loaded successfully")
        return True
    except ImportError as e:
        logger.warning(f"Could not import botok: {e}. Install with: pip install botok")
        _botok_loaded = False
        return False


def normalize_tibetan_text(text: str, use_botok: bool = True) -> str:
    """
    Normalize Tibetan text using botok normalization.
    
    Args:
        text: Input text
        use_botok: If True, use botok normalization (recommended)
    
    Returns:
        Normalized text
    """
    if not text or not text.strip():
        return ""
    
    if use_botok and _load_botok():
        # Apply Unicode normalization
        text = _normalize_unicode(text)
        # Apply graphical normalization
        text = _normalize_graphical(text)
    
    return text


def pretokenize_on_tsheg(text: str, keep_tsheg: bool = True) -> str:
    """
    Pre-tokenize Tibetan text on tsheg (་) boundaries.
    
    This splits the text at tsheg marks and replaces them with spaces
    (or keeps them as separate tokens) so that SentencePiece learns
    syllable-level patterns instead of treating entire sentences as units.
    
    Args:
        text: Input Tibetan text
        keep_tsheg: If True, keep tsheg as separate tokens; if False, replace with space
    
    Returns:
        Pre-tokenized text with tsheg boundaries marked
    
    Example:
        Input: "བོད་སྐད་ནི་རིག་གནས་ཡིན།"
        Output (keep_tsheg=True): "བོད ་ སྐད ་ ནི ་ རིག ་ གནས ་ ཡིན །"
        Output (keep_tsheg=False): "བོད སྐད ནི རིག གནས ཡིན །"
    """
    if not text or not text.strip():
        return ""
    
    if keep_tsheg:
        # Split on tsheg but keep it as a separate token
        # བོད་སྐད → བོད ་ སྐད
        text = re.sub(r'(་)', r' \1 ', text)
    else:
        # Replace tsheg with space
        # བོད་སྐད → བོད སྐད
        text = text.replace('་', ' ')
    
    # Clean up multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def extract_tibetan_only(text: str, keep_spaces: bool = True) -> str:
    """
    Extract only Tibetan Unicode characters from text.
    
    Useful for filtering out non-Tibetan content in OCR output.
    
    Args:
        text: Input text (may contain mixed scripts)
        keep_spaces: If True, preserve ASCII spaces
    
    Returns:
        Text with only Tibetan characters
    """
    if not text or not text.strip():
        return ""
    
    # Tibetan Unicode range: U+0F00 to U+0FFF
    if keep_spaces:
        pattern = r'[\u0F00-\u0FFF\u0020]+'
    else:
        pattern = r'[\u0F00-\u0FFF]+'
    
    # Extract all Tibetan segments
    tibetan_segments = re.findall(pattern, text)
    
    # Join with spaces and clean up
    cleaned_text = ' '.join(tibetan_segments)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    return cleaned_text


def preprocess_for_training(
    text: str,
    normalize: bool = True,
    pretokenize_tsheg: bool = True,
    keep_tsheg: bool = True,
    tibetan_only: bool = False
) -> str:
    """
    Complete preprocessing pipeline for training a Tibetan tokenizer.
    
    Args:
        text: Input text
        normalize: Apply botok normalization
        pretokenize_tsheg: Split on tsheg boundaries
        keep_tsheg: Keep tsheg as separate token (recommended for training)
        tibetan_only: Remove non-Tibetan characters
    
    Returns:
        Preprocessed text ready for tokenizer training
    """
    if not text or not text.strip():
        return ""
    
    # Step 1: Normalize Unicode (if enabled)
    if normalize:
        text = normalize_tibetan_text(text, use_botok=True)
    
    # Step 2: Extract Tibetan only (if enabled)
    if tibetan_only:
        text = extract_tibetan_only(text, keep_spaces=True)
    
    # Step 3: Pre-tokenize on tsheg (if enabled)
    if pretokenize_tsheg:
        text = pretokenize_on_tsheg(text, keep_tsheg=keep_tsheg)
    
    return text


def preprocess_for_inference(
    text: str,
    normalize: bool = True,
    tibetan_only: bool = False
) -> str:
    """
    Preprocessing for inference (tokenizing new text).
    
    Note: Don't pre-tokenize on tsheg for inference if the model
    was trained on pre-tokenized text - the model handles that.
    
    Args:
        text: Input text
        normalize: Apply botok normalization
        tibetan_only: Remove non-Tibetan characters
    
    Returns:
        Preprocessed text ready for tokenization
    """
    if not text or not text.strip():
        return ""
    
    # Normalize
    if normalize:
        text = normalize_tibetan_text(text, use_botok=True)
    
    # Extract Tibetan only (if needed)
    if tibetan_only:
        text = extract_tibetan_only(text, keep_spaces=True)
    
    return text


def preprocess_corpus_for_training(
    input_path: str,
    output_path: str,
    normalize: bool = True,
    pretokenize_tsheg: bool = True,
    keep_tsheg: bool = True,
    tibetan_only: bool = False,
    show_progress: bool = True
) -> str:
    """
    Preprocess an entire corpus file for tokenizer training.
    
    Args:
        input_path: Path to input corpus file
        output_path: Path to save preprocessed corpus
        normalize: Apply botok normalization
        pretokenize_tsheg: Split on tsheg boundaries
        keep_tsheg: Keep tsheg as separate token
        tibetan_only: Remove non-Tibetan characters
        show_progress: Show progress during processing
    
    Returns:
        Path to preprocessed corpus file
    """
    logger.info(f"Preprocessing corpus: {input_path}")
    logger.info(f"Settings: normalize={normalize}, pretokenize_tsheg={pretokenize_tsheg}, "
                f"keep_tsheg={keep_tsheg}, tibetan_only={tibetan_only}")
    
    lines_processed = 0
    
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        
        for line in fin:
            line = line.strip()
            if not line:
                fout.write('\n')
                continue
            
            # Preprocess line
            preprocessed = preprocess_for_training(
                line,
                normalize=normalize,
                pretokenize_tsheg=pretokenize_tsheg,
                keep_tsheg=keep_tsheg,
                tibetan_only=tibetan_only
            )
            
            fout.write(preprocessed + '\n')
            lines_processed += 1
            
            if show_progress and lines_processed % 10000 == 0:
                logger.info(f"Processed {lines_processed:,} lines")
    
    logger.info(f"Preprocessing complete. Processed {lines_processed:,} lines")
    logger.info(f"Output saved to: {output_path}")
    
    return output_path


# Example usage
if __name__ == "__main__":
    # Test normalization
    text = "བོད་སྐད་ནི་རིག་གནས་ཡིན།"
    print(f"Original: {text}")
    
    normalized = normalize_tibetan_text(text)
    print(f"Normalized: {normalized}")
    
    pretokenized = pretokenize_on_tsheg(text, keep_tsheg=True)
    print(f"Pre-tokenized (keep tsheg): {pretokenized}")
    
    pretokenized_no_tsheg = pretokenize_on_tsheg(text, keep_tsheg=False)
    print(f"Pre-tokenized (no tsheg): {pretokenized_no_tsheg}")
    
    # Test with mixed content
    mixed_text = "བོད་སྐད་ Page 123 ISBN 978-1234567890"
    print(f"\nMixed text: {mixed_text}")
    
    tibetan_extracted = extract_tibetan_only(mixed_text)
    print(f"Tibetan only: {tibetan_extracted}")
    
    # Full pipeline
    preprocessed = preprocess_for_training(
        mixed_text,
        normalize=True,
        pretokenize_tsheg=True,
        keep_tsheg=True,
        tibetan_only=True
    )
    print(f"Fully preprocessed: {preprocessed}")
