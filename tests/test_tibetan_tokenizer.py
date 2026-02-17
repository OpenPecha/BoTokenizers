"""
Test suite for Tibetan tokenizer improvements.

Tests cover:
1. Tibetan normalization
2. Tsheg-aware pre-tokenization
3. Tokenizer training
4. Inference with normalization
"""

import unittest
import tempfile
import os
from pathlib import Path

from BoTokenizers.tibetan_preprocessor import (
    normalize_tibetan_text,
    pretokenize_on_tsheg,
    extract_tibetan_only,
    preprocess_for_training,
    preprocess_for_inference
)


class TestTibetanNormalization(unittest.TestCase):
    """Test botok normalization"""
    
    def test_normalize_unicode(self):
        """Test Unicode normalization preserves Tibetan text"""
        text = "བོད་སྐད་ནི་རིག་གནས་ཡིན།"
        normalized = normalize_tibetan_text(text)
        # Should not be empty
        self.assertTrue(len(normalized) > 0)
        # Should still be Tibetan
        self.assertIn('བ', normalized)
    
    def test_normalize_empty(self):
        """Test normalization handles empty text"""
        self.assertEqual(normalize_tibetan_text(""), "")
        self.assertEqual(normalize_tibetan_text("   "), "")


class TestTshegPreTokenization(unittest.TestCase):
    """Test tsheg-aware pre-tokenization"""
    
    def test_pretokenize_keep_tsheg(self):
        """Test pre-tokenization keeps tsheg as separate token"""
        text = "བོད་སྐད་ནི།"
        result = pretokenize_on_tsheg(text, keep_tsheg=True)
        
        # Should have spaces around tsheg
        self.assertIn('་', result)
        # Syllables should be separated
        self.assertIn('བོད', result)
        self.assertIn('སྐད', result)
        self.assertIn('ནི', result)
    
    def test_pretokenize_replace_tsheg(self):
        """Test pre-tokenization replaces tsheg with space"""
        text = "བོད་སྐད་ནི།"
        result = pretokenize_on_tsheg(text, keep_tsheg=False)
        
        # Tsheg should be gone
        self.assertNotIn('་', result)
        # Should have spaces between syllables
        parts = result.split()
        self.assertIn('བོད', parts)
        self.assertIn('སྐད', parts)
        self.assertIn('ནི', parts)
    
    def test_pretokenize_empty(self):
        """Test pre-tokenization handles empty text"""
        self.assertEqual(pretokenize_on_tsheg(""), "")
        self.assertEqual(pretokenize_on_tsheg("   "), "")
    
    def test_pretokenize_no_tsheg(self):
        """Test text without tsheg"""
        text = "འདི"
        result = pretokenize_on_tsheg(text)
        self.assertEqual(result.strip(), text)


class TestTibetanExtraction(unittest.TestCase):
    """Test Tibetan-only extraction"""
    
    def test_extract_pure_tibetan(self):
        """Test extraction preserves pure Tibetan"""
        text = "བོད་སྐད་ནི་རིག་གནས་ཡིན།"
        result = extract_tibetan_only(text)
        self.assertEqual(result.replace(' ', ''), text.replace(' ', ''))
    
    def test_extract_mixed_content(self):
        """Test extraction removes non-Tibetan"""
        text = "བོད་སྐད་ Page 123 ISBN 978-1234567890 ནི་རིག་གནས།"
        result = extract_tibetan_only(text, keep_spaces=True)
        
        # Should have Tibetan
        self.assertIn('བོད', result)
        self.assertIn('ནི', result)
        
        # Should not have English/numbers
        self.assertNotIn('Page', result)
        self.assertNotIn('ISBN', result)
        self.assertNotIn('123', result)
    
    def test_extract_no_tibetan(self):
        """Test extraction of text with no Tibetan"""
        text = "This is English text 123"
        result = extract_tibetan_only(text)
        # Should be empty or whitespace only
        self.assertEqual(result.strip(), "")


class TestPreprocessingPipeline(unittest.TestCase):
    """Test complete preprocessing pipeline"""
    
    def test_training_preprocessing_all_enabled(self):
        """Test training preprocessing with all options"""
        text = "བོད་སྐད་ English text ནི་རིག་གནས།"
        result = preprocess_for_training(
            text,
            normalize=True,
            pretokenize_tsheg=True,
            keep_tsheg=True,
            tibetan_only=True
        )
        
        # Should have Tibetan
        self.assertIn('བོད', result)
        # Should not have English
        self.assertNotIn('English', result)
        # Should have tsheg as separate token (with spaces)
        self.assertIn('་', result)
    
    def test_inference_preprocessing(self):
        """Test inference preprocessing"""
        text = "བོད་སྐད་ English ནི།"
        
        # Without filtering
        result1 = preprocess_for_inference(
            text,
            normalize=True,
            tibetan_only=False
        )
        self.assertIn('English', result1)
        
        # With filtering
        result2 = preprocess_for_inference(
            text,
            normalize=True,
            tibetan_only=True
        )
        self.assertNotIn('English', result2)
    
    def test_empty_text_handling(self):
        """Test preprocessing handles empty text gracefully"""
        self.assertEqual(preprocess_for_training(""), "")
        self.assertEqual(preprocess_for_inference(""), "")


class TestCorpusPreprocessing(unittest.TestCase):
    """Test corpus file preprocessing"""
    
    def test_corpus_preprocessing(self):
        """Test preprocessing a corpus file"""
        # Create temporary input file
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, suffix='.txt') as f:
            f.write("བོད་སྐད་ནི་རིག་གནས་ཡིན།\n")
            f.write("English mixed content དེ་མཛེས་པོ་རེད།\n")
            f.write("\n")  # Empty line
            f.write("དགའ་བོ།\n")
            input_file = f.name
        
        try:
            # Create temporary output file
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, suffix='.txt') as f:
                output_file = f.name
            
            # Import here to avoid circular imports
            from BoTokenizers.tibetan_preprocessor import preprocess_corpus_for_training
            
            # Preprocess corpus
            result = preprocess_corpus_for_training(
                input_path=input_file,
                output_path=output_file,
                normalize=True,
                pretokenize_tsheg=True,
                keep_tsheg=True,
                tibetan_only=True,
                show_progress=False
            )
            
            # Check output file exists
            self.assertTrue(os.path.exists(output_file))
            
            # Read and check content
            with open(output_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Should have same number of lines (including empty)
            self.assertEqual(len(lines), 4)
            
            # First line should have Tibetan with tsheg separation
            self.assertIn('བོད', lines[0])
            self.assertIn('་', lines[0])
            
            # Second line should not have English
            self.assertNotIn('English', lines[1])
            
            # Third line should be empty
            self.assertEqual(lines[2].strip(), '')
            
            # Fourth line should have Tibetan
            self.assertIn('དགའ', lines[3])
        
        finally:
            # Cleanup
            if os.path.exists(input_file):
                os.unlink(input_file)
            if os.path.exists(output_file):
                os.unlink(output_file)


class TestTokenizationComparison(unittest.TestCase):
    """Test tokenization before and after improvements"""
    
    def test_tsheg_preservation(self):
        """Test that tsheg boundaries are preserved"""
        text = "བོད་སྐད་ནི་རིག་གནས་ཡིན།"
        
        # Pre-tokenize with tsheg awareness
        preprocessed = pretokenize_on_tsheg(text, keep_tsheg=True)
        
        # Should have more tokens due to tsheg separation
        original_syllables = text.count('་')
        preprocessed_tokens = len(preprocessed.split())
        
        # Should have at least as many tokens as tsheg marks
        # (each tsheg becomes a separate token, plus syllables)
        self.assertGreaterEqual(preprocessed_tokens, original_syllables)
    
    def test_whitespace_error_handling(self):
        """Test handling of OCR whitespace errors"""
        # Correct text
        correct = "བོད་སྐད་ནི་རིག་གནས།"
        
        # OCR error with extra space
        ocr_error = "བོད་སྐད་ ནི་རིག་གནས།"
        
        # Pre-tokenize both
        correct_preprocessed = pretokenize_on_tsheg(correct, keep_tsheg=False)
        error_preprocessed = pretokenize_on_tsheg(ocr_error, keep_tsheg=False)
        
        # Both should have all syllables (space is normalized)
        for syllable in ['བོད', 'སྐད', 'ནི', 'རིག', 'གནས']:
            self.assertIn(syllable, correct_preprocessed)
            self.assertIn(syllable, error_preprocessed)


def run_tests():
    """Run all tests"""
    unittest.main(argv=[''], verbosity=2, exit=False)


if __name__ == '__main__':
    run_tests()
