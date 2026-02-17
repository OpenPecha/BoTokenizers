import argparse
import os
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer
import sentencepiece as spm
from typing import List
from BoTokenizers.config import BO_BPE_MODEL_ID, BO_SENTENCEPIECE_MODEL_ID

# Import Tibetan preprocessor (lazy import to avoid hard dependency)
try:
    from BoTokenizers.tibetan_preprocessor import preprocess_for_inference
    TIBETAN_PREPROCESSOR_AVAILABLE = True
except ImportError:
    TIBETAN_PREPROCESSOR_AVAILABLE = False


class BpeTokenizer:
    """
    A BPE tokenizer that loads a model from the Hugging Face Hub.
    """
    def __init__(self, repo_id: str = BO_BPE_MODEL_ID):
        """
        Initializes the tokenizer and downloads the model from the Hub.

        Args:
            repo_id (str): The repository ID on the Hugging Face Hub.
        """
        print(f"Loading BPE tokenizer from {repo_id}...")
        tokenizer_path = hf_hub_download(repo_id=repo_id, filename="tokenizer.json")
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        print("BPE tokenizer loaded successfully.")

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenizes a string of text.

        Args:
            text (str): The text to tokenize.

        Returns:
            A list of tokens.
        """
        return self.tokenizer.encode(text).tokens


class SentencePieceTokenizer:
    """
    A SentencePiece tokenizer that loads a model from the Hugging Face Hub.
    """
    def __init__(self, repo_id: str = BO_SENTENCEPIECE_MODEL_ID):
        """
        Initializes the tokenizer and downloads the model from the Hub.

        Args:
            repo_id (str): The repository ID on the Hugging Face Hub.
        """
        print(f"Loading SentencePiece tokenizer from {repo_id}...")
        model_path = hf_hub_download(repo_id=repo_id, filename=f"{repo_id.split('/')[-1]}.model")
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)
        print("SentencePiece tokenizer loaded successfully.")

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenizes a string of text.

        Args:
            text (str): The text to tokenize.

        Returns:
            A list of tokens.
        """
        return self.sp.encode_as_pieces(text)


class TibetanSentencePieceTokenizer:
    """
    Tibetan-optimized SentencePiece tokenizer with built-in normalization.
    
    This tokenizer automatically applies botok normalization before tokenization
    and is designed to work with Tibetan-aware models (trained with tsheg-awareness).
    
    Usage:
        tokenizer = TibetanSentencePieceTokenizer(normalize=True)
        tokens = tokenizer.tokenize("བོད་སྐད་ནི་རིག་གནས་ཡིན།")
    """
    def __init__(
        self, 
        repo_id: str = BO_SENTENCEPIECE_MODEL_ID,
        normalize: bool = True,
        tibetan_only: bool = False
    ):
        """
        Initialize Tibetan-optimized tokenizer.
        
        Args:
            repo_id: Repository ID on Hugging Face Hub
            normalize: Apply botok normalization before tokenization
            tibetan_only: Filter to Tibetan characters only
        """
        print(f"Loading Tibetan-optimized SentencePiece tokenizer from {repo_id}...")
        model_path = hf_hub_download(repo_id=repo_id, filename=f"{repo_id.split('/')[-1]}.model")
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)
        self.normalize = normalize
        self.tibetan_only = tibetan_only
        
        if (normalize or tibetan_only) and not TIBETAN_PREPROCESSOR_AVAILABLE:
            print("Warning: Tibetan preprocessor not available. Install botok for normalization.")
            print("  pip install botok")
            self.normalize = False
            self.tibetan_only = False
        
        print("Tibetan-optimized tokenizer loaded successfully.")
        if TIBETAN_PREPROCESSOR_AVAILABLE:
            print(f"  Normalization: {'enabled' if normalize else 'disabled'}")
            print(f"  Tibetan-only: {'enabled' if tibetan_only else 'disabled'}")
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text with automatic Tibetan preprocessing.
        
        Args:
            text: Input text
        
        Returns:
            List of tokens
        """
        # Preprocess text if enabled and available
        if TIBETAN_PREPROCESSOR_AVAILABLE and (self.normalize or self.tibetan_only):
            text = preprocess_for_inference(
                text,
                normalize=self.normalize,
                tibetan_only=self.tibetan_only
            )
        
        # Tokenize
        return self.sp.encode_as_pieces(text)
    
    def encode(self, text: str) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text
        
        Returns:
            List of token IDs
        """
        # Preprocess text if enabled and available
        if TIBETAN_PREPROCESSOR_AVAILABLE and (self.normalize or self.tibetan_only):
            text = preprocess_for_inference(
                text,
                normalize=self.normalize,
                tibetan_only=self.tibetan_only
            )
        
        # Encode
        return self.sp.encode_as_ids(text)
    
    def decode(self, ids: List[int]) -> str:
        """
        Decode token IDs to text.
        
        Args:
            ids: List of token IDs
        
        Returns:
            Decoded text
        """
        return self.sp.decode_ids(ids)


def main():
    parser = argparse.ArgumentParser(
        description="Tokenize text or a file using a model from the Hugging Face Hub."
    )
    parser.add_argument(
        "--repo_id", 
        type=str, 
        help="Optional: Repository ID on the Hugging Face Hub. "
             "If not provided, the default from config.py will be used."
    )
    parser.add_argument(
        "--tokenizer_type", 
        type=str, 
        required=True, 
        choices=["bpe", "sentencepiece", "tibetan"],
        help="The type of tokenizer to use. 'tibetan' uses Tibetan-optimized tokenizer with normalization."
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", type=str, help="A string of text to tokenize.")
    group.add_argument("--file_path", type=str, help="Path to a text file to tokenize.")
    
    # Tibetan-specific options
    parser.add_argument(
        "--normalize",
        action="store_true",
        default=True,
        help="Apply botok normalization (default: True for tibetan tokenizer)"
    )
    parser.add_argument(
        "--no-normalize",
        dest="normalize",
        action="store_false",
        help="Disable botok normalization"
    )
    parser.add_argument(
        "--tibetan-only",
        action="store_true",
        help="Filter to Tibetan characters only"
    )
    
    args = parser.parse_args()

    # Determine repo_id
    repo_id = args.repo_id
    if not repo_id:
        if args.tokenizer_type == "bpe":
            repo_id = BO_BPE_MODEL_ID
        else:
            repo_id = BO_SENTENCEPIECE_MODEL_ID

    # Initialize tokenizer
    if args.tokenizer_type == "bpe":
        tokenizer = BpeTokenizer(repo_id=repo_id)
    elif args.tokenizer_type == "tibetan":
        tokenizer = TibetanSentencePieceTokenizer(
            repo_id=repo_id,
            normalize=args.normalize,
            tibetan_only=args.tibetan_only
        )
    else:
        tokenizer = SentencePieceTokenizer(repo_id=repo_id)

    # Get text content
    if args.text:
        content = args.text
    else:
        with open(args.file_path, 'r', encoding='utf-8') as f:
            content = f.read()

    # Tokenize
    tokens = tokenizer.tokenize(content)
    
    # Display results
    print("\n" + "="*80)
    print("TOKENIZATION RESULT")
    print("="*80)
    print(f"Original text: {content[:100]}{'...' if len(content) > 100 else ''}")
    print(f"\nTokens ({len(tokens)} total):")
    print(tokens)
    print("="*80)


if __name__ == "__main__":
    main()
