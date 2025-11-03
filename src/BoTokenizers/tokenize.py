import argparse
import os
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer
import sentencepiece as spm
from BoTokenizers.config import BO_BPE_MODEL_ID, BO_SENTENCEPIECE_MODEL_ID


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

    def tokenize(self, text: str):
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

    def tokenize(self, text: str):
        """
        Tokenizes a string of text.

        Args:
            text (str): The text to tokenize.

        Returns:
            A list of tokens.
        """
        return self.sp.encode_as_pieces(text)

def main():
    parser = argparse.ArgumentParser(description="Tokenize text or a file using a model from the Hugging Face Hub.")
    parser.add_argument("--repo_id", type=str, help="Optional: Repository ID on the Hugging Face Hub. If not provided, the default from config.py will be used.")
    parser.add_argument("--tokenizer_type", type=str, required=True, choices=["bpe", "sentencepiece"], help="The type of tokenizer to use.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", type=str, help="A string of text to tokenize.")
    group.add_argument("--file_path", type=str, help="Path to a text file to tokenize.")

    args = parser.parse_args()

    repo_id = args.repo_id
    if not repo_id:
        if args.tokenizer_type == "bpe":
            repo_id = config.BO_BPE_MODEL_ID
        else:
            repo_id = config.BO_SENTENCEPIECE_MODEL_ID

    if args.tokenizer_type == "bpe":
        tokenizer = BpeTokenizer(repo_id=repo_id)
    else:
        tokenizer = SentencePieceTokenizer(repo_id=repo_id)

    if args.text:
        content = args.text
    else:
        with open(args.file_path, 'r', encoding='utf-8') as f:
            content = f.read()

    tokens = tokenizer.tokenize(content)
    print("\n--- Tokens ---")
    print(tokens)
    print(f"\nTotal tokens: {len(tokens)}")

if __name__ == "__main__":
    main()
