import os
import argparse
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from huggingface_hub import HfApi, login, HfFolder

def train_bpe_tokenizer(corpus_path, output_dir, vocab_size):
    """
    Trains a Byte-level BPE tokenizer on a given corpus.

    Args:
        corpus_path (str): Path to the text file to train on.
        output_dir (str): Directory to save the trained tokenizer.
        vocab_size (int): The desired size of the vocabulary.
    """
    # Initialize a new tokenizer with a BPE model
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))

    # Set the pre-tokenizer to ByteLevel
    tokenizer.pre_tokenizer = ByteLevel()

    # Define the trainer, including special tokens
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<s>", "<pad>", "</s>", "<unk>"]
    )

    # Train the tokenizer
    print(f"Training tokenizer on {corpus_path}...")
    tokenizer.train(files=[corpus_path], trainer=trainer)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the tokenizer
    output_path = os.path.join(output_dir, "tokenizer.json")
    tokenizer.save(output_path)
    
    print(f"Tokenizer training complete. Saved to {output_path}")
    return output_path

def upload_to_hf_hub(tokenizer_path, repo_id, private):
    """
    Uploads the trained tokenizer to the Hugging Face Hub.

    Args:
        tokenizer_path (str): Path to the tokenizer.json file.
        repo_id (str): The ID of the repository on the Hugging Face Hub.
        private (bool): Whether to create a private repository.
    """
    api = HfApi()
    
    print(f"Uploading to Hugging Face Hub repository: {repo_id}")
    
    # Create the repository
    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=private,
        exist_ok=True,
    )
    
    # Upload the tokenizer file
    api.upload_file(
        path_or_fileobj=tokenizer_path,
        path_in_repo=os.path.basename(tokenizer_path),
        repo_id=repo_id,
    )
    
    print(f"Successfully uploaded to https://huggingface.co/{repo_id}")

def main():
    parser = argparse.ArgumentParser(
        description="Train a Byte-level BPE tokenizer and optionally upload to Hugging Face Hub."
    )
    parser.add_argument(
        "--corpus",
        type=str,
        required=True,
        help="Path to the clean corpus file."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the trained tokenizer file (tokenizer.json)."
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=32000,
        help="The desired size of the vocabulary."
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Upload the trained model to Hugging Face Hub.")
    parser.add_argument("--repo_id", type=str, help="The ID of the repository on the Hugging Face Hub (e.g., 'username/repo_name').")
    parser.add_argument("--private", action="store_true", help="Create a private repository on the Hugging Face Hub.")
    
    args = parser.parse_args()

    # Log in to Hugging Face
    if args.push_to_hub:
        if not args.repo_id:
            raise ValueError("repo_id must be specified when pushing to hub.")
        
        token = HfFolder.get_token()
        if token is None:
            print("Hugging Face token not found. Please log in.")
            login()

    tokenizer_path = train_bpe_tokenizer(args.corpus, args.output_dir, args.vocab_size)
    
    if args.push_to_hub:
        upload_to_hf_hub(tokenizer_path, args.repo_id, args.private)

if __name__ == "__main__":
    main()
