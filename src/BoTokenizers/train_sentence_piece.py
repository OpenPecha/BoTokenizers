
import argparse
import sentencepiece as spm
from huggingface_hub import HfApi, login, HfFolder
import os

def train_sentencepiece(corpus_path, model_prefix, vocab_size):
    """
    Trains a SentencePiece model.

    Args:
        corpus_path (str): Path to the training corpus.
        model_prefix (str): Prefix for the model and vocab files.
        vocab_size (int): The size of the vocabulary.
    
    Returns:
        A tuple containing the path to the model and vocab files.
    """
    print("Training SentencePiece tokenizer...")
    spm.SentencePieceTrainer.train(
        f'--input={corpus_path} --model_prefix={model_prefix} '
        f'--vocab_size={vocab_size} --model_type=bpe'
    )
    model_file = f"{model_prefix}.model"
    vocab_file = f"{model_prefix}.vocab"
    print(f"SentencePiece model saved as {model_file} and {vocab_file}")
    return model_file, vocab_file

def upload_to_hf_hub(model_file, vocab_file, repo_id, private):
    """
    Uploads the trained SentencePiece model to the Hugging Face Hub.

    Args:
        model_file (str): Path to the model file.
        vocab_file (str): Path to the vocab file.
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
    
    # Upload the model and vocab files
    api.upload_file(
        path_or_fileobj=model_file,
        path_in_repo=os.path.basename(model_file),
        repo_id=repo_id,
    )
    api.upload_file(
        path_or_fileobj=vocab_file,
        path_in_repo=os.path.basename(vocab_file),
        repo_id=repo_id,
    )
    
    print(f"Successfully uploaded to https://huggingface.co/{repo_id}")

def main():
    parser = argparse.ArgumentParser(description="Train a SentencePiece BPE model and optionally upload to Hugging Face Hub.")
    parser.add_argument("--corpus_path", type=str, required=True, help="Path to the training corpus.")
    parser.add_argument("--model_prefix", type=str, required=True, help="Prefix for the model and vocab files.")
    parser.add_argument("--vocab_size", type=int, default=32000, help="The size of the vocabulary.")
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
    
    model_file, vocab_file = train_sentencepiece(args.corpus_path, args.model_prefix, args.vocab_size)
    
    if args.push_to_hub:
        upload_to_hf_hub(model_file, vocab_file, args.repo_id, args.private)

if __name__ == "__main__":
    main()