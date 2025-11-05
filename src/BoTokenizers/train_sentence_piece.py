
import os
import argparse
import sentencepiece as spm
from huggingface_hub import HfApi, login, HfFolder
from transformers import AlbertTokenizer

def train_sentencepiece(corpus_path, output_dir, vocab_size):
    """
    Trains a SentencePiece model and saves it in a format compatible with Hugging Face Hub.

    Args:
        corpus_path (str): Path to the training corpus.
        output_dir (str): Directory to save the trained tokenizer files.
        vocab_size (int): The size of the vocabulary.
        
    Returns:
        The path to the output directory containing the tokenizer files.
    """
    print("Training SentencePiece tokenizer...")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    model_prefix = os.path.join(output_dir, "sentencepiece")
    special_tokens = ["<s>", "<pad>", "</s>", "<unk>"]
    
    spm.SentencePieceTrainer.train(
        f'--input={corpus_path} --model_prefix={model_prefix} '
        f'--vocab_size={vocab_size} --model_type=bpe '
        f'--user_defined_symbols={",".join(special_tokens)}'
    )
    
    model_file = f"{model_prefix}.model"
    vocab_file = f"{model_prefix}.vocab"
    print(f"SentencePiece model saved as {model_file} and {vocab_file}")
    
    # Load the trained model and save it using transformers
    print("Saving tokenizer in transformers format...")
    tokenizer = AlbertTokenizer(
        vocab_file=vocab_file,
        spm_model_file=model_file,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
    )
    
    tokenizer.save_pretrained(output_dir)
    print(f"Tokenizer files saved to {output_dir}")
    return output_dir

def upload_to_hf_hub(output_dir, repo_id, private, hf_token):
    """
    Uploads the trained tokenizer files to the Hugging Face Hub.

    Args:
        output_dir (str): The directory containing the tokenizer files.
        repo_id (str): The ID of the repository on the Hugging Face Hub.
        private (bool): Whether to create a private repository.
    """
    api = HfApi(token=hf_token)
    
    print(f"Uploading tokenizer files to Hugging Face Hub repository: {repo_id}")
    
    # Create the repository
    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=private,
        exist_ok=True,
    )
    
    # Upload the entire folder
    api.upload_folder(
        folder_path=output_dir,
        repo_id=repo_id,
        repo_type="model",
    )
    
    print(f"Successfully uploaded to https://huggingface.co/{repo_id}")

def train(corpus_path, output_dir, vocab_size, push_to_hub, repo_id, private, hf_token):
    output_dir = train_sentencepiece(corpus_path, output_dir, vocab_size)
    if push_to_hub:
        upload_to_hf_hub(output_dir, repo_id, private, hf_token)