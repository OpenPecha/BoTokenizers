import os
import argparse
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from huggingface_hub import HfApi, login, HfFolder
from transformers import PreTrainedTokenizerFast

def train_bpe_tokenizer(corpus_path, output_dir, vocab_size):
    """
    Trains a Byte-level BPE tokenizer on a given corpus.

    Args:
        corpus_path (str): Path to the text file to train on.
        output_dir (str): Directory to save the trained tokenizer files.
        vocab_size (int): The desired size of the vocabulary.
        
    Returns:
        The path to the output directory containing the tokenizer files.
    """
    # Initialize a new tokenizer with a BPE model
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))

    # Set the pre-tokenizer to ByteLevel
    tokenizer.pre_tokenizer = ByteLevel()

    # Define the trainer, including special tokens
    special_tokens = ["<s>", "<pad>", "</s>", "<unk>"]
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens
    )

    # Train the tokenizer
    print(f"Training tokenizer on {corpus_path}...")
    tokenizer.train(files=[corpus_path], trainer=trainer)
    
    # Wrap the trained tokenizer in a PreTrainedTokenizerFast
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
    )
    
    # Ensure output directory exists and save the tokenizer
    os.makedirs(output_dir, exist_ok=True)
    fast_tokenizer.save_pretrained(output_dir)
    
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
    output_dir = train_bpe_tokenizer(corpus_path, output_dir, vocab_size)
    
    if push_to_hub:
        upload_to_hf_hub(output_dir, repo_id, private, hf_token)


