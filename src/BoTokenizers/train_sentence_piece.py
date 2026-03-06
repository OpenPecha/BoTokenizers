
import os
import shutil
import sentencepiece as spm
from huggingface_hub import HfApi
from transformers import AlbertTokenizer

def create_model_card(output_dir, vocab_size):
    """Creates a README.md model card for the tokenizer."""
    readme_content = f"""---
language:
- bo
library_name: transformers
tags:
- tokenizer
- sentencepiece
- tibetan
- unigram
license: apache-2.0
---

# BoSentencePiece - Tibetan SentencePiece Tokenizer

A SentencePiece tokenizer trained on Tibetan text using the Unigram language model algorithm.

## Model Details

| Parameter | Value |
|-----------|-------|
| **Model Type** | Unigram |
| **Vocabulary Size** | {vocab_size:,} |
| **Character Coverage** | 100% |
| **Max Token Length** | 16 |

## Special Tokens

| Token | ID | Description |
|-------|-----|-------------|
| `<unk>` | 0 | Unknown token |
| `<s>` | 1 | Beginning of sequence |
| `</s>` | 2 | End of sequence |
| `<pad>` | 3 | Padding token |

## Usage

### With Transformers

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("openpecha/BoSentencePiece")

text = "བོད་སྐད་ཀྱི་ཚིག་གྲུབ་འདི་ཡིན།"
tokens = tokenizer.tokenize(text)
print(tokens)

# Encode
encoded = tokenizer.encode(text)
print(encoded)

# Decode
decoded = tokenizer.decode(encoded)
print(decoded)
```

### With SentencePiece Directly

```python
from huggingface_hub import hf_hub_download
import sentencepiece as spm

# Download the model file
model_path = hf_hub_download("openpecha/BoSentencePiece", "spiece.model")

sp = spm.SentencePieceProcessor()
sp.load(model_path)

text = "བོད་སྐད་ཀྱི་ཚིག་གྲུབ་འདི་ཡིན།"
tokens = sp.encode_as_pieces(text)
print(tokens)
```

## License

Apache 2.0
"""
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)
    print(f"Model card saved to {readme_path}")


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
    special_tokens = ["<s>", "<pad>", "</s>"]
    
    spm.SentencePieceTrainer.train(
        f'--input={corpus_path} --model_prefix={model_prefix} '
        f'--vocab_size={vocab_size} --model_type=unigram '
        f'--character_coverage=1.0 '
        f'--user_defined_symbols={",".join(special_tokens)}'
    )
    
    model_file = f"{model_prefix}.model"
    vocab_file = f"{model_prefix}.vocab"
    print(f"SentencePiece model saved as {model_file} and {vocab_file}")
    
    # Copy the model to spiece.model (required name for AlbertTokenizer)
    spiece_model_path = os.path.join(output_dir, "spiece.model")
    shutil.copy(model_file, spiece_model_path)
    print(f"Copied model to {spiece_model_path}")
    
    # Load the trained model and save it using transformers
    print("Saving tokenizer in transformers format...")
    tokenizer = AlbertTokenizer(
        vocab_file=spiece_model_path,  # Use the correctly named spiece.model
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
    )
    
    tokenizer.save_pretrained(output_dir)
    
    # Remove tokenizer.json as it doesn't properly include the full vocabulary
    tokenizer_json_path = os.path.join(output_dir, "tokenizer.json")
    if os.path.exists(tokenizer_json_path):
        os.remove(tokenizer_json_path)
        print("Removed tokenizer.json (use slow tokenizer instead)")
    
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
    create_model_card(output_dir, vocab_size)
    if push_to_hub:
        upload_to_hf_hub(output_dir, repo_id, private, hf_token)

if __name__ == "__main__":
    corpus_path = "data/bo_corpus.txt"
    output_dir = "data/bo_sentencepiece"
    vocab_size = 20000
    push_to_hub = True
    repo_id = "openpecha/BoSentencePiece"
    private = False
    hf_token = os.getenv("HF_TOKEN")
    train(corpus_path, output_dir, vocab_size, push_to_hub, repo_id, private, hf_token)
    print("Done")