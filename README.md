# BoTokenizers

A Python package for training and using BPE and SentencePiece tokenizers for Tibetan text, with support for uploading to and downloading from the Hugging Face Hub.

## Installation

To install the package, clone the repository and install the dependencies:

```bash
git clone https://github.com/OpenPecha/BoTokenizers.git
cd BoTokenizers
pip install .
```

For development, install with the `dev` dependencies:

```bash
pip install -e ".[dev]"
```

## Usage

### Training a BPE Tokenizer

You can train a Byte-level BPE tokenizer using the `tokenizers` library. This script also supports uploading the trained tokenizer to the Hugging Face Hub.

**Usage:**

To train a new BPE tokenizer, run the `train_bpe.py` script:

```bash
python src/BoTokenizers/train_bpe.py \
    --corpus <path_to_your_corpus.txt> \
    --output_dir <directory_to_save_tokenizer> \
    --vocab_size <vocabulary_size> \
    --push_to_hub \
    --repo_id <your_hf_username/your_repo_name> \
    --private # Optional: to make the repo private
```

### Training a SentencePiece Tokenizer

This project includes a script to train a SentencePiece BPE model on your own corpus. You can also upload the trained tokenizer to the Hugging Face Hub.

**Usage:**

To train a new tokenizer, run the `train_sentence_piece.py` script:

```bash
python src/BoTokenizers/train_sentence_piece.py \
    --corpus_path <path_to_your_corpus.txt> \
    --model_prefix <your_model_name> \
    --vocab_size <vocabulary_size> \
    --push_to_hub \
    --repo_id <your_hf_username/your_repo_name> \
    --private  # Optional: to make the repo private
```

### Tokenizing Text

You can use the `tokenize.py` script to tokenize text or files using the trained models from the Hugging Face Hub.

**Usage from Command Line:**

```bash
# Tokenize a string using the default BPE model
python -m BoTokenizers.tokenize --tokenizer_type bpe --text "བཀྲ་ཤིས་བདེ་ལེགས།"

# Tokenize a file using the default SentencePiece model
python -m BoTokenizers.tokenize --tokenizer_type sentencepiece --file_path ./data/corpus.txt
```

**Programmatic Usage:**

```python
from BoTokenizers.tokenize import BpeTokenizer, SentencePieceTokenizer

# Initialize with default models from config
bpe_tokenizer = BpeTokenizer()
sp_tokenizer = SentencePieceTokenizer()

text = "བཀྲ་ཤིས་བདེ་ལེགས།"
bpe_tokens = bpe_tokenizer.tokenize(text)
sp_tokens = sp_tokenizer.tokenize(text)

print("BPE Tokens:", bpe_tokens)
print("SentencePiece Tokens:", sp_tokens)
```

## References

- [What is BPE Tokenization? (YouTube)](https://www.youtube.com/watch?v=BcxJk4WQVIw)

## Contributing

If you'd like to help out, check out our [contributing guidelines](/CONTRIBUTING.md).

## How to get help

* File an issue.
* Email us at openpecha[at]gmail.com.
* Join our [discord](https://discord.com/invite/7GFpPFSTeA).

## License

This project is licensed under the [MIT License](/LICENSE.md).
