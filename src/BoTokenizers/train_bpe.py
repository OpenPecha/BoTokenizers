import os
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

def train_bpe_tokenizer(corpus_path, output_dir, vocab_size):
    """
    Trains a Byte-level BPE tokenizer on a given corpus.

    Args:
        corpus_path (str): Path to the text file to train on.
        output_dir (str): Directory to save the trained tokenizer.
        vocab_size (int): The desired size of the vocabulary.
    """
    # Initialize a new tokenizer with a BPE model
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))

    # Set the pre-tokenizer to ByteLevel
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()

    # Define the trainer, including special tokens
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<s>", "<pad>", "</s>", "<unk>"]
    )

    # Train the tokenizer
    print(f"Training tokenizer on {corpus_path}...")
    tokenizer.train(files=[corpus_path], trainer=trainer)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the tokenizer
    output_path = os.path.join(output_dir, "tokenizer.json", )
    tokenizer.save(output_path)
    
    print(f"Tokenizer training complete. Saved to {output_path}")
    return output_path

def main():
    
    corpus = "./data/clean_corpus.txt"
    output_dir = "./data"
    vocab_size = 32000
    
    train_bpe_tokenizer(corpus, output_dir, vocab_size)

if __name__ == "__main__":
    main()
