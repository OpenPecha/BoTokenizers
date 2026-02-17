"""
Train a Tibetan-optimized SentencePiece tokenizer with:
1. Tsheg-aware pre-tokenization
2. Botok normalization
3. Larger vocabulary (40k default)
4. Proper configuration for Tibetan syllable structure
"""

import os
import argparse
import sentencepiece as spm
from huggingface_hub import HfApi
from transformers import AlbertTokenizer
import logging

from BoTokenizers.tibetan_preprocessor import preprocess_corpus_for_training

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train_tibetan_sentencepiece(
    corpus_path: str,
    output_dir: str,
    vocab_size: int = 40000,
    normalize: bool = True,
    pretokenize_tsheg: bool = True,
    keep_tsheg: bool = True,
    tibetan_only: bool = False
):
    """
    Train a Tibetan-optimized SentencePiece model.
    
    Args:
        corpus_path: Path to training corpus
        output_dir: Directory to save trained tokenizer
        vocab_size: Vocabulary size (default 40k, recommended for Tibetan)
        normalize: Apply botok normalization during preprocessing
        pretokenize_tsheg: Split on tsheg before training
        keep_tsheg: Keep tsheg as separate token
        tibetan_only: Filter to Tibetan characters only
    
    Returns:
        Path to output directory with tokenizer files
    """
    logger.info("="*80)
    logger.info("TRAINING TIBETAN-OPTIMIZED SENTENCEPIECE TOKENIZER")
    logger.info("="*80)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Preprocess corpus
    logger.info("\nStep 1: Preprocessing corpus for Tibetan...")
    preprocessed_corpus = os.path.join(output_dir, "preprocessed_corpus.txt")
    
    preprocess_corpus_for_training(
        input_path=corpus_path,
        output_path=preprocessed_corpus,
        normalize=normalize,
        pretokenize_tsheg=pretokenize_tsheg,
        keep_tsheg=keep_tsheg,
        tibetan_only=tibetan_only,
        show_progress=True
    )
    
    # Step 2: Train SentencePiece model
    logger.info("\nStep 2: Training SentencePiece model...")
    
    model_prefix = os.path.join(output_dir, "sentencepiece")
    
    # Special tokens
    special_tokens = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
    
    # Training arguments optimized for Tibetan
    training_args = [
        f'--input={preprocessed_corpus}',
        f'--model_prefix={model_prefix}',
        f'--vocab_size={vocab_size}',
        '--model_type=unigram',  # Unigram is good for Tibetan
        '--character_coverage=1.0',  # Cover all Tibetan characters
        
        # Critical: Don't use whitespace as primary delimiter
        '--split_by_whitespace=false',  # Tibetan uses tsheg, not whitespace
        '--split_by_number=false',  # Keep numbers with context
        '--split_by_unicode_script=false',  # Keep Sanskrit in Tibetan script
        
        # Whitespace handling
        '--treat_whitespace_as_suffix=false',  # Whitespace not suffix marker
        '--allow_whitespace_only_pieces=true',  # Whitespace is just a character
        
        # Normalization (disabled because we normalize in preprocessing)
        '--normalization_rule_name=identity',  # No normalization (we did it already)
        
        # Special tokens
        f'--user_defined_symbols={",".join(special_tokens)}',
        
        # Training parameters
        '--num_threads=8',  # Use multiple threads
        '--max_sentence_length=4096',  # Support long texts
    ]
    
    training_command = ' '.join(training_args)
    logger.info(f"Training command:\n{training_command}\n")
    
    spm.SentencePieceTrainer.train(training_command)
    
    model_file = f"{model_prefix}.model"
    vocab_file = f"{model_prefix}.vocab"
    
    logger.info(f"✓ SentencePiece model saved: {model_file}")
    logger.info(f"✓ Vocabulary file saved: {vocab_file}")
    
    # Step 3: Save in transformers format
    logger.info("\nStep 3: Saving in transformers format...")
    
    tokenizer = AlbertTokenizer(
        vocab_file=vocab_file,
        spm_model_file=model_file,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        mask_token="<mask>",
    )
    
    tokenizer.save_pretrained(output_dir)
    logger.info(f"✓ Tokenizer files saved to: {output_dir}")
    
    # Step 4: Print statistics
    logger.info("\n" + "="*80)
    logger.info("TRAINING COMPLETE")
    logger.info("="*80)
    logger.info(f"Vocabulary size: {vocab_size:,}")
    logger.info(f"Model type: Unigram")
    logger.info(f"Preprocessing: normalize={normalize}, pretokenize_tsheg={pretokenize_tsheg}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("="*80)
    
    return output_dir


def upload_to_hf_hub(output_dir: str, repo_id: str, private: bool = False, hf_token: str = None):
    """
    Upload trained tokenizer to Hugging Face Hub.
    
    Args:
        output_dir: Directory containing tokenizer files
        repo_id: Repository ID on Hugging Face Hub
        private: Whether to create private repository
        hf_token: Hugging Face API token
    """
    if not hf_token:
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            logger.error("No Hugging Face token provided. Set HF_TOKEN environment variable.")
            return
    
    logger.info(f"\nUploading to Hugging Face Hub: {repo_id}")
    
    api = HfApi(token=hf_token)
    
    # Create repository
    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=private,
        exist_ok=True,
    )
    
    # Upload folder
    api.upload_folder(
        folder_path=output_dir,
        repo_id=repo_id,
        repo_type="model",
    )
    
    logger.info(f"✓ Successfully uploaded to https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser(
        description='Train Tibetan-optimized SentencePiece tokenizer'
    )
    
    # Required arguments
    parser.add_argument('--corpus_path', type=str, required=True,
                       help='Path to training corpus file')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save trained tokenizer')
    
    # Tokenizer parameters
    parser.add_argument('--vocab_size', type=int, default=40000,
                       help='Vocabulary size (default: 40000, recommended for Tibetan)')
    
    # Preprocessing options
    parser.add_argument('--normalize', action='store_true', default=True,
                       help='Apply botok normalization (default: True)')
    parser.add_argument('--no-normalize', dest='normalize', action='store_false',
                       help='Disable botok normalization')
    
    parser.add_argument('--pretokenize-tsheg', action='store_true', default=True,
                       help='Pre-tokenize on tsheg boundaries (default: True)')
    parser.add_argument('--no-pretokenize-tsheg', dest='pretokenize_tsheg', action='store_false',
                       help='Disable tsheg pre-tokenization')
    
    parser.add_argument('--keep-tsheg', action='store_true', default=True,
                       help='Keep tsheg as separate token (default: True)')
    parser.add_argument('--no-keep-tsheg', dest='keep_tsheg', action='store_false',
                       help='Replace tsheg with space')
    
    parser.add_argument('--tibetan-only', action='store_true', default=False,
                       help='Filter to Tibetan characters only (default: False)')
    
    # Hugging Face Hub options
    parser.add_argument('--push-to-hub', action='store_true',
                       help='Upload tokenizer to Hugging Face Hub')
    parser.add_argument('--repo-id', type=str,
                       help='Repository ID on Hugging Face Hub (required if --push-to-hub)')
    parser.add_argument('--private', action='store_true',
                       help='Create private repository on Hub')
    parser.add_argument('--hf-token', type=str,
                       help='Hugging Face API token (or set HF_TOKEN env var)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.push_to_hub and not args.repo_id:
        parser.error("--repo-id is required when --push-to-hub is used")
    
    # Train tokenizer
    output_dir = train_tibetan_sentencepiece(
        corpus_path=args.corpus_path,
        output_dir=args.output_dir,
        vocab_size=args.vocab_size,
        normalize=args.normalize,
        pretokenize_tsheg=args.pretokenize_tsheg,
        keep_tsheg=args.keep_tsheg,
        tibetan_only=args.tibetan_only
    )
    
    # Upload to Hub if requested
    if args.push_to_hub:
        upload_to_hf_hub(
            output_dir=output_dir,
            repo_id=args.repo_id,
            private=args.private,
            hf_token=args.hf_token
        )


if __name__ == "__main__":
    main()
