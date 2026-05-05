"""
High-quality GPT-2 Training Script

Features:
- Automatic data ingestion from data/ directory (parquet, json, csv, txt)
- Mixed precision training (AMP)
- Gradient accumulation
- Learning rate scheduling with warmup
- Checkpoint saving
- Logging with tqdm and optional wandb
- Configurable via command-line arguments

Usage:
    python train.py --output_dir ./models/gpt2-finetuned --num_train_epochs 3 --batch_size 8

One-line training command after setup:
    python train.py --output_dir ./models/my-gpt2 --num_train_epochs 3
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    GPT2Config,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup,
)
from transformers.trainer_callback import TrainerCallback, ProgressCallback
import numpy as np

# Import custom dataloader
sys.path.insert(0, str(Path(__file__).parent))
from dataloader import create_dataloader, AutoIngestDataset


class CustomProgressCallback(ProgressCallback):
    """Custom progress callback with better formatting."""
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            # Format the log output nicely
            step = logs.get('step', state.global_step)
            loss = logs.get('loss', 0)
            learning_rate = logs.get('learning_rate', 0)
            
            if 'loss' in logs:
                print(f"\n[Step {step}] Loss: {loss:.4f} | LR: {learning_rate:.2e}")


def setup_logging(output_dir: str):
    """Setup logging configuration."""
    log_file = os.path.join(output_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train GPT-2 on custom data')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory containing training data files')
    parser.add_argument('--text_column', type=str, default=None,
                        help='Specific column name to use for text')
    
    # Model arguments
    parser.add_argument('--model_name', type=str, default='gpt2',
                        choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'],
                        help='Base GPT-2 model to fine-tune')
    parser.add_argument('--max_length', type=int, default=512,
                        help='Maximum sequence length')
    
    # Training arguments
    parser.add_argument('--output_dir', type=str, default='./models/gpt2-finetuned',
                        help='Output directory for checkpoints')
    parser.add_argument('--num_train_epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Training batch size per device')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                        help='Number of gradient accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                        help='Learning rate')
    parser.add_argument('--warmup_steps', type=int, default=1000,
                        help='Number of warmup steps')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Maximum gradient norm for clipping')
    
    # Performance arguments
    parser.add_argument('--fp16', action='store_true',
                        help='Use mixed precision training')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--streaming', action='store_true',
                        help='Use streaming mode for large datasets')
    
    # Logging and checkpointing
    parser.add_argument('--logging_steps', type=int, default=100,
                        help='Log every X steps')
    parser.add_argument('--save_steps', type=int, default=500,
                        help='Save checkpoint every X steps')
    parser.add_argument('--save_total_limit', type=int, default=3,
                        help='Maximum number of checkpoints to keep')
    
    # Other
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                        help='Resume training from checkpoint')
    
    return parser.parse_args()


def load_model_and_tokenizer(model_name: str, max_length: int):
    """Load GPT-2 model and tokenizer."""
    print(f"\nLoading model: {model_name}")
    
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    
    config = GPT2Config.from_pretrained(model_name)
    config.pad_token_id = tokenizer.eos_token_id
    
    model = GPT2LMHeadModel.from_pretrained(model_name, config=config)
    model.resize_token_embeddings(len(tokenizer))
    
    print(f"Model loaded with {model.num_parameters():,} parameters")
    
    return model, tokenizer


def main():
    """Main training function."""
    args = parse_args()
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(args.output_dir)
    logger.info("=" * 60)
    logger.info("GPT-2 Training Started")
    logger.info("=" * 60)
    logger.info(f"Configuration: {vars(args)}")
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_name, args.max_length)
    
    # Create dataloader
    logger.info(f"\nLoading data from: {args.data_dir}")
    
    dataloader = create_dataloader(
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        streaming=args.streaming,
        num_workers=args.num_workers,
        text_column=args.text_column,
        shuffle=True,
        drop_last=True
    )
    
    dataset = dataloader.dataset
    logger.info(f"Dataset size: {len(dataset):,} samples")
    
    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        warmup_steps=args.warmup_steps,
        fp16=args.fp16,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        logging_first_step=True,
        logging_dir=os.path.join(args.output_dir, 'logs'),
        report_to=[],  # Disable wandb/tensorboard by default
        seed=args.seed,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,
        gradient_checkpointing=False,
        optim='adamw_torch',
        lr_scheduler_type='linear',
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # GPT-2 is causal LM, not masked LM
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    
    # Remove default progress callback and add custom one
    trainer.remove_callback(ProgressCallback)
    trainer.add_callback(CustomProgressCallback())
    
    # Start training
    logger.info("\n" + "=" * 60)
    logger.info("Starting Training")
    logger.info("=" * 60)
    
    checkpoint = args.resume_from_checkpoint if args.resume_from_checkpoint else None
    
    try:
        trainer.train(resume_from_checkpoint=checkpoint)
        
        # Save final model
        logger.info("\nSaving final model...")
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        
        logger.info(f"\n✓ Training completed successfully!")
        logger.info(f"Model saved to: {args.output_dir}")
        
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
        # Save checkpoint on interrupt
        logger.info("Saving checkpoint...")
        trainer.save_model(os.path.join(args.output_dir, 'interrupted'))
        tokenizer.save_pretrained(os.path.join(args.output_dir, 'interrupted'))
        logger.info("Checkpoint saved")
    
    logger.info("=" * 60)
    logger.info("Training Finished")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
