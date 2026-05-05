"""
Production-Ready GPT-2 Training Script

Advanced Features:
- Automatic data ingestion from data/ directory (parquet, json, csv, txt)
- Mixed precision training (AMP) with BF16 support
- Gradient accumulation and checkpointing
- Advanced learning rate scheduling with warmup
- Production-grade checkpointing system with metadata
- Real-time metrics tracking and export
- Configuration management (YAML/JSON)
- Early stopping and validation
- Performance profiling
- Memory monitoring
- Resume capability with full state restoration
- Async checkpoint saving
- Experiment tracking and comparison

Usage:
    # Basic training
    python train.py --output_dir ./models/gpt2-finetuned --num_train_epochs 3 --batch_size 8
    
    # With config file
    python train.py --config config.yaml
    
    # Resume from checkpoint
    python train.py --resume_from_checkpoint ./models/gpt2-finetuned/checkpoints/checkpoint-xxxxx
    
    # Enable profiling
    python train.py --profile --output_dir ./models/gpt2-profiled

One-line training command after setup:
    python train.py --output_dir ./models/my-gpt2 --num_train_epochs 3
"""

import os
import sys
import argparse
import logging
import time
import traceback
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
import signal
import atexit

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
from transformers.trainer_callback import TrainerCallback, ProgressCallback, TrainerControl, TrainerState
import numpy as np

# Import custom modules
sys.path.insert(0, str(Path(__file__).parent))
from dataloader import create_dataloader, AutoIngestDataset
from config import ExperimentConfig, get_default_config, TrainingConfig
from checkpoint import CheckpointManager, resume_from_checkpoint
from metrics import MetricsTracker


class ProductionTrainerCallback(TrainerCallback):
    """
    Advanced callback for production training with checkpointing and metrics.
    """
    
    def __init__(
        self,
        checkpoint_manager: CheckpointManager,
        metrics_tracker: MetricsTracker,
        model_config: Dict[str, Any],
        training_config: Dict[str, Any],
        early_stopping: bool = False,
        early_stopping_patience: int = 3,
        early_stopping_threshold: float = 0.0,
    ):
        self.checkpoint_manager = checkpoint_manager
        self.metrics_tracker = metrics_tracker
        self.model_config = model_config
        self.training_config = training_config
        self.early_stopping = early_stopping
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.epoch_start_time = None
        self.last_global_step = 0
    
    def on_train_begin(self, args, state, control, **kwargs):
        """Initialize tracking at training start."""
        self.metrics_tracker.start_tracking()
        print("\n" + "="*60)
        print("TRAINING STARTED")
        print("="*60)
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        """Track epoch start time."""
        self.epoch_start_time = time.time()
    
    def on_step_end(self, args, state, control, **kwargs):
        """Log metrics and handle checkpointing at each step."""
        if state.global_step <= self.last_global_step:
            return control
        
        self.last_global_step = state.global_step
        
        # Get current metrics
        logs = kwargs.get('logs', {})
        loss = logs.get('loss', 0)
        learning_rate = logs.get('learning_rate', args.learning_rate)
        
        # Calculate throughput
        elapsed = time.time() - self.epoch_start_time if self.epoch_start_time else 0
        steps_in_epoch = state.global_step % args.num_train_epochs
        samples_per_second = (args.per_device_train_batch_size * args.gradient_accumulation_steps * steps_in_epoch / elapsed) if elapsed > 0 else 0
        
        # Get memory usage
        gpu_memory = 0.0
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1e6
        
        # Log to metrics tracker
        self.metrics_tracker.log(
            global_step=state.global_step,
            epoch=state.epoch,
            loss=loss,
            learning_rate=learning_rate,
            samples_per_second=samples_per_second,
            gpu_memory_mb=gpu_memory,
        )
        
        # Handle checkpointing
        if state.global_step % args.save_steps == 0 and args.save_steps > 0:
            print(f"\n\nSaving checkpoint at step {state.global_step}...")
            
            try:
                self.checkpoint_manager.save_checkpoint(
                    model=kwargs.get('model'),
                    optimizer=kwargs.get('optimizer'),
                    scheduler=kwargs.get('lr_scheduler'),
                    global_step=state.global_step,
                    epoch=state.epoch,
                    loss=loss,
                    learning_rate=learning_rate,
                    model_config=self.model_config,
                    training_config=self.training_config,
                )
            except Exception as e:
                print(f"Warning: Checkpoint saving failed: {e}")
        
        # Early stopping check
        if self.early_stopping and loss < self.best_loss - self.early_stopping_threshold:
            self.best_loss = loss
            self.patience_counter = 0
        elif self.early_stopping:
            self.patience_counter += 1
            if self.patience_counter >= self.early_stopping_patience:
                print(f"\n\nEarly stopping triggered after {self.patience_counter} steps without improvement")
                control.should_training_stop = True
        
        return control
    
    def on_epoch_end(self, args, state, control, **kwargs):
        """Handle end-of-epoch tasks."""
        elapsed = time.time() - self.epoch_start_time if self.epoch_start_time else 0
        print(f"\n\n✓ Epoch {int(state.epoch)} completed in {elapsed:.1f}s")
        
        # Save epoch checkpoint
        logs = kwargs.get('logs', {})
        loss = logs.get('loss', 0)
        learning_rate = logs.get('learning_rate', args.learning_rate)
        
        try:
            self.checkpoint_manager.save_checkpoint(
                model=kwargs.get('model'),
                optimizer=kwargs.get('optimizer'),
                scheduler=kwargs.get('lr_scheduler'),
                global_step=state.global_step,
                epoch=state.epoch,
                loss=loss,
                learning_rate=learning_rate,
                model_config=self.model_config,
                training_config=self.training_config,
                notes=f"Epoch {int(state.epoch)} end",
            )
        except Exception as e:
            print(f"Warning: Epoch checkpoint saving failed: {e}")
        
        return control
    
    def on_train_end(self, args, state, control, **kwargs):
        """Finalize training and export metrics."""
        print("\n\n" + "="*60)
        print("TRAINING COMPLETED")
        print("="*60)
        
        # Export metrics
        self.metrics_tracker.export_csv()
        self.metrics_tracker.export_json()
        
        # Generate plot if matplotlib available
        try:
            self.metrics_tracker.plot_losses(
                output_path=os.path.join(args.output_dir, 'training_loss.png')
            )
        except Exception as e:
            print(f"Note: Could not generate loss plot: {e}")
        
        # Print summary
        summary = self.metrics_tracker.get_metrics_summary()
        print("\nTraining Summary:")
        print(f"  Total steps: {summary.get('total_steps', 0)}")
        print(f"  Final loss: {summary.get('final_loss', 0):.4f}")
        print(f"  Best loss: {summary.get('min_loss', 0):.4f}")
        print(f"  Avg step time: {summary.get('avg_step_time', 0):.3f}s")
        print(f"  Steps/second: {summary.get('steps_per_second', 0):.2f}")


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
    parser = argparse.ArgumentParser(
        description='Production-Ready GPT-2 Training Script',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML/JSON configuration file (overrides other args)')
    
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
    
    # Performance arguments - optimized for CPU-only 2-core systems
    parser.add_argument('--fp16', action='store_true', default=False,
                        help='Use mixed precision training (disabled for CPU)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of data loading workers (0 for single-process, optimal for 2-core CPU)')
    parser.add_argument('--streaming', action='store_true',
                        help='Use streaming mode for large datasets')
    parser.add_argument('--bf16', action='store_true', default=False,
                        help='Use bfloat16 precision (if supported by CPU)')
    parser.add_argument('--optim', type=str, default='adamw_torch',
                        choices=['adamw_torch', 'adamw_hf', 'sgd', 'adafactor'],
                        help='Optimizer to use')
    parser.add_argument('--gradient_checkpointing', action='store_true',
                        help='Enable gradient checkpointing to save memory')
    parser.add_argument('--compile', action='store_true',
                        help='Use torch.compile for faster training (PyTorch 2.0+)')
    
    # Logging and checkpointing
    parser.add_argument('--logging_steps', type=int, default=100,
                        help='Log every X steps')
    parser.add_argument('--save_steps', type=int, default=500,
                        help='Save checkpoint every X steps')
    parser.add_argument('--save_total_limit', type=int, default=3,
                        help='Maximum number of checkpoints to keep')
    
    # Advanced checkpointing
    parser.add_argument('--async_checkpoint', action='store_true',
                        help='Enable async checkpoint saving (non-blocking)')
    parser.add_argument('--checkpoint_compression', action='store_true',
                        help='Enable checkpoint compression')
    parser.add_argument('--save_optimizer_state', action='store_true', default=True,
                        help='Save optimizer state in checkpoints')
    parser.add_argument('--save_scheduler_state', action='store_true', default=True,
                        help='Save scheduler state in checkpoints')
    
    # Early stopping
    parser.add_argument('--early_stopping', action='store_true',
                        help='Enable early stopping')
    parser.add_argument('--early_stopping_patience', type=int, default=3,
                        help='Early stopping patience')
    parser.add_argument('--early_stopping_threshold', type=float, default=0.0,
                        help='Minimum improvement threshold for early stopping')
    
    # Profiling
    parser.add_argument('--profile', action='store_true',
                        help='Enable PyTorch profiling')
    parser.add_argument('--profile_steps', type=int, default=10,
                        help='Number of steps to profile')
    
    # Other
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                        help='Resume training from checkpoint')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Experiment name for tracking')
    
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
    """Main training function with production features."""
    args = parse_args()
    
    # Load config file if provided
    config = None
    if args.config:
        logger.info(f"Loading configuration from {args.config}")
        config = ExperimentConfig.load(args.config)
        # Override args with config values
        args.data_dir = config.data.data_dir
        args.model_name = config.model.model_name
        args.max_length = config.data.max_length
        args.num_train_epochs = config.training.num_train_epochs
        args.batch_size = config.training.batch_size
        args.learning_rate = config.training.learning_rate
        args.output_dir = config.training.output_dir
        args.save_steps = config.training.save_steps
        args.save_total_limit = config.checkpoint.keep_last_n
        args.early_stopping = config.training.early_stopping
        args.early_stopping_patience = config.training.early_stopping_patience
    
    # Create output directory
    output_path = Path(args.output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    
    experiment_name = args.experiment_name or f"gpt2_finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Setup logging
    logger = setup_logging(args.output_dir)
    logger.info("=" * 60)
    logger.info("PRODUCTION GPT-2 TRAINING")
    logger.info("=" * 60)
    logger.info(f"Experiment: {experiment_name}")
    logger.info(f"Configuration: {vars(args)}")
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager(
        output_dir=args.output_dir,
        save_total_limit=args.save_total_limit,
        save_optimizer=args.save_optimizer_state,
        save_scheduler=args.save_scheduler_state,
        async_save=args.async_checkpoint,
        compression=args.checkpoint_compression,
    )
    
    # Initialize metrics tracker
    metrics_tracker = MetricsTracker(
        output_dir=args.output_dir,
        experiment_name=experiment_name,
    )
    
    # Save initial config for reproducibility
    try:
        initial_config = get_default_config()
        initial_config.training.output_dir = args.output_dir
        initial_config.training.num_train_epochs = args.num_train_epochs
        initial_config.training.batch_size = args.batch_size
        initial_config.model.model_name = args.model_name
        initial_config.save(os.path.join(args.output_dir, "training_config.yaml"))
    except Exception as e:
        logger.warning(f"Could not save config: {e}")
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_name, args.max_length)
    
    # Model config for checkpoint metadata
    model_config = {
        "model_name": args.model_name,
        "max_length": args.max_length,
        "num_parameters": model.num_parameters(),
        "gradient_checkpointing": args.gradient_checkpointing,
    }
    
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
    
    # Setup training arguments - optimized for CPU 2-core systems
    device = 'cpu'
    
    # Detect CPU capabilities
    use_bf16 = args.bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = args.fp16 and not use_bf16  # FP16 typically not beneficial on CPU
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        warmup_steps=args.warmup_steps,
        fp16=use_fp16,
        bf16=use_bf16,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        logging_first_step=True,
        logging_dir=os.path.join(args.output_dir, 'logs'),
        report_to=[],  # Disable wandb/tensorboard by default
        seed=args.seed,
        dataloader_num_workers=args.num_workers,  # 0 for single-process (optimal for 2-core)
        dataloader_pin_memory=False,  # No benefit on CPU
        gradient_checkpointing=args.gradient_checkpointing,
        optim=args.optim,
        lr_scheduler_type='linear',
        tf32=False,  # Not relevant for CPU
        dataloader_prefetch_factor=2 if args.num_workers > 0 else None,
        prediction_loss_only=False,
        remove_unused_columns=False,  # Keep all columns for proper handling
        group_by_length=False,  # Don't group by length (variable length handled by collator)
        ddp_find_unused_parameters=None,
        skip_memory_metrics=True,  # Skip memory metrics on CPU
        eval_on_start=False,
        metric_for_best_model=None,
        greater_is_better=None,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # GPT-2 is causal LM, not masked LM
    )
    
    # Training config for checkpoint metadata
    training_config = {
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "warmup_steps": args.warmup_steps,
        "weight_decay": args.weight_decay,
        "optimizer": args.optim,
        "fp16": use_fp16,
        "bf16": use_bf16,
    }
    
    # Initialize Trainer with CPU optimizations
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    
    # Apply torch.compile for faster training if requested (PyTorch 2.0+)
    if args.compile and hasattr(torch, 'compile'):
        logger.info("Applying torch.compile for faster training...")
        trainer.model = torch.compile(trainer.model, mode='reduce-overhead')
        logger.info("torch.compile applied successfully")
    
    # Remove default progress callback and add production callback
    trainer.remove_callback(ProgressCallback)
    
    # Add production callback with checkpointing and metrics
    production_callback = ProductionTrainerCallback(
        checkpoint_manager=checkpoint_manager,
        metrics_tracker=metrics_tracker,
        model_config=model_config,
        training_config=training_config,
        early_stopping=args.early_stopping,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_threshold=args.early_stopping_threshold,
    )
    trainer.add_callback(production_callback)
    
    # Setup signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        logger.info("\nReceived interrupt signal, saving checkpoint...")
        checkpoint_manager.wait_for_async_saves()
        trainer.save_model(os.path.join(args.output_dir, 'interrupted'))
        tokenizer.save_pretrained(os.path.join(args.output_dir, 'interrupted'))
        metrics_tracker.export_csv()
        metrics_tracker.export_json()
        logger.info("Checkpoint saved. Exiting gracefully.")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Register cleanup on exit
    atexit.register(lambda: checkpoint_manager.wait_for_async_saves())
    
    # Start training
    logger.info("\n" + "=" * 60)
    logger.info("Starting Training")
    logger.info("=" * 60)
    
    checkpoint = args.resume_from_checkpoint if args.resume_from_checkpoint else None
    
    # Setup profiler if requested
    if args.profile:
        logger.info("PyTorch profiling enabled")
        profile_schedule = torch.profiler.schedule(
            wait=5,
            warmup=5,
            active=args.profile_steps,
            repeat=1
        )
        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA if torch.cuda.is_available() else None,
            ],
            schedule=profile_schedule,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                os.path.join(args.output_dir, 'profiling')
            ),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
        profiler.start()
    
    try:
        trainer.train(resume_from_checkpoint=checkpoint)
        
        # Stop profiler
        if args.profile:
            profiler.stop()
            logger.info(f"Profiling data saved to {os.path.join(args.output_dir, 'profiling')}")
        
        # Save final model
        logger.info("\nSaving final model...")
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        
        # Export inference model
        latest_checkpoint = checkpoint_manager.get_latest_checkpoint()
        if latest_checkpoint:
            inference_path = os.path.join(args.output_dir, 'inference_model.pt')
            checkpoint_manager.export_checkpoint_for_inference(latest_checkpoint, inference_path)
        
        logger.info(f"\n✓ Training completed successfully!")
        logger.info(f"Model saved to: {args.output_dir}")
        
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
        logger.info("Saving checkpoint...")
        checkpoint_manager.wait_for_async_saves()
        trainer.save_model(os.path.join(args.output_dir, 'interrupted'))
        tokenizer.save_pretrained(os.path.join(args.output_dir, 'interrupted'))
        metrics_tracker.export_csv()
        metrics_tracker.export_json()
        logger.info("Checkpoint saved")
    
    except Exception as e:
        logger.error(f"\nTraining failed with error: {e}")
        logger.error(traceback.format_exc())
        
        # Try to save checkpoint on error
        try:
            checkpoint_manager.wait_for_async_saves()
            trainer.save_model(os.path.join(args.output_dir, 'error_checkpoint'))
            tokenizer.save_pretrained(os.path.join(args.output_dir, 'error_checkpoint'))
            logger.info("Emergency checkpoint saved")
        except Exception as save_error:
            logger.error(f"Failed to save emergency checkpoint: {save_error}")
        
        raise
    
    finally:
        # Ensure async saves complete
        checkpoint_manager.wait_for_async_saves()
    
    logger.info("=" * 60)
    logger.info("Training Finished")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
