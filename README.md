# GPT-2 Training Setup

Everything is configured and ready to train GPT-2 on your custom data.

## Files Created

- `src/dataloader.py` - High-performance DataLoader with enhanced Parquet support
- `src/train.py` - Complete training script with best practices

## Data Downloaded

- `data/synaxarium_dataset.parquet` - Synaxarium dataset
- `data/conon-biblical-am-en.parquet` - Conon Biblical AM-EN dataset

**Total samples loaded: 32,286**

## Features

### DataLoader (`src/dataloader.py`)
- ✅ Auto-ingests any file in `data/` directory
- ✅ Supports: `.parquet`, `.json`, `.jsonl`, `.csv`, `.txt`
- ✅ Intelligent text column detection
- ✅ Streaming mode for large datasets (memory efficient)
- ✅ Automatic tokenization with HuggingFace tokenizers
- ✅ Variable length sequence padding

### Training Script (`src/trainer.py`)
- ✅ Mixed precision training (AMP)
- ✅ Gradient accumulation
- ✅ Learning rate scheduling with warmup
- ✅ Checkpoint saving
- ✅ Resume from checkpoint support
- ✅ Configurable via CLI arguments

---

## ONE-LINE TRAINING COMMAND

```bash
cd /workspace && python src/train.py --output_dir ./models/gpt2-finetuned --num_train_epochs 3 --batch_size 8 --gradient_accumulation_steps 4 --fp16
```

---

## Customization Options

```bash
python src/train.py \
  --output_dir ./models/my-gpt2 \
  --model_name gpt2-medium \
  --num_train_epochs 5 \
  --batch_size 16 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-5 \
  --warmup_steps 500 \
  --max_length 1024 \
  --fp16 \
  --streaming
```

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_name` | gpt2 | Base model: gpt2, gpt2-medium, gpt2-large, gpt2-xl |
| `--output_dir` | ./models/gpt2-finetuned | Where to save checkpoints |
| `--num_train_epochs` | 3 | Number of training epochs |
| `--batch_size` | 8 | Batch size per device |
| `--gradient_accumulation_steps` | 4 | Gradient accumulation steps |
| `--learning_rate` | 5e-5 | Learning rate |
| `--max_length` | 512 | Maximum sequence length |
| `--fp16` | False | Use mixed precision training |
| `--streaming` | False | Stream data for large datasets |

---

## Verification Results

✅ **Syntax Check**: All Python files pass syntax validation  
✅ **DataLoader Test**: Successfully loads 32,286 samples from parquet files  
✅ **Batch Generation**: Correct tensor shapes produced  

---

## Notes

- The current environment has limited memory (~1GB RAM, ~500MB disk)
- Training requires more resources - run on your machine with adequate GPU/memory
- For large datasets, use `--streaming` flag to avoid loading everything into memory
- Model will be saved to the specified `--output_dir` after training completes
