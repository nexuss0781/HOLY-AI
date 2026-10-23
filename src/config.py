"""
Production Configuration Management

Supports YAML and JSON configuration files for reproducible experiments.
"""

import os
import json
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime


@dataclass
class DataConfig:
    """Data loading configuration."""
    data_dir: str = "data"
    text_column: Optional[str] = None
    max_length: int = 512
    streaming: bool = False
    num_workers: int = 0
    train_test_split: float = 0.9
    seed: int = 42


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    model_name: str = "gpt2"
    tokenizer_name: Optional[str] = None
    vocab_size: Optional[int] = None
    n_positions: int = 1024
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12
    gradient_checkpointing: bool = False
    compile_model: bool = False


@dataclass
class TrainingConfig:
    """Training hyperparameters and settings."""
    output_dir: str = "./models/gpt2-finetuned"
    num_train_epochs: int = 3
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "linear"
    optimizer: str = "adamw_torch"
    
    # Precision
    fp16: bool = False
    bf16: bool = False
    
    # Checkpointing
    save_steps: int = 500
    save_total_limit: int = 3
    save_strategy: str = "steps"  # steps, epoch, no
    
    # Logging
    logging_steps: int = 100
    logging_first_step: bool = True
    report_to: list = field(default_factory=lambda: [])
    
    # Performance
    dataloader_num_workers: int = 0
    dataloader_pin_memory: bool = False
    dataloader_prefetch_factor: Optional[int] = None
    
    # Validation
    eval_strategy: str = "no"  # no, steps, epoch
    eval_steps: Optional[int] = None
    per_device_eval_batch_size: Optional[int] = None
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    
    # Early stopping
    early_stopping: bool = False
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.0
    
    # Resume
    resume_from_checkpoint: Optional[str] = None


@dataclass
class CheckpointConfig:
    """Advanced checkpointing configuration."""
    enable: bool = True
    save_optimizer_state: bool = True
    save_scheduler_state: bool = True
    save_random_state: bool = True
    async_save: bool = False
    compression: bool = False
    remote_storage: Optional[str] = None  # S3, GCS path
    keep_last_n: int = 3
    save_every_n_epochs: int = 1


@dataclass
class DistributedConfig:
    """Distributed training configuration."""
    enable: bool = False
    backend: str = "nccl"  # nccl, gloo
    world_size: int = 1
    rank: int = 0
    local_rank: int = -1
    master_addr: str = "localhost"
    master_port: str = "29500"
    gradient_bucket_size_mb: int = 25
    static_graph: bool = False


@dataclass
class ProfilingConfig:
    """Performance profiling configuration."""
    enable: bool = False
    profile_steps: int = 10
    wait_steps: int = 5
    warmup_steps: int = 5
    active_steps: int = 5
    trace_dir: str = "./profiling"
    export_chrome_trace: bool = True
    memory_profile: bool = False


@dataclass
class ExperimentConfig:
    """Root configuration for the entire experiment."""
    experiment_name: str = "gpt2_finetune"
    run_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    profiling: ProfilingConfig = field(default_factory=ProfilingConfig)
    
    # Metadata
    notes: str = ""
    tags: list = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to nested dictionary."""
        return {
            "experiment_name": self.experiment_name,
            "run_id": self.run_id,
            "notes": self.notes,
            "tags": self.tags,
            "data": asdict(self.data),
            "model": asdict(self.model),
            "training": asdict(self.training),
            "checkpoint": asdict(self.checkpoint),
            "distributed": asdict(self.distributed),
            "profiling": asdict(self.profiling),
        }
    
    def save(self, path: str):
        """Save configuration to YAML or JSON file."""
        path = Path(path)
        config_dict = self.to_dict()
        
        if path.suffix in ['.yml', '.yaml']:
            with open(path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        elif path.suffix == '.json':
            with open(path, 'w') as f:
                json.dump(config_dict, f, indent=2)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        
        print(f"Configuration saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'ExperimentConfig':
        """Load configuration from YAML or JSON file."""
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        with open(path, 'r') as f:
            if path.suffix in ['.yml', '.yaml']:
                config_dict = yaml.safe_load(f)
            elif path.suffix == '.json':
                config_dict = json.load(f)
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")
        
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ExperimentConfig':
        """Create ExperimentConfig from dictionary."""
        config = cls()
        
        config.experiment_name = config_dict.get("experiment_name", config.experiment_name)
        config.run_id = config_dict.get("run_id", config.run_id)
        config.notes = config_dict.get("notes", "")
        config.tags = config_dict.get("tags", [])
        
        if "data" in config_dict:
            config.data = DataConfig(**config_dict["data"])
        if "model" in config_dict:
            config.model = ModelConfig(**config_dict["model"])
        if "training" in config_dict:
            config.training = TrainingConfig(**config_dict["training"])
        if "checkpoint" in config_dict:
            config.checkpoint = CheckpointConfig(**config_dict["checkpoint"])
        if "distributed" in config_dict:
            config.distributed = DistributedConfig(**config_dict["distributed"])
        if "profiling" in config_dict:
            config.profiling = ProfilingConfig(**config_dict["profiling"])
        
        return config


def get_default_config() -> ExperimentConfig:
    """Get default configuration with sensible defaults for CPU training."""
    config = ExperimentConfig()
    
    # Optimize for CPU-only systems
    config.training.fp16 = False
    config.training.bf16 = False
    config.training.dataloader_num_workers = 0
    config.training.dataloader_pin_memory = False
    config.model.gradient_checkpointing = False
    
    return config


def create_config_template(output_path: str = "config.yaml"):
    """Create a template configuration file."""
    config = get_default_config()
    config.save(output_path)
    print(f"Template configuration created at {output_path}")
    return config


if __name__ == "__main__":
    # Create template config
    create_config_template("config_template.yaml")
    
    # Example: Load and modify config
    # config = ExperimentConfig.load("config.yaml")
    # config.training.num_train_epochs = 5
    # config.save("config_updated.yaml")
