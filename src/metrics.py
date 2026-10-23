"""
Production Training Metrics and Logging System

Features:
- Real-time metrics tracking
- CSV and JSON export
- Visualization-ready data
- Performance profiling
- Memory tracking
- Experiment comparison
"""

import os
import json
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass, field, asdict
import threading


@dataclass
class TrainingMetric:
    """Single training metric entry."""
    timestamp: str
    global_step: int
    epoch: float
    loss: float
    learning_rate: float
    samples_per_second: float = 0.0
    gpu_memory_mb: float = 0.0
    cpu_memory_mb: float = 0.0
    custom_metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MetricsTracker:
    """
    Production-grade metrics tracking system.
    
    Features:
    - Real-time metric collection
    - Automatic averaging and smoothing
    - Multiple export formats (CSV, JSON)
    - Performance profiling
    - Memory monitoring
    - Thread-safe operations
    """
    
    def __init__(self, output_dir: str, experiment_name: str = "training"):
        self.output_dir = Path(output_dir).resolve()
        self.metrics_dir = self.output_dir / "metrics"
        self.experiment_name = experiment_name
        
        # Create directories
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage
        self._metrics: List[TrainingMetric] = []
        self._epoch_metrics: Dict[int, List[TrainingMetric]] = defaultdict(list)
        self._lock = threading.Lock()
        
        # Running statistics
        self._running_loss = 0.0
        self._running_loss_count = 0
        self._smoothed_loss = None
        self._smoothing_alpha = 0.95  # EMA smoothing factor
        
        # Timing
        self._start_time: Optional[datetime] = None
        self._last_step_time: Optional[datetime] = None
        self._step_times: List[float] = []
        
        print(f"MetricsTracker initialized at {self.metrics_dir}")
    
    def start_tracking(self):
        """Start timing for the training run."""
        self._start_time = datetime.now()
        self._last_step_time = self._start_time
        print(f"\n✓ Started tracking at {self._start_time.isoformat()}")
    
    def log(
        self,
        global_step: int,
        epoch: float,
        loss: float,
        learning_rate: float,
        samples_per_second: Optional[float] = None,
        gpu_memory_mb: Optional[float] = None,
        cpu_memory_mb: Optional[float] = None,
        **custom_metrics,
    ):
        """
        Log a training metric.
        
        Args:
            global_step: Current training step
            epoch: Current epoch
            loss: Current loss value
            learning_rate: Current learning rate
            samples_per_second: Throughput metric
            gpu_memory_mb: GPU memory usage (optional)
            cpu_memory_mb: CPU memory usage (optional)
            **custom_metrics: Additional custom metrics
        """
        current_time = datetime.now()
        
        # Calculate step time
        step_time = 0.0
        if self._last_step_time is not None:
            step_time = (current_time - self._last_step_time).total_seconds()
            self._step_times.append(step_time)
        self._last_step_time = current_time
        
        # Update running loss
        self._running_loss = (
            self._smoothing_alpha * self._running_loss + 
            (1 - self._smoothing_alpha) * loss
        )
        self._running_loss_count += 1
        
        if self._smoothed_loss is None:
            self._smoothed_loss = loss
        else:
            self._smoothed_loss = (
                self._smoothing_alpha * self._smoothed_loss + 
                (1 - self._smoothing_alpha) * loss
            )
        
        # Create metric entry
        metric = TrainingMetric(
            timestamp=current_time.isoformat(),
            global_step=global_step,
            epoch=epoch,
            loss=loss,
            learning_rate=learning_rate,
            samples_per_second=samples_per_second or 0.0,
            gpu_memory_mb=gpu_memory_mb or 0.0,
            cpu_memory_mb=cpu_memory_mb or 0.0,
            custom_metrics=custom_metrics,
        )
        
        with self._lock:
            self._metrics.append(metric)
            self._epoch_metrics[int(epoch)].append(metric)
        
        # Print progress
        avg_step_time = sum(self._step_times[-10:]) / min(len(self._step_times), 10) if self._step_times else 0
        elapsed = (current_time - self._start_time).total_seconds() if self._start_time else 0
        
        print(
            f"\r[Step {global_step:6d}] Epoch: {epoch:.2f} | "
            f"Loss: {loss:.4f} (smoothed: {self._smoothed_loss:.4f}) | "
            f"LR: {learning_rate:.2e} | "
            f"Step: {avg_step_time:.3f}s | "
            f"Elapsed: {elapsed:.1f}s",
            end="",
            flush=True
        )
    
    def get_smoothed_loss(self) -> Optional[float]:
        """Get the exponentially smoothed loss."""
        return self._smoothed_loss
    
    def get_average_loss(self, window: int = 100) -> float:
        """Get average loss over the last N steps."""
        with self._lock:
            if len(self._metrics) == 0:
                return 0.0
            recent = self._metrics[-window:]
            return sum(m.loss for m in recent) / len(recent)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary."""
        with self._lock:
            if len(self._metrics) == 0:
                return {}
            
            losses = [m.loss for m in self._metrics]
            step_times = self._step_times[-100:] if self._step_times else []
            
            summary = {
                "experiment_name": self.experiment_name,
                "total_steps": len(self._metrics),
                "total_epochs": max(m.epoch for m in self._metrics),
                "start_time": self._start_time.isoformat() if self._start_time else None,
                "end_time": datetime.now().isoformat(),
                "duration_seconds": (datetime.now() - self._start_time).total_seconds() if self._start_time else 0,
                
                # Loss statistics
                "initial_loss": losses[0],
                "final_loss": losses[-1],
                "min_loss": min(losses),
                "max_loss": max(losses),
                "avg_loss": sum(losses) / len(losses),
                "smoothed_loss": self._smoothed_loss,
                
                # Performance
                "avg_step_time": sum(step_times) / len(step_times) if step_times else 0,
                "min_step_time": min(step_times) if step_times else 0,
                "max_step_time": max(step_times) if step_times else 0,
                "steps_per_second": 1.0 / (sum(step_times) / len(step_times)) if step_times else 0,
                
                # Learning rate
                "initial_lr": self._metrics[0].learning_rate,
                "final_lr": self._metrics[-1].learning_rate,
            }
            
            # Add memory stats if available
            gpu_memories = [m.gpu_memory_mb for m in self._metrics if m.gpu_memory_mb > 0]
            if gpu_memories:
                summary["avg_gpu_memory_mb"] = sum(gpu_memories) / len(gpu_memories)
                summary["max_gpu_memory_mb"] = max(gpu_memories)
            
            cpu_memories = [m.cpu_memory_mb for m in self._metrics if m.cpu_memory_mb > 0]
            if cpu_memories:
                summary["avg_cpu_memory_mb"] = sum(cpu_memories) / len(cpu_memories)
                summary["max_cpu_memory_mb"] = max(cpu_memories)
            
            return summary
    
    def export_csv(self, filename: Optional[str] = None) -> Path:
        """Export metrics to CSV file."""
        if filename is None:
            filename = f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        filepath = self.metrics_dir / filename
        
        with self._lock:
            if not self._metrics:
                print("No metrics to export")
                return filepath
            
            # Get all keys from custom metrics
            custom_keys = set()
            for m in self._metrics:
                custom_keys.update(m.custom_metrics.keys())
            
            fieldnames = [
                'timestamp', 'global_step', 'epoch', 'loss', 'learning_rate',
                'samples_per_second', 'gpu_memory_mb', 'cpu_memory_mb'
            ] + sorted(custom_keys)
            
            with open(filepath, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for metric in self._metrics:
                    row = {
                        'timestamp': metric.timestamp,
                        'global_step': metric.global_step,
                        'epoch': metric.epoch,
                        'loss': metric.loss,
                        'learning_rate': metric.learning_rate,
                        'samples_per_second': metric.samples_per_second,
                        'gpu_memory_mb': metric.gpu_memory_mb,
                        'cpu_memory_mb': metric.cpu_memory_mb,
                    }
                    row.update(metric.custom_metrics)
                    writer.writerow(row)
        
        print(f"\n✓ Metrics exported to {filepath}")
        return filepath
    
    def export_json(self, filename: Optional[str] = None) -> Path:
        """Export metrics to JSON file."""
        if filename is None:
            filename = f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = self.metrics_dir / filename
        
        with self._lock:
            data = {
                "experiment_name": self.experiment_name,
                "summary": self.get_metrics_summary(),
                "metrics": [m.to_dict() for m in self._metrics],
                "epoch_summaries": {}
            }
            
            # Add per-epoch summaries
            for epoch, metrics in self._epoch_metrics.items():
                losses = [m.loss for m in metrics]
                data["epoch_summaries"][str(epoch)] = {
                    "steps": len(metrics),
                    "avg_loss": sum(losses) / len(losses) if losses else 0,
                    "min_loss": min(losses) if losses else 0,
                    "max_loss": max(losses) if losses else 0,
                }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        
        print(f"\n✓ Metrics exported to {filepath}")
        return filepath
    
    def plot_losses(self, output_path: Optional[str] = None, show: bool = False):
        """
        Generate loss plot (requires matplotlib).
        
        Args:
            output_path: Path to save plot (optional)
            show: Whether to display plot
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib not installed. Install with: pip install matplotlib")
            return
        
        with self._lock:
            if not self._metrics:
                print("No metrics to plot")
                return
            
            steps = [m.global_step for m in self._metrics]
            losses = [m.loss for m in self._metrics]
            
            # Calculate smoothed loss
            smoothed = []
            s = losses[0]
            alpha = 0.95
            for loss in losses:
                s = alpha * s + (1 - alpha) * loss
                smoothed.append(s)
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(steps, losses, 'b-', alpha=0.3, label='Raw Loss')
            ax.plot(steps, smoothed, 'r-', linewidth=2, label='Smoothed Loss')
            ax.set_xlabel('Step')
            ax.set_ylabel('Loss')
            ax.set_title(f'Training Loss - {self.experiment_name}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(output_path, dpi=150, bbox_inches='tight')
                print(f"Plot saved to {output_path}")
            
            if show:
                plt.show()
            
            plt.close()
    
    def get_epoch_progress(self, epoch: int) -> Dict[str, Any]:
        """Get progress summary for a specific epoch."""
        with self._lock:
            if epoch not in self._epoch_metrics:
                return {}
            
            metrics = self._epoch_metrics[epoch]
            losses = [m.loss for m in metrics]
            
            return {
                "epoch": epoch,
                "steps": len(metrics),
                "avg_loss": sum(losses) / len(losses),
                "min_loss": min(losses),
                "max_loss": max(losses),
                "start_time": metrics[0].timestamp,
                "end_time": metrics[-1].timestamp,
            }


class ExperimentComparator:
    """Compare multiple training experiments."""
    
    def __init__(self):
        self.experiments: Dict[str, MetricsTracker] = {}
    
    def add_experiment(self, name: str, metrics_tracker: MetricsTracker):
        """Add an experiment for comparison."""
        self.experiments[name] = metrics_tracker
    
    def compare_final_losses(self) -> List[Tuple[str, float]]:
        """Compare final losses across experiments."""
        results = []
        for name, tracker in self.experiments.items():
            summary = tracker.get_metrics_summary()
            if summary:
                results.append((name, summary.get("final_loss", float('inf'))))
        
        return sorted(results, key=lambda x: x[1])
    
    def compare_convergence_speed(self, threshold: float) -> List[Tuple[str, int]]:
        """Compare how fast experiments reach a loss threshold."""
        results = []
        for name, tracker in self.experiments.items():
            with tracker._lock:
                for i, metric in enumerate(tracker._metrics):
                    if metric.loss < threshold:
                        results.append((name, metric.global_step))
                        break
        
        return sorted(results, key=lambda x: x[1])
    
    def plot_comparison(self, output_path: str = "experiment_comparison.png"):
        """Plot loss curves for all experiments."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib not installed")
            return
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']
        
        for idx, (name, tracker) in enumerate(self.experiments.items()):
            with tracker._lock:
                if not tracker._metrics:
                    continue
                
                steps = [m.global_step for m in tracker._metrics]
                losses = [m.loss for m in tracker._metrics]
                
                color = colors[idx % len(colors)]
                ax.plot(steps, losses, color=color, alpha=0.5, label=f'{name} (raw)')
                
                # Smoothed
                smoothed = []
                s = losses[0]
                alpha = 0.95
                for loss in losses:
                    s = alpha * s + (1 - alpha) * loss
                    smoothed.append(s)
                
                ax.plot(steps, smoothed, color=color, linewidth=2, linestyle='--', label=f'{name} (smoothed)')
        
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.set_title('Experiment Comparison')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Comparison plot saved to {output_path}")
        
        plt.close()


if __name__ == "__main__":
    # Demo
    print("Metrics Tracker Demo")
    print("=" * 50)
    
    tracker = MetricsTracker("./test_metrics", "demo_experiment")
    tracker.start_tracking()
    
    # Simulate training
    import math
    for step in range(100):
        epoch = step / 100
        loss = 2.0 * math.exp(-step / 50) + 0.1 * math.sin(step / 10)
        lr = 5e-5 * (1 - step / 100)
        
        tracker.log(
            global_step=step,
            epoch=epoch,
            loss=loss,
            learning_rate=lr,
        )
    
    print("\n")
    
    # Export
    tracker.export_csv()
    tracker.export_json()
    
    # Summary
    summary = tracker.get_metrics_summary()
    print("\nSummary:")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # Cleanup
    import shutil
    if Path("./test_metrics").exists():
        shutil.rmtree("./test_metrics")
        print("\nTest directory cleaned up")
