#!/usr/bin/env python3
"""
TensorBoard logging service for model training and evaluation metrics.

This module provides comprehensive logging capabilities for:
- Training metrics (loss, accuracy, learning rate)
- Model performance evaluation
- Surrogate execution monitoring
- System performance metrics
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import numpy as np

try:
    from torch.utils.tensorboard import SummaryWriter
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    SummaryWriter = None

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TensorBoardLogger:
    """
    TensorBoard logging service for SWAI Cursor project.
    
    Provides structured logging for:
    - Model training metrics
    - Surrogate execution performance
    - System resource usage
    - Module interaction patterns
    """
    
    def __init__(self, log_dir: str = "logs/tensorboard", experiment_name: Optional[str] = None):
        """
        Initialize TensorBoard logger.
        
        Args:
            log_dir: Directory to store TensorBoard logs
            experiment_name: Name for this experiment run
        """
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name or f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.log_path = self.log_dir / self.experiment_name
        
        # Create log directory
        self.log_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize TensorBoard writer if available
        if TORCH_AVAILABLE:
            self.writer = SummaryWriter(log_dir=str(self.log_path))
            logger.info(f"TensorBoard logging initialized at {self.log_path}")
        else:
            self.writer = None
            logger.warning("PyTorch not available - TensorBoard logging disabled")
        
        # Track experiment metadata
        self.experiment_metadata = {
            "start_time": datetime.now().isoformat(),
            "log_dir": str(self.log_path),
            "torch_available": TORCH_AVAILABLE
        }
        
        # Store torch availability for testing
        self.torch_available = TORCH_AVAILABLE
        
        # Performance tracking
        self.step_counters = {}
        self.metric_history = {}
    
    def log_scalar(self, tag: str, value: Union[float, int], step: Optional[int] = None, 
                   category: str = "general") -> None:
        """
        Log a scalar metric to TensorBoard.
        
        Args:
            tag: Metric name/tag
            value: Metric value
            step: Step number (auto-incremented if None)
            category: Category for organizing metrics
        """
        # Auto-increment step if not provided
        if step is None:
            step = self.step_counters.get(category, 0)
            self.step_counters[category] = step + 1
        
        # Store in history (always, regardless of TensorBoard availability)
        if category not in self.metric_history:
            self.metric_history[category] = {}
        if tag not in self.metric_history[category]:
            self.metric_history[category][tag] = []
        self.metric_history[category][tag].append((step, value))
        
        # Log to TensorBoard if available
        if self.writer:
            full_tag = f"{category}/{tag}"
            self.writer.add_scalar(full_tag, value, step)
            logger.debug(f"Logged scalar {full_tag}: {value} at step {step}")
        else:
            logger.debug(f"Stored scalar {category}/{tag}: {value} at step {step} (TensorBoard unavailable)")
    
    def log_surrogate_metrics(self, surrogate_name: str, execution_time: float, 
                            input_size: int, output_size: int, success: bool) -> None:
        """
        Log surrogate execution metrics.
        
        Args:
            surrogate_name: Name of the surrogate
            execution_time: Time taken for execution (seconds)
            input_size: Size of input data
            output_size: Size of output data
            success: Whether execution was successful
        """
        category = "surrogate"
        
        # Log execution time
        self.log_scalar(f"{surrogate_name}/execution_time", execution_time, category=category)
        
        # Log data throughput
        throughput = (input_size + output_size) / execution_time if execution_time > 0 else 0
        self.log_scalar(f"{surrogate_name}/throughput", throughput, category=category)
        
        # Log success rate
        success_rate = 1.0 if success else 0.0
        self.log_scalar(f"{surrogate_name}/success_rate", success_rate, category=category)
        
        # Log input/output sizes
        self.log_scalar(f"{surrogate_name}/input_size", input_size, category=category)
        self.log_scalar(f"{surrogate_name}/output_size", output_size, category=category)
    
    def log_model_metrics(self, model_name: str, metrics: Dict[str, float], 
                         epoch: int, phase: str = "train") -> None:
        """
        Log model training/evaluation metrics.
        
        Args:
            model_name: Name of the model
            metrics: Dictionary of metric name -> value
            epoch: Current epoch
            phase: Training phase (train/val/test)
        """
        category = f"model/{model_name}/{phase}"
        
        for metric_name, value in metrics.items():
            self.log_scalar(metric_name, value, step=epoch, category=category)
    
    def log_system_metrics(self, cpu_usage: float, memory_usage: float, 
                          gpu_usage: Optional[float] = None) -> None:
        """
        Log system resource usage metrics.
        
        Args:
            cpu_usage: CPU usage percentage
            memory_usage: Memory usage percentage
            gpu_usage: GPU usage percentage (optional)
        """
        category = "system"
        
        self.log_scalar("cpu_usage", cpu_usage, category=category)
        self.log_scalar("memory_usage", memory_usage, category=category)
        
        if gpu_usage is not None:
            self.log_scalar("gpu_usage", gpu_usage, category=category)
    
    def log_module_interaction(self, source_module: str, target_module: str, 
                             interaction_type: str, latency: float) -> None:
        """
        Log module interaction metrics.
        
        Args:
            source_module: Source module name
            target_module: Target module name
            interaction_type: Type of interaction (call/response/data)
            latency: Interaction latency in seconds
        """
        category = "interactions"
        tag = f"{source_module}->{target_module}/{interaction_type}"
        
        self.log_scalar(tag, latency, category=category)
    
    def log_hyperparameters(self, hparams: Dict[str, Any]) -> None:
        """
        Log hyperparameters for the experiment.
        
        Args:
            hparams: Dictionary of hyperparameters
        """
        if not self.writer:
            return
            
        # Convert all values to strings for TensorBoard
        hparams_str = {k: str(v) for k, v in hparams.items()}
        
        # Create dummy metrics for hyperparameter logging
        metrics = {"dummy_metric": 0.0}
        
        self.writer.add_hparams(hparams_str, metrics)
        logger.info(f"Logged hyperparameters: {hparams_str}")
    
    def log_text(self, tag: str, text: str, step: Optional[int] = None) -> None:
        """
        Log text data to TensorBoard.
        
        Args:
            tag: Tag for the text
            text: Text content
            step: Step number
        """
        if not self.writer:
            return
            
        if step is None:
            step = self.step_counters.get("text", 0)
            self.step_counters["text"] = step + 1
        
        self.writer.add_text(tag, text, step)
        logger.debug(f"Logged text {tag} at step {step}")
    
    def log_graph(self, model, input_to_model: Any) -> None:
        """
        Log model graph to TensorBoard.
        
        Args:
            model: PyTorch model
            input_to_model: Sample input for the model
        """
        if not self.writer or not TORCH_AVAILABLE:
            return
            
        try:
            self.writer.add_graph(model, input_to_model)
            logger.info("Logged model graph to TensorBoard")
        except Exception as e:
            logger.error(f"Failed to log model graph: {e}")
    
    def get_metric_summary(self, category: str = None) -> Dict[str, Any]:
        """
        Get summary of logged metrics.
        
        Args:
            category: Specific category to summarize (None for all)
            
        Returns:
            Dictionary with metric summaries
        """
        if category:
            categories = {category: self.metric_history.get(category, {})}
        else:
            categories = self.metric_history
        
        summary = {}
        for cat, metrics in categories.items():
            summary[cat] = {}
            for metric_name, history in metrics.items():
                if history:
                    values = [v for _, v in history]
                    summary[cat][metric_name] = {
                        "count": len(values),
                        "mean": np.mean(values),
                        "std": np.std(values),
                        "min": np.min(values),
                        "max": np.max(values),
                        "latest": values[-1]
                    }
        
        return summary
    
    def save_experiment_config(self, config: Dict[str, Any]) -> None:
        """
        Save experiment configuration to file.
        
        Args:
            config: Experiment configuration dictionary
        """
        config_path = self.log_path / "experiment_config.json"
        
        full_config = {
            "experiment_metadata": self.experiment_metadata,
            "config": config,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(config_path, 'w') as f:
            json.dump(full_config, f, indent=2)
        
        logger.info(f"Saved experiment config to {config_path}")
    
    def close(self) -> None:
        """Close the TensorBoard writer and finalize logging."""
        if self.writer:
            self.writer.close()
            logger.info("TensorBoard writer closed")
        
        # Save final summary
        summary = self.get_metric_summary()
        summary_path = self.log_path / "metric_summary.json"
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Saved metric summary to {summary_path}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# Global logger instance
_global_logger: Optional[TensorBoardLogger] = None


def get_global_logger() -> Optional[TensorBoardLogger]:
    """Get the global TensorBoard logger instance."""
    return _global_logger


def initialize_global_logger(log_dir: str = "logs/tensorboard", 
                           experiment_name: Optional[str] = None) -> TensorBoardLogger:
    """
    Initialize the global TensorBoard logger.
    
    Args:
        log_dir: Directory for TensorBoard logs
        experiment_name: Name for the experiment
        
    Returns:
        Initialized TensorBoard logger
    """
    global _global_logger
    _global_logger = TensorBoardLogger(log_dir, experiment_name)
    return _global_logger


def log_metric(tag: str, value: Union[float, int], step: Optional[int] = None, 
               category: str = "general") -> None:
    """
    Log a metric using the global logger.
    
    Args:
        tag: Metric name
        value: Metric value
        step: Step number
        category: Metric category
    """
    if _global_logger:
        _global_logger.log_scalar(tag, value, step, category)
