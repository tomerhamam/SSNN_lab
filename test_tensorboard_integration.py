#!/usr/bin/env python3
"""
Tests for TensorBoard integration in SWAI Cursor.

Tests the TensorBoard logging service and API endpoints.
"""

import json
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from backend.services.tensorboard_logger import TensorBoardLogger, initialize_global_logger
from backend.services.surrogate import StaticStubSurrogate, MockLLMSurrogate


class TestTensorBoardLogger:
    """Test TensorBoard logger functionality."""
    
    def test_logger_initialization(self):
        """Test logger initialization without PyTorch."""
        with patch('backend.services.tensorboard_logger.TORCH_AVAILABLE', False):
            with tempfile.TemporaryDirectory() as temp_dir:
                logger = TensorBoardLogger(log_dir=temp_dir, experiment_name="test")
                
                assert logger.experiment_name == "test"
                assert logger.log_dir == Path(temp_dir)
                assert logger.writer is None  # No PyTorch available
                assert logger.torch_available is False
    
    def test_logger_with_torch(self):
        """Test logger initialization with PyTorch available."""
        with patch('backend.services.tensorboard_logger.TORCH_AVAILABLE', True):
            with patch('backend.services.tensorboard_logger.SummaryWriter') as mock_writer:
                with tempfile.TemporaryDirectory() as temp_dir:
                    logger = TensorBoardLogger(log_dir=temp_dir, experiment_name="test")
                    
                    assert logger.writer is not None
                    mock_writer.assert_called_once()
    
    def test_scalar_logging(self):
        """Test scalar metric logging."""
        with patch('backend.services.tensorboard_logger.TORCH_AVAILABLE', False):
            with tempfile.TemporaryDirectory() as temp_dir:
                logger = TensorBoardLogger(log_dir=temp_dir, experiment_name="test")
                
                # Log some scalars
                logger.log_scalar("test_metric", 0.5, step=1, category="test")
                logger.log_scalar("test_metric", 0.7, step=2, category="test")
                
                # Check metric history
                assert "test" in logger.metric_history
                assert "test_metric" in logger.metric_history["test"]
                assert len(logger.metric_history["test"]["test_metric"]) == 2
    
    def test_surrogate_metrics_logging(self):
        """Test surrogate execution metrics logging."""
        with patch('backend.services.tensorboard_logger.TORCH_AVAILABLE', False):
            with tempfile.TemporaryDirectory() as temp_dir:
                logger = TensorBoardLogger(log_dir=temp_dir, experiment_name="test")
                
                # Log surrogate metrics
                logger.log_surrogate_metrics(
                    surrogate_name="test_surrogate",
                    execution_time=0.1,
                    input_size=100,
                    output_size=200,
                    success=True
                )
                
                # Check that metrics were logged
                assert "surrogate" in logger.metric_history
                surrogate_metrics = logger.metric_history["surrogate"]
                assert "test_surrogate/execution_time" in surrogate_metrics
                assert "test_surrogate/throughput" in surrogate_metrics
                assert "test_surrogate/success_rate" in surrogate_metrics
    
    def test_model_metrics_logging(self):
        """Test model training metrics logging."""
        with patch('backend.services.tensorboard_logger.TORCH_AVAILABLE', False):
            with tempfile.TemporaryDirectory() as temp_dir:
                logger = TensorBoardLogger(log_dir=temp_dir, experiment_name="test")
                
                # Log model metrics
                metrics = {"loss": 0.5, "accuracy": 0.9}
                logger.log_model_metrics("test_model", metrics, epoch=1, phase="train")
                
                # Check that metrics were logged
                assert "model/test_model/train" in logger.metric_history
                model_metrics = logger.metric_history["model/test_model/train"]
                assert "loss" in model_metrics
                assert "accuracy" in model_metrics
    
    def test_system_metrics_logging(self):
        """Test system resource metrics logging."""
        with patch('backend.services.tensorboard_logger.TORCH_AVAILABLE', False):
            with tempfile.TemporaryDirectory() as temp_dir:
                logger = TensorBoardLogger(log_dir=temp_dir, experiment_name="test")
                
                # Log system metrics
                logger.log_system_metrics(
                    cpu_usage=50.0,
                    memory_usage=75.0,
                    gpu_usage=25.0
                )
                
                # Check that metrics were logged
                assert "system" in logger.metric_history
                system_metrics = logger.metric_history["system"]
                assert "cpu_usage" in system_metrics
                assert "memory_usage" in system_metrics
                assert "gpu_usage" in system_metrics
    
    def test_text_logging(self):
        """Test text data logging."""
        with patch('backend.services.tensorboard_logger.TORCH_AVAILABLE', False):
            with tempfile.TemporaryDirectory() as temp_dir:
                logger = TensorBoardLogger(log_dir=temp_dir, experiment_name="test")
                
                # Log text
                logger.log_text("test_tag", "This is test text", step=1)
                
                # Check that text was logged (no history tracking for text)
                # Just verify no errors occurred
                assert True
    
    def test_metric_summary(self):
        """Test metric summary generation."""
        with patch('backend.services.tensorboard_logger.TORCH_AVAILABLE', False):
            with tempfile.TemporaryDirectory() as temp_dir:
                logger = TensorBoardLogger(log_dir=temp_dir, experiment_name="test")
                
                # Log some metrics
                logger.log_scalar("metric1", 1.0, step=1, category="test")
                logger.log_scalar("metric1", 2.0, step=2, category="test")
                logger.log_scalar("metric2", 3.0, step=1, category="test")
                
                # Get summary
                summary = logger.get_metric_summary()
                
                assert "test" in summary
                assert "metric1" in summary["test"]
                assert "metric2" in summary["test"]
                
                # Check summary statistics
                metric1_summary = summary["test"]["metric1"]
                assert metric1_summary["count"] == 2
                assert metric1_summary["mean"] == 1.5
                assert metric1_summary["min"] == 1.0
                assert metric1_summary["max"] == 2.0
    
    def test_experiment_config_saving(self):
        """Test experiment configuration saving."""
        with patch('backend.services.tensorboard_logger.TORCH_AVAILABLE', False):
            with tempfile.TemporaryDirectory() as temp_dir:
                logger = TensorBoardLogger(log_dir=temp_dir, experiment_name="test")
                
                # Save config
                config = {"learning_rate": 0.001, "batch_size": 32}
                logger.save_experiment_config(config)
                
                # Check that config file was created
                config_file = logger.log_path / "experiment_config.json"
                assert config_file.exists()
                
                # Check config content
                with open(config_file, 'r') as f:
                    saved_config = json.load(f)
                
                assert "config" in saved_config
                assert saved_config["config"] == config


class TestSurrogateTensorBoardIntegration:
    """Test TensorBoard integration with surrogate execution."""
    
    def test_static_stub_with_metrics(self):
        """Test StaticStubSurrogate with TensorBoard metrics."""
        surrogate = StaticStubSurrogate()
        
        # Mock the TensorBoard logger
        with patch('backend.services.surrogate.get_global_logger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            
            # Run surrogate with metrics
            inputs = {"test": "data"}
            result = surrogate.run_with_metrics(inputs)
            
            # Check that metrics were logged
            mock_logger.log_surrogate_metrics.assert_called_once()
            
            # Check that execution metrics were added to result
            assert "_execution_metrics" in result
            metrics = result["_execution_metrics"]
            assert "execution_time" in metrics
            assert "input_size" in metrics
            assert "output_size" in metrics
            assert "surrogate_type" in metrics
    
    def test_mock_llm_with_metrics(self):
        """Test MockLLMSurrogate with TensorBoard metrics."""
        surrogate = MockLLMSurrogate()
        
        # Mock the TensorBoard logger
        with patch('backend.services.surrogate.get_global_logger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            
            # Run surrogate with metrics
            inputs = {"text": "Hello world"}
            result = surrogate.run_with_metrics(inputs)
            
            # Check that metrics were logged
            mock_logger.log_surrogate_metrics.assert_called_once()
            
            # Check that execution metrics were added to result
            assert "_execution_metrics" in result
            metrics = result["_execution_metrics"]
            assert "execution_time" in metrics
            assert "input_size" in metrics
            assert "output_size" in metrics
            assert "surrogate_type" in metrics
    
    def test_surrogate_failure_metrics(self):
        """Test TensorBoard metrics logging when surrogate fails."""
        surrogate = StaticStubSurrogate()
        
        # Mock the TensorBoard logger
        with patch('backend.services.surrogate.get_global_logger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            
            # Mock the run method to raise an exception
            with patch.object(surrogate, 'run', side_effect=Exception("Test error")):
                inputs = {"test": "data"}
                
                # Should raise the exception
                with pytest.raises(Exception, match="Test error"):
                    surrogate.run_with_metrics(inputs)
                
                # Check that failure metrics were logged
                mock_logger.log_surrogate_metrics.assert_called_once()
                call_args = mock_logger.log_surrogate_metrics.call_args
                assert call_args[1]["success"] is False


class TestGlobalLogger:
    """Test global logger functionality."""
    
    def test_global_logger_initialization(self):
        """Test global logger initialization."""
        with patch('backend.services.tensorboard_logger.TORCH_AVAILABLE', False):
            with tempfile.TemporaryDirectory() as temp_dir:
                # Initialize global logger
                logger = initialize_global_logger(log_dir=temp_dir, experiment_name="global_test")
                
                # Check that it was set as global
                from backend.services.tensorboard_logger import get_global_logger
                global_logger = get_global_logger()
                assert global_logger is logger
                assert global_logger.experiment_name == "global_test"
                
                # Clean up
                logger.close()


if __name__ == "__main__":
    pytest.main([__file__])
