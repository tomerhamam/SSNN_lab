#!/usr/bin/env python3
"""
TensorBoard Demo Script for SWAI Cursor

This script demonstrates how to use TensorBoard integration for:
1. Model training simulation
2. Surrogate execution monitoring
3. System resource tracking
4. Module interaction analysis

Run this script to see TensorBoard capabilities in action.
"""

import json
import random
import time
import requests
from datetime import datetime
from pathlib import Path

# Add backend to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from backend.services.tensorboard_logger import TensorBoardLogger, initialize_global_logger
from backend.services.surrogate import registry


def simulate_model_training():
    """Simulate a model training process with TensorBoard logging."""
    print("🚀 Starting model training simulation...")
    
    # Initialize TensorBoard logger
    logger = initialize_global_logger(
        log_dir="logs/tensorboard",
        experiment_name=f"model_training_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    # Log hyperparameters
    hyperparams = {
        "model": "DemoModel",
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 50,
        "optimizer": "Adam",
        "loss_function": "CrossEntropy"
    }
    logger.log_hyperparameters(hyperparams)
    
    # Simulate training loop
    for epoch in range(50):
        # Simulate training metrics
        train_loss = 1.0 / (epoch + 1) + random.uniform(-0.1, 0.1)
        train_accuracy = min(0.95, epoch * 0.02 + random.uniform(-0.05, 0.05))
        
        # Simulate validation metrics
        val_loss = train_loss * 1.1 + random.uniform(-0.05, 0.05)
        val_accuracy = train_accuracy * 0.95 + random.uniform(-0.02, 0.02)
        
        # Log training metrics
        logger.log_model_metrics(
            model_name="demo_model",
            metrics={
                "loss": train_loss,
                "accuracy": train_accuracy
            },
            epoch=epoch,
            phase="train"
        )
        
        # Log validation metrics
        logger.log_model_metrics(
            model_name="demo_model",
            metrics={
                "loss": val_loss,
                "accuracy": val_accuracy
            },
            epoch=epoch,
            phase="val"
        )
        
        # Log learning rate (simulate learning rate schedule)
        lr = 0.001 * (0.95 ** epoch)
        logger.log_scalar("learning_rate", lr, epoch, "training")
        
        # Log some custom metrics
        logger.log_scalar("gradient_norm", random.uniform(0.1, 2.0), epoch, "training")
        logger.log_scalar("weight_decay", 0.0001, epoch, "training")
        
        if epoch % 10 == 0:
            print(f"  Epoch {epoch}: Train Loss={train_loss:.3f}, Val Acc={val_accuracy:.3f}")
    
    # Log final model info
    logger.log_text("model_info", f"Training completed. Final accuracy: {val_accuracy:.3f}")
    
    print("✅ Model training simulation completed!")
    return logger


def simulate_surrogate_execution():
    """Simulate surrogate execution with performance monitoring."""
    print("🔄 Starting surrogate execution simulation...")
    
    logger = initialize_global_logger(
        log_dir="logs/tensorboard",
        experiment_name=f"surrogate_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    # Test different surrogate types
    surrogate_types = ["static_stub", "mock_llm"]
    test_inputs = [
        {"text": "Hello world", "context": "greeting"},
        {"data": [1, 2, 3, 4, 5], "operation": "sum"},
        {"query": "What is AI?", "model": "gpt-4"},
        {"image": "base64_data", "task": "classification"},
        {"code": "def hello(): print('world')", "language": "python"}
    ]
    
    for surrogate_type in surrogate_types:
        print(f"  Testing {surrogate_type} surrogate...")
        
        surrogate = registry.create(surrogate_type)
        if not surrogate:
            print(f"    ⚠️  Surrogate {surrogate_type} not available")
            continue
        
        for i, inputs in enumerate(test_inputs):
            # Simulate execution time
            execution_time = random.uniform(0.01, 0.5)
            input_size = len(json.dumps(inputs))
            output_size = random.randint(100, 1000)
            success = random.random() > 0.1  # 90% success rate
            
            # Log surrogate metrics
            logger.log_surrogate_metrics(
                surrogate_name=surrogate_type,
                execution_time=execution_time,
                input_size=input_size,
                output_size=output_size,
                success=success
            )
            
            # Log throughput
            throughput = (input_size + output_size) / execution_time if execution_time > 0 else 0
            logger.log_scalar(f"{surrogate_type}/throughput", throughput, i, "surrogate")
            
            time.sleep(0.1)  # Small delay for realistic simulation
    
    print("✅ Surrogate execution simulation completed!")
    return logger


def simulate_system_monitoring():
    """Simulate system resource monitoring."""
    print("📊 Starting system monitoring simulation...")
    
    logger = initialize_global_logger(
        log_dir="logs/tensorboard",
        experiment_name=f"system_monitoring_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    # Simulate system metrics over time
    for step in range(100):
        # Simulate realistic system metrics
        cpu_usage = 20 + 30 * random.random() + 10 * (step % 20) / 20
        memory_usage = 40 + 20 * random.random() + 5 * (step % 30) / 30
        gpu_usage = 0 if step < 20 else 60 + 30 * random.random()
        
        # Log system metrics
        logger.log_system_metrics(
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            gpu_usage=gpu_usage
        )
        
        # Log some additional system metrics
        logger.log_scalar("disk_usage", 50 + 10 * random.random(), step, "system")
        logger.log_scalar("network_io", random.uniform(0, 100), step, "system")
        logger.log_scalar("active_connections", random.randint(10, 100), step, "system")
        
        if step % 20 == 0:
            print(f"  Step {step}: CPU={cpu_usage:.1f}%, Memory={memory_usage:.1f}%")
        
        time.sleep(0.05)  # Simulate monitoring interval
    
    print("✅ System monitoring simulation completed!")
    return logger


def simulate_module_interactions():
    """Simulate module interaction patterns in AI architecture."""
    print("🔗 Starting module interaction simulation...")
    
    logger = initialize_global_logger(
        log_dir="logs/tensorboard",
        experiment_name=f"module_interactions_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    # Define module architecture
    modules = [
        "data_loader",
        "preprocessor", 
        "feature_extractor",
        "model",
        "postprocessor",
        "output_formatter"
    ]
    
    interaction_types = ["data_flow", "control_flow", "error_handling", "status_update"]
    
    # Simulate interactions over time
    for step in range(200):
        # Randomly select modules and interaction type
        source_idx = random.randint(0, len(modules) - 2)
        target_idx = random.randint(source_idx + 1, len(modules) - 1)
        
        source_module = modules[source_idx]
        target_module = modules[target_idx]
        interaction_type = random.choice(interaction_types)
        
        # Simulate latency based on interaction type
        base_latency = {
            "data_flow": 0.05,
            "control_flow": 0.01,
            "error_handling": 0.1,
            "status_update": 0.005
        }
        
        latency = base_latency[interaction_type] + random.uniform(-0.01, 0.01)
        
        # Log interaction
        logger.log_module_interaction(
            source_module=source_module,
            target_module=target_module,
            interaction_type=interaction_type,
            latency=latency
        )
        
        # Log some aggregate metrics
        if step % 50 == 0:
            avg_latency = random.uniform(0.02, 0.08)
            logger.log_scalar("avg_interaction_latency", avg_latency, step, "interactions")
            logger.log_scalar("active_modules", random.randint(3, 6), step, "interactions")
        
        if step % 50 == 0:
            print(f"  Step {step}: {source_module} -> {target_module} ({interaction_type})")
    
    print("✅ Module interaction simulation completed!")
    return logger


def demonstrate_api_usage():
    """Demonstrate TensorBoard API usage."""
    print("🌐 Demonstrating TensorBoard API usage...")
    
    base_url = "http://localhost:5000/api/tensorboard"
    
    try:
        # Check if API is available
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print("  ✅ TensorBoard API is available")
        else:
            print("  ⚠️  TensorBoard API not responding")
            return
    except requests.exceptions.RequestException:
        print("  ⚠️  TensorBoard API not available (Flask server not running?)")
        return
    
    # Start an experiment via API
    experiment_data = {
        "experiment_name": f"api_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "log_dir": "logs/tensorboard"
    }
    
    try:
        response = requests.post(f"{base_url}/experiment/start", json=experiment_data)
        if response.status_code == 200:
            print("  ✅ Experiment started via API")
        else:
            print(f"  ⚠️  Failed to start experiment: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"  ⚠️  API request failed: {e}")
    
    # Log some metrics via API
    for i in range(10):
        metric_data = {
            "tag": "api_demo_metric",
            "value": random.uniform(0, 1),
            "step": i,
            "category": "api_demo"
        }
        
        try:
            response = requests.post(f"{base_url}/log/scalar", json=metric_data)
            if response.status_code == 200:
                print(f"  📊 Logged metric {i+1}/10")
        except requests.exceptions.RequestException:
            print(f"  ⚠️  Failed to log metric {i+1}")
        
        time.sleep(0.1)
    
    print("✅ API usage demonstration completed!")


def main():
    """Run all TensorBoard demonstrations."""
    print("🎯 TensorBoard Integration Demo for SWAI Cursor")
    print("=" * 50)
    
    # Create logs directory
    Path("logs/tensorboard").mkdir(parents=True, exist_ok=True)
    
    demos = [
        ("Model Training", simulate_model_training),
        ("Surrogate Execution", simulate_surrogate_execution),
        ("System Monitoring", simulate_system_monitoring),
        ("Module Interactions", simulate_module_interactions),
        ("API Usage", demonstrate_api_usage)
    ]
    
    for demo_name, demo_func in demos:
        print(f"\n📋 Running {demo_name} Demo...")
        try:
            logger = demo_func()
            if logger:
                logger.close()
        except Exception as e:
            print(f"  ❌ Demo failed: {e}")
    
    print("\n🎉 All demonstrations completed!")
    print("\n📝 Next Steps:")
    print("1. Start TensorBoard: tensorboard --logdir=logs/tensorboard --port=6006")
    print("2. Open http://localhost:6006 in your browser")
    print("3. Explore the logged metrics and visualizations")
    print("4. Check the integration guide: docs/tensorboard_integration_guide.md")


if __name__ == "__main__":
    main()
