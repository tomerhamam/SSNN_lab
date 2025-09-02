# TensorBoard Integration Guide for SWAI Cursor

## Overview

This guide explains how to use TensorBoard (TB) capabilities in the SWAI Cursor project for model training and evaluation. TensorBoard provides comprehensive logging and visualization for:

- **Model Training Metrics**: Loss, accuracy, learning rate curves
- **Surrogate Execution Performance**: Execution times, throughput, success rates
- **System Resource Monitoring**: CPU, memory, GPU usage
- **Module Interaction Analysis**: Latency, dependency patterns
- **Experiment Management**: Configuration tracking, hyperparameter logging

## Quick Start

### 1. Install Dependencies

```bash
# Install TensorBoard and ML dependencies
pip install -r requirements.txt

# Verify installation
python -c "import tensorboard; print('TensorBoard ready')"
```

### 2. Start TensorBoard Service

```bash
# Start the Flask backend with TensorBoard routes
python -m flask --app app.py run --debug

# In another terminal, start TensorBoard
tensorboard --logdir=logs/tensorboard --port=6006
```

### 3. Access TensorBoard

- **TensorBoard UI**: http://localhost:6006
- **API Endpoints**: http://localhost:5000/api/tensorboard/*

## API Usage

### Starting an Experiment

```python
import requests

# Start a new experiment
response = requests.post('http://localhost:5000/api/tensorboard/experiment/start', 
                        json={
                            'experiment_name': 'my_training_run',
                            'log_dir': 'logs/tensorboard'
                        })

# Start TensorBoard service
requests.post('http://localhost:5000/api/tensorboard/start',
              json={'log_dir': 'logs/tensorboard', 'port': 6006})
```

### Logging Metrics

```python
# Log scalar metrics
requests.post('http://localhost:5000/api/tensorboard/log/scalar',
              json={
                  'tag': 'loss',
                  'value': 0.25,
                  'step': 100,
                  'category': 'training'
              })

# Log text data
requests.post('http://localhost:5000/api/tensorboard/log/text',
              json={
                  'tag': 'model_config',
                  'text': 'Model: ResNet50, LR: 0.001, Batch: 32'
              })
```

## Python Integration

### Using the TensorBoard Logger

```python
from backend.services.tensorboard_logger import TensorBoardLogger, initialize_global_logger

# Initialize global logger
logger = initialize_global_logger(
    log_dir="logs/tensorboard",
    experiment_name="model_training_v1"
)

# Log training metrics
for epoch in range(100):
    # Simulate training
    train_loss = 1.0 / (epoch + 1)
    val_accuracy = min(0.95, epoch * 0.01)
    
    # Log metrics
    logger.log_model_metrics(
        model_name="my_model",
        metrics={
            "loss": train_loss,
            "accuracy": val_accuracy,
            "learning_rate": 0.001
        },
        epoch=epoch,
        phase="train"
    )

# Log hyperparameters
logger.log_hyperparameters({
    "learning_rate": 0.001,
    "batch_size": 32,
    "optimizer": "Adam",
    "model_architecture": "ResNet50"
})

# Close logger
logger.close()
```

### Surrogate Execution with Metrics

```python
from backend.services.surrogate import registry

# Create a surrogate
surrogate = registry.create("mock_llm")

# Run with automatic metrics logging
result = surrogate.run_with_metrics({
    "input_text": "Hello world",
    "context": "test"
})

# Metrics are automatically logged to TensorBoard
print(result["_execution_metrics"])
```

## Use Cases for SWAI Cursor

### 1. Model Training Monitoring

Track the performance of AI models during training:

```python
# Training loop with TensorBoard logging
def train_model(model, train_loader, val_loader, epochs=100):
    logger = initialize_global_logger("model_training")
    
    for epoch in range(epochs):
        # Training phase
        train_loss = train_epoch(model, train_loader)
        logger.log_scalar("train/loss", train_loss, epoch)
        
        # Validation phase
        val_accuracy = validate_epoch(model, val_loader)
        logger.log_scalar("val/accuracy", val_accuracy, epoch)
        
        # Log learning rate
        lr = optimizer.param_groups[0]['lr']
        logger.log_scalar("train/learning_rate", lr, epoch)
    
    logger.close()
```

### 2. Surrogate Performance Analysis

Monitor surrogate execution performance:

```python
# Analyze surrogate performance over time
def benchmark_surrogates():
    logger = initialize_global_logger("surrogate_benchmark")
    
    surrogates = ["static_stub", "mock_llm"]
    test_inputs = generate_test_data()
    
    for surrogate_type in surrogates:
        surrogate = registry.create(surrogate_type)
        
        for i, inputs in enumerate(test_inputs):
            start_time = time.time()
            result = surrogate.run(inputs)
            execution_time = time.time() - start_time
            
            # Log performance metrics
            logger.log_surrogate_metrics(
                surrogate_name=surrogate_type,
                execution_time=execution_time,
                input_size=len(str(inputs)),
                output_size=len(str(result)),
                success=True
            )
    
    logger.close()
```

### 3. System Resource Monitoring

Track system performance during experiments:

```python
import psutil

def monitor_system_resources():
    logger = initialize_global_logger("system_monitoring")
    
    for step in range(1000):
        # Get system metrics
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        
        # Log to TensorBoard
        logger.log_system_metrics(
            cpu_usage=cpu_percent,
            memory_usage=memory_percent
        )
        
        time.sleep(1)  # Monitor every second
    
    logger.close()
```

### 4. Module Interaction Analysis

Track how modules interact in your AI architecture:

```python
def analyze_module_interactions():
    logger = initialize_global_logger("module_interactions")
    
    # Simulate module interactions
    modules = ["data_loader", "preprocessor", "model", "postprocessor"]
    
    for i in range(100):
        for j in range(len(modules) - 1):
            source = modules[j]
            target = modules[j + 1]
            
            # Simulate interaction latency
            latency = random.uniform(0.01, 0.1)
            
            logger.log_module_interaction(
                source_module=source,
                target_module=target,
                interaction_type="data_flow",
                latency=latency
            )
    
    logger.close()
```

## Advanced Features

### Experiment Comparison

Compare multiple experiments:

```python
# Run multiple experiments with different configurations
configs = [
    {"lr": 0.001, "batch_size": 32},
    {"lr": 0.01, "batch_size": 64},
    {"lr": 0.0001, "batch_size": 16}
]

for i, config in enumerate(configs):
    logger = initialize_global_logger(f"experiment_{i}")
    logger.log_hyperparameters(config)
    
    # Run training with this config
    train_with_config(config, logger)
    logger.close()
```

### Custom Metrics

Log custom metrics for your specific use case:

```python
def log_custom_metrics():
    logger = initialize_global_logger("custom_metrics")
    
    # Log custom business metrics
    logger.log_scalar("business/conversion_rate", 0.15, category="business")
    logger.log_scalar("business/user_satisfaction", 4.2, category="business")
    
    # Log technical metrics
    logger.log_scalar("technical/api_response_time", 0.05, category="technical")
    logger.log_scalar("technical/error_rate", 0.02, category="technical")
    
    logger.close()
```

## Best Practices

### 1. Organize Metrics with Categories

```python
# Use consistent category naming
logger.log_scalar("model/loss", loss, category="training")
logger.log_scalar("model/accuracy", accuracy, category="training")
logger.log_scalar("system/cpu_usage", cpu, category="system")
logger.log_scalar("business/revenue", revenue, category="business")
```

### 2. Log Hyperparameters Early

```python
# Log configuration at experiment start
config = {
    "model": "ResNet50",
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 100,
    "optimizer": "Adam"
}

logger.log_hyperparameters(config)
```

### 3. Use Context Managers

```python
# Automatic cleanup with context manager
with TensorBoardLogger("my_experiment") as logger:
    # Your training code here
    for epoch in range(100):
        logger.log_scalar("loss", loss, epoch)
    # Logger automatically closed
```

### 4. Monitor Long-Running Experiments

```python
# For long experiments, save progress periodically
def long_training_experiment():
    logger = initialize_global_logger("long_experiment")
    
    try:
        for epoch in range(10000):
            # Training code
            logger.log_scalar("loss", loss, epoch)
            
            # Save checkpoint every 100 epochs
            if epoch % 100 == 0:
                logger.log_text("checkpoint", f"Epoch {epoch} completed")
                
    except KeyboardInterrupt:
        logger.log_text("experiment_info", "Training interrupted by user")
    finally:
        logger.close()
```

## Troubleshooting

### Common Issues

1. **TensorBoard not starting**: Check if port 6006 is available
2. **Metrics not appearing**: Ensure experiment is started before logging
3. **Permission errors**: Check log directory permissions
4. **Import errors**: Verify all dependencies are installed

### Debug Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check TensorBoard status
response = requests.get('http://localhost:5000/api/tensorboard/status')
print(response.json())
```

## Integration with Frontend

The TensorBoard integration provides API endpoints that can be consumed by the Vue.js frontend:

```javascript
// Frontend integration example
async function startExperiment(experimentName) {
    const response = await fetch('/api/tensorboard/experiment/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ experiment_name: experimentName })
    });
    return response.json();
}

async function logMetric(tag, value, step) {
    await fetch('/api/tensorboard/log/scalar', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ tag, value, step, category: 'frontend' })
    });
}
```

## Conclusion

TensorBoard integration provides powerful monitoring and visualization capabilities for your SWAI Cursor project. Use it to:

- **Track model performance** during training and evaluation
- **Monitor system resources** and optimize performance
- **Analyze surrogate execution** patterns
- **Compare experiments** with different configurations
- **Debug issues** with detailed logging

The integration is designed to be lightweight and optional - your existing code will work without TensorBoard, but you can enhance it with comprehensive monitoring when needed.
