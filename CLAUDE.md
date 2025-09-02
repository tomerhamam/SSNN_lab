# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an educational Self-Supervised Neural Networks (SSL) lab that teaches SSL principles through hands-on implementation. The project has two parallel structures:
1. **Standalone implementation** (`tests/snn_lab.py`) - Complete working code with all classes
2. **Interactive tutorial** (`snn_lab_complete.ipynb`) - 2-hour guided learning experience with exercises

The codebase demonstrates SSL across two domains:
- **Computer Vision**: Rotation prediction on 8×8 handwritten digits → transfer to digit classification
- **Time Series**: Autoencoder reconstruction of sine waves → transfer to frequency classification

## Key Commands

### Running Tests and Experiments
```bash
# Run all SSL experiments and tests
python tests/snn_lab.py

# Run individual test functions
python -c "from tests.snn_lab import test_rotation_accuracy; test_rotation_accuracy()"
python -c "from tests.snn_lab import test_digit_transfer; test_digit_transfer()"
python -c "from tests.snn_lab import test_time_series_transfer; test_time_series_transfer()"

# Run advanced SSL experiments with various configurations
python run_ssl_experiments.py          # Systematic testing with comparison tables
python run_ssl_experiments.py quick    # Quick test mode
python run_advanced_ssl_experiments.py # Advanced hyperparameter testing

# Run specific SSL experiments
python run_ssl_experiment.py           # Interactive digit SSL with plots
python ssl_fashion_mnist.py            # Fashion-MNIST SSL experiments
python ssl_cifar10_experiment.py       # CIFAR-10 SSL experiments

# Test the modular SSL framework
python test_ssl_framework.py           # Test different datasets and pretext tasks
python test_ssl_improvements.py        # Test SSL improvements
python test_complex_ssl.py             # Complex SSL testing
python comprehensive_ssl_test.py       # Comprehensive test suite
```

### API Testing (for assessment features)
```bash
# Quick 30-second verification
python quick_test.py

# Comprehensive API and environment test
python test_api_connection.py
```

### Interactive Learning
```bash
# Main tutorial notebook (recommended starting point)
jupyter notebook snn_lab_complete.ipynb
```

### Package Management
```bash
# Using uv (preferred - manages Python versions)
uv pip install -r pyproject.toml

# Standard pip (requires Python >= 3.12)
pip install anthropic>=0.40.0 numpy>=2.3.2 scikit-learn>=1.7.1 matplotlib>=3.10.5 seaborn>=0.13.0

# Check Python version requirement
python --version  # Must be >= 3.12
```

## Architecture & Code Structure

### Two-Phase SSL Workflow
Both vision and time series modules follow the same pattern:
1. **Pretext Task Phase**: Train on self-supervised objective (rotation/reconstruction)
2. **Transfer Phase**: Extract learned features → evaluate on downstream classification

### Project File Organization

#### Core Implementations
- `tests/snn_lab.py` - Complete standalone SSL implementation with test functions
- `snn_lab_complete.ipynb` - Main interactive tutorial notebook (2-hour guided experience)
- `ssl_framework.py` - Modular SSL framework with dataset-agnostic design

#### Experiment Runners
- `run_ssl_experiments.py` - Systematic testing with multiple configurations
- `run_advanced_ssl_experiments.py` - Advanced hyperparameter optimization
- `run_ssl_experiment.py` - Interactive experiment with visualization
- `ssl_fashion_mnist.py` - Fashion-MNIST specific SSL experiments
- `ssl_cifar10_experiment.py` - CIFAR-10 SSL experiments
- `comprehensive_ssl_test.py` - Full test suite across all components

#### Utilities and Visualization
- `utils/vis.py` - Visualization utilities
- `ssl_transfer_diagram.py` - Generate SSL concept diagrams
- `push_ssl_accuracy.py` - Advanced accuracy optimization experiments

### Core Neural Network Classes

#### `TwoLayerNet` (Vision SSL)
- Location: Defined in `tests/snn_lab.py`, `snn_lab_complete.ipynb`, and various experiment files
- Architecture: input(64) → hidden(32) → output(4) [configurable in some versions]
- Activation: tanh hidden layer, softmax output
- Training: Mini-batch SGD with manual backpropagation
- Key methods: `forward()`, `backward()`, `hidden_representation()`, `train()`

#### `Autoencoder` (Time Series SSL)
- Location: Defined in `tests/snn_lab.py` and `snn_lab_complete.ipynb`
- Architecture: input(50) → hidden(16) → output(50)
- Activation: tanh encoder, linear decoder
- Loss: Mean Squared Error (MSE)
- Key methods: `forward()`, `encode()`, `reconstruct()`, `train()`

#### Modular Framework Classes (`ssl_framework.py`)
- `DatasetInterface` - Abstract base for dataset loading
- `PretextTask` - Abstract base for pretext tasks (rotation, jigsaw, etc.)
- `SSLNetwork` - Configurable deep network architecture
- `SSLTrainer` - Training orchestration with metrics tracking

### Data Generation Functions
- `load_digit_data()`: Loads sklearn digits (1797 samples, 8×8 pixels), normalizes to [0,1]
- `create_rotation_dataset()`: Generates 4x data with rotation labels (configurable angles)
- `generate_sine_sequences()`: Creates binary classification dataset (freq=1.0 vs freq=3.0)
- `load_fashion_mnist()`: Loads Fashion-MNIST dataset for advanced experiments
- Dataset loaders in `ssl_framework.py` support CIFAR-10, Fashion-MNIST, and digits

### Assessment System
The notebook includes AI-powered evaluation using Claude API:
- Config: `assessment_config_anthropic.json`
- Environment: Requires `MY_APP_ANTHROPIC_KEY`
- Class: `OpenEndedAssessment` handles question evaluation
- Fallback: Manual evaluation mode when API unavailable

## Testing Requirements

The test suite validates SSL effectiveness:
- **Rotation accuracy**: Must exceed 40% (random = 25%)
- **Digit transfer**: SSL features must achieve ≥50% accuracy
- **Time series transfer**: Embeddings must preserve discriminative power

### Running Comprehensive Tests
```bash
# Full test suite
python comprehensive_ssl_test.py

# Individual module tests
python test_ssl_framework.py      # Framework components
python test_ssl_improvements.py   # Improvement strategies
python test_complex_ssl.py        # Complex scenarios
```

## Implementation Details

### Reproducibility
- Fixed random seeds: `rng(0)` for vision, `rng(1)` for time series
- Consistent data splits: `random_state=42` for train/test splits

### Data Preprocessing
- **Vision**: Digits normalized to [0,1] by dividing by 16
- **Time Series**: Per-sequence normalization (zero mean, unit variance)

### Training Hyperparameters
- **Vision**: learning_rate=0.3, epochs=15, batch_size=256
- **Time Series**: learning_rate=0.05, epochs=30, batch_size=128

## Development Tips

### Working with the Notebook
- Cells must be run sequentially (class definitions before usage)
- `test_net` is instantiated at the end of cell 14 (TwoLayerNet definition)
- Solutions are provided after each exercise for self-checking

### Common Issues
- If `test_net` is undefined: Ensure cell 14 (TwoLayerNet class) ran completely
- For API assessment: Check `MY_APP_ANTHROPIC_KEY` is exported in environment
- Notebook exercises use "FILL" placeholders for student completion
- For matplotlib display issues: Backend is set to 'TkAgg' in experiment runners

### Extending the Framework
The modular design in `ssl_framework.py` allows easy extension:
1. Implement new `PretextTask` subclasses for novel SSL objectives
2. Add dataset loaders by extending `DatasetInterface`
3. Configure network depth via `hidden_dims` parameter in `SSLNetwork`
4. Experiment runners accept command-line arguments for different modes