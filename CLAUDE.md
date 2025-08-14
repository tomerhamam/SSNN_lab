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
# Using uv (preferred)
uv pip install -r pyproject.toml

# Standard pip
pip install anthropic numpy scikit-learn matplotlib seaborn
```

## Architecture & Code Structure

### Two-Phase SSL Workflow
Both vision and time series modules follow the same pattern:
1. **Pretext Task Phase**: Train on self-supervised objective (rotation/reconstruction)
2. **Transfer Phase**: Extract learned features → evaluate on downstream classification

### Core Neural Network Classes

#### `TwoLayerNet` (Vision SSL)
- Location: Defined in both `tests/snn_lab.py` and `snn_lab_complete.ipynb`
- Architecture: input(64) → hidden(32) → output(4)
- Activation: tanh hidden layer, softmax output
- Training: Mini-batch SGD with manual backpropagation
- Key methods: `forward()`, `backward()`, `hidden_representation()`, `train()`

#### `Autoencoder` (Time Series SSL)
- Location: Defined in both files
- Architecture: input(50) → hidden(16) → output(50)
- Activation: tanh encoder, linear decoder
- Loss: Mean Squared Error (MSE)
- Key methods: `forward()`, `encode()`, `reconstruct()`, `train()`

### Data Generation Functions
- `load_digit_data()`: Loads sklearn digits, normalizes to [0,1]
- `create_rotation_dataset()`: Generates 4x data with rotation labels (0°, 90°, 180°, 270°)
- `generate_sine_sequences()`: Creates binary classification dataset (freq=1.0 vs freq=3.0)

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