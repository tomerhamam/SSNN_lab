# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Self-Supervised Neural Networks (SNN) lab implementation that demonstrates self-supervised learning principles across two domains:
- **Computer Vision**: Using handwritten digits (8×8 pixels) with rotation prediction as a pretext task
- **Time Series**: Using synthetic sine sequences with autoencoder-based reconstruction

The codebase uses scikit-learn and NumPy without external deep learning frameworks to keep implementations lightweight and educational.

## Key Commands

### Running the Lab
```bash
python tests/snn_lab.py  # Run all experiments and tests
```

### Running the Jupyter Tutorial
```bash
jupyter notebook snn_lab_tutorial.ipynb
```

### Package Management
- Uses `uv` package manager (pyproject.toml configured)
- Python >=3.12 required
- Dependencies: numpy, scikit-learn

## Architecture & Structure

### Core Components

1. **Vision Module** (`TwoLayerNet` class)
   - Simple 2-layer neural network with tanh activation
   - Trained on rotation prediction (0°, 90°, 180°, 270°)
   - Hidden representations transferred to digit classification

2. **Time Series Module** (`Autoencoder` class)
   - Single hidden layer autoencoder
   - Reconstructs noisy sine sequences
   - Embeddings used for frequency classification

3. **Data Generation**
   - `load_digit_data()`: Normalizes sklearn digits to [0,1]
   - `create_rotation_dataset()`: Generates rotated versions with labels
   - `generate_sine_sequences()`: Creates synthetic sine waves with two frequencies

### Key Design Patterns

- All neural networks implemented from scratch using NumPy
- Gradient descent with manual backpropagation
- Transfer learning workflow: pretext task → feature extraction → downstream classification
- Test-driven development with assertion-based validation

## Testing Strategy

The codebase includes three main test functions in `tests/snn_lab.py`:
- `test_rotation_accuracy()`: Verifies rotation classifier beats random chance (>40%)
- `test_digit_transfer()`: Ensures SSL features achieve ≥50% digit classification accuracy
- `test_time_series_transfer()`: Validates autoencoder embeddings preserve discriminative power

Run all tests via `run_all_tests()` or execute the main script.

## Important Notes

- The lab prioritizes educational clarity over performance
- Models intentionally use small architectures to demonstrate concepts
- Random seeds are fixed for reproducibility (rng seed=0 for vision, seed=1 for time series)
- Normalization is applied to time series data for training stability