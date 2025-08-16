"""
Advanced SSL experimentation script to push accuracy higher.
Tests various hyperparameters and strategies systematically.
"""

import numpy as np
import sys
import time
from typing import List, Tuple, Dict
import json
from itertools import product

# Import base functionality
from ssl_fashion_mnist import load_fashion_mnist, create_rotation_dataset, TwoLayerNet
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# ============================================================================
# ADVANCED CONFIGURATIONS
# ============================================================================

# Hyperparameter search space
HYPERPARAMETER_GRID = {
    'rotation_experiments': [
        # (num_rotations, epochs, hidden_dim, learning_rate, batch_size, description)
        # Exploring rotation granularity
        (4, 20, 128, 0.1, 128, "4 rot baseline"),
        (6, 25, 128, 0.1, 128, "6 rot sweet spot?"),
        (8, 30, 128, 0.1, 128, "8 rot standard"),
        (12, 35, 128, 0.08, 128, "12 rot fine-grained"),
        (16, 40, 128, 0.05, 128, "16 rot challenging"),
        (24, 45, 128, 0.05, 64, "24 rot very fine"),
        
        # Deeper networks
        (8, 30, 256, 0.1, 128, "8 rot, wider network"),
        (8, 30, 512, 0.08, 128, "8 rot, very wide network"),
        (12, 35, 256, 0.08, 128, "12 rot, wider network"),
        
        # Longer training
        (8, 50, 128, 0.1, 128, "8 rot, extended training"),
        (8, 75, 128, 0.08, 128, "8 rot, long training"),
        (12, 60, 128, 0.08, 128, "12 rot, extended training"),
        
        # Smaller batch sizes (better gradients)
        (8, 30, 128, 0.1, 64, "8 rot, smaller batch"),
        (8, 30, 128, 0.1, 32, "8 rot, tiny batch"),
        (12, 35, 128, 0.08, 64, "12 rot, smaller batch"),
        
        # Learning rate variations
        (8, 30, 128, 0.2, 128, "8 rot, higher LR"),
        (8, 30, 128, 0.05, 128, "8 rot, lower LR"),
        (12, 35, 128, 0.15, 128, "12 rot, higher LR"),
    ],
    
    'augmentation_experiments': [
        # Additional pretext tasks beyond rotation
        ('noise', 30, 128, 0.1, 128, "Noise prediction task"),
        ('crop', 30, 128, 0.1, 128, "Crop position task"),
        ('flip', 30, 128, 0.1, 128, "Flip detection task"),
    ]
}

# ============================================================================
# ENHANCED NEURAL NETWORK
# ============================================================================

class EnhancedTwoLayerNet(TwoLayerNet):
    """Enhanced network with additional features."""
    
    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate=0.1, 
                 dropout_rate=0.0, momentum=0.0):
        super().__init__(input_dim, hidden_dim, output_dim, learning_rate)
        self.dropout_rate = dropout_rate
        self.momentum = momentum
        
        # Initialize momentum terms
        self.v_W1 = np.zeros_like(self.W1)
        self.v_b1 = np.zeros_like(self.b1)
        self.v_W2 = np.zeros_like(self.W2)
        self.v_b2 = np.zeros_like(self.b2)
    
    def forward_with_dropout(self, X, training=True):
        """Forward pass with optional dropout."""
        z1 = X @ self.W1 + self.b1
        a1 = np.tanh(z1)
        
        # Apply dropout during training
        if training and self.dropout_rate > 0:
            dropout_mask = np.random.binomial(1, 1-self.dropout_rate, a1.shape) / (1-self.dropout_rate)
            a1 = a1 * dropout_mask
        
        z2 = a1 @ self.W2 + self.b2
        exp_scores = np.exp(z2 - np.max(z2, axis=1, keepdims=True))
        probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
        
        cache = (X, z1, a1, z2, probs)
        return probs, cache
    
    def update_params_with_momentum(self, dW1, db1, dW2, db2):
        """Update parameters with momentum."""
        # Update velocity
        self.v_W1 = self.momentum * self.v_W1 - self.learning_rate * dW1
        self.v_b1 = self.momentum * self.v_b1 - self.learning_rate * db1
        self.v_W2 = self.momentum * self.v_W2 - self.learning_rate * dW2
        self.v_b2 = self.momentum * self.v_b2 - self.learning_rate * db2
        
        # Update parameters
        self.W1 += self.v_W1
        self.b1 += self.v_b1
        self.W2 += self.v_W2
        self.b2 += self.v_b2

# ============================================================================
# ALTERNATIVE PRETEXT TASKS
# ============================================================================

def create_noise_prediction_dataset(X, noise_levels=(0, 0.1, 0.2, 0.3)):
    """Create dataset for noise level prediction task."""
    noisy_images = []
    noise_labels = []
    
    for idx, noise_std in enumerate(noise_levels):
        for img in X:
            img_2d = img.reshape(28, 28)
            if noise_std > 0:
                noise = np.random.randn(*img_2d.shape) * noise_std
                noisy_img = np.clip(img_2d + noise, 0, 1)
            else:
                noisy_img = img_2d
            noisy_images.append(noisy_img.flatten())
            noise_labels.append(idx)
    
    return np.array(noisy_images, dtype=np.float32), np.array(noise_labels, dtype=np.int64)

def create_crop_position_dataset(X, crop_size=14):
    """Create dataset for crop position prediction (4 quadrants)."""
    cropped_images = []
    position_labels = []
    
    positions = [(0, 0), (14, 0), (0, 14), (14, 14)]  # Top-left, top-right, bottom-left, bottom-right
    
    for idx, (y, x) in enumerate(positions):
        for img in X:
            img_2d = img.reshape(28, 28)
            crop = img_2d[y:y+crop_size, x:x+crop_size]
            # Pad back to 28x28
            padded = np.zeros((28, 28))
            padded[:crop_size, :crop_size] = crop
            cropped_images.append(padded.flatten())
            position_labels.append(idx)
    
    return np.array(cropped_images, dtype=np.float32), np.array(position_labels, dtype=np.int64)

def create_combined_pretext_dataset(X, tasks=['rotation', 'noise']):
    """Combine multiple pretext tasks for richer representations."""
    combined_X = []
    combined_y = []
    
    if 'rotation' in tasks:
        rot_X, rot_y = create_rotation_dataset(X, [0, 90, 180, 270])
        combined_X.append(rot_X)
        combined_y.append(rot_y)
    
    if 'noise' in tasks:
        noise_X, noise_y = create_noise_prediction_dataset(X)
        combined_X.append(noise_X)
        combined_y.append(noise_y + 4)  # Offset labels
    
    if 'crop' in tasks:
        crop_X, crop_y = create_crop_position_dataset(X)
        combined_X.append(crop_X)
        combined_y.append(crop_y + 8)  # Offset labels
    
    return np.vstack(combined_X), np.hstack(combined_y)

# ============================================================================
# ADVANCED EXPERIMENT RUNNER
# ============================================================================

def run_advanced_experiment(config: Dict, n_samples: int = 5000, verbose: bool = True) -> Dict:
    """
    Run advanced SSL experiment with given configuration.
    
    Args:
        config: Dictionary with experiment configuration
        n_samples: Number of training samples
        verbose: Whether to print progress
        
    Returns:
        Dictionary with results
    """
    start_time = time.time()
    
    # Load data
    X_train, y_train, X_test, y_test, class_names = load_fashion_mnist(n_samples)
    
    # Create pretext task dataset
    if config['task_type'] == 'rotation':
        num_rotations = config['num_rotations']
        rotation_angles = np.linspace(0, 360, num_rotations, endpoint=False)
        pretext_X, pretext_y = create_rotation_dataset(X_train, rotation_angles)
        num_classes = num_rotations
    elif config['task_type'] == 'noise':
        pretext_X, pretext_y = create_noise_prediction_dataset(X_train)
        num_classes = 4
    elif config['task_type'] == 'crop':
        pretext_X, pretext_y = create_crop_position_dataset(X_train)
        num_classes = 4
    elif config['task_type'] == 'combined':
        pretext_X, pretext_y = create_combined_pretext_dataset(X_train, config.get('tasks', ['rotation', 'noise']))
        num_classes = len(np.unique(pretext_y))
    else:
        raise ValueError(f"Unknown task type: {config['task_type']}")
    
    # Split pretext data
    X_pre_train, X_pre_val, y_pre_train, y_pre_val = train_test_split(
        pretext_X, pretext_y, test_size=0.2, random_state=42
    )
    
    # Create and train model
    if config.get('use_enhanced', False):
        net = EnhancedTwoLayerNet(
            input_dim=784,
            hidden_dim=config['hidden_dim'],
            output_dim=num_classes,
            learning_rate=config['learning_rate'],
            dropout_rate=config.get('dropout_rate', 0.0),
            momentum=config.get('momentum', 0.0)
        )
    else:
        net = TwoLayerNet(
            input_dim=784,
            hidden_dim=config['hidden_dim'],
            output_dim=num_classes,
            learning_rate=config['learning_rate']
        )
    
    # Train on pretext task
    losses, accuracies = net.train(
        X_pre_train, y_pre_train,
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        verbose=False
    )
    
    pretext_acc = net.evaluate(X_pre_val, y_pre_val)
    
    # Extract features for transfer learning
    ssl_features_train = net.hidden_representation(X_train)
    ssl_features_test = net.hidden_representation(X_test)
    
    # Try different downstream classifiers
    results = {}
    
    # 1. Logistic Regression
    clf_lr = LogisticRegression(max_iter=200, random_state=42)
    clf_lr.fit(ssl_features_train, y_train)
    results['lr_ssl'] = clf_lr.score(ssl_features_test, y_test)
    
    # 2. Random Forest
    clf_rf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_rf.fit(ssl_features_train, y_train)
    results['rf_ssl'] = clf_rf.score(ssl_features_test, y_test)
    
    # 3. MLP Classifier
    clf_mlp = MLPClassifier(hidden_layer_sizes=(64,), max_iter=200, random_state=42)
    clf_mlp.fit(ssl_features_train, y_train)
    results['mlp_ssl'] = clf_mlp.score(ssl_features_test, y_test)
    
    # Baseline with raw pixels
    clf_baseline = LogisticRegression(max_iter=200, random_state=42)
    clf_baseline.fit(X_train, y_train)
    baseline_acc = clf_baseline.score(X_test, y_test)
    
    elapsed_time = time.time() - start_time
    
    # Best SSL result
    best_ssl = max(results['lr_ssl'], results['rf_ssl'], results['mlp_ssl'])
    
    if verbose:
        print(f"   Pretext: {pretext_acc:.3f}, SSL: {best_ssl:.3f}, Baseline: {baseline_acc:.3f}, "
              f"Improvement: {(best_ssl-baseline_acc)*100:+.1f}% ({elapsed_time:.1f}s)")
    
    return {
        'config': config,
        'pretext_acc': pretext_acc,
        'ssl_lr': results['lr_ssl'],
        'ssl_rf': results['rf_ssl'],
        'ssl_mlp': results['mlp_ssl'],
        'best_ssl': best_ssl,
        'baseline': baseline_acc,
        'improvement': best_ssl - baseline_acc,
        'elapsed_time': elapsed_time
    }

# ============================================================================
# HYPERPARAMETER SEARCH
# ============================================================================

def hyperparameter_search(n_samples: int = 5000):
    """Run comprehensive hyperparameter search."""
    print("\n" + "="*80)
    print("🔬 HYPERPARAMETER SEARCH")
    print("="*80)
    
    all_results = []
    
    # Test rotation experiments
    print("\n📊 Testing rotation configurations...")
    for num_rot, epochs, hidden, lr, batch, desc in HYPERPARAMETER_GRID['rotation_experiments']:
        print(f"\n{desc}:")
        config = {
            'task_type': 'rotation',
            'num_rotations': num_rot,
            'epochs': epochs,
            'hidden_dim': hidden,
            'learning_rate': lr,
            'batch_size': batch,
            'description': desc
        }
        result = run_advanced_experiment(config, n_samples)
        all_results.append(result)
    
    # Test alternative pretext tasks
    print("\n📊 Testing alternative pretext tasks...")
    for task_type, epochs, hidden, lr, batch, desc in HYPERPARAMETER_GRID['augmentation_experiments']:
        print(f"\n{desc}:")
        config = {
            'task_type': task_type,
            'epochs': epochs,
            'hidden_dim': hidden,
            'learning_rate': lr,
            'batch_size': batch,
            'description': desc
        }
        try:
            result = run_advanced_experiment(config, n_samples)
            all_results.append(result)
        except Exception as e:
            print(f"   Failed: {e}")
    
    # Test combined pretext tasks
    print("\n📊 Testing combined pretext tasks...")
    config = {
        'task_type': 'combined',
        'tasks': ['rotation', 'noise'],
        'epochs': 40,
        'hidden_dim': 256,
        'learning_rate': 0.08,
        'batch_size': 128,
        'description': 'Combined: rotation + noise'
    }
    try:
        result = run_advanced_experiment(config, n_samples)
        all_results.append(result)
    except Exception as e:
        print(f"   Failed: {e}")
    
    # Test enhanced network features
    print("\n📊 Testing enhanced network features...")
    enhanced_configs = [
        {'dropout_rate': 0.2, 'momentum': 0.9, 'desc': 'Dropout + Momentum'},
        {'dropout_rate': 0.3, 'momentum': 0.0, 'desc': 'Dropout only'},
        {'dropout_rate': 0.0, 'momentum': 0.95, 'desc': 'High momentum'},
    ]
    
    for enh_config in enhanced_configs:
        print(f"\n{enh_config['desc']}:")
        config = {
            'task_type': 'rotation',
            'num_rotations': 12,
            'epochs': 35,
            'hidden_dim': 256,
            'learning_rate': 0.1,
            'batch_size': 128,
            'use_enhanced': True,
            'dropout_rate': enh_config['dropout_rate'],
            'momentum': enh_config['momentum'],
            'description': enh_config['desc']
        }
        result = run_advanced_experiment(config, n_samples)
        all_results.append(result)
    
    return all_results

def ensemble_experiment(n_samples: int = 5000):
    """Try ensemble of multiple SSL models."""
    print("\n" + "="*80)
    print("🎭 ENSEMBLE EXPERIMENT")
    print("="*80)
    
    # Load data once
    X_train, y_train, X_test, y_test, _ = load_fashion_mnist(n_samples)
    
    # Train multiple SSL models with different pretext tasks
    pretext_configs = [
        ('rotation', 8, "8 rotations"),
        ('rotation', 12, "12 rotations"),
        ('noise', 4, "Noise levels"),
        ('crop', 4, "Crop positions"),
    ]
    
    all_features_train = []
    all_features_test = []
    
    for task_type, num_classes, desc in pretext_configs:
        print(f"\nTraining {desc} model...")
        
        if task_type == 'rotation':
            if num_classes == 8:
                angles = np.linspace(0, 360, 8, endpoint=False)
            else:
                angles = np.linspace(0, 360, 12, endpoint=False)
            pretext_X, pretext_y = create_rotation_dataset(X_train, angles)
        elif task_type == 'noise':
            pretext_X, pretext_y = create_noise_prediction_dataset(X_train)
        elif task_type == 'crop':
            pretext_X, pretext_y = create_crop_position_dataset(X_train)
        
        # Train model
        X_pre_train, X_pre_val, y_pre_train, y_pre_val = train_test_split(
            pretext_X, pretext_y, test_size=0.2, random_state=42
        )
        
        net = TwoLayerNet(784, 128, num_classes if task_type == 'rotation' else 4, 0.1)
        net.train(X_pre_train, y_pre_train, epochs=25, batch_size=128, verbose=False)
        
        # Extract features
        features_train = net.hidden_representation(X_train)
        features_test = net.hidden_representation(X_test)
        
        all_features_train.append(features_train)
        all_features_test.append(features_test)
        
        print(f"   Pretext accuracy: {net.evaluate(X_pre_val, y_pre_val):.3f}")
    
    # Concatenate all features
    ensemble_features_train = np.hstack(all_features_train)
    ensemble_features_test = np.hstack(all_features_test)
    
    print(f"\n📊 Ensemble feature dimensions: {ensemble_features_train.shape}")
    
    # Train classifier on ensemble features
    clf_ensemble = LogisticRegression(max_iter=300, random_state=42)
    clf_ensemble.fit(ensemble_features_train, y_train)
    ensemble_acc = clf_ensemble.score(ensemble_features_test, y_test)
    
    # Compare with baseline
    clf_baseline = LogisticRegression(max_iter=200, random_state=42)
    clf_baseline.fit(X_train, y_train)
    baseline_acc = clf_baseline.score(X_test, y_test)
    
    print(f"\n🎯 Results:")
    print(f"   Ensemble SSL: {ensemble_acc:.3f}")
    print(f"   Baseline: {baseline_acc:.3f}")
    print(f"   Improvement: {(ensemble_acc-baseline_acc)*100:+.1f}%")
    
    return ensemble_acc, baseline_acc

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run advanced experiments to push SSL accuracy higher."""
    print("🚀 ADVANCED SSL EXPERIMENTS")
    print("="*80)
    
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    else:
        mode = 'search'
    
    if mode == 'search':
        # Run hyperparameter search
        results = hyperparameter_search(n_samples=3000)
        
        # Find best configuration
        best_result = max(results, key=lambda x: x['best_ssl'])
        
        print("\n" + "="*80)
        print("🏆 BEST CONFIGURATION FOUND:")
        print("="*80)
        print(f"Configuration: {best_result['config']['description']}")
        print(f"SSL Accuracy: {best_result['best_ssl']:.3f}")
        print(f"Baseline: {best_result['baseline']:.3f}")
        print(f"Improvement: {best_result['improvement']*100:+.1f}%")
        print(f"\nDetails:")
        for key, value in best_result['config'].items():
            if key != 'description':
                print(f"   {key}: {value}")
        
        # Save results
        with open(f'advanced_ssl_results_{int(time.time())}.json', 'w') as f:
            json.dump([{k: v for k, v in r.items() if k != 'config'} for r in results], f, indent=2)
        
    elif mode == 'ensemble':
        # Run ensemble experiment
        ensemble_acc, baseline_acc = ensemble_experiment(n_samples=3000)
        
    elif mode == 'best':
        # Run the historically best configuration
        print("\n🎯 Running best known configuration...")
        config = {
            'task_type': 'rotation',
            'num_rotations': 12,
            'epochs': 50,
            'hidden_dim': 256,
            'learning_rate': 0.08,
            'batch_size': 64,
            'description': 'Best known config'
        }
        result = run_advanced_experiment(config, n_samples=5000, verbose=True)
        print(f"\nFinal results:")
        print(f"   Best SSL: {result['best_ssl']:.3f}")
        print(f"   Baseline: {result['baseline']:.3f}")
        print(f"   Improvement: {result['improvement']*100:+.1f}%")
    
    else:
        print(f"Unknown mode: {mode}")
        print("Available modes: search, ensemble, best")

if __name__ == "__main__":
    main()