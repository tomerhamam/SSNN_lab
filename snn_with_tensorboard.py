#!/usr/bin/env python3
"""
Enhanced SNN Training with TensorBoard Integration

This script demonstrates how to integrate TensorBoard logging into existing
SNN training scripts for comprehensive monitoring and visualization.

Features:
- Real-time training metrics logging
- Model performance visualization
- System resource monitoring
- Experiment comparison capabilities
- Interpretability analysis integration
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass
import warnings
import time
import psutil
warnings.filterwarnings('ignore')

# Import TensorBoard and interpretability tools
from tensorboard_logger import TensorBoardLogger, initialize_global_logger
from interpretability_analyzer import InterpretabilityAnalyzer, initialize_global_analyzer

# Set random seeds for reproducibility
np.random.seed(42)

# Configure plotting
plt.style.use('default')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11

print("🔧 Enhanced SNN Environment setup complete!")
print("📊 TensorBoard integration ready!")
print("🧠 Interpretability tools loaded!")
print("🚀 Ready to start enhanced SSL training!")


def create_rotation_dataset(X: np.ndarray, rotations: Tuple[int, ...] = (0, 90, 180, 270)) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a dataset of rotated images for the pretext task.
    
    Args:
        X: Array of flattened images, shape (n_samples, 64)
        rotations: Tuple of rotation angles in degrees
    
    Returns:
        rot_X: Array of rotated images, shape (n_samples * len(rotations), 64)
        rot_y: Array of rotation labels, shape (n_samples * len(rotations),)
    """
    # Reshape flattened images back to 8x8
    images = X.reshape(-1, 8, 8)
    
    rot_images = []
    rot_labels = []
    
    for idx, angle in enumerate(rotations):
        # Calculate number of 90-degree rotations needed
        k = (angle // 90) % 4

        for img in images:
            # Rotate the image k times by 90 degrees
            rotated = np.rot90(img, k=k)
            # Flatten and add to lists
            rot_images.append(rotated.flatten())
            rot_labels.append(idx)

    return np.array(rot_images, dtype=np.float32), np.array(rot_labels, dtype=np.int64)


@dataclass
class EnhancedTwoLayerNet:
    """
    Enhanced two-layer neural network with TensorBoard integration.
    
    This network includes comprehensive logging capabilities for:
    - Training metrics (loss, accuracy, learning rate)
    - Model performance evaluation
    - System resource monitoring
    - Interpretability analysis
    """
    input_dim: int
    hidden_dim: int  
    output_dim: int
    learning_rate: float = 0.5
    
    def __post_init__(self):
        """Initialize network parameters."""
        # Use fixed seed for reproducible results
        rng = np.random.default_rng(0)
        
        # Initialize weights and biases
        self.W1 = rng.standard_normal((self.input_dim, self.hidden_dim)) * 0.01
        self.b1 = np.zeros(self.hidden_dim)
        self.W2 = rng.standard_normal((self.hidden_dim, self.output_dim)) * 0.01
        self.b2 = np.zeros(self.output_dim)
        
        # Initialize TensorBoard logger
        self.logger = None
        self.interpretability_analyzer = None
        
        # Training history for analysis
        self.training_history = {
            'losses': [],
            'accuracies': [],
            'learning_rates': [],
            'gradient_norms': [],
            'system_metrics': []
        }

    def set_tensorboard_logger(self, experiment_name: str = None):
        """Set up TensorBoard logging for this network."""
        self.logger = initialize_global_logger(
            log_dir="logs/snn_tensorboard",
            experiment_name=experiment_name or f"snn_experiment_{int(time.time())}"
        )
        
        # Log hyperparameters
        self.logger.log_hyperparameters({
            "model_type": "TwoLayerNet",
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "learning_rate": self.learning_rate,
            "activation": "tanh",
            "optimizer": "gradient_descent"
        })
        
        print(f"📊 TensorBoard logger initialized: {self.logger.experiment_name}")

    def set_interpretability_analyzer(self):
        """Set up interpretability analysis for this network."""
        # Create a PyTorch-like model wrapper for interpretability
        class ModelWrapper:
            def __init__(self, net):
                self.net = net
            
            def __call__(self, x):
                # Convert numpy to torch-like tensor
                import torch
                x_tensor = torch.FloatTensor(x)
                
                # Forward pass
                z1 = x_tensor @ torch.FloatTensor(self.net.W1) + torch.FloatTensor(self.net.b1)
                a1 = torch.tanh(z1)
                z2 = a1 @ torch.FloatTensor(self.net.W2) + torch.FloatTensor(self.net.b2)
                
                # Softmax
                exp_scores = torch.exp(z2 - torch.max(z2, dim=1, keepdims=True)[0])
                probs = exp_scores / torch.sum(exp_scores, dim=1, keepdims=True)
                
                return probs
        
        # Initialize analyzer with wrapper
        try:
            self.interpretability_analyzer = initialize_global_analyzer(
                ModelWrapper(self), device="cpu"
            )
            print("🧠 Interpretability analyzer initialized")
        except Exception as e:
            print(f"⚠️  Interpretability analyzer not available: {e}")
            self.interpretability_analyzer = None

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, Tuple]:
        """Forward pass through the network."""
        # Layer 1: Linear transformation + tanh activation
        z1 = X @ self.W1 + self.b1
        a1 = np.tanh(z1)

        # Layer 2: Linear transformation
        z2 = a1 @ self.W2 + self.b2

        # Softmax activation (numerically stable)
        exp_scores = np.exp(z2 - np.max(z2, axis=1, keepdims=True))
        probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
        
        # Cache intermediate values for backprop
        cache = (X, z1, a1, z2, probs)
        return probs, cache
    
    def backward(self, cache, y_true: np.ndarray):
        """Backward pass (backpropagation)."""
        X, z1, a1, z2, probs = cache
        n_samples = X.shape[0]
        
        # Convert labels to one-hot encoding
        one_hot = np.zeros_like(probs)
        one_hot[np.arange(n_samples), y_true] = 1
        
        # Gradients for output layer
        dz2 = (probs - one_hot) / n_samples
        dW2 = a1.T @ dz2
        db2 = dz2.sum(axis=0)
        
        # Gradients for hidden layer
        da1 = dz2 @ self.W2.T
        dz1 = da1 * (1.0 - np.tanh(z1)**2)  # derivative of tanh
        dW1 = X.T @ dz1
        db1 = dz1.sum(axis=0)
        
        return dW1, db1, dW2, db2
    
    def update_params(self, dW1, db1, dW2, db2):
        """Update parameters using gradients."""
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
    
    def log_system_metrics(self):
        """Log system resource metrics to TensorBoard."""
        if not self.logger:
            return
            
        try:
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Log to TensorBoard
            self.logger.log_system_metrics(
                cpu_usage=cpu_percent,
                memory_usage=memory_percent
            )
            
            # Store in history
            self.training_history['system_metrics'].append({
                'cpu': cpu_percent,
                'memory': memory_percent,
                'timestamp': time.time()
            })
            
        except Exception as e:
            print(f"⚠️  System metrics logging failed: {e}")
    
    def train(self, X, y, epochs=20, batch_size=128, verbose=True):
        """
        Train the network with comprehensive TensorBoard logging.
        
        This enhanced training method includes:
        - Real-time metric logging
        - System resource monitoring
        - Gradient norm tracking
        - Learning rate scheduling
        - Interpretability analysis
        """
        n_samples = X.shape[0]
        
        # Initialize TensorBoard if not already done
        if not self.logger:
            self.set_tensorboard_logger()
        
        # Initialize interpretability analyzer
        if not self.interpretability_analyzer:
            self.set_interpretability_analyzer()
        
        print(f"🚀 Starting enhanced training with TensorBoard logging...")
        print(f"📊 Experiment: {self.logger.experiment_name}")
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # Shuffle data
            idx = np.random.permutation(n_samples)
            X_shuf, y_shuf = X[idx], y[idx]
            
            epoch_loss = 0
            epoch_accuracy = 0
            n_batches = 0
            gradient_norms = []
            
            # Mini-batch training
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                X_batch = X_shuf[start:end]
                y_batch = y_shuf[start:end]
                
                # Forward pass
                probs, cache = self.forward(X_batch)
                
                # Calculate loss
                batch_loss = -np.log(probs[np.arange(len(y_batch)), y_batch] + 1e-8).mean()
                epoch_loss += batch_loss
                
                # Calculate batch accuracy
                batch_predictions = np.argmax(probs, axis=1)
                batch_accuracy = np.mean(batch_predictions == y_batch)
                epoch_accuracy += batch_accuracy
                
                # Backward pass and parameter update
                grads = self.backward(cache, y_batch)
                self.update_params(*grads)
                
                # Calculate gradient norms
                grad_norm = np.sqrt(sum(np.sum(g**2) for g in grads))
                gradient_norms.append(grad_norm)
                
                n_batches += 1
                
                # Log batch-level metrics to TensorBoard
                if self.logger and start % (batch_size * 5) == 0:  # Log every 5 batches
                    global_step = epoch * n_batches + start // batch_size
                    self.logger.log_scalar("batch/loss", batch_loss, global_step, "training")
                    self.logger.log_scalar("batch/accuracy", batch_accuracy, global_step, "training")
                    self.logger.log_scalar("batch/gradient_norm", grad_norm, global_step, "training")
            
            # Calculate epoch metrics
            avg_loss = epoch_loss / n_batches
            avg_accuracy = epoch_accuracy / n_batches
            avg_grad_norm = np.mean(gradient_norms)
            
            # Store in history
            self.training_history['losses'].append(avg_loss)
            self.training_history['accuracies'].append(avg_accuracy)
            self.training_history['learning_rates'].append(self.learning_rate)
            self.training_history['gradient_norms'].append(avg_grad_norm)
            
            # Log epoch-level metrics to TensorBoard
            if self.logger:
                self.logger.log_model_metrics(
                    model_name="snn_two_layer",
                    metrics={
                        "loss": avg_loss,
                        "accuracy": avg_accuracy,
                        "gradient_norm": avg_grad_norm,
                        "learning_rate": self.learning_rate
                    },
                    epoch=epoch,
                    phase="train"
                )
            
            # Log system metrics every 5 epochs
            if epoch % 5 == 0:
                self.log_system_metrics()
            
            # Learning rate scheduling
            if epoch > 0 and epoch % 10 == 0:
                self.learning_rate *= 0.9  # Decay learning rate
                if self.logger:
                    self.logger.log_scalar("learning_rate", self.learning_rate, epoch, "training")
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time
            
            if verbose and (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1:2d}/{epochs}: "
                      f"Loss = {avg_loss:.4f}, "
                      f"Accuracy = {avg_accuracy:.3f}, "
                      f"Grad Norm = {avg_grad_norm:.4f}, "
                      f"Time = {epoch_time:.2f}s")
        
        # Final evaluation and logging
        final_accuracy = self.evaluate(X, y)
        if self.logger:
            self.logger.log_text("training_summary", 
                               f"Final training accuracy: {final_accuracy:.3f}\n"
                               f"Total epochs: {epochs}\n"
                               f"Final learning rate: {self.learning_rate:.6f}")
        
        print(f"✅ Training completed! Final accuracy: {final_accuracy:.3f}")
        
        return self.training_history['losses'], self.training_history['accuracies']
    
    def predict(self, X):
        """Predict class labels."""
        probs, _ = self.forward(X)
        return probs.argmax(axis=1)
    
    def evaluate(self, X, y):
        """Evaluate accuracy on given data."""
        predictions = self.predict(X)
        return (predictions == y).mean()
    
    def hidden_representation(self, X):
        """Extract hidden layer features for transfer learning."""
        z1 = X @ self.W1 + self.b1
        return np.tanh(z1)
    
    def run_interpretability_analysis(self, X_sample, y_sample, feature_names=None):
        """
        Run comprehensive interpretability analysis on the trained model.
        
        Args:
            X_sample: Sample input data for analysis
            y_sample: Corresponding labels
            feature_names: Names of input features (optional)
        """
        if not self.interpretability_analyzer:
            print("⚠️  Interpretability analyzer not available")
            return None
        
        print("\n🧠 Running interpretability analysis...")
        
        try:
            # Get a sample for analysis
            sample_input = X_sample[0:1]  # Keep batch dimension
            sample_target = y_sample[0]
            
            print(f"Analyzing sample (true class: {sample_target})")
            
            # Feature importance analysis
            print("📊 Analyzing feature importance...")
            feature_importance = self.interpretability_analyzer.analyze_feature_importance(
                sample_input, method="integrated_gradients"
            )
            
            if "error" not in feature_importance:
                print(f"✅ Feature importance analysis completed")
                print(f"   Target class: {feature_importance['target_class']}")
                print(f"   Predicted class: {feature_importance['predicted_class']}")
                
                # Show top features if feature names provided
                if feature_names:
                    attributions = np.array(feature_importance['attributions']).flatten()
                    top_indices = np.argsort(np.abs(attributions))[-5:][::-1]
                    print("   Top 5 most important features:")
                    for i, idx in enumerate(top_indices):
                        print(f"     {i+1}. {feature_names[idx]}: {attributions[idx]:.4f}")
            
            # Decision boundary analysis
            print("🎯 Analyzing decision boundary...")
            decision_boundary = self.interpretability_analyzer.analyze_decision_boundary(
                sample_input, num_samples=30
            )
            
            if "error" not in decision_boundary:
                robustness = decision_boundary['perturbation_analysis']['robustness_ratio']
                print(f"✅ Decision boundary analysis completed")
                print(f"   Robustness ratio: {robustness:.4f}")
                
                if robustness < 0.8:
                    print("   ⚠️  Low robustness detected")
                else:
                    print("   ✅ Good robustness")
            
            # Generate comprehensive report
            report = self.interpretability_analyzer.create_interpretability_report(
                "snn_interpretability_report.json"
            )
            
            print(f"📋 Comprehensive report saved to 'snn_interpretability_report.json'")
            
            return {
                'feature_importance': feature_importance,
                'decision_boundary': decision_boundary,
                'report': report
            }
            
        except Exception as e:
            print(f"❌ Interpretability analysis failed: {e}")
            return None
    
    def close_logging(self):
        """Close TensorBoard logging and save final summary."""
        if self.logger:
            self.logger.close()
            print("📊 TensorBoard logging closed")


def run_enhanced_ssl_experiment(num_rotations=8, epochs=20, enable_interpretability=True):
    """
    Run enhanced SSL experiment with TensorBoard integration.
    
    Args:
        num_rotations: Number of rotation angles
        epochs: Number of training epochs
        enable_interpretability: Whether to run interpretability analysis
    """
    print("="*80)
    print("🚀 ENHANCED SSL EXPERIMENT WITH TENSORBOARD INTEGRATION")
    print("="*80)
    print(f"📊 Rotations: {num_rotations}")
    print(f"🔄 Epochs: {epochs}")
    print(f"🧠 Interpretability: {enable_interpretability}")
    print("="*80)
    
    # 1. Load and prepare data
    print("\n📊 Loading digits dataset...")
    digits = load_digits()
    X = digits.data.astype(np.float32) / 16.0
    y = digits.target
    
    print(f"   • Original dataset: {X.shape} with {len(np.unique(y))} classes")
    
    # Create rotation dataset
    print(f"\n🔄 Creating rotation dataset with {num_rotations} angles...")
    rotation_res_angle = 2 * np.pi / num_rotations
    rotations = np.arange(num_rotations) * rotation_res_angle * 360.0/(2*np.pi)
    rot_X, rot_y = create_rotation_dataset(X, rotations)
    
    print(f"   • Rotation dataset: {rot_X.shape}")
    print(f"   • Random baseline: {1/num_rotations:.1%}")
    
    # Split data
    print("\n✂️  Splitting data...")
    X_train, X_val, y_train, y_val = train_test_split(
        rot_X, rot_y, test_size=0.2, random_state=42
    )
    print(f"   • Training: {X_train.shape}")
    print(f"   • Validation: {X_val.shape}")
    
    # 2. Create and train enhanced model
    print(f"\n🧠 Creating enhanced neural network...")
    net = EnhancedTwoLayerNet(
        input_dim=64,
        hidden_dim=32,
        output_dim=num_rotations,
        learning_rate=0.3
    )
    
    # Train with TensorBoard logging
    print(f"\n🚀 Starting enhanced training...")
    losses, accuracies = net.train(
        X_train, y_train, 
        epochs=epochs, 
        batch_size=256, 
        verbose=True
    )
    
    # 3. Evaluate model
    val_acc = net.evaluate(X_val, y_val)
    print(f"\n🎯 Final validation accuracy: {val_acc:.3f}")
    
    random_baseline = 1.0 / num_rotations
    if val_acc > random_baseline:
        improvement = (val_acc - random_baseline) * 100
        print(f"✅ Beat random guessing ({random_baseline:.1%}) by {improvement:.1f} percentage points")
    else:
        print("❌ Performance didn't beat random guessing")
    
    # 4. Run interpretability analysis
    if enable_interpretability:
        print(f"\n🧠 Running interpretability analysis...")
        
        # Create feature names for interpretability
        feature_names = [f"pixel_{i:02d}" for i in range(64)]
        
        interpretability_results = net.run_interpretability_analysis(
            X_val[:10], y_val[:10], feature_names
        )
    
    # 5. Transfer learning evaluation
    print(f"\n🔄 Transfer learning evaluation...")
    
    # Get original digit data splits
    X_orig_train, X_orig_test, y_orig_train, y_orig_test = train_test_split(
        X, y, test_size=0.3, random_state=1
    )
    
    # Extract learned features
    ssl_features_train = net.hidden_representation(X_orig_train)
    ssl_features_test = net.hidden_representation(X_orig_test)
    
    # Train classifier on SSL features
    print("🚀 Training downstream classifiers...")
    clf_ssl = LogisticRegression(max_iter=200, random_state=42)
    clf_ssl.fit(ssl_features_train, y_orig_train)
    ssl_acc = clf_ssl.score(ssl_features_test, y_orig_test)
    
    # Train baseline classifier on raw pixels
    clf_baseline = LogisticRegression(max_iter=200, random_state=42)
    clf_baseline.fit(X_orig_train, y_orig_train)
    baseline_acc = clf_baseline.score(X_orig_test, y_orig_test)
    
    print(f"\n📊 Transfer Learning Results:")
    print(f"   🧠 SSL Features Accuracy: {ssl_acc:.3f}")
    print(f"   📸 Raw Pixels Accuracy: {baseline_acc:.3f}")
    print(f"   📈 SSL vs Baseline: {(ssl_acc - baseline_acc)*100:+.1f} percentage points")
    
    # Log transfer learning results to TensorBoard
    if net.logger:
        net.logger.log_model_metrics(
            model_name="transfer_learning",
            metrics={
                "ssl_accuracy": ssl_acc,
                "baseline_accuracy": baseline_acc,
                "improvement": ssl_acc - baseline_acc
            },
            epoch=0,
            phase="transfer"
        )
    
    # 6. Visualize results
    print(f"\n📈 Creating visualizations...")
    
    # Plot training curves
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss curve
    ax1.plot(losses, 'b-', linewidth=2, label='Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Cross-Entropy Loss')
    ax1.set_title('Training Loss Over Time')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Accuracy curve
    ax2.plot(accuracies, 'g-', linewidth=2, label='Training Accuracy')
    ax2.axhline(y=random_baseline, color='r', linestyle='--', alpha=0.7, label=f'Random ({random_baseline:.1%})')
    ax2.axhline(y=val_acc, color='orange', linestyle='--', alpha=0.7, label=f'Val Acc ({val_acc:.3f})')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training Accuracy Over Time')
    ax2.set_ylim([0, 1])
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Gradient norms
    if net.training_history['gradient_norms']:
        ax3.plot(net.training_history['gradient_norms'], 'purple', linewidth=2, label='Gradient Norm')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Gradient Norm')
        ax3.set_title('Gradient Norms Over Time')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
    
    # System metrics
    if net.training_history['system_metrics']:
        system_data = net.training_history['system_metrics']
        epochs_with_metrics = list(range(0, len(system_data) * 5, 5))
        cpu_usage = [m['cpu'] for m in system_data]
        memory_usage = [m['memory'] for m in system_data]
        
        ax4.plot(epochs_with_metrics, cpu_usage, 'red', linewidth=2, label='CPU %')
        ax4.plot(epochs_with_metrics, memory_usage, 'blue', linewidth=2, label='Memory %')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Usage %')
        ax4.set_title('System Resource Usage')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
    
    plt.tight_layout()
    plt.savefig('enhanced_snn_training_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 7. Final summary
    print(f"\n{'='*80}")
    print("📊 ENHANCED SSL EXPERIMENT SUMMARY")
    print(f"{'='*80}")
    print(f"🎯 Rotation Task Accuracy: {val_acc:.3f}")
    print(f"🧠 SSL Transfer Accuracy: {ssl_acc:.3f}")
    print(f"📸 Baseline Accuracy: {baseline_acc:.3f}")
    print(f"📈 SSL Improvement: {(ssl_acc - baseline_acc)*100:+.1f} percentage points")
    print(f"📊 TensorBoard Experiment: {net.logger.experiment_name if net.logger else 'N/A'}")
    
    if enable_interpretability and interpretability_results:
        print(f"🧠 Interpretability Analysis: Completed")
        print(f"📋 Report saved: snn_interpretability_report.json")
    
    print(f"\n📁 Generated files:")
    print(f"   • enhanced_snn_training_results.png")
    print(f"   • logs/snn_tensorboard/{net.logger.experiment_name}/ (TensorBoard logs)")
    if enable_interpretability:
        print(f"   • snn_interpretability_report.json")
    
    print(f"\n🚀 To view TensorBoard:")
    print(f"   tensorboard --logdir=logs/snn_tensorboard --port=6006")
    print(f"   Then open: http://localhost:6006")
    
    # Close logging
    net.close_logging()
    
    return {
        'num_rotations': num_rotations,
        'epochs': epochs,
        'rotation_acc': val_acc,
        'ssl_transfer_acc': ssl_acc,
        'baseline_acc': baseline_acc,
        'improvement': ssl_acc - baseline_acc,
        'net': net,
        'interpretability_results': interpretability_results if enable_interpretability else None
    }


def compare_enhanced_experiments():
    """Compare different configurations with TensorBoard integration."""
    configurations = [
        (4, 15),    # Easy task, few epochs
        (8, 20),    # Medium task
        (16, 25),   # Harder task
        (32, 30),   # Very hard task, more epochs
    ]
    
    results = []
    for num_rot, epochs in configurations:
        print(f"\n{'='*80}")
        print(f"🧪 ENHANCED EXPERIMENT: {num_rot} rotations, {epochs} epochs")
        print(f"{'='*80}")
        
        result = run_enhanced_ssl_experiment(
            num_rot, epochs, 
            enable_interpretability=(num_rot == 8)  # Only run interpretability for medium task
        )
        results.append(result)
        
        print(f"✅ Completed: {num_rot} rotations → SSL acc: {result['ssl_transfer_acc']:.3f}")
    
    # Summary
    print(f"\n{'='*80}")
    print("📊 ENHANCED COMPARISON SUMMARY")
    print(f"{'='*80}")
    print("Rotations | Epochs | Rotation Acc | SSL Transfer | Baseline | Improvement")
    print("-" * 80)
    for r in results:
        print(f"{r['num_rotations']:8d} | {r['epochs']:6d} | {r['rotation_acc']:11.3f} | {r['ssl_transfer_acc']:11.3f} | {r['baseline_acc']:8.3f} | {r['improvement']:+10.3f}")
    
    print(f"\n🚀 All experiments logged to TensorBoard!")
    print(f"   tensorboard --logdir=logs/snn_tensorboard --port=6006")
    
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "compare":
        # Run comparison across configurations
        results = compare_enhanced_experiments()
    else:
        # Run single experiment
        if len(sys.argv) >= 3:
            num_rotations = int(sys.argv[1])
            epochs = int(sys.argv[2])
        else:
            num_rotations = 8
            epochs = 20
        
        result = run_enhanced_ssl_experiment(
            num_rotations, epochs, 
            enable_interpretability=True
        )
        
        print(f"\n🎉 Enhanced experiment completed!")
        print(f"📊 Final results: {result['ssl_transfer_acc']:.3f} SSL accuracy vs {result['baseline_acc']:.3f} baseline")
        print(f"🚀 View results in TensorBoard: tensorboard --logdir=logs/snn_tensorboard --port=6006")
