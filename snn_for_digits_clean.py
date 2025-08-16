"""
Clean, simplified SSL digit recognition script.
Self-supervised learning using rotation prediction as pretext task.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for debugging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# Configuration
np.random.seed(42)

# Detect if running in debugger
import sys
DEBUG_MODE = hasattr(sys, 'gettrace') and sys.gettrace() is not None

if not DEBUG_MODE:
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 11

print("🔧 Environment setup complete!")
print(f"🐛 Debug mode: {DEBUG_MODE}")
print("🚀 Ready to start SSL learning!")

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def visualize_samples(images, labels=None, title="Sample Visualization", n_samples=10, figsize=(12, 2)):
    """Visualize a grid of image samples."""
    if DEBUG_MODE:
        print(f"📊 {title} (plot skipped in debug mode)")
        return
        
    n_show = min(n_samples, len(images))
    fig, axes = plt.subplots(1, n_show, figsize=figsize)
    
    if n_show == 1:
        axes = [axes]
    
    for i in range(n_show):
        img = images[i]
        if len(img.shape) == 1:  # If flattened
            img = img.reshape(8, 8)
        
        axes[i].imshow(img, cmap='gray')
        if labels is not None:
            axes[i].set_title(f"Label: {labels[i]}")
        axes[i].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def create_rotation_dataset(X, rotations=(0, 90, 180, 270)):
    """Create dataset with rotated images."""
    images = X.reshape(-1, 8, 8)
    rot_images = []
    rot_labels = []
    
    for idx, angle in enumerate(rotations):
        k = (angle // 90) % 4  # Number of 90-degree rotations
        for img in images:
            rotated = np.rot90(img, k=k)
            rot_images.append(rotated.flatten())
            rot_labels.append(idx)
    
    return np.array(rot_images, dtype=np.float32), np.array(rot_labels, dtype=np.int64)

# ============================================================================
# NEURAL NETWORK
# ============================================================================

@dataclass
class TwoLayerNet:
    """Simple two-layer neural network for rotation prediction."""
    input_dim: int
    hidden_dim: int  
    output_dim: int
    learning_rate: float = 0.3
    
    def __post_init__(self):
        """Initialize weights and biases."""
        rng = np.random.default_rng(0)
        self.W1 = rng.standard_normal((self.input_dim, self.hidden_dim)) * 0.01
        self.b1 = np.zeros(self.hidden_dim)
        self.W2 = rng.standard_normal((self.hidden_dim, self.output_dim)) * 0.01
        self.b2 = np.zeros(self.output_dim)

    def forward(self, X):
        """Forward pass through network."""
        z1 = X @ self.W1 + self.b1
        a1 = np.tanh(z1)
        z2 = a1 @ self.W2 + self.b2
        
        # Softmax
        exp_scores = np.exp(z2 - np.max(z2, axis=1, keepdims=True))
        probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
        
        cache = (X, z1, a1, z2, probs)
        return probs, cache
    
    def backward(self, cache, y_true):
        """Backward pass (backpropagation)."""
        X, z1, a1, z2, probs = cache
        n_samples = X.shape[0]
        
        # One-hot encoding
        one_hot = np.zeros_like(probs)
        one_hot[np.arange(n_samples), y_true] = 1
        
        # Gradients
        dz2 = (probs - one_hot) / n_samples
        dW2 = a1.T @ dz2
        db2 = dz2.sum(axis=0)
        
        da1 = dz2 @ self.W2.T
        dz1 = da1 * (1.0 - np.tanh(z1)**2)
        dW1 = X.T @ dz1
        db1 = dz1.sum(axis=0)
        
        return dW1, db1, dW2, db2
    
    def update_params(self, dW1, db1, dW2, db2):
        """Update parameters."""
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
    
    def train(self, X, y, epochs=15, batch_size=256, verbose=True):
        """Train the network."""
        n_samples = X.shape[0]
        losses = []
        accuracies = []
        
        for epoch in range(epochs):
            # Shuffle data
            idx = np.random.permutation(n_samples)
            X_shuf, y_shuf = X[idx], y[idx]
            
            epoch_loss = 0
            n_batches = 0
            
            # Mini-batch training
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                X_batch = X_shuf[start:end]
                y_batch = y_shuf[start:end]
                
                # Forward + backward + update
                probs, cache = self.forward(X_batch)
                batch_loss = -np.log(probs[np.arange(len(y_batch)), y_batch] + 1e-8).mean()
                epoch_loss += batch_loss
                n_batches += 1
                
                grads = self.backward(cache, y_batch)
                self.update_params(*grads)
            
            # Track metrics
            avg_loss = epoch_loss / n_batches
            train_acc = self.evaluate(X, y)
            losses.append(avg_loss)
            accuracies.append(train_acc)
            
            if verbose and (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1:2d}/{epochs}: Loss = {avg_loss:.4f}, Accuracy = {train_acc:.3f}")
        
        return losses, accuracies
    
    def predict(self, X):
        """Predict class labels."""
        probs, _ = self.forward(X)
        return probs.argmax(axis=1)
    
    def evaluate(self, X, y):
        """Calculate accuracy."""
        predictions = self.predict(X)
        return (predictions == y).mean()
    
    def hidden_representation(self, X):
        """Extract hidden features for transfer learning."""
        z1 = X @ self.W1 + self.b1
        return np.tanh(z1)

# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def main(show_plots=True, epochs=15, verbose=True):
    print("\n" + "="*60)
    print("🎯 SSL DIGIT RECOGNITION - CLEAN VERSION")
    print("="*60)
    
    # 1. Load and prepare data
    print("\n📊 Loading digits dataset...")
    digits = load_digits()
    X = digits.data.astype(np.float32) / 16.0  # Normalize to [0, 1]
    y = digits.target
    print(f"   • Original dataset: {X.shape} with {len(np.unique(y))} classes")
    
    # Show sample digits
    # visualize_samples(digits.images, digits.target, "Original Handwritten Digits", n_samples=10)
    
    # 2. Create rotation dataset
    print("\n🔄 Creating rotation dataset...")
    num_of_rotations = 32
    rotation_res_angle = 2 * np.pi / num_of_rotations
    # rotations = [k * rotation_res_angle for k in range(num_of_rotations)]*360.0/2/np.pi
    rotations = np.arange(num_of_rotations) * rotation_res_angle * 360.0/(2*np.pi)
    rot_X, rot_y = create_rotation_dataset(X, rotations)
    print(f"   • Rotation dataset: {rot_X.shape}")
    rotation_labels = [f"{angle:.0f}°" for angle in rotations]
    print(f"   • Rotation classes: {np.unique(rot_y)} ({', '.join([f'{i}={label}' for i, label in enumerate(rotation_labels)])})")
    
    # Show rotation example
    sample_idx = 0
    sample_rotations = []
    for i in range(num_of_rotations):
        rot_sample_idx = sample_idx + i * len(X)
        sample_rotations.append(rot_X[rot_sample_idx])
    
    if show_plots:
        visualize_samples(
            sample_rotations, 
            rotation_labels,
            f"Rotations of Digit {y[sample_idx]}", 
            n_samples=num_of_rotations,
            figsize=(min(12, num_of_rotations * 1.5), 2)
        )
    
    # 3. Split data
    print("\n✂️  Splitting data...")
    X_train, X_val, y_train, y_val = train_test_split(
        rot_X, rot_y, test_size=0.2, random_state=42
    )
    print(f"   • Training: {X_train.shape}")
    print(f"   • Validation: {X_val.shape}")
    
    # 4. Create and train model
    print("\n🧠 Training neural network...")
    print(f"📊 Task difficulty: {num_of_rotations} classes (random baseline: {1/num_of_rotations:.1%})")
    
    # Adjust learning rate and epochs based on task difficulty
    if num_of_rotations > 16:
        learning_rate = 0.1  # Lower LR for harder tasks
        print(f"🎯 Using lower learning rate ({learning_rate}) for harder task")
    else:
        learning_rate = 0.3
    
    net = TwoLayerNet(input_dim=64, hidden_dim=32, output_dim=num_of_rotations, learning_rate=learning_rate)
    losses, accuracies = net.train(X_train, y_train, epochs=epochs, batch_size=256, verbose=verbose)
    
    # 5. Evaluate
    val_acc = net.evaluate(X_val, y_val)
    print(f"\n🎯 Final validation accuracy: {val_acc:.3f}")
    
    random_baseline = 1.0 / num_of_rotations
    if val_acc > random_baseline:
        improvement = (val_acc - random_baseline) * 100
        print(f"✅ Beat random guessing ({random_baseline:.1%}) by {improvement:.1f} percentage points")
        if val_acc > 0.5:
            print("🎉 Excellent! Network learned meaningful rotation features!")
    else:
        print("❌ Performance didn't beat random guessing")
    
    # 6. Plot training progress
    print("\n📈 Plotting training curves...")
    if show_plots and not DEBUG_MODE:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss curve
        ax1.plot(losses, 'b-', linewidth=2, label='Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Cross-Entropy Loss')
        ax1.set_title('Training Loss')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Accuracy curve
        ax2.plot(accuracies, 'g-', linewidth=2, label='Training Accuracy')
        ax2.axhline(y=random_baseline, color='r', linestyle='--', alpha=0.7, label=f'Random ({random_baseline:.1%})')
        ax2.axhline(y=val_acc, color='orange', linestyle='--', alpha=0.7, label=f'Val Acc ({val_acc:.3f})')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title(f'Training Accuracy ({num_of_rotations} classes)')
        ax2.set_ylim([0, 1])
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
    else:
        print("📊 Training curves (plot skipped)")
        print(f"📈 Final loss: {losses[-1]:.4f}, Final accuracy: {accuracies[-1]:.3f}")
        print(f"📊 Training progress: {accuracies[0]:.3f} → {accuracies[-1]:.3f} (Δ{accuracies[-1]-accuracies[0]:+.3f})")
    
    # 7. Transfer learning demo
    print("\n🔄 Transfer learning demonstration...")
    try:
        # Get original digit data splits
        X_orig_train, X_orig_test, y_orig_train, y_orig_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Extract learned features
        features_train = net.hidden_representation(X_orig_train)
        features_test = net.hidden_representation(X_orig_test)
        
        # Train classifier on learned features
        clf = LogisticRegression(random_state=42, max_iter=1000)
        clf.fit(features_train, y_orig_train)
        transfer_acc = clf.score(features_test, y_orig_test)
        
        print(f"   • Transfer learning accuracy: {transfer_acc:.3f}")
        
        # Compare with random features
        random_features = np.random.randn(*features_test.shape)
        clf_random = LogisticRegression(random_state=42, max_iter=1000)
        clf_random.fit(np.random.randn(*features_train.shape), y_orig_train)
        random_acc = clf_random.score(random_features, y_orig_test)
        
        print(f"   • Random features baseline: {random_acc:.3f}")
        print(f"   • Improvement: {(transfer_acc - random_acc) * 100:.1f} percentage points")
        
        if transfer_acc > random_acc:
            print("✅ Learned features are useful for digit classification!")
        
    except ImportError:
        print("   ⚠️  Scikit-learn required for transfer learning demo")
    
    print(f"\n🎉 SSL workflow completed successfully!")
    
    # Transfer Learning Analysis

    print("🔄 Setting up transfer learning experiment...")

    # TODO: Extract hidden features for all original digits
    ssl_features = net.hidden_representation(X) # FILL: net.hidden_representation(X)

    print(f"📊 Original digit data: {X.shape}")
    print(f"🧠 SSL features: {ssl_features.shape}")
    print(f"📉 Dimensionality reduction: {X.shape[1]} → {ssl_features.shape[1]} features")

    # TODO: Split data for downstream classification
    # Create train/test splits for both SSL features and raw pixels
    X_train_ssl, X_test_ssl, y_train_ssl, y_test_ssl = train_test_split(
        ssl_features,  # FILL: ssl_features
        y,  # FILL: y
        test_size=0.3,
        random_state=1
    )

    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X,  # FILL: X
        y,  # FILL: y
        test_size=0.3,
        random_state=1
    )

    print(f"\n📊 Downstream task splits:")
    print(f"   • SSL features train/test: {X_train_ssl.shape} / {X_test_ssl.shape}")
    print(f"   • Raw pixels train/test: {X_train_raw.shape} / {X_test_raw.shape}")

    # Solution and training
    ssl_features = net.hidden_representation(X)

    X_train_ssl, X_test_ssl, y_train_ssl, y_test_ssl = train_test_split(
        ssl_features, y, test_size=0.3, random_state=1
    )

    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X, y, test_size=0.3, random_state=1
    )

    print("🚀 Training downstream classifiers...")

    # Train classifier on SSL features
    clf_ssl = LogisticRegression(max_iter=200, random_state=42)
    clf_ssl.fit(X_train_ssl, y_train_ssl)
    ssl_acc = clf_ssl.score(X_test_ssl, y_test_ssl)

    # Train baseline classifier on raw pixels
    clf_baseline = LogisticRegression(max_iter=200, random_state=42)
    clf_baseline.fit(X_train_raw, y_train_raw)
    baseline_acc = clf_baseline.score(X_test_raw, y_test_raw)

    print(f"\n📊 Transfer Learning Results:")
    print(f"   🧠 SSL Features Accuracy: {ssl_acc:.3f}")
    print(f"   📸 Raw Pixels Accuracy: {baseline_acc:.3f}")
    print(f"   📈 SSL vs Baseline: {(ssl_acc - baseline_acc)*100:+.1f} percentage points")

    if ssl_acc > baseline_acc:
        print(f"   ✅ SSL features outperform raw pixels!")
    elif abs(ssl_acc - baseline_acc) < 0.02:
        print(f"   📊 SSL features perform similarly to raw pixels")
    else:
        print(f"   📝 Raw pixels perform better (dataset might be too simple for SSL to shine)")


    # Detailed performence Analysis
    # Generate detailed performance comparison
    ssl_pred = clf_ssl.predict(X_test_ssl)
    baseline_pred = clf_baseline.predict(X_test_raw)

    # Confusion matrices
    if show_plots and not DEBUG_MODE:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # SSL features confusion matrix
        cm_ssl = confusion_matrix(y_test_ssl, ssl_pred)
        sns.heatmap(cm_ssl, annot=True, fmt='d', cmap='Blues', ax=ax1, cbar=False)
        ax1.set_title(f'SSL Features (Acc: {ssl_acc:.3f})')
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('True')

        # Baseline confusion matrix  
        cm_baseline = confusion_matrix(y_test_raw, baseline_pred)
        sns.heatmap(cm_baseline, annot=True, fmt='d', cmap='Oranges', ax=ax2, cbar=False)
        ax2.set_title(f'Raw Pixels (Acc: {baseline_acc:.3f})')
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('True')

        plt.tight_layout()
        plt.show()
    else:
        print("📊 Confusion matrices (plots skipped)")

    # Per-class performance
    print("\n📊 Per-class Performance Comparison:")
    print("\nSSL Features:")
    print(classification_report(y_test_ssl, ssl_pred, target_names=[str(i) for i in range(10)]))

    print("\nRaw Pixels:")
    print(classification_report(y_test_raw, baseline_pred, target_names=[str(i) for i in range(10)]))

    return net, val_acc

if __name__ == "__main__":
    # Experiment with different configurations
    print("🧪 Choose experiment:")
    print("1. Quick test (4 rotations, 15 epochs)")
    print("2. Harder task (32 rotations, 30 epochs)")
    print("3. Custom configuration")
    
    import sys
    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        choice = "2"  # Default to harder task
    
    if choice == "1":
        # Quick test
        num_rotations = 4
        epochs = 15
        print("🚀 Running quick test...")
    elif choice == "2":
        # Harder task - needs more epochs
        num_rotations = 32  
        epochs = 30
        print("🚀 Running harder task with more epochs...")
    else:
        # Custom - you can modify these
        num_rotations = 32
        epochs = 25
        print("🚀 Running custom configuration...")
    
    # Temporarily override the hardcoded values in main()
    original_main = main
    def main_wrapper(show_plots=True, epochs=epochs, verbose=True):
        global num_of_rotations
        # This is a hack - normally you'd pass num_rotations as parameter
        import types
        main_code = original_main.__code__
        main_globals = original_main.__globals__.copy()
        main_globals['num_of_rotations'] = num_rotations
        new_main = types.FunctionType(main_code, main_globals)
        return new_main(show_plots, epochs, verbose)
    
    model, accuracy = main_wrapper(show_plots=True, epochs=epochs)
    
    print(f"\n🎯 Experiment completed!")
    print(f"📊 Configuration: {num_rotations} rotations, {epochs} epochs")
    print(f"🎯 Final accuracy: {accuracy:.3f}")