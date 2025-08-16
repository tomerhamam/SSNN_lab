"""
SSL experiment with Fashion-MNIST dataset.
Fashion-MNIST has more complex patterns than digits, making it better for SSL demonstration.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for debugging
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Detect if running in debugger
import sys
DEBUG_MODE = hasattr(sys, 'gettrace') and sys.gettrace() is not None

# Configuration
np.random.seed(42)
if not DEBUG_MODE:
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 11

print("🔧 Environment setup complete!")
print(f"🐛 Debug mode: {DEBUG_MODE}")
print("🚀 Ready to start Fashion-MNIST SSL!")

# ============================================================================
# DATA LOADING
# ============================================================================

def load_fashion_mnist(n_samples=5000):
    """
    Load Fashion-MNIST dataset.
    Falls back to synthetic data if Fashion-MNIST not available.
    """
    try:
        # Try to load Fashion-MNIST using torchvision
        import torchvision
        import torchvision.transforms as transforms
        
        print("📦 Loading Fashion-MNIST from torchvision...")
        transform = transforms.Compose([transforms.ToTensor()])
        
        # Load training data
        dataset = torchvision.datasets.FashionMNIST(
            root='./data', 
            train=True, 
            download=True, 
            transform=transform
        )
        
        # Load test data
        test_dataset = torchvision.datasets.FashionMNIST(
            root='./data',
            train=False,
            download=True,
            transform=transform
        )
        
        # Extract subset for faster training
        if n_samples < len(dataset):
            indices = np.random.choice(len(dataset), n_samples, replace=False)
        else:
            indices = np.arange(len(dataset))
        
        X_train = []
        y_train = []
        for i in indices[:n_samples]:
            img, label = dataset[i]
            X_train.append(img.numpy().flatten())
            y_train.append(label)
        
        # Get test data
        X_test = []
        y_test = []
        test_samples = min(1000, len(test_dataset))
        for i in range(test_samples):
            img, label = test_dataset[i]
            X_test.append(img.numpy().flatten())
            y_test.append(label)
        
        X_train = np.array(X_train, dtype=np.float32)
        y_train = np.array(y_train)
        X_test = np.array(X_test, dtype=np.float32)
        y_test = np.array(y_test)
        
        print(f"✅ Loaded Fashion-MNIST: train={X_train.shape}, test={X_test.shape}")
        
        # Class names for Fashion-MNIST
        class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
                      'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        
        return X_train, y_train, X_test, y_test, class_names
        
    except ImportError:
        print("⚠️  Fashion-MNIST not available, using sklearn's load_digits as fallback...")
        from sklearn.datasets import load_digits
        
        # Load digits and upsample to 28x28
        digits = load_digits()
        X = digits.data.astype(np.float32) / 16.0
        y = digits.target
        
        # Upsample from 8x8 to 28x28 to match Fashion-MNIST size
        from scipy import ndimage
        X_upsampled = []
        for img in X:
            img_8x8 = img.reshape(8, 8)
            img_28x28 = ndimage.zoom(img_8x8, 3.5, order=1)  # 8 * 3.5 = 28
            X_upsampled.append(img_28x28.flatten())
        
        X = np.array(X_upsampled, dtype=np.float32)
        X = np.clip(X, 0, 1)  # Ensure values are in [0, 1]
        
        # Split into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Subsample if needed
        if n_samples < len(X_train):
            X_train = X_train[:n_samples]
            y_train = y_train[:n_samples]
        
        print(f"✅ Using upsampled digits: train={X_train.shape}, test={X_test.shape}")
        
        class_names = [str(i) for i in range(10)]
        
        return X_train, y_train, X_test, y_test, class_names

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def visualize_samples(images, labels=None, title="Sample Visualization", 
                     n_samples=10, figsize=(12, 2), class_names=None):
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
            img = img.reshape(28, 28)  # Fashion-MNIST is 28x28
        
        axes[i].imshow(img, cmap='gray')
        if labels is not None:
            if class_names is not None and isinstance(labels[i], (int, np.integer)):
                axes[i].set_title(class_names[labels[i]])
            else:
                axes[i].set_title(f"{labels[i]}")
        axes[i].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def create_rotation_dataset(X, rotations=(0, 90, 180, 270)):
    """Create dataset with rotated images."""
    images = X.reshape(-1, 28, 28)  # Fashion-MNIST is 28x28
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
    """Two-layer neural network for rotation prediction."""
    input_dim: int
    hidden_dim: int  
    output_dim: int
    learning_rate: float = 0.1
    
    def __post_init__(self):
        """Initialize weights and biases."""
        rng = np.random.default_rng(0)
        # Better initialization for larger networks
        scale = np.sqrt(2.0 / self.input_dim)
        self.W1 = rng.standard_normal((self.input_dim, self.hidden_dim)) * scale
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
    
    def train(self, X, y, epochs=20, batch_size=128, verbose=True):
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
# MAIN EXPERIMENT
# ============================================================================

def run_fashion_mnist_ssl(num_rotations=8, epochs=20, n_samples=5000, 
                         show_plots=True, verbose=True):
    """
    Run SSL experiment on Fashion-MNIST.
    
    Args:
        num_rotations: Number of rotation angles for pretext task
        epochs: Number of training epochs
        n_samples: Number of training samples to use
        show_plots: Whether to show visualizations
        verbose: Whether to print progress
    
    Returns:
        Dictionary with results
    """
    print("\n" + "="*60)
    print("🎯 SSL FASHION-MNIST EXPERIMENT")
    print("="*60)
    
    # 1. Load Fashion-MNIST data
    print("\n📊 Loading Fashion-MNIST dataset...")
    X_train, y_train, X_test, y_test, class_names = load_fashion_mnist(n_samples)
    print(f"   • Training samples: {X_train.shape}")
    print(f"   • Test samples: {X_test.shape}")
    print(f"   • Classes: {class_names}")
    
    # Show sample images
    if show_plots:
        sample_indices = np.random.choice(len(X_train), 10, replace=False)
        visualize_samples(
            X_train[sample_indices], 
            y_train[sample_indices], 
            "Fashion-MNIST Samples",
            n_samples=10,
            class_names=class_names
        )
    
    # 2. Create rotation dataset
    print(f"\n🔄 Creating rotation dataset with {num_rotations} angles...")
    rotation_res_angle = 2 * np.pi / num_rotations
    rotations = np.arange(num_rotations) * rotation_res_angle * 360.0/(2*np.pi)
    rot_X, rot_y = create_rotation_dataset(X_train, rotations)
    
    print(f"   • Rotation dataset: {rot_X.shape}")
    rotation_labels = [f"{angle:.0f}°" for angle in rotations]
    print(f"   • Rotation classes: {rotation_labels}")
    print(f"   • Random baseline: {1/num_rotations:.1%}")
    
    # 3. Split rotation data
    print("\n✂️  Splitting rotation data...")
    X_rot_train, X_rot_val, y_rot_train, y_rot_val = train_test_split(
        rot_X, rot_y, test_size=0.2, random_state=42
    )
    print(f"   • Training: {X_rot_train.shape}")
    print(f"   • Validation: {X_rot_val.shape}")
    
    # 4. Train SSL model
    print("\n🧠 Training neural network on rotation task...")
    print(f"📊 Task difficulty: {num_rotations} classes (random baseline: {1/num_rotations:.1%})")
    
    # Adjust hyperparameters based on dataset size and complexity
    hidden_dim = 128  # Larger hidden layer for Fashion-MNIST
    if num_rotations > 16:
        learning_rate = 0.05  # Lower LR for harder tasks
        print(f"🎯 Using lower learning rate ({learning_rate}) for harder task")
    else:
        learning_rate = 0.1
    
    net = TwoLayerNet(
        input_dim=784,  # 28*28
        hidden_dim=hidden_dim, 
        output_dim=num_rotations, 
        learning_rate=learning_rate
    )
    
    losses, accuracies = net.train(
        X_rot_train, y_rot_train, 
        epochs=epochs, 
        batch_size=128, 
        verbose=verbose
    )
    
    # 5. Evaluate rotation task
    val_acc = net.evaluate(X_rot_val, y_rot_val)
    print(f"\n🎯 Rotation task validation accuracy: {val_acc:.3f}")
    
    random_baseline = 1.0 / num_rotations
    if val_acc > random_baseline:
        improvement = (val_acc - random_baseline) * 100
        print(f"✅ Beat random guessing ({random_baseline:.1%}) by {improvement:.1f} percentage points")
    else:
        print("❌ Performance didn't beat random guessing")
    
    # 6. Plot training curves
    if show_plots and not DEBUG_MODE:
        print("\n📈 Plotting training curves...")
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
        ax2.axhline(y=random_baseline, color='r', linestyle='--', alpha=0.7, 
                   label=f'Random ({random_baseline:.1%})')
        ax2.axhline(y=val_acc, color='orange', linestyle='--', alpha=0.7, 
                   label=f'Val Acc ({val_acc:.3f})')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title(f'Training Accuracy ({num_rotations} classes)')
        ax2.set_ylim([0, 1])
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
    
    # 7. Transfer learning evaluation
    print("\n🔄 Transfer learning evaluation...")
    
    # Extract SSL features
    print("🔍 Extracting learned features...")
    ssl_features_train = net.hidden_representation(X_train)
    ssl_features_test = net.hidden_representation(X_test)
    
    print(f"   • SSL features shape: {ssl_features_train.shape}")
    print(f"   • Dimensionality: 784 → {ssl_features_train.shape[1]} features")
    
    # Train classifier on SSL features
    print("🚀 Training downstream classifier on SSL features...")
    clf_ssl = LogisticRegression(max_iter=100, random_state=42)
    clf_ssl.fit(ssl_features_train, y_train)
    ssl_acc = clf_ssl.score(ssl_features_test, y_test)
    
    # Train baseline classifier on raw pixels
    print("📸 Training baseline classifier on raw pixels...")
    clf_baseline = LogisticRegression(max_iter=100, random_state=42)
    clf_baseline.fit(X_train, y_train)
    baseline_acc = clf_baseline.score(X_test, y_test)
    
    # Compare results
    print(f"\n📊 Transfer Learning Results:")
    print(f"   🧠 SSL Features Accuracy: {ssl_acc:.3f}")
    print(f"   📸 Raw Pixels Accuracy: {baseline_acc:.3f}")
    print(f"   📈 SSL vs Baseline: {(ssl_acc - baseline_acc)*100:+.1f} percentage points")
    
    if ssl_acc > baseline_acc:
        print(f"   ✅ SSL features outperform raw pixels! 🎉")
    elif abs(ssl_acc - baseline_acc) < 0.02:
        print(f"   📊 SSL features perform similarly to raw pixels")
    else:
        print(f"   📝 Raw pixels perform better")
    
    return {
        'num_rotations': num_rotations,
        'epochs': epochs,
        'rotation_acc': val_acc,
        'ssl_transfer_acc': ssl_acc,
        'baseline_acc': baseline_acc,
        'improvement': ssl_acc - baseline_acc,
        'model': net
    }

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    if len(sys.argv) >= 3:
        num_rotations = int(sys.argv[1])
        epochs = int(sys.argv[2])
        n_samples = int(sys.argv[3]) if len(sys.argv) > 3 else 5000
    else:
        # Default configuration
        num_rotations = 8
        epochs = 20
        n_samples = 5000
    
    print(f"🧪 Running SSL experiment with:")
    print(f"   • Rotations: {num_rotations}")
    print(f"   • Epochs: {epochs}")
    print(f"   • Samples: {n_samples}")
    
    # Run experiment
    result = run_fashion_mnist_ssl(
        num_rotations=num_rotations,
        epochs=epochs,
        n_samples=n_samples,
        show_plots=True,
        verbose=True
    )
    
    print(f"\n🎉 Experiment completed!")
    print(f"📊 Final SSL accuracy: {result['ssl_transfer_acc']:.3f}")
    print(f"📊 Baseline accuracy: {result['baseline_acc']:.3f}")
    
    if result['improvement'] > 0:
        print(f"✨ SSL improved accuracy by {result['improvement']*100:.1f} percentage points!")