# %% Setup and imports
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
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)

# Configure plotting
plt.style.use('default')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11
# %matplotlib inline
print("🔧 Environment setup complete!")
print(f"📊 NumPy version: {np.__version__}")
print("🚀 Ready to start learning SSL!")


# %% 
# Test your implementation
digits = load_digits()
print(f"📊 Dataset loaded: {digits.data.shape[0]} samples, {digits.data.shape[1]} features")
print(f"🔢 Classes: {np.unique(digits.target)}")

# visualize_samples(digits.images, digits.target, "Handwritten Digits Dataset", n_samples=10)

# Solution (run this cell if you need help)
def visualize_samples_solution(images, labels=None, title="Sample Visualization", n_samples=10, figsize=(12, 2)):
    """Reference implementation for visualization function."""
    n_show = min(n_samples, len(images))
    
    fig, axes = plt.subplots(1, n_show, figsize=figsize)
    
    if n_show == 1:
        axes = [axes]
    
    for i in range(n_show):
        img = images[i]
        if len(img.shape) == 1:
            img = img.reshape(8, 8)
        
        axes[i].imshow(img, cmap='gray')
        
        if labels is not None:
            axes[i].set_title(f"Label: {labels[i]}")
        
        axes[i].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# # Test the solution
# visualize_samples_solution(digits.images, digits.target, "Handwritten Digits - Solution", n_samples=10)

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
    # TODO: Reshape flattened images back to 8x8
    images = X.reshape(-1, 8, 8) # FILL: (-1, 8, 8)
    
    rot_images = []
    rot_labels = []
    
    for idx, angle in enumerate(rotations):
        # TODO: Calculate number of 90-degree rotations needed
        # Hint: np.rot90 rotates by 90 degrees k times
        k = (angle // 90) % 4  # FILL: (angle // 90) % 4

        for img in images:
            # TODO: Rotate the image k times by 90 degrees
            rotated = np.rot90(img, k=k)

            # TODO: Flatten and add to lists
            rot_images.append(rotated.flatten())
            rot_labels.append(idx)

    return np.array(rot_images, dtype=np.float32), np.array(rot_labels, dtype=np.int64)

# Load and normalize digit data
digits = load_digits()
X = digits.data.astype(np.float32) / 16.0  # Normalize to [0, 1]
y = digits.target

print(f"📊 Original dataset: {X.shape}")
print(f"🎯 Original classes: {len(np.unique(y))} digits (0-9)")

# Create rotation dataset using first 100 samples for testing
rot_X, rot_y = create_rotation_dataset(X[:100])
print(f"🔄 Rotation dataset: {rot_X.shape}")
print(f"🏷️ Rotation classes: {np.unique(rot_y)}")


# Solution and visualization
def create_rotation_dataset_solution(X: np.ndarray, rotations: Tuple[int, ...] = (0, 90, 180, 270)) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Create a dataset of rotated images.    
    '''
    images = X.reshape(-1, 8, 8)
    rot_images = []
    rot_labels = []
    
    for idx, angle in enumerate(rotations):
        k = (angle // 90) % 4
        for img in images:
            rotated = np.rot90(img, k=k)
            rot_images.append(rotated.flatten())
            rot_labels.append(idx)
    
    return np.array(rot_images, dtype=np.float32), np.array(rot_labels, dtype=np.int64)

# Create rotation dataset
rot_X, rot_y = create_rotation_dataset_solution(X[:100])
print(f"🔄 Rotation dataset shape: {rot_X.shape}")
print(f"🏷️ Rotation labels: {np.unique(rot_y)} (0=0°, 1=90°, 2=180°, 3=270°)")

# Visualize rotations of a single digit
sample_idx = 0  # First digit
rotations = [0, 90, 180, 270]
sample_rotations = []

for i in range(4):
    # Each rotation class contains the same digit rotated differently
    rot_sample_idx = sample_idx + i * 100  # 100 original samples per rotation
    sample_rotations.append(rot_X[rot_sample_idx])

visualize_samples_solution(
    sample_rotations, 
    [f"{angle}°" for angle in rotations],
    f"Rotations of Digit {y[sample_idx]}", 
    n_samples=4,
    figsize=(8, 2)
)


# %% Neural Network Implementation 
@dataclass
class TwoLayerNet:
    """A simple two-layer neural network for rotation prediction."""
    input_dim: int
    hidden_dim: int  
    output_dim: int
    learning_rate: float = 0.5
    
    def __post_init__(self):
        """Initialize network parameters."""
        # Use fixed seed for reproducible results
        rng = np.random.default_rng(0)
        
        # TODO: Initialize weights and biases
        # Hint: Use small random weights (multiply by 0.01) and zero biases
        self.W1 = rng.standard_normal((self.input_dim, self.hidden_dim)) * 0.01
        self.b1 = np.zeros(self.hidden_dim)
        self.W2 = rng.standard_normal((self.hidden_dim, self.output_dim)) * 0.01
        self.b2 = np.zeros(self.output_dim)

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, Tuple]:
        """Forward pass through the network."""
        # TODO: Implement forward pass
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
        """Backward pass (backpropagation) - provided for you."""
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
    
    def train(self, X, y, epochs=20, batch_size=128, verbose=True):
        """Train the network using mini-batch gradient descent."""
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
                
                # Forward pass
                probs, cache = self.forward(X_batch)
                
                # Calculate loss
                batch_loss = -np.log(probs[np.arange(len(y_batch)), y_batch] + 1e-8).mean()
                epoch_loss += batch_loss
                n_batches += 1
                
                # Backward pass and parameter update
                grads = self.backward(cache, y_batch)
                self.update_params(*grads)
            
            avg_loss = epoch_loss / n_batches
            losses.append(avg_loss)
            
            # Calculate accuracy
            train_acc = self.evaluate(X, y)
            accuracies.append(train_acc)
            
            if verbose and (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1:2d}/{epochs}: Loss = {avg_loss:.4f}, Accuracy = {train_acc:.3f}")
        
        return losses, accuracies
    
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

# Test network initialization
print("🧠 Testing network initialization...")
test_net = TwoLayerNet(input_dim=64, hidden_dim=32, output_dim=4, learning_rate=0.3)
print(f"✅ Network created: {test_net.input_dim} → {test_net.hidden_dim} → {test_net.output_dim}")
print(f"📊 Weight shapes: W1={test_net.W1.shape}, W2={test_net.W2.shape}")


# %% Training 
# Create full rotation dataset
print("🔄 Creating full rotation dataset...")
rot_X, rot_y = create_rotation_dataset_solution(X)  # Use all samples
print(f"📊 Full rotation dataset: {rot_X.shape}")

# TODO: Split into training and validation sets
# Hint: Use 80/20 split with random_state=42 for reproducibility
X_train, X_val, y_train, y_val = train_test_split(
    rot_X,  # FILL: rot_X
    rot_y,  # FILL: rot_y
    test_size=0.2,  # FILL: 0.2
    random_state=42
)

print(f"📊 Training set: {X_train.shape}")
print(f"📊 Validation set: {X_val.shape}")

# TODO: Create and configure the network
# Hint: Use input_dim=64, hidden_dim=32, output_dim=4
net = TwoLayerNet(
    input_dim=64,  # FILL: 64
    hidden_dim=32,  # FILL: 32
    output_dim=4,  # FILL: 4
    learning_rate=0.3
)

print("\n🚀 Starting training...")

# TODO: Train the network
# Hint: Use 15 epochs, batch_size=256
# losses, accuracies = net.train(X_train, y_train, epochs=___, batch_size=___)
losses, accuracies = net.train(X_train, y_train, epochs=15, batch_size=256)

# Solution: Train the network
X_train, X_val, y_train, y_val = train_test_split(
    rot_X, rot_y, test_size=0.2, random_state=42
)

net = TwoLayerNet(input_dim=64, hidden_dim=32, output_dim=4, learning_rate=0.3)

print("🚀 Training rotation classifier...")
losses, accuracies = net.train(X_train, y_train, epochs=15, batch_size=256)

# Evaluate on validation set
val_acc = net.evaluate(X_val, y_val)
print(f"\n🎯 Final validation accuracy: {val_acc:.3f}")

# Check if we beat random guessing (25% for 4 classes)
if val_acc > 0.25:
    print(f"✅ Great! We beat random guessing (25%)")
    if val_acc > 0.5:
        print(f"🎉 Excellent! The network learned meaningful rotation features!")
else:
    print(f"❌ Hmm, we didn't beat random guessing. Try adjusting hyperparameters.")


# %% Visualize the training progress
# Plot training curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Loss curve
ax1.plot(losses, 'b-', linewidth=2, label='Training Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Cross-Entropy Loss')
ax1.set_title('Training Loss Over Time')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Accuracy curve
ax2.plot(accuracies, 'g-', linewidth=2, label='Training Accuracy')
ax2.axhline(y=0.25, color='r', linestyle='--', alpha=0.7, label='Random Guess (25%)')
ax2.axhline(y=val_acc, color='orange', linestyle='--', alpha=0.7, label=f'Final Val Acc ({val_acc:.3f})')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title('Training Accuracy Over Time')
ax2.set_ylim([0, 1])
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.show()

print(f"📊 Training Summary:")
print(f"   • Final training accuracy: {accuracies[-1]:.3f}")
print(f"   • Final validation accuracy: {val_acc:.3f}")
print(f"   • Improvement over random: {(val_acc - 0.25) * 100:.1f} percentage points")