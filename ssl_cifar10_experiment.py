"""
SSL experiment adapted for CIFAR-10 dataset.
This should show clearer SSL benefits compared to digits.
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from dataclasses import dataclass

def load_cifar10_subset(n_samples=2000):
    """
    Load a subset of CIFAR-10 for quick experimentation.
    Falls back to synthetic data if CIFAR-10 not available.
    """
    try:
        # Try to load CIFAR-10
        import torchvision
        import torchvision.transforms as transforms
        
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = torchvision.datasets.CIFAR10(root='./data', train=True, 
                                               download=True, transform=transform)
        
        # Extract subset
        indices = np.random.choice(len(dataset), n_samples, replace=False)
        images = []
        labels = []
        
        for i in indices:
            img, label = dataset[i]
            # Convert to grayscale and flatten
            img_gray = np.mean(img.numpy(), axis=0)  # Average RGB channels
            images.append(img_gray.flatten())
            labels.append(label)
        
        X = np.array(images, dtype=np.float32)
        y = np.array(labels)
        
        print(f"✅ Loaded CIFAR-10 subset: {X.shape}")
        return X, y, (32, 32)  # image dimensions
        
    except ImportError:
        print("⚠️  CIFAR-10 not available, generating synthetic complex data...")
        
        # Generate more complex synthetic data
        np.random.seed(42)
        n_features = 32 * 32  # Same as CIFAR-10 grayscale
        X = np.random.randn(n_samples, n_features).astype(np.float32)
        
        # Add structure to make it more realistic
        for i in range(n_samples):
            # Add some spatial structure
            img = X[i].reshape(32, 32)
            # Add random shapes
            center_x, center_y = np.random.randint(8, 24, 2)
            radius = np.random.randint(3, 8)
            y_coords, x_coords = np.ogrid[:32, :32]
            mask = (x_coords - center_x)**2 + (y_coords - center_y)**2 <= radius**2
            img[mask] += np.random.randn() * 2
            X[i] = img.flatten()
        
        # Normalize
        X = (X - X.min()) / (X.max() - X.min())
        
        # Create 10 classes
        y = np.random.randint(0, 10, n_samples)
        
        print(f"✅ Generated synthetic complex data: {X.shape}")
        return X, y, (32, 32)

def create_rotation_dataset(X, rotations, img_shape):
    """Create rotation dataset for any image shape."""
    h, w = img_shape
    images = X.reshape(-1, h, w)
    rot_images = []
    rot_labels = []
    
    for idx, angle in enumerate(rotations):
        k = int(angle // 90) % 4  # Number of 90-degree rotations
        for img in images:
            rotated = np.rot90(img, k=k)
            rot_images.append(rotated.flatten())
            rot_labels.append(idx)
    
    return np.array(rot_images, dtype=np.float32), np.array(rot_labels, dtype=np.int64)

@dataclass
class TwoLayerNet:
    """Enhanced neural network for larger inputs."""
    input_dim: int
    hidden_dim: int  
    output_dim: int
    learning_rate: float = 0.1
    
    def __post_init__(self):
        rng = np.random.default_rng(0)
        scale = np.sqrt(2.0 / self.input_dim)  # Better initialization for larger networks
        self.W1 = rng.standard_normal((self.input_dim, self.hidden_dim)) * scale
        self.b1 = np.zeros(self.hidden_dim)
        self.W2 = rng.standard_normal((self.hidden_dim, self.output_dim)) * 0.01
        self.b2 = np.zeros(self.output_dim)

    def forward(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = np.tanh(z1)
        z2 = a1 @ self.W2 + self.b2
        
        # Stable softmax
        exp_scores = np.exp(z2 - np.max(z2, axis=1, keepdims=True))
        probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
        
        cache = (X, z1, a1, z2, probs)
        return probs, cache
    
    def backward(self, cache, y_true):
        X, z1, a1, z2, probs = cache
        n_samples = X.shape[0]
        
        one_hot = np.zeros_like(probs)
        one_hot[np.arange(n_samples), y_true] = 1
        
        dz2 = (probs - one_hot) / n_samples
        dW2 = a1.T @ dz2
        db2 = dz2.sum(axis=0)
        
        da1 = dz2 @ self.W2.T
        dz1 = da1 * (1.0 - np.tanh(z1)**2)
        dW1 = X.T @ dz1
        db1 = dz1.sum(axis=0)
        
        return dW1, db1, dW2, db2
    
    def update_params(self, dW1, db1, dW2, db2):
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
    
    def train(self, X, y, epochs=20, batch_size=128, verbose=True):
        n_samples = X.shape[0]
        losses = []
        accuracies = []
        
        for epoch in range(epochs):
            idx = np.random.permutation(n_samples)
            X_shuf, y_shuf = X[idx], y[idx]
            
            epoch_loss = 0
            n_batches = 0
            
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                X_batch = X_shuf[start:end]
                y_batch = y_shuf[start:end]
                
                probs, cache = self.forward(X_batch)
                batch_loss = -np.log(probs[np.arange(len(y_batch)), y_batch] + 1e-8).mean()
                epoch_loss += batch_loss
                n_batches += 1
                
                grads = self.backward(cache, y_batch)
                self.update_params(*grads)
            
            avg_loss = epoch_loss / n_batches
            train_acc = self.evaluate(X, y)
            losses.append(avg_loss)
            accuracies.append(train_acc)
            
            if verbose and (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1:2d}/{epochs}: Loss = {avg_loss:.4f}, Accuracy = {train_acc:.3f}")
        
        return losses, accuracies
    
    def predict(self, X):
        probs, _ = self.forward(X)
        return probs.argmax(axis=1)
    
    def evaluate(self, X, y):
        predictions = self.predict(X)
        return (predictions == y).mean()
    
    def hidden_representation(self, X):
        z1 = X @ self.W1 + self.b1
        return np.tanh(z1)

def run_complex_ssl_experiment(num_rotations=8, epochs=25, n_samples=2000):
    """Run SSL experiment on more complex data."""
    print(f"🧪 SSL Experiment on Complex Data")
    print(f"📊 Rotations: {num_rotations}, Epochs: {epochs}, Samples: {n_samples}")
    print("=" * 60)
    
    # Load complex dataset
    X, y, img_shape = load_cifar10_subset(n_samples)
    print(f"📊 Dataset: {X.shape}, Classes: {len(np.unique(y))}")
    
    # Create rotation dataset
    rotation_res_angle = 2 * np.pi / num_rotations
    rotations = np.arange(num_rotations) * rotation_res_angle * 360.0/(2*np.pi)
    rot_X, rot_y = create_rotation_dataset(X, rotations, img_shape)
    
    print(f"🔄 Rotation dataset: {rot_X.shape}")
    print(f"📊 Random baseline: {1/num_rotations:.1%}")
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        rot_X, rot_y, test_size=0.2, random_state=42
    )
    
    # Train SSL model
    print(f"\n🧠 Training SSL model...")
    hidden_dim = 128  # Larger network for complex data
    net = TwoLayerNet(
        input_dim=X.shape[1], 
        hidden_dim=hidden_dim, 
        output_dim=num_rotations,
        learning_rate=0.01  # Lower LR for complex data
    )
    
    losses, accuracies = net.train(X_train, y_train, epochs=epochs, batch_size=128)
    
    val_acc = net.evaluate(X_val, y_val)
    random_baseline = 1.0 / num_rotations
    
    print(f"\n🎯 Rotation task results:")
    print(f"   Validation accuracy: {val_acc:.3f}")
    print(f"   Beat random ({random_baseline:.1%}): {(val_acc-random_baseline)*100:+.1f} pp")
    
    # Transfer learning evaluation
    print(f"\n🔄 Transfer learning evaluation...")
    X_orig_train, X_orig_test, y_orig_train, y_orig_test = train_test_split(
        X, y, test_size=0.3, random_state=1
    )
    
    # SSL features
    ssl_features_train = net.hidden_representation(X_orig_train)
    ssl_features_test = net.hidden_representation(X_orig_test)
    
    clf_ssl = LogisticRegression(max_iter=1000, random_state=42)
    clf_ssl.fit(ssl_features_train, y_orig_train)
    ssl_acc = clf_ssl.score(ssl_features_test, y_orig_test)
    
    # Baseline (raw pixels)
    clf_baseline = LogisticRegression(max_iter=1000, random_state=42)
    clf_baseline.fit(X_orig_train, y_orig_train)
    baseline_acc = clf_baseline.score(X_orig_test, y_orig_test)
    
    print(f"\n📊 Transfer Learning Results:")
    print(f"   🧠 SSL Features: {ssl_acc:.3f}")
    print(f"   📸 Raw Pixels:   {baseline_acc:.3f}")
    print(f"   📈 Improvement:  {(ssl_acc-baseline_acc)*100:+.1f} percentage points")
    
    if ssl_acc > baseline_acc:
        print(f"   ✅ SSL features WIN! 🎉")
    else:
        print(f"   📊 Raw pixels still better")
    
    # Plot results
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(losses, 'b-', linewidth=2, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(accuracies, 'g-', linewidth=2, label='Training Accuracy')
    plt.axhline(y=random_baseline, color='r', linestyle='--', alpha=0.7, 
                label=f'Random ({random_baseline:.1%})')
    plt.axhline(y=val_acc, color='orange', linestyle='--', alpha=0.7, 
                label=f'Val Acc ({val_acc:.3f})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Rotation Training ({num_rotations} classes)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return {
        'ssl_acc': ssl_acc,
        'baseline_acc': baseline_acc,
        'improvement': ssl_acc - baseline_acc,
        'rotation_acc': val_acc
    }

if __name__ == "__main__":
    # Test different configurations on complex data
    configs = [
        (4, 20),   # Easy task
        (8, 25),   # Medium task  
        (16, 30),  # Hard task
    ]
    
    results = []
    for num_rot, epochs in configs:
        print(f"\n{'='*70}")
        result = run_complex_ssl_experiment(num_rot, epochs, n_samples=1500)
        results.append((num_rot, epochs, result))
        print(f"Result: SSL {result['ssl_acc']:.3f} vs Baseline {result['baseline_acc']:.3f}")
    
    print(f"\n{'='*70}")
    print("📊 SUMMARY ON COMPLEX DATA")
    print(f"{'='*70}")
    print("Rotations | Epochs | SSL Acc | Baseline | Improvement")
    print("-" * 50)
    for num_rot, epochs, result in results:
        print(f"{num_rot:8d} | {epochs:6d} | {result['ssl_acc']:7.3f} | {result['baseline_acc']:8.3f} | {result['improvement']:+10.3f}")