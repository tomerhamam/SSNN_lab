"""
Quick test of SSL on more complex synthetic data.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def generate_complex_synthetic_data(n_samples=1000, img_size=16):
    """Generate more complex synthetic data with structure."""
    np.random.seed(42)
    
    X = []
    y = []
    
    for i in range(n_samples):
        # Create a 16x16 image with structure
        img = np.zeros((img_size, img_size))
        
        # Add different patterns for different classes
        class_id = i % 5  # 5 classes
        
        if class_id == 0:  # Circles
            center = np.random.randint(4, 12, 2)
            radius = np.random.randint(2, 4)
            y_coords, x_coords = np.ogrid[:img_size, :img_size]
            mask = (x_coords - center[0])**2 + (y_coords - center[1])**2 <= radius**2
            img[mask] = 1.0
            
        elif class_id == 1:  # Horizontal lines
            y_pos = np.random.randint(2, 14)
            x_start = np.random.randint(0, 8)
            x_end = np.random.randint(8, 16)
            img[y_pos, x_start:x_end] = 1.0
            
        elif class_id == 2:  # Vertical lines
            x_pos = np.random.randint(2, 14)
            y_start = np.random.randint(0, 8)
            y_end = np.random.randint(8, 16)
            img[y_start:y_end, x_pos] = 1.0
            
        elif class_id == 3:  # Squares
            top_left = np.random.randint(2, 10, 2)
            size = np.random.randint(3, 6)
            img[top_left[0]:top_left[0]+size, top_left[1]:top_left[1]+size] = 1.0
            
        else:  # Random noise pattern
            for _ in range(np.random.randint(5, 15)):
                x, y_pos = np.random.randint(0, img_size, 2)
                img[y_pos, x] = 1.0
        
        # Add some noise
        noise = np.random.randn(img_size, img_size) * 0.1
        img = np.clip(img + noise, 0, 1)
        
        X.append(img.flatten())
        y.append(class_id)
    
    return np.array(X, dtype=np.float32), np.array(y)

def create_rotation_dataset(X, rotations, img_size):
    """Create rotation dataset."""
    images = X.reshape(-1, img_size, img_size)
    rot_images = []
    rot_labels = []
    
    for idx, angle in enumerate(rotations):
        k = int(angle // 90) % 4
        for img in images:
            rotated = np.rot90(img, k=k)
            rot_images.append(rotated.flatten())
            rot_labels.append(idx)
    
    return np.array(rot_images, dtype=np.float32), np.array(rot_labels, dtype=np.int64)

class SimpleNet:
    """Simple network for complex data."""
    def __init__(self, input_dim, hidden_dim, output_dim, lr=0.01):
        self.lr = lr
        scale = np.sqrt(2.0 / input_dim)
        self.W1 = np.random.randn(input_dim, hidden_dim) * scale
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.01
        self.b2 = np.zeros(output_dim)
    
    def forward(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = np.tanh(z1)
        z2 = a1 @ self.W2 + self.b2
        exp_scores = np.exp(z2 - np.max(z2, axis=1, keepdims=True))
        probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
        return probs, a1
    
    def train_epoch(self, X, y):
        probs, a1 = self.forward(X)
        n = X.shape[0]
        
        # Backward pass
        one_hot = np.zeros_like(probs)
        one_hot[np.arange(n), y] = 1
        
        dz2 = (probs - one_hot) / n
        dW2 = a1.T @ dz2
        db2 = dz2.sum(axis=0)
        
        da1 = dz2 @ self.W2.T
        z1 = X @ self.W1 + self.b1
        dz1 = da1 * (1.0 - np.tanh(z1)**2)
        dW1 = X.T @ dz1
        db1 = dz1.sum(axis=0)
        
        # Update
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        
        return -np.log(probs[np.arange(n), y] + 1e-8).mean()
    
    def predict(self, X):
        probs, _ = self.forward(X)
        return probs.argmax(axis=1)
    
    def hidden_features(self, X):
        _, a1 = self.forward(X)
        return a1

def run_experiment(num_rotations=8, epochs=15):
    print(f"🧪 Testing SSL on complex synthetic data")
    print(f"📊 {num_rotations} rotations, {epochs} epochs")
    
    # Generate complex data
    X, y = generate_complex_synthetic_data(1000, img_size=16)
    print(f"📊 Dataset: {X.shape}, {len(np.unique(y))} classes")
    
    # Create rotations
    rotation_angles = np.linspace(0, 360, num_rotations, endpoint=False)
    rot_X, rot_y = create_rotation_dataset(X, rotation_angles, 16)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(rot_X, rot_y, test_size=0.2, random_state=42)
    
    # Train SSL model
    print(f"🧠 Training SSL model...")
    net = SimpleNet(input_dim=256, hidden_dim=64, output_dim=num_rotations, lr=0.01)
    
    for epoch in range(epochs):
        loss = net.train_epoch(X_train, y_train)
        if (epoch + 1) % 5 == 0:
            val_acc = (net.predict(X_val) == y_val).mean()
            print(f"Epoch {epoch+1}: Loss = {loss:.4f}, Val Acc = {val_acc:.3f}")
    
    # Final rotation accuracy
    rot_acc = (net.predict(X_val) == y_val).mean()
    random_baseline = 1.0 / num_rotations
    print(f"🎯 Rotation accuracy: {rot_acc:.3f} (random: {random_baseline:.3f})")
    
    # Transfer learning
    print(f"🔄 Transfer learning test...")
    X_orig_train, X_orig_test, y_orig_train, y_orig_test = train_test_split(
        X, y, test_size=0.3, random_state=1
    )
    
    # SSL features
    ssl_features_train = net.hidden_features(X_orig_train)
    ssl_features_test = net.hidden_features(X_orig_test)
    
    clf_ssl = LogisticRegression(max_iter=500, random_state=42)
    clf_ssl.fit(ssl_features_train, y_orig_train)
    ssl_acc = clf_ssl.score(ssl_features_test, y_orig_test)
    
    # Raw pixel baseline
    clf_baseline = LogisticRegression(max_iter=500, random_state=42)
    clf_baseline.fit(X_orig_train, y_orig_train)
    baseline_acc = clf_baseline.score(X_orig_test, y_orig_test)
    
    print(f"📊 Results:")
    print(f"   SSL features: {ssl_acc:.3f}")
    print(f"   Raw pixels:   {baseline_acc:.3f}")
    print(f"   Improvement:  {(ssl_acc-baseline_acc)*100:+.1f} pp")
    
    if ssl_acc > baseline_acc:
        print(f"   ✅ SSL WINS! 🎉")
    else:
        print(f"   📊 Raw pixels better")
    
    return ssl_acc, baseline_acc, ssl_acc - baseline_acc

if __name__ == "__main__":
    configs = [(4, 15), (8, 20), (16, 25)]
    
    print("Testing SSL on complex synthetic data...")
    print("=" * 50)
    
    results = []
    for num_rot, epochs in configs:
        print(f"\n--- {num_rot} rotations, {epochs} epochs ---")
        ssl_acc, baseline_acc, improvement = run_experiment(num_rot, epochs)
        results.append((num_rot, ssl_acc, baseline_acc, improvement))
    
    print(f"\n" + "="*50)
    print("📊 SUMMARY")
    print(f"="*50)
    print("Rotations | SSL Acc | Baseline | Improvement")
    print("-" * 40)
    for num_rot, ssl_acc, baseline_acc, improvement in results:
        print(f"{num_rot:8d} | {ssl_acc:7.3f} | {baseline_acc:8.3f} | {improvement:+10.3f}")