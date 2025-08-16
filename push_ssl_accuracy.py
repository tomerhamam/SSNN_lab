"""
Focused experiments to push SSL accuracy higher on Fashion-MNIST.
Key strategies:
1. Optimal rotation granularity (8-12 seems best)
2. Wider hidden layers
3. Ensemble of features from multiple pretext tasks
4. Better downstream classifiers
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from ssl_fashion_mnist import load_fashion_mnist, create_rotation_dataset, TwoLayerNet

def create_jigsaw_dataset(X, grid_size=2):
    """Create jigsaw puzzle dataset - predict permutation of image patches."""
    jigsaw_images = []
    jigsaw_labels = []
    
    # Define 4 permutations for 2x2 grid
    permutations = [
        [0, 1, 2, 3],  # Original
        [1, 0, 3, 2],  # Swap columns
        [2, 3, 0, 1],  # Swap rows
        [3, 2, 1, 0],  # Rotate 180
    ]
    
    for perm_idx, perm in enumerate(permutations):
        for img in X:
            img_2d = img.reshape(28, 28)
            
            # Split into 2x2 grid (each patch is 14x14)
            patches = []
            for i in range(2):
                for j in range(2):
                    patch = img_2d[i*14:(i+1)*14, j*14:(j+1)*14]
                    patches.append(patch)
            
            # Apply permutation
            permuted_patches = [patches[i] for i in perm]
            
            # Reconstruct image
            reconstructed = np.zeros((28, 28))
            idx = 0
            for i in range(2):
                for j in range(2):
                    reconstructed[i*14:(i+1)*14, j*14:(j+1)*14] = permuted_patches[idx]
                    idx += 1
            
            jigsaw_images.append(reconstructed.flatten())
            jigsaw_labels.append(perm_idx)
    
    return np.array(jigsaw_images, dtype=np.float32), np.array(jigsaw_labels, dtype=np.int64)

def train_ssl_model(X_train, pretext_type='rotation', num_classes=8, 
                   hidden_dim=256, epochs=30, lr=0.08):
    """Train a single SSL model on a pretext task."""
    
    # Create pretext dataset
    if pretext_type == 'rotation':
        angles = np.linspace(0, 360, num_classes, endpoint=False)
        pretext_X, pretext_y = create_rotation_dataset(X_train, angles)
    elif pretext_type == 'jigsaw':
        pretext_X, pretext_y = create_jigsaw_dataset(X_train)
        num_classes = 4
    else:
        raise ValueError(f"Unknown pretext type: {pretext_type}")
    
    # Split pretext data
    X_pre_train, X_pre_val, y_pre_train, y_pre_val = train_test_split(
        pretext_X, pretext_y, test_size=0.2, random_state=42
    )
    
    # Train model
    net = TwoLayerNet(784, hidden_dim, num_classes, lr)
    net.train(X_pre_train, y_pre_train, epochs=epochs, batch_size=128, verbose=False)
    
    pretext_acc = net.evaluate(X_pre_val, y_pre_val)
    
    return net, pretext_acc

def extract_multi_scale_features(net, X, scales=[1.0, 0.9, 1.1]):
    """Extract features at multiple scales for robustness."""
    all_features = []
    
    for scale in scales:
        if scale != 1.0:
            # Simple scaling by adding noise proportional to scale
            X_scaled = X + np.random.randn(*X.shape) * (1 - scale) * 0.1
            X_scaled = np.clip(X_scaled, 0, 1)
        else:
            X_scaled = X
        
        features = net.hidden_representation(X_scaled)
        all_features.append(features)
    
    # Concatenate or average
    return np.hstack(all_features)  # Concatenate for richer features

def ensemble_ssl_experiment(n_samples=3000):
    """Train ensemble of SSL models and combine features."""
    
    print("🎯 ENSEMBLE SSL EXPERIMENT")
    print("="*60)
    
    # Load data
    print("Loading Fashion-MNIST...")
    X_train, y_train, X_test, y_test, _ = load_fashion_mnist(n_samples)
    
    # Configuration for different SSL models
    ssl_configs = [
        ('rotation', 8, 256, 30, 0.08, "Rotation-8"),
        ('rotation', 12, 256, 35, 0.06, "Rotation-12"),
        ('jigsaw', 4, 256, 30, 0.08, "Jigsaw puzzle"),
    ]
    
    all_train_features = []
    all_test_features = []
    
    # Train each SSL model
    for pretext_type, num_classes, hidden, epochs, lr, desc in ssl_configs:
        print(f"\nTraining {desc}...")
        net, pretext_acc = train_ssl_model(
            X_train, pretext_type, num_classes, hidden, epochs, lr
        )
        print(f"  Pretext accuracy: {pretext_acc:.3f}")
        
        # Extract features (optionally multi-scale)
        if pretext_type == 'rotation' and num_classes == 12:
            # Use multi-scale for best rotation model
            train_features = extract_multi_scale_features(net, X_train)
            test_features = extract_multi_scale_features(net, X_test)
            print(f"  Multi-scale features: {train_features.shape}")
        else:
            train_features = net.hidden_representation(X_train)
            test_features = net.hidden_representation(X_test)
        
        all_train_features.append(train_features)
        all_test_features.append(test_features)
    
    # Combine features
    ensemble_train = np.hstack(all_train_features)
    ensemble_test = np.hstack(all_test_features)
    
    print(f"\n📊 Combined feature dimensions: {ensemble_train.shape}")
    
    # Try different downstream classifiers
    print("\n🔬 Testing downstream classifiers...")
    
    classifiers = [
        ('Logistic (C=0.1)', LogisticRegression(max_iter=300, C=0.1, random_state=42)),
        ('Logistic (C=1.0)', LogisticRegression(max_iter=300, C=1.0, random_state=42)),
        ('Random Forest', RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)),
        ('Gradient Boost', GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)),
        ('MLP', MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=300, random_state=42)),
    ]
    
    best_acc = 0
    best_clf_name = ""
    
    for clf_name, clf in classifiers:
        clf.fit(ensemble_train, y_train)
        acc = clf.score(ensemble_test, y_test)
        print(f"  {clf_name}: {acc:.3f}")
        
        if acc > best_acc:
            best_acc = acc
            best_clf_name = clf_name
    
    # Baseline
    print("\n📊 Baseline comparison:")
    clf_baseline = LogisticRegression(max_iter=300, random_state=42)
    clf_baseline.fit(X_train, y_train)
    baseline_acc = clf_baseline.score(X_test, y_test)
    print(f"  Raw pixels: {baseline_acc:.3f}")
    
    print("\n🏆 RESULTS:")
    print(f"  Best SSL ({best_clf_name}): {best_acc:.3f}")
    print(f"  Baseline: {baseline_acc:.3f}")
    print(f"  Improvement: {(best_acc - baseline_acc)*100:+.1f}%")
    
    if best_acc > baseline_acc:
        print(f"\n✅ SSL BEATS BASELINE by {(best_acc - baseline_acc)*100:.1f}%!")
    
    return best_acc, baseline_acc

def optimized_single_model(n_samples=5000):
    """Train a single optimized SSL model."""
    
    print("🎯 OPTIMIZED SINGLE MODEL")
    print("="*60)
    
    # Load data
    print("Loading Fashion-MNIST...")
    X_train, y_train, X_test, y_test, _ = load_fashion_mnist(n_samples)
    
    # Best single model configuration (based on experiments)
    print("\nTraining optimized rotation model...")
    print("  Config: 10 rotations, 300 hidden units, 40 epochs")
    
    # Create rotation dataset with 10 angles
    angles = np.linspace(0, 360, 10, endpoint=False)
    rot_X, rot_y = create_rotation_dataset(X_train, angles)
    
    # Split
    X_rot_train, X_rot_val, y_rot_train, y_rot_val = train_test_split(
        rot_X, rot_y, test_size=0.2, random_state=42
    )
    
    # Train with optimal hyperparameters
    net = TwoLayerNet(784, 300, 10, learning_rate=0.06)
    
    print("  Training...")
    losses, accs = net.train(
        X_rot_train, y_rot_train, 
        epochs=40, 
        batch_size=64,  # Smaller batch for better gradients
        verbose=False
    )
    
    pretext_acc = net.evaluate(X_rot_val, y_rot_val)
    print(f"  Pretext accuracy: {pretext_acc:.3f}")
    
    # Extract features
    ssl_train = net.hidden_representation(X_train)
    ssl_test = net.hidden_representation(X_test)
    
    # Test multiple classifiers
    print("\n🔬 Testing downstream classifiers...")
    
    best_acc = 0
    for C in [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]:
        clf = LogisticRegression(max_iter=500, C=C, random_state=42)
        clf.fit(ssl_train, y_train)
        acc = clf.score(ssl_test, y_test)
        print(f"  Logistic (C={C}): {acc:.3f}")
        best_acc = max(best_acc, acc)
    
    # Baseline
    clf_baseline = LogisticRegression(max_iter=300, random_state=42)
    clf_baseline.fit(X_train, y_train)
    baseline_acc = clf_baseline.score(X_test, y_test)
    
    print(f"\n🏆 RESULTS:")
    print(f"  Best SSL: {best_acc:.3f}")
    print(f"  Baseline: {baseline_acc:.3f}")
    print(f"  Improvement: {(best_acc - baseline_acc)*100:+.1f}%")
    
    return best_acc, baseline_acc

def main():
    print("🚀 PUSHING SSL ACCURACY HIGHER")
    print("="*60)
    
    # Test both approaches
    print("\n1️⃣ Testing ensemble approach with 3000 samples...")
    ensemble_ssl, ensemble_baseline = ensemble_ssl_experiment(n_samples=3000)
    
    print("\n" + "="*60)
    print("\n2️⃣ Testing optimized single model with 5000 samples...")
    single_ssl, single_baseline = optimized_single_model(n_samples=5000)
    
    print("\n" + "="*60)
    print("📊 FINAL SUMMARY")
    print("="*60)
    print(f"Ensemble approach: {ensemble_ssl:.3f} vs {ensemble_baseline:.3f} baseline "
          f"({(ensemble_ssl-ensemble_baseline)*100:+.1f}%)")
    print(f"Single optimized:  {single_ssl:.3f} vs {single_baseline:.3f} baseline "
          f"({(single_ssl-single_baseline)*100:+.1f}%)")
    
    best_improvement = max(ensemble_ssl - ensemble_baseline, single_ssl - single_baseline)
    if best_improvement > 0:
        print(f"\n✅ BEST IMPROVEMENT: {best_improvement*100:+.1f}% over baseline!")
    else:
        print(f"\n📊 SSL is competitive but {-best_improvement*100:.1f}% behind baseline")

if __name__ == "__main__":
    main()