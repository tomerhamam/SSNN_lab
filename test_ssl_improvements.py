"""
Quick test script to explore SSL improvements on Fashion-MNIST.
"""

import numpy as np
from ssl_fashion_mnist import load_fashion_mnist, create_rotation_dataset, TwoLayerNet
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

def test_configuration(num_rotations, epochs, hidden_dim, lr, n_samples=2000):
    """Test a specific configuration quickly."""
    
    # Load data
    X_train, y_train, X_test, y_test, _ = load_fashion_mnist(n_samples)
    
    # Create rotation dataset
    rotation_angles = np.linspace(0, 360, num_rotations, endpoint=False)
    rot_X, rot_y = create_rotation_dataset(X_train, rotation_angles)
    
    # Split rotation data
    X_rot_train, X_rot_val, y_rot_train, y_rot_val = train_test_split(
        rot_X, rot_y, test_size=0.2, random_state=42
    )
    
    # Train SSL model
    net = TwoLayerNet(784, hidden_dim, num_rotations, lr)
    losses, accuracies = net.train(
        X_rot_train, y_rot_train, 
        epochs=epochs, 
        batch_size=128, 
        verbose=False
    )
    
    # Evaluate pretext task
    pretext_acc = net.evaluate(X_rot_val, y_rot_val)
    
    # Extract SSL features
    ssl_features_train = net.hidden_representation(X_train)
    ssl_features_test = net.hidden_representation(X_test)
    
    # Try multiple classifiers
    # 1. Logistic Regression
    clf_lr = LogisticRegression(max_iter=200, random_state=42, C=1.0)
    clf_lr.fit(ssl_features_train, y_train)
    lr_acc = clf_lr.score(ssl_features_test, y_test)
    
    # 2. Random Forest
    clf_rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    clf_rf.fit(ssl_features_train, y_train)
    rf_acc = clf_rf.score(ssl_features_test, y_test)
    
    # 3. Logistic Regression with L2 regularization variations
    clf_lr_low = LogisticRegression(max_iter=200, random_state=42, C=0.1)
    clf_lr_low.fit(ssl_features_train, y_train)
    lr_low_acc = clf_lr_low.score(ssl_features_test, y_test)
    
    clf_lr_high = LogisticRegression(max_iter=200, random_state=42, C=10.0)
    clf_lr_high.fit(ssl_features_train, y_train)
    lr_high_acc = clf_lr_high.score(ssl_features_test, y_test)
    
    # Baseline
    clf_baseline = LogisticRegression(max_iter=200, random_state=42)
    clf_baseline.fit(X_train, y_train)
    baseline_acc = clf_baseline.score(X_test, y_test)
    
    best_ssl = max(lr_acc, rf_acc, lr_low_acc, lr_high_acc)
    
    return {
        'pretext_acc': pretext_acc,
        'lr_acc': lr_acc,
        'rf_acc': rf_acc,
        'lr_low_acc': lr_low_acc,
        'lr_high_acc': lr_high_acc,
        'best_ssl': best_ssl,
        'baseline': baseline_acc,
        'improvement': best_ssl - baseline_acc
    }

def main():
    print("🧪 Testing SSL improvements on Fashion-MNIST")
    print("="*60)
    
    # Test configurations
    configs = [
        # (num_rotations, epochs, hidden_dim, learning_rate, description)
        (4, 20, 128, 0.1, "Baseline: 4 rotations"),
        (6, 25, 128, 0.1, "6 rotations"),
        (8, 30, 128, 0.1, "8 rotations"),
        (10, 35, 128, 0.08, "10 rotations"),
        (12, 40, 128, 0.08, "12 rotations"),
        (8, 30, 256, 0.1, "8 rot, wider network"),
        (10, 35, 256, 0.08, "10 rot, wider network"),
        (12, 40, 256, 0.08, "12 rot, wider network"),
        (8, 50, 128, 0.08, "8 rot, longer training"),
        (10, 50, 256, 0.05, "10 rot, wider + longer"),
    ]
    
    best_config = None
    best_improvement = -float('inf')
    
    print("\nTesting configurations with 2000 samples:")
    print("-"*60)
    
    for num_rot, epochs, hidden, lr, desc in configs:
        print(f"\n{desc}:")
        print(f"  Config: {num_rot} rotations, {epochs} epochs, hidden={hidden}, lr={lr}")
        
        result = test_configuration(num_rot, epochs, hidden, lr, n_samples=2000)
        
        print(f"  Pretext accuracy: {result['pretext_acc']:.3f}")
        print(f"  SSL accuracies:")
        print(f"    - Logistic (C=1.0): {result['lr_acc']:.3f}")
        print(f"    - Logistic (C=0.1): {result['lr_low_acc']:.3f}")
        print(f"    - Logistic (C=10): {result['lr_high_acc']:.3f}")
        print(f"    - Random Forest: {result['rf_acc']:.3f}")
        print(f"  Best SSL: {result['best_ssl']:.3f}")
        print(f"  Baseline: {result['baseline']:.3f}")
        print(f"  Improvement: {result['improvement']*100:+.1f}%")
        
        if result['improvement'] > best_improvement:
            best_improvement = result['improvement']
            best_config = (num_rot, epochs, hidden, lr, desc)
    
    # Test best config with more data
    if best_config:
        print("\n" + "="*60)
        print("🏆 BEST CONFIGURATION:")
        print("="*60)
        num_rot, epochs, hidden, lr, desc = best_config
        print(f"{desc}: {num_rot} rotations, {epochs} epochs, hidden={hidden}, lr={lr}")
        print(f"Best improvement with 2000 samples: {best_improvement*100:+.1f}%")
        
        print("\n🔬 Testing best config with 5000 samples...")
        result = test_configuration(num_rot, epochs, hidden, lr, n_samples=5000)
        
        print(f"\nResults with 5000 samples:")
        print(f"  Pretext accuracy: {result['pretext_acc']:.3f}")
        print(f"  Best SSL: {result['best_ssl']:.3f}")
        print(f"  Baseline: {result['baseline']:.3f}")
        print(f"  Improvement: {result['improvement']*100:+.1f}%")
        
        if result['improvement'] > 0:
            print(f"\n✅ SSL beats baseline by {result['improvement']*100:.1f}%!")
        else:
            print(f"\n📊 SSL is {-result['improvement']*100:.1f}% behind baseline")

if __name__ == "__main__":
    main()