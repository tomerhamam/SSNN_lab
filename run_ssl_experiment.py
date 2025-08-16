"""
Easy experiment runner for SSL digit recognition.
Run different configurations to see the effect of rotation count and epochs.
"""

import matplotlib
matplotlib.use('TkAgg')  # Interactive backend for plots
import sys
import os

# Add current directory to path to import our module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def run_ssl_experiment(num_rotations=32, epochs=30, show_plots=True):
    """
    Run SSL experiment with specific configuration.
    
    Args:
        num_rotations: Number of rotation angles (4, 8, 16, 32, etc.)
        epochs: Number of training epochs 
        show_plots: Whether to show matplotlib plots
    """
    print(f"🧪 Running SSL experiment:")
    print(f"   📊 Rotations: {num_rotations}")
    print(f"   🔄 Epochs: {epochs}")
    print(f"   📈 Show plots: {show_plots}")
    print("=" * 50)
    
    # Import and modify the script
    from snn_for_digits_clean import (
        load_digits, train_test_split, LogisticRegression,
        create_rotation_dataset, TwoLayerNet, visualize_samples,
        confusion_matrix, sns, plt, np, classification_report
    )
    
    # 1. Load data
    print("\n📊 Loading digits dataset...")
    digits = load_digits()
    X = digits.data.astype(np.float32) / 16.0
    y = digits.target
    print(f"   • Original dataset: {X.shape} with {len(np.unique(y))} classes")
    
    # 2. Create rotation dataset
    print(f"\n🔄 Creating rotation dataset with {num_rotations} angles...")
    rotation_res_angle = 2 * np.pi / num_rotations
    rotations = np.arange(num_rotations) * rotation_res_angle * 360.0/(2*np.pi)
    rot_X, rot_y = create_rotation_dataset(X, rotations)
    print(f"   • Rotation dataset: {rot_X.shape}")
    
    rotation_labels = [f"{angle:.0f}°" for angle in rotations]
    print(f"   • Rotation classes: {np.unique(rot_y)}")
    print(f"   • Random baseline: {1/num_rotations:.1%}")
    
    # Show sample rotations (first few only to avoid clutter)
    if show_plots:
        sample_idx = 0
        sample_rotations = []
        n_show = min(8, num_rotations)  # Show max 8 rotations
        for i in range(n_show):
            rot_sample_idx = sample_idx + i * len(X)
            sample_rotations.append(rot_X[rot_sample_idx])
        
        visualize_samples(
            sample_rotations, 
            rotation_labels[:n_show],
            f"First {n_show} Rotations of Digit {y[sample_idx]}", 
            n_samples=n_show,
            figsize=(min(12, n_show * 1.5), 2)
        )
    
    # 3. Split data
    print("\n✂️  Splitting data...")
    X_train, X_val, y_train, y_val = train_test_split(
        rot_X, rot_y, test_size=0.2, random_state=42
    )
    print(f"   • Training: {X_train.shape}")
    print(f"   • Validation: {X_val.shape}")
    
    # 4. Create and train model
    print(f"\n🧠 Training neural network for {epochs} epochs...")
    print(f"📊 Task difficulty: {num_rotations} classes (random baseline: {1/num_rotations:.1%})")
    
    # Adjust learning rate based on task difficulty
    if num_rotations > 16:
        learning_rate = 0.1
        print(f"🎯 Using lower learning rate ({learning_rate}) for harder task")
    else:
        learning_rate = 0.3
    
    net = TwoLayerNet(input_dim=64, hidden_dim=32, output_dim=num_rotations, learning_rate=learning_rate)
    losses, accuracies = net.train(X_train, y_train, epochs=epochs, batch_size=256, verbose=True)
    
    # 5. Evaluate
    val_acc = net.evaluate(X_val, y_val)
    print(f"\n🎯 Final validation accuracy: {val_acc:.3f}")
    
    random_baseline = 1.0 / num_rotations
    if val_acc > random_baseline:
        improvement = (val_acc - random_baseline) * 100
        print(f"✅ Beat random guessing ({random_baseline:.1%}) by {improvement:.1f} percentage points")
        if val_acc > 0.5:
            print("🎉 Excellent! Network learned meaningful rotation features!")
    else:
        print("❌ Performance didn't beat random guessing")
    
    # 6. Plot training curves
    if show_plots:
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
        ax2.axhline(y=random_baseline, color='r', linestyle='--', alpha=0.7, label=f'Random ({random_baseline:.1%})')
        ax2.axhline(y=val_acc, color='orange', linestyle='--', alpha=0.7, label=f'Val Acc ({val_acc:.3f})')
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
    
    if ssl_acc > baseline_acc:
        print(f"   ✅ SSL features outperform raw pixels!")
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
        'net': net
    }

def compare_configurations():
    """Compare different rotation counts and epoch settings."""
    configurations = [
        (4, 15),    # Easy task, few epochs
        (8, 20),    # Medium task
        (16, 25),   # Harder task
        (32, 30),   # Very hard task, more epochs
    ]
    
    results = []
    for num_rot, epochs in configurations:
        print(f"\n{'='*60}")
        print(f"🧪 EXPERIMENT: {num_rot} rotations, {epochs} epochs")
        print(f"{'='*60}")
        
        result = run_ssl_experiment(num_rot, epochs, show_plots=False)
        results.append(result)
        
        print(f"✅ Completed: {num_rot} rotations → SSL acc: {result['ssl_transfer_acc']:.3f}")
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 COMPARISON SUMMARY")
    print(f"{'='*60}")
    print("Rotations | Epochs | Rotation Acc | SSL Transfer | Baseline | Improvement")
    print("-" * 70)
    for r in results:
        print(f"{r['num_rotations']:8d} | {r['epochs']:6d} | {r['rotation_acc']:11.3f} | {r['ssl_transfer_acc']:11.3f} | {r['baseline_acc']:8.3f} | {r['improvement']:+10.3f}")
    
    return results

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "compare":
        # Run comparison across configurations
        results = compare_configurations()
    else:
        # Run single experiment
        if len(sys.argv) >= 3:
            num_rotations = int(sys.argv[1])
            epochs = int(sys.argv[2])
        else:
            num_rotations = 32
            epochs = 30
        
        result = run_ssl_experiment(num_rotations, epochs, show_plots=True)
        print(f"\n🎉 Experiment completed!")
        print(f"📊 Final results: {result['ssl_transfer_acc']:.3f} SSL accuracy vs {result['baseline_acc']:.3f} baseline")