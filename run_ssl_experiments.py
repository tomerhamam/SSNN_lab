"""
Systematic testing script for SSL experiments.
Tests multiple configurations and generates comparison tables.
"""

import numpy as np
import sys
import time
from typing import List, Tuple, Dict
import json

# Import our SSL modules
from ssl_fashion_mnist import run_fashion_mnist_ssl, load_fashion_mnist
from snn_for_digits_clean import main as run_digits_ssl

# Suppress matplotlib warnings
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# EXPERIMENT CONFIGURATIONS
# ============================================================================

# Standard configurations to test
STANDARD_CONFIGS = [
    # (num_rotations, epochs, description)
    (4, 15, "Easy task, quick training"),
    (4, 30, "Easy task, more training"),
    (8, 20, "Medium task, balanced"),
    (8, 40, "Medium task, extended"),
    (16, 25, "Hard task, standard"),
    (16, 50, "Hard task, extended"),
    (32, 30, "Very hard task, standard"),
    (32, 60, "Very hard task, extended"),
]

# Quick test configurations
QUICK_CONFIGS = [
    (4, 10, "Quick easy"),
    (8, 15, "Quick medium"),
    (16, 20, "Quick hard"),
]

# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

def run_fashion_mnist_experiments(configs: List[Tuple], n_samples: int = 5000) -> List[Dict]:
    """
    Run multiple Fashion-MNIST SSL experiments.
    
    Args:
        configs: List of (num_rotations, epochs, description) tuples
        n_samples: Number of training samples to use
    
    Returns:
        List of result dictionaries
    """
    print("\n" + "="*70)
    print("🧪 FASHION-MNIST SSL EXPERIMENTS")
    print("="*70)
    
    results = []
    
    for num_rot, epochs, desc in configs:
        print(f"\n--- Configuration: {desc} ---")
        print(f"   Rotations: {num_rot}, Epochs: {epochs}")
        
        start_time = time.time()
        
        try:
            result = run_fashion_mnist_ssl(
                num_rotations=num_rot,
                epochs=epochs,
                n_samples=n_samples,
                show_plots=False,  # Disable plots for batch testing
                verbose=False  # Less verbose output
            )
            
            elapsed_time = time.time() - start_time
            
            # Add metadata
            result['description'] = desc
            result['elapsed_time'] = elapsed_time
            result['dataset'] = 'Fashion-MNIST'
            
            results.append(result)
            
            print(f"✅ Completed in {elapsed_time:.1f}s")
            print(f"   Rotation acc: {result['rotation_acc']:.3f}")
            print(f"   SSL transfer: {result['ssl_transfer_acc']:.3f}")
            print(f"   Baseline: {result['baseline_acc']:.3f}")
            print(f"   Improvement: {result['improvement']*100:+.1f} pp")
            
        except Exception as e:
            print(f"❌ Failed: {e}")
            results.append({
                'num_rotations': num_rot,
                'epochs': epochs,
                'description': desc,
                'error': str(e),
                'dataset': 'Fashion-MNIST'
            })
    
    return results

def run_digits_experiments(configs: List[Tuple]) -> List[Dict]:
    """
    Run multiple Digits SSL experiments for comparison.
    
    Args:
        configs: List of (num_rotations, epochs, description) tuples
    
    Returns:
        List of result dictionaries
    """
    print("\n" + "="*70)
    print("🧪 DIGITS SSL EXPERIMENTS (for comparison)")
    print("="*70)
    
    results = []
    
    # Import necessary functions from digits script
    from snn_for_digits_clean import (
        load_digits, create_rotation_dataset, TwoLayerNet,
        train_test_split, LogisticRegression
    )
    
    for num_rot, epochs, desc in configs:
        print(f"\n--- Configuration: {desc} ---")
        print(f"   Rotations: {num_rot}, Epochs: {epochs}")
        
        start_time = time.time()
        
        try:
            # Load digits data
            digits = load_digits()
            X = digits.data.astype(np.float32) / 16.0
            y = digits.target
            
            # Create rotation dataset
            rotation_res_angle = 2 * np.pi / num_rot
            rotations = np.arange(num_rot) * rotation_res_angle * 360.0/(2*np.pi)
            rot_X, rot_y = create_rotation_dataset(X, rotations)
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                rot_X, rot_y, test_size=0.2, random_state=42
            )
            
            # Train model
            learning_rate = 0.1 if num_rot > 16 else 0.3
            net = TwoLayerNet(
                input_dim=64, 
                hidden_dim=32, 
                output_dim=num_rot, 
                learning_rate=learning_rate
            )
            losses, accuracies = net.train(X_train, y_train, epochs=epochs, verbose=False)
            
            # Evaluate rotation task
            rotation_acc = net.evaluate(X_val, y_val)
            
            # Transfer learning evaluation
            X_orig_train, X_orig_test, y_orig_train, y_orig_test = train_test_split(
                X, y, test_size=0.3, random_state=1
            )
            
            ssl_features_train = net.hidden_representation(X_orig_train)
            ssl_features_test = net.hidden_representation(X_orig_test)
            
            clf_ssl = LogisticRegression(max_iter=200, random_state=42)
            clf_ssl.fit(ssl_features_train, y_orig_train)
            ssl_acc = clf_ssl.score(ssl_features_test, y_orig_test)
            
            clf_baseline = LogisticRegression(max_iter=200, random_state=42)
            clf_baseline.fit(X_orig_train, y_orig_train)
            baseline_acc = clf_baseline.score(X_orig_test, y_orig_test)
            
            elapsed_time = time.time() - start_time
            
            result = {
                'num_rotations': num_rot,
                'epochs': epochs,
                'description': desc,
                'rotation_acc': rotation_acc,
                'ssl_transfer_acc': ssl_acc,
                'baseline_acc': baseline_acc,
                'improvement': ssl_acc - baseline_acc,
                'elapsed_time': elapsed_time,
                'dataset': 'Digits'
            }
            
            results.append(result)
            
            print(f"✅ Completed in {elapsed_time:.1f}s")
            print(f"   Rotation acc: {rotation_acc:.3f}")
            print(f"   SSL transfer: {ssl_acc:.3f}")
            print(f"   Baseline: {baseline_acc:.3f}")
            print(f"   Improvement: {(ssl_acc - baseline_acc)*100:+.1f} pp")
            
        except Exception as e:
            print(f"❌ Failed: {e}")
            results.append({
                'num_rotations': num_rot,
                'epochs': epochs,
                'description': desc,
                'error': str(e),
                'dataset': 'Digits'
            })
    
    return results

# ============================================================================
# ANALYSIS AND REPORTING
# ============================================================================

def print_results_table(results: List[Dict], dataset_name: str = ""):
    """Print a formatted table of results."""
    print(f"\n{'='*80}")
    print(f"📊 RESULTS SUMMARY{' - ' + dataset_name if dataset_name else ''}")
    print(f"{'='*80}")
    print(f"{'Config':<25} | {'Rot':<4} | {'Eps':<4} | {'Rot Acc':<8} | {'SSL Acc':<8} | {'Baseline':<8} | {'Improve':<8} | {'Time':<6}")
    print("-" * 80)
    
    for r in results:
        if 'error' in r:
            print(f"{r['description']:<25} | {r['num_rotations']:4d} | {r['epochs']:4d} | {'ERROR':<8} | {'-':<8} | {'-':<8} | {'-':<8} | {'-':<6}")
        else:
            print(f"{r['description']:<25} | {r['num_rotations']:4d} | {r['epochs']:4d} | {r['rotation_acc']:8.3f} | {r['ssl_transfer_acc']:8.3f} | {r['baseline_acc']:8.3f} | {r['improvement']*100:+7.1f}% | {r['elapsed_time']:6.1f}s")

def analyze_results(results: List[Dict]) -> Dict:
    """Analyze results and find best configurations."""
    if not results:
        return {}
    
    # Filter out errors
    valid_results = [r for r in results if 'error' not in r]
    
    if not valid_results:
        return {'error': 'No valid results'}
    
    # Find best configurations
    best_ssl = max(valid_results, key=lambda x: x['ssl_transfer_acc'])
    best_improvement = max(valid_results, key=lambda x: x['improvement'])
    best_rotation = max(valid_results, key=lambda x: x['rotation_acc'])
    
    # Calculate averages by rotation count
    rotation_groups = {}
    for r in valid_results:
        rot = r['num_rotations']
        if rot not in rotation_groups:
            rotation_groups[rot] = []
        rotation_groups[rot].append(r)
    
    avg_by_rotation = {}
    for rot, group in rotation_groups.items():
        avg_by_rotation[rot] = {
            'avg_ssl_acc': np.mean([r['ssl_transfer_acc'] for r in group]),
            'avg_baseline': np.mean([r['baseline_acc'] for r in group]),
            'avg_improvement': np.mean([r['improvement'] for r in group]),
            'avg_rotation_acc': np.mean([r['rotation_acc'] for r in group]),
            'count': len(group)
        }
    
    return {
        'best_ssl': best_ssl,
        'best_improvement': best_improvement,
        'best_rotation': best_rotation,
        'avg_by_rotation': avg_by_rotation
    }

def save_results(results: List[Dict], filename: str):
    """Save results to JSON file."""
    with open(filename, 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        clean_results = []
        for r in results:
            clean_r = {}
            for k, v in r.items():
                if k == 'model':  # Skip model object
                    continue
                elif isinstance(v, (np.integer, np.floating)):
                    clean_r[k] = float(v)
                else:
                    clean_r[k] = v
            clean_results.append(clean_r)
        
        json.dump(clean_results, f, indent=2)
    print(f"💾 Results saved to {filename}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run systematic SSL experiments."""
    print("🚀 SSL SYSTEMATIC EXPERIMENTS")
    print("="*80)
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    else:
        mode = 'quick'
    
    if mode == 'full':
        configs = STANDARD_CONFIGS
        print("📊 Running FULL experiment suite...")
    elif mode == 'quick':
        configs = QUICK_CONFIGS
        print("⚡ Running QUICK experiments...")
    elif mode == 'compare':
        # Run both datasets with quick configs
        print("🔄 Running COMPARISON between datasets...")
        
        # Fashion-MNIST experiments
        fashion_results = run_fashion_mnist_experiments(QUICK_CONFIGS, n_samples=2000)
        print_results_table(fashion_results, "Fashion-MNIST")
        
        # Digits experiments
        digits_results = run_digits_experiments(QUICK_CONFIGS)
        print_results_table(digits_results, "Digits")
        
        # Compare results
        print(f"\n{'='*80}")
        print("📊 DATASET COMPARISON")
        print(f"{'='*80}")
        
        fashion_analysis = analyze_results(fashion_results)
        digits_analysis = analyze_results(digits_results)
        
        if 'best_improvement' in fashion_analysis:
            print(f"\n🎯 Fashion-MNIST best improvement: {fashion_analysis['best_improvement']['improvement']*100:+.1f}% "
                  f"({fashion_analysis['best_improvement']['num_rotations']} rotations, "
                  f"{fashion_analysis['best_improvement']['epochs']} epochs)")
        
        if 'best_improvement' in digits_analysis:
            print(f"🎯 Digits best improvement: {digits_analysis['best_improvement']['improvement']*100:+.1f}% "
                  f"({digits_analysis['best_improvement']['num_rotations']} rotations, "
                  f"{digits_analysis['best_improvement']['epochs']} epochs)")
        
        # Save results
        save_results(fashion_results, 'fashion_mnist_results.json')
        save_results(digits_results, 'digits_results.json')
        
        return
    elif mode == 'custom':
        # Custom configuration from command line
        if len(sys.argv) >= 4:
            num_rot = int(sys.argv[2])
            epochs = int(sys.argv[3])
            configs = [(num_rot, epochs, "Custom config")]
        else:
            print("❌ Custom mode requires: python run_ssl_experiments.py custom <rotations> <epochs>")
            return
    else:
        print(f"❌ Unknown mode: {mode}")
        print("Available modes: quick, full, compare, custom")
        return
    
    # Run Fashion-MNIST experiments
    results = run_fashion_mnist_experiments(configs, n_samples=5000 if mode == 'full' else 2000)
    
    # Print results table
    print_results_table(results, "Fashion-MNIST")
    
    # Analyze results
    analysis = analyze_results(results)
    
    if 'best_ssl' in analysis:
        print(f"\n🏆 BEST CONFIGURATIONS:")
        print(f"   Best SSL accuracy: {analysis['best_ssl']['ssl_transfer_acc']:.3f} "
              f"({analysis['best_ssl']['num_rotations']} rotations, {analysis['best_ssl']['epochs']} epochs)")
        print(f"   Best improvement: {analysis['best_improvement']['improvement']*100:+.1f}% "
              f"({analysis['best_improvement']['num_rotations']} rotations, {analysis['best_improvement']['epochs']} epochs)")
        print(f"   Best rotation learning: {analysis['best_rotation']['rotation_acc']:.3f} "
              f"({analysis['best_rotation']['num_rotations']} rotations, {analysis['best_rotation']['epochs']} epochs)")
    
    # Print averages by rotation count
    if 'avg_by_rotation' in analysis:
        print(f"\n📈 AVERAGES BY ROTATION COUNT:")
        for rot, stats in sorted(analysis['avg_by_rotation'].items()):
            print(f"   {rot:2d} rotations: SSL={stats['avg_ssl_acc']:.3f}, "
                  f"Baseline={stats['avg_baseline']:.3f}, "
                  f"Improvement={stats['avg_improvement']*100:+.1f}%, "
                  f"Rotation={stats['avg_rotation_acc']:.3f} "
                  f"(n={stats['count']})")
    
    # Save results
    save_results(results, f'ssl_results_{mode}_{int(time.time())}.json')
    
    print(f"\n✅ Experiments completed!")

if __name__ == "__main__":
    main()