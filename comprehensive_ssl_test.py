"""
Comprehensive SSL testing with the new framework.
Tests different architectures and finds optimal configurations.
"""

from ssl_framework import run_ssl_experiment
import numpy as np
import json
import time
import warnings
warnings.filterwarnings('ignore')

def architecture_search():
    """Test different network architectures."""
    print("🏗️  ARCHITECTURE SEARCH")
    print("="*50)
    
    architectures = [
        ([128], "Shallow"),
        ([256, 128], "Medium-2Layer"),
        ([512, 256], "Wide-2Layer"),
        ([256, 128, 64], "Deep-3Layer"),
        ([512, 256, 128], "Deep-Wide"),
        ([1024, 512, 256], "Very-Deep-Wide"),
    ]
    
    results = []
    
    for arch, desc in architectures:
        print(f"\n🧪 Testing {desc}: {arch}")
        
        try:
            result = run_ssl_experiment(
                dataset_name='cifar10',
                pretext_task='rotation',
                architecture=arch,
                epochs=15,
                learning_rate=0.015,
                num_rotations=8,
                verbose=False
            )
            
            transfer = result['transfer_results']
            improvement = transfer['improvement']
            
            results.append({
                'architecture': arch,
                'description': desc,
                'ssl_acc': transfer['best_ssl'],
                'baseline_acc': transfer['baseline'],
                'improvement': improvement
            })
            
            print(f"   SSL: {transfer['best_ssl']:.3f}, Baseline: {transfer['baseline']:.3f}, "
                  f"Improvement: {improvement*100:+.1f}%")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
    
    # Find best architecture
    if results:
        best = max(results, key=lambda x: x['improvement'])
        print(f"\n🏆 Best architecture: {best['description']} {best['architecture']}")
        print(f"   Improvement: {best['improvement']*100:+.1f}%")
    
    return results

def pretext_task_comparison():
    """Compare different pretext tasks."""
    print("\n🎯 PRETEXT TASK COMPARISON")
    print("="*50)
    
    tasks = [
        ('rotation', {'num_rotations': 6}, "Rotation-6"),
        ('rotation', {'num_rotations': 8}, "Rotation-8"),
        ('rotation', {'num_rotations': 12}, "Rotation-12"),
        ('jigsaw', {'grid_size': 2}, "Jigsaw-2x2"),
        ('contrastive', {'num_augmentations': 4}, "Contrastive-4"),
    ]
    
    results = []
    
    for task_name, kwargs, desc in tasks:
        print(f"\n🧪 Testing {desc}")
        
        try:
            result = run_ssl_experiment(
                dataset_name='cifar10',
                pretext_task=task_name,
                architecture=[512, 256, 128],
                epochs=20,
                learning_rate=0.01,
                verbose=False,
                **kwargs
            )
            
            transfer = result['transfer_results']
            train = result['train_results']
            
            results.append({
                'task': desc,
                'pretext_acc': train['final_pretext_acc'],
                'ssl_acc': transfer['best_ssl'],
                'baseline_acc': transfer['baseline'],
                'improvement': transfer['improvement']
            })
            
            print(f"   Pretext: {train['final_pretext_acc']:.3f}, "
                  f"Transfer: {transfer['best_ssl']:.3f}, "
                  f"Improvement: {transfer['improvement']*100:+.1f}%")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
    
    # Find best pretext task
    if results:
        best = max(results, key=lambda x: x['improvement'])
        print(f"\n🏆 Best pretext task: {best['task']}")
        print(f"   Improvement: {best['improvement']*100:+.1f}%")
    
    return results

def hyperparameter_optimization():
    """Optimize key hyperparameters."""
    print("\n⚙️  HYPERPARAMETER OPTIMIZATION")
    print("="*50)
    
    # Test learning rates
    print("\n📊 Testing learning rates...")
    lr_results = []
    
    for lr in [0.005, 0.01, 0.02, 0.03]:
        print(f"   LR = {lr}")
        try:
            result = run_ssl_experiment(
                dataset_name='cifar10',
                pretext_task='rotation',
                architecture=[512, 256, 128],
                epochs=15,
                learning_rate=lr,
                num_rotations=8,
                verbose=False
            )
            
            improvement = result['transfer_results']['improvement']
            lr_results.append((lr, improvement))
            print(f"      Improvement: {improvement*100:+.1f}%")
            
        except Exception as e:
            print(f"      Failed: {e}")
    
    # Test dropout rates
    print("\n📊 Testing dropout rates...")
    dropout_results = []
    
    for dropout in [0.0, 0.1, 0.2, 0.3]:
        print(f"   Dropout = {dropout}")
        try:
            result = run_ssl_experiment(
                dataset_name='cifar10',
                pretext_task='rotation',
                architecture=[512, 256, 128],
                epochs=15,
                learning_rate=0.015,
                dropout_rate=dropout,
                num_rotations=8,
                verbose=False
            )
            
            improvement = result['transfer_results']['improvement']
            dropout_results.append((dropout, improvement))
            print(f"      Improvement: {improvement*100:+.1f}%")
            
        except Exception as e:
            print(f"      Failed: {e}")
    
    # Find best hyperparameters
    if lr_results:
        best_lr = max(lr_results, key=lambda x: x[1])
        print(f"\n🏆 Best learning rate: {best_lr[0]} ({best_lr[1]*100:+.1f}%)")
    
    if dropout_results:
        best_dropout = max(dropout_results, key=lambda x: x[1])
        print(f"🏆 Best dropout rate: {best_dropout[0]} ({best_dropout[1]*100:+.1f}%)")
    
    return lr_results, dropout_results

def final_optimized_test():
    """Run final test with optimized configuration."""
    print("\n🚀 FINAL OPTIMIZED TEST")
    print("="*50)
    
    print("Running with optimized configuration...")
    print("  Dataset: CIFAR-10")
    print("  Architecture: [512, 256, 128] (Deep-Wide)")
    print("  Pretext: Rotation-8")
    print("  Learning Rate: 0.015")
    print("  Dropout: 0.1")
    print("  Epochs: 25")
    
    result = run_ssl_experiment(
        dataset_name='cifar10',
        pretext_task='rotation',
        architecture=[512, 256, 128],
        epochs=25,
        learning_rate=0.015,
        dropout_rate=0.1,
        num_rotations=8,
        verbose=True
    )
    
    transfer = result['transfer_results']
    train = result['train_results']
    
    print(f"\n🎯 FINAL RESULTS:")
    print(f"   Pretext accuracy: {train['final_pretext_acc']:.3f}")
    print(f"   SSL accuracy: {transfer['best_ssl']:.3f}")
    print(f"   Baseline accuracy: {transfer['baseline']:.3f}")
    print(f"   Improvement: {transfer['improvement']*100:+.1f}%")
    
    if transfer['improvement'] > 0.05:  # 5% improvement
        print("🎉 EXCELLENT! SSL significantly outperforms baseline!")
    elif transfer['improvement'] > 0:
        print("✅ GOOD! SSL beats baseline!")
    else:
        print("📊 SSL is competitive with baseline")
    
    return result

def main():
    """Run comprehensive SSL testing."""
    print("🔬 COMPREHENSIVE SSL FRAMEWORK TESTING")
    print("="*80)
    
    start_time = time.time()
    
    # Run all tests
    print("\n1️⃣ Architecture Search")
    arch_results = architecture_search()
    
    print("\n2️⃣ Pretext Task Comparison") 
    task_results = pretext_task_comparison()
    
    print("\n3️⃣ Hyperparameter Optimization")
    lr_results, dropout_results = hyperparameter_optimization()
    
    print("\n4️⃣ Final Optimized Test")
    final_result = final_optimized_test()
    
    # Summary
    elapsed = time.time() - start_time
    print(f"\n{'='*80}")
    print("📊 COMPREHENSIVE TEST SUMMARY")
    print(f"{'='*80}")
    print(f"Total testing time: {elapsed:.1f} seconds")
    
    # Save results
    all_results = {
        'architecture_search': arch_results,
        'pretext_task_comparison': task_results,
        'learning_rate_search': lr_results,
        'dropout_search': dropout_results,
        'final_optimized': {
            'ssl_acc': final_result['transfer_results']['best_ssl'],
            'baseline_acc': final_result['transfer_results']['baseline'],
            'improvement': final_result['transfer_results']['improvement']
        },
        'test_duration': elapsed
    }
    
    # Save to file
    with open(f'comprehensive_ssl_results_{int(time.time())}.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"💾 Results saved to comprehensive_ssl_results_{int(time.time())}.json")
    
    # Final verdict
    best_improvement = final_result['transfer_results']['improvement']
    print(f"\n🎯 FINAL VERDICT:")
    print(f"   Best SSL improvement: {best_improvement*100:+.1f}%")
    
    if best_improvement > 0.1:
        print("🏆 OUTSTANDING! SSL shows significant advantages!")
    elif best_improvement > 0.05:
        print("🎉 EXCELLENT! SSL clearly outperforms baseline!")
    elif best_improvement > 0:
        print("✅ SUCCESS! SSL beats baseline!")
    else:
        print("📊 SSL is competitive - good for feature learning!")

if __name__ == "__main__":
    main()