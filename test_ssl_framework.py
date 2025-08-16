"""
Test script for the modular SSL framework.
Tests different datasets and pretext tasks.
"""

from ssl_framework import run_ssl_experiment
import warnings
warnings.filterwarnings('ignore')

def test_framework():
    """Test the SSL framework with different configurations."""
    
    print("🚀 TESTING MODULAR SSL FRAMEWORK")
    print("="*60)
    
    # Test configurations
    configs = [
        # (dataset, pretext_task, architecture, epochs, description)
        ('cifar10', 'rotation', [256, 128], 20, "CIFAR-10 + Rotation (Medium)"),
        ('cifar10', 'jigsaw', [256, 128], 20, "CIFAR-10 + Jigsaw (Medium)"),
        ('cifar10', 'rotation', [512, 256, 128], 15, "CIFAR-10 + Rotation (Deep)"),
        ('fashion_mnist', 'rotation', [256, 128], 20, "Fashion-MNIST + Rotation"),
    ]
    
    results = []
    
    for dataset, task, arch, epochs, desc in configs:
        print(f"\n{'='*60}")
        print(f"🧪 {desc}")
        print(f"{'='*60}")
        
        try:
            # Run experiment
            result = run_ssl_experiment(
                dataset_name=dataset,
                pretext_task=task,
                architecture=arch,
                epochs=epochs,
                learning_rate=0.01,
                dropout_rate=0.1,
                num_rotations=8,
                verbose=True
            )
            
            # Extract key metrics
            transfer = result['transfer_results']
            train = result['train_results']
            
            summary = {
                'config': desc,
                'dataset': dataset,
                'task': task,
                'architecture': arch,
                'pretext_acc': train['final_pretext_acc'],
                'ssl_acc': transfer['best_ssl'],
                'baseline_acc': transfer['baseline'],
                'improvement': transfer['improvement']
            }
            
            results.append(summary)
            
            print(f"\n✅ Completed: SSL {transfer['best_ssl']:.3f} vs Baseline {transfer['baseline']:.3f}")
            print(f"📈 Improvement: {transfer['improvement']*100:+.1f}%")
            
        except Exception as e:
            print(f"❌ Failed: {e}")
            results.append({
                'config': desc,
                'error': str(e)
            })
    
    # Summary table
    print(f"\n{'='*80}")
    print("📊 EXPERIMENT SUMMARY")
    print(f"{'='*80}")
    print(f"{'Configuration':<30} | {'Pretext':<8} | {'SSL Acc':<8} | {'Baseline':<8} | {'Improve':<8}")
    print("-" * 80)
    
    for r in results:
        if 'error' in r:
            print(f"{r['config']:<30} | {'ERROR':<8} | {'-':<8} | {'-':<8} | {'-':<8}")
        else:
            print(f"{r['config']:<30} | {r['pretext_acc']:<8.3f} | {r['ssl_acc']:<8.3f} | {r['baseline_acc']:<8.3f} | {r['improvement']*100:+7.1f}%")
    
    # Find best result
    valid_results = [r for r in results if 'error' not in r]
    if valid_results:
        best = max(valid_results, key=lambda x: x['improvement'])
        print(f"\n🏆 BEST CONFIGURATION:")
        print(f"   {best['config']}")
        print(f"   Improvement: {best['improvement']*100:+.1f}%")
        print(f"   SSL: {best['ssl_acc']:.3f}, Baseline: {best['baseline_acc']:.3f}")
    
    return results

def quick_test():
    """Quick test with small dataset."""
    print("⚡ Quick test with CIFAR-10...")
    
    result = run_ssl_experiment(
        dataset_name='cifar10',
        pretext_task='rotation',
        architecture=[256, 128],
        epochs=10,
        learning_rate=0.02,
        num_rotations=6,
        verbose=True
    )
    
    transfer = result['transfer_results']
    print(f"\nQuick test result:")
    print(f"  SSL: {transfer['best_ssl']:.3f}")
    print(f"  Baseline: {transfer['baseline']:.3f}")
    print(f"  Improvement: {transfer['improvement']*100:+.1f}%")
    
    return result

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'quick':
        quick_test()
    else:
        test_framework()