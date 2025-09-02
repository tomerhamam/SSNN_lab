#!/usr/bin/env python3
"""
Interpretability and Explainability Demo for Neural Networks

This demo provides comprehensive examples of neural network interpretability
techniques including feature importance analysis, layer activation analysis,
decision boundary analysis, and bias assessment.

The demo is designed to be educational with:
- Descriptive explanations of each technique
- Elaborate examples with real data
- Guiding questions for deeper understanding
- Complementary tasks for hands-on learning
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_classification, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Import our interpretability tools
from interpretability_analyzer import InterpretabilityAnalyzer, initialize_global_analyzer
from tensorboard_logger import TensorBoardLogger, initialize_global_logger

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class SimpleNeuralNetwork(nn.Module):
    """
    A simple neural network for demonstration purposes.
    
    This network is designed to be interpretable while still being
    complex enough to demonstrate various interpretability techniques.
    """
    
    def __init__(self, input_size, hidden_sizes, num_classes, dropout_rate=0.2):
        super(SimpleNeuralNetwork, self).__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.num_classes = num_classes
        
        # Build layers dynamically
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)
    
    def get_layer_activations(self, x):
        """Get activations from each layer for analysis."""
        activations = {}
        current_input = x
        
        for i, layer in enumerate(self.network):
            current_input = layer(current_input)
            if isinstance(layer, nn.ReLU):
                activations[f'layer_{i//3}_relu'] = current_input.clone()
        
        return activations


def create_synthetic_dataset(n_samples=1000, n_features=20, n_classes=2, noise=0.1):
    """
    Create a synthetic dataset for interpretability analysis.
    
    This function creates a classification dataset with known feature
    importance patterns, making it ideal for demonstrating and validating
    interpretability techniques.
    
    Args:
        n_samples: Number of samples to generate
        n_features: Number of features
        n_classes: Number of classes
        noise: Amount of noise to add
        
    Returns:
        X: Feature matrix
        y: Target labels
        feature_names: List of feature names
        true_importance: True feature importance (for validation)
    """
    print("Creating synthetic dataset for interpretability analysis...")
    
    # Create dataset with known structure
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=int(n_features * 0.6),  # 60% of features are informative
        n_redundant=int(n_features * 0.2),    # 20% are redundant
        n_classes=n_classes,
        random_state=42
    )
    
    # Create meaningful feature names
    feature_names = [f'feature_{i:02d}' for i in range(n_features)]
    
    # Simulate true feature importance (for validation)
    true_importance = np.random.exponential(1.0, n_features)
    true_importance = true_importance / np.sum(true_importance)  # Normalize
    
    # Add some structure to make it more realistic
    true_importance[:5] *= 2.0  # First 5 features are more important
    true_importance[5:10] *= 0.5  # Next 5 are less important
    
    print(f"Dataset created: {X.shape[0]} samples, {X.shape[1]} features, {n_classes} classes")
    print(f"Class distribution: {np.bincount(y)}")
    
    return X, y, feature_names, true_importance


def create_breast_cancer_dataset():
    """
    Create a real-world dataset for interpretability analysis.
    
    The breast cancer dataset is ideal for interpretability analysis
    because it has meaningful feature names and is well-studied.
    """
    print("Loading breast cancer dataset...")
    
    # Load the dataset
    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = data.feature_names
    
    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Class distribution: {np.bincount(y)}")
    print(f"Classes: {data.target_names}")
    
    return X, y, feature_names


def prepare_data(X, y, test_size=0.2, random_state=42):
    """
    Prepare data for training and testing.
    
    This function handles train-test splitting, scaling, and conversion
    to PyTorch tensors.
    """
    print("Preparing data for training...")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.LongTensor(y_test)
    
    print(f"Training set: {X_train_tensor.shape}")
    print(f"Test set: {X_test_tensor.shape}")
    
    return (X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor), scaler


def train_model(model, train_data, test_data, epochs=100, learning_rate=0.001):
    """
    Train a neural network model with comprehensive logging.
    
    This function demonstrates how to integrate TensorBoard logging
    with model training for comprehensive monitoring.
    """
    print(f"Training model for {epochs} epochs...")
    
    # Initialize TensorBoard logger
    logger = initialize_global_logger(
        log_dir="logs/interpretability_demo",
        experiment_name="neural_network_training"
    )
    
    # Prepare data loaders
    X_train, y_train, X_test, y_test = train_data + test_data
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    # Log hyperparameters
    logger.log_hyperparameters({
        "learning_rate": learning_rate,
        "batch_size": 32,
        "epochs": epochs,
        "optimizer": "Adam",
        "scheduler": "StepLR",
        "model_architecture": str(model)
    })
    
    # Training loop
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()
            
            # Log batch-level metrics
            if batch_idx % 10 == 0:
                logger.log_scalar("batch/loss", loss.item(), 
                                step=epoch * len(train_loader) + batch_idx,
                                category="training")
        
        # Calculate epoch metrics
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total
        
        # Testing phase
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                loss = criterion(output, target)
                
                test_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                test_total += target.size(0)
                test_correct += (predicted == target).sum().item()
        
        avg_test_loss = test_loss / len(test_loader)
        test_accuracy = 100 * test_correct / test_total
        
        # Log epoch-level metrics
        logger.log_model_metrics(
            model_name="neural_network",
            metrics={
                "loss": avg_train_loss,
                "accuracy": train_accuracy,
                "val_loss": avg_test_loss,
                "val_accuracy": test_accuracy
            },
            epoch=epoch,
            phase="train"
        )
        
        # Log learning rate
        current_lr = optimizer.param_groups[0]['lr']
        logger.log_scalar("learning_rate", current_lr, epoch, "training")
        
        # Store metrics
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)
        test_losses.append(avg_test_loss)
        test_accuracies.append(test_accuracy)
        
        # Update scheduler
        scheduler.step()
        
        # Print progress
        if epoch % 10 == 0:
            print(f'Epoch {epoch:3d}: Train Loss: {avg_train_loss:.4f}, '
                  f'Train Acc: {train_accuracy:.2f}%, '
                  f'Test Loss: {avg_test_loss:.4f}, '
                  f'Test Acc: {test_accuracy:.2f}%')
    
    # Final evaluation
    final_accuracy = test_accuracies[-1]
    print(f"\nTraining completed. Final test accuracy: {final_accuracy:.2f}%")
    
    # Log final summary
    logger.log_text("training_summary", 
                   f"Final test accuracy: {final_accuracy:.2f}%\n"
                   f"Best test accuracy: {max(test_accuracies):.2f}%\n"
                   f"Training epochs: {epochs}")
    
    logger.close()
    
    return {
        "train_losses": train_losses,
        "train_accuracies": train_accuracies,
        "test_losses": test_losses,
        "test_accuracies": test_accuracies,
        "final_accuracy": final_accuracy
    }


def demo_feature_importance_analysis(model, test_data, feature_names, true_importance=None):
    """
    Demonstrate comprehensive feature importance analysis.
    
    This function shows how to use multiple attribution methods
    to understand which features are most important for predictions.
    """
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE ANALYSIS DEMO")
    print("="*60)
    
    X_test, y_test = test_data[0], test_data[1]
    
    # Initialize analyzer
    analyzer = initialize_global_analyzer(model, device="cpu")
    
    # Get a sample for analysis
    sample_idx = 0
    sample_input = X_test[sample_idx:sample_idx+1]  # Keep batch dimension
    sample_target = y_test[sample_idx].item()
    
    print(f"Analyzing sample {sample_idx} (true class: {sample_target})")
    
    # Test different attribution methods
    methods = ["integrated_gradients", "gradient_shap", "saliency", "guided_backprop"]
    attribution_results = {}
    
    for method in methods:
        print(f"\n--- {method.upper()} Analysis ---")
        
        try:
            result = analyzer.analyze_feature_importance(
                inputs=sample_input,
                method=method
            )
            
            if "error" not in result:
                attributions = np.array(result['attributions']).flatten()
                attribution_results[method] = attributions
                
                print(f"Target class: {result['target_class']}")
                print(f"Predicted class: {result['predicted_class']}")
                print(f"Attribution stats:")
                print(f"  Mean: {result['attribution_stats']['mean']:.4f}")
                print(f"  Std: {result['attribution_stats']['std']:.4f}")
                print(f"  Min: {result['attribution_stats']['min']:.4f}")
                print(f"  Max: {result['attribution_stats']['max']:.4f}")
                
                # Show top 5 most important features
                top_indices = np.argsort(np.abs(attributions))[-5:][::-1]
                print(f"Top 5 most important features:")
                for i, idx in enumerate(top_indices):
                    print(f"  {i+1}. {feature_names[idx]}: {attributions[idx]:.4f}")
                
            else:
                print(f"Error: {result['error']}")
                
        except Exception as e:
            print(f"Error with {method}: {e}")
    
    # Compare methods
    if len(attribution_results) > 1:
        print(f"\n--- METHOD COMPARISON ---")
        compare_attribution_methods(attribution_results, feature_names)
    
    # Validate against true importance (if available)
    if true_importance is not None and len(attribution_results) > 0:
        print(f"\n--- VALIDATION AGAINST TRUE IMPORTANCE ---")
        validate_attributions(attribution_results, true_importance, feature_names)
    
    # Visualize attributions
    visualize_attributions(attribution_results, feature_names)
    
    return attribution_results


def compare_attribution_methods(attribution_results, feature_names):
    """
    Compare different attribution methods to understand their agreement.
    """
    methods = list(attribution_results.keys())
    
    if len(methods) < 2:
        print("Need at least 2 methods for comparison")
        return
    
    print("Correlation between attribution methods:")
    for i, method1 in enumerate(methods):
        for method2 in methods[i+1:]:
            corr = np.corrcoef(
                attribution_results[method1],
                attribution_results[method2]
            )[0, 1]
            print(f"  {method1} vs {method2}: {corr:.4f}")
    
    # Find features where methods disagree most
    print("\nFeatures with highest disagreement between methods:")
    disagreements = []
    
    for i in range(len(feature_names)):
        values = [attribution_results[method][i] for method in methods]
        disagreement = np.std(values)
        disagreements.append((i, disagreement))
    
    disagreements.sort(key=lambda x: x[1], reverse=True)
    
    for i, (feature_idx, disagreement) in enumerate(disagreements[:5]):
        print(f"  {i+1}. {feature_names[feature_idx]}: {disagreement:.4f}")
        for method in methods:
            print(f"     {method}: {attribution_results[method][feature_idx]:.4f}")


def validate_attributions(attribution_results, true_importance, feature_names):
    """
    Validate attribution methods against known true importance.
    """
    print("Correlation with true feature importance:")
    
    for method, attributions in attribution_results.items():
        # Use absolute values for correlation
        corr = np.corrcoef(
            np.abs(attributions),
            true_importance
        )[0, 1]
        print(f"  {method}: {corr:.4f}")
    
    # Show where attributions differ most from true importance
    print("\nFeatures where attributions differ most from true importance:")
    differences = []
    
    for i in range(len(feature_names)):
        true_val = true_importance[i]
        # Use the first available attribution method
        method = list(attribution_results.keys())[0]
        attr_val = np.abs(attribution_results[method][i])
        difference = abs(true_val - attr_val)
        differences.append((i, difference, true_val, attr_val))
    
    differences.sort(key=lambda x: x[1], reverse=True)
    
    for i, (feature_idx, diff, true_val, attr_val) in enumerate(differences[:5]):
        print(f"  {i+1}. {feature_names[feature_idx]}:")
        print(f"     True: {true_val:.4f}, Attributed: {attr_val:.4f}, Diff: {diff:.4f}")


def visualize_attributions(attribution_results, feature_names):
    """
    Create visualizations of attribution results.
    """
    if not attribution_results:
        print("No attribution results to visualize")
        return
    
    n_methods = len(attribution_results)
    fig, axes = plt.subplots(2, (n_methods + 1) // 2, figsize=(15, 10))
    if n_methods == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, (method, attributions) in enumerate(attribution_results.items()):
        ax = axes[i]
        
        # Create bar plot
        sorted_indices = np.argsort(np.abs(attributions))[-10:]  # Top 10 features
        sorted_attributions = attributions[sorted_indices]
        sorted_names = [feature_names[idx] for idx in sorted_indices]
        
        colors = ['red' if x < 0 else 'blue' for x in sorted_attributions]
        bars = ax.barh(range(len(sorted_names)), sorted_attributions, color=colors, alpha=0.7)
        
        ax.set_yticks(range(len(sorted_names)))
        ax.set_yticklabels(sorted_names)
        ax.set_xlabel('Attribution Value')
        ax.set_title(f'{method.replace("_", " ").title()}')
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for j, (bar, val) in enumerate(zip(bars, sorted_attributions)):
            ax.text(val + (0.01 if val >= 0 else -0.01), j, f'{val:.3f}', 
                   va='center', ha='left' if val >= 0 else 'right')
    
    # Hide unused subplots
    for i in range(n_methods, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('attribution_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Attribution visualization saved as 'attribution_analysis.png'")


def demo_layer_activation_analysis(model, test_data):
    """
    Demonstrate layer-wise activation analysis.
    
    This function shows how to analyze what each layer learns
    and identify potential issues in the network.
    """
    print("\n" + "="*60)
    print("LAYER ACTIVATION ANALYSIS DEMO")
    print("="*60)
    
    X_test, y_test = test_data[0], test_data[1]
    
    # Initialize analyzer
    analyzer = initialize_global_analyzer(model, device="cpu")
    
    # Get a sample for analysis
    sample_input = X_test[0:1]  # Keep batch dimension
    
    print("Analyzing layer activations...")
    
    try:
        result = analyzer.analyze_layer_activations(sample_input)
        
        if "error" not in result:
            print(f"Analyzed {result['total_layers']} layers")
            
            for layer_name, activation_data in result["layer_activations"].items():
                print(f"\n--- {layer_name.upper()} ---")
                print(f"Shape: {activation_data['shape']}")
                print(f"Mean: {activation_data['mean']:.4f}")
                print(f"Std: {activation_data['std']:.4f}")
                print(f"Min: {activation_data['min']:.4f}")
                print(f"Max: {activation_data['max']:.4f}")
                print(f"Sparsity: {activation_data['sparsity']:.4f}")
                
                # Identify potential issues
                issues = []
                if activation_data['sparsity'] > 0.9:
                    issues.append("High sparsity - consider regularization")
                if activation_data['std'] < 0.01:
                    issues.append("Low variance - potential dead neurons")
                if activation_data['mean'] < -1.0 or activation_data['mean'] > 1.0:
                    issues.append("Mean far from zero - check initialization")
                
                if issues:
                    print("⚠️  Potential issues:")
                    for issue in issues:
                        print(f"   - {issue}")
                else:
                    print("✅ No obvious issues detected")
        
        else:
            print(f"Error: {result['error']}")
            
    except Exception as e:
        print(f"Error in layer activation analysis: {e}")


def demo_decision_boundary_analysis(model, test_data):
    """
    Demonstrate decision boundary analysis through perturbation.
    
    This function shows how to assess model robustness and
    identify potential adversarial vulnerabilities.
    """
    print("\n" + "="*60)
    print("DECISION BOUNDARY ANALYSIS DEMO")
    print("="*60)
    
    X_test, y_test = test_data[0], test_data[1]
    
    # Initialize analyzer
    analyzer = initialize_global_analyzer(model, device="cpu")
    
    # Get a sample for analysis
    sample_input = X_test[0:1]  # Keep batch dimension
    sample_target = y_test[0].item()
    
    print(f"Analyzing decision boundary for sample (true class: {sample_target})")
    
    try:
        result = analyzer.analyze_decision_boundary(sample_input, num_samples=50)
        
        if "error" not in result:
            print(f"Original prediction: {result['original_prediction']}")
            print(f"Original confidence: {result['original_confidence']:.4f}")
            
            perturbation_analysis = result['perturbation_analysis']
            print(f"\nPerturbation Analysis:")
            print(f"  Number of samples: {perturbation_analysis['num_samples']}")
            print(f"  Prediction changes: {perturbation_analysis['prediction_changes']}")
            print(f"  Robustness ratio: {perturbation_analysis['robustness_ratio']:.4f}")
            print(f"  Average confidence: {perturbation_analysis['avg_confidence']:.4f}")
            print(f"  Confidence std: {perturbation_analysis['confidence_std']:.4f}")
            
            # Interpret results
            robustness = perturbation_analysis['robustness_ratio']
            if robustness < 0.8:
                print("\n⚠️  Low robustness detected!")
                print("   - Model is sensitive to input perturbations")
                print("   - Consider adversarial training or data augmentation")
            elif robustness < 0.9:
                print("\n⚠️  Moderate robustness")
                print("   - Model shows some sensitivity to perturbations")
                print("   - Monitor for potential adversarial vulnerabilities")
            else:
                print("\n✅ Good robustness")
                print("   - Model is relatively stable to input perturbations")
            
            # Visualize robustness
            visualize_robustness_analysis(perturbation_analysis)
        
        else:
            print(f"Error: {result['error']}")
            
    except Exception as e:
        print(f"Error in decision boundary analysis: {e}")


def visualize_robustness_analysis(perturbation_analysis):
    """
    Visualize robustness analysis results.
    """
    perturbations = perturbation_analysis['perturbations']
    confidences = perturbation_analysis['confidences']
    predictions = perturbation_analysis['predictions']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot confidence vs perturbation level
    ax1.scatter(perturbations, confidences, alpha=0.6, c='blue')
    ax1.set_xlabel('Perturbation Level')
    ax1.set_ylabel('Model Confidence')
    ax1.set_title('Model Confidence vs Perturbation Level')
    ax1.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(perturbations, confidences, 1)
    p = np.poly1d(z)
    ax1.plot(perturbations, p(perturbations), "r--", alpha=0.8, label=f'Trend (slope: {z[0]:.3f})')
    ax1.legend()
    
    # Plot prediction changes
    unique_perturbations = sorted(set(perturbations))
    prediction_change_rates = []
    
    for pert_level in unique_perturbations:
        indices = [i for i, p in enumerate(perturbations) if p == pert_level]
        if indices:
            original_pred = predictions[0]  # Assuming first prediction is original
            changes = sum(1 for i in indices if predictions[i] != original_pred)
            change_rate = changes / len(indices)
            prediction_change_rates.append(change_rate)
        else:
            prediction_change_rates.append(0)
    
    ax2.plot(unique_perturbations, prediction_change_rates, 'o-', color='red', linewidth=2)
    ax2.set_xlabel('Perturbation Level')
    ax2.set_ylabel('Prediction Change Rate')
    ax2.set_title('Prediction Change Rate vs Perturbation Level')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('robustness_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Robustness visualization saved as 'robustness_analysis.png'")


def demo_bias_and_fairness_analysis(model, test_data):
    """
    Demonstrate bias and fairness assessment.
    
    This function shows how to assess model fairness across
    different groups and identify potential bias.
    """
    print("\n" + "="*60)
    print("BIAS AND FAIRNESS ANALYSIS DEMO")
    print("="*60)
    
    X_test, y_test = test_data[0], test_data[1]
    
    # Initialize analyzer
    analyzer = initialize_global_analyzer(model, device="cpu")
    
    # Get a sample for analysis
    sample_input = X_test[0:1]  # Keep batch dimension
    
    # Simulate different demographic groups
    # In a real scenario, you would have actual demographic data
    sensitive_attributes = {
        "group_a": "demographic_group_1",
        "group_b": "demographic_group_2", 
        "group_c": "demographic_group_3"
    }
    
    print("Analyzing bias and fairness across demographic groups...")
    print("Note: This is a simulated analysis for demonstration purposes")
    
    try:
        result = analyzer.analyze_bias_and_fairness(sample_input, sensitive_attributes)
        
        if "error" not in result:
            print(f"\nGroup Predictions:")
            for group_name, group_data in result['group_predictions'].items():
                print(f"  {group_name}:")
                print(f"    Prediction: {group_data['prediction']}")
                print(f"    Confidence: {group_data['confidence']:.4f}")
            
            fairness_metrics = result['fairness_metrics']
            print(f"\nFairness Metrics:")
            print(f"  Confidence gap: {fairness_metrics['confidence_gap']:.4f}")
            print(f"  Prediction diversity: {fairness_metrics['prediction_diversity']:.4f}")
            print(f"  Average confidence: {fairness_metrics['avg_confidence']:.4f}")
            print(f"  Confidence std: {fairness_metrics['confidence_std']:.4f}")
            
            bias_indicators = result['bias_indicators']
            print(f"\nBias Indicators:")
            print(f"  High confidence gap: {bias_indicators['high_confidence_gap']}")
            print(f"  Low prediction diversity: {bias_indicators['low_prediction_diversity']}")
            print(f"  Potential bias: {bias_indicators['potential_bias']}")
            
            # Interpret results
            if bias_indicators['potential_bias']:
                print("\n⚠️  Potential bias detected!")
                print("   - Review training data for representation issues")
                print("   - Consider bias mitigation techniques")
                print("   - Ensure diverse training data")
            else:
                print("\n✅ No significant bias detected")
                print("   - Model appears to treat groups fairly")
                print("   - Continue monitoring for bias")
            
            # Visualize fairness metrics
            visualize_fairness_analysis(result)
        
        else:
            print(f"Error: {result['error']}")
            
    except Exception as e:
        print(f"Error in bias and fairness analysis: {e}")


def visualize_fairness_analysis(result):
    """
    Visualize fairness analysis results.
    """
    group_predictions = result['group_predictions']
    fairness_metrics = result['fairness_metrics']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot confidence by group
    groups = list(group_predictions.keys())
    confidences = [group_predictions[group]['confidence'] for group in groups]
    predictions = [group_predictions[group]['prediction'] for group in groups]
    
    colors = ['red' if p == 0 else 'blue' for p in predictions]
    bars = ax1.bar(groups, confidences, color=colors, alpha=0.7)
    
    ax1.set_ylabel('Model Confidence')
    ax1.set_title('Model Confidence by Demographic Group')
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    
    # Add confidence values on bars
    for bar, conf in zip(bars, confidences):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{conf:.3f}', ha='center', va='bottom')
    
    # Plot fairness metrics
    metrics = ['Confidence Gap', 'Prediction Diversity', 'Avg Confidence']
    values = [
        fairness_metrics['confidence_gap'],
        fairness_metrics['prediction_diversity'],
        fairness_metrics['avg_confidence']
    ]
    
    bars = ax2.bar(metrics, values, color=['red', 'orange', 'green'], alpha=0.7)
    ax2.set_ylabel('Metric Value')
    ax2.set_title('Fairness Metrics')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    
    # Add values on bars
    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{val:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('fairness_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Fairness visualization saved as 'fairness_analysis.png'")


def demo_explanation_generation(model, test_data, feature_names):
    """
    Demonstrate explanation generation using different methods.
    
    This function shows how to generate human-readable explanations
    for model predictions using SHAP, LIME, and Grad-CAM.
    """
    print("\n" + "="*60)
    print("EXPLANATION GENERATION DEMO")
    print("="*60)
    
    X_test, y_test = test_data[0], test_data[1]
    
    # Initialize analyzer
    analyzer = initialize_global_analyzer(model, device="cpu")
    
    # Get a sample for analysis
    sample_input = X_test[0:1]  # Keep batch dimension
    sample_target = y_test[0].item()
    
    print(f"Generating explanations for sample (true class: {sample_target})")
    
    # Test different explanation methods
    methods = ["shap", "lime", "grad_cam"]
    
    for method in methods:
        print(f"\n--- {method.upper()} Explanation ---")
        
        try:
            explanation = analyzer.generate_explanations(sample_input, method=method)
            
            if "error" not in explanation:
                prediction = explanation['prediction']
                print(f"Predicted class: {prediction['class']}")
                print(f"Confidence: {prediction['confidence']:.4f}")
                print(f"All probabilities: {[f'{p:.3f}' for p in prediction['all_probabilities']]}")
                
                if method == "shap" and "shap_values" in explanation:
                    print("SHAP values generated successfully")
                    print("Note: SHAP values are complex objects - see visualization for details")
                
                elif method == "lime" and "lime_explanation" in explanation:
                    lime_data = explanation['lime_explanation']
                    print(f"Local prediction: {lime_data['local_prediction']:.4f}")
                    print("Top feature contributions:")
                    for feature, importance in lime_data['feature_importance'][:5]:
                        print(f"  {feature}: {importance:.4f}")
                
                elif method == "grad_cam" and "grad_cam" in explanation:
                    cam_data = explanation['grad_cam']
                    print(f"Grad-CAM generated for layer: {cam_data['target_layer']}")
                    print("Note: Grad-CAM values are complex objects - see visualization for details")
                
            else:
                print(f"Error: {explanation['error']}")
                
        except Exception as e:
            print(f"Error with {method}: {e}")


def create_comprehensive_report(model, test_data, feature_names, attribution_results):
    """
    Create a comprehensive interpretability report.
    """
    print("\n" + "="*60)
    print("CREATING COMPREHENSIVE INTERPRETABILITY REPORT")
    print("="*60)
    
    # Initialize analyzer
    analyzer = initialize_global_analyzer(model, device="cpu")
    
    # Run all analyses
    print("Running comprehensive analysis...")
    
    # Feature importance (already done)
    print("✓ Feature importance analysis completed")
    
    # Layer activations
    sample_input = test_data[0][0:1]
    layer_result = analyzer.analyze_layer_activations(sample_input)
    print("✓ Layer activation analysis completed")
    
    # Decision boundary
    decision_result = analyzer.analyze_decision_boundary(sample_input, num_samples=30)
    print("✓ Decision boundary analysis completed")
    
    # Bias and fairness
    sensitive_attributes = {"group_a": "demo_1", "group_b": "demo_2", "group_c": "demo_3"}
    bias_result = analyzer.analyze_bias_and_fairness(sample_input, sensitive_attributes)
    print("✓ Bias and fairness analysis completed")
    
    # Generate comprehensive report
    report = analyzer.create_interpretability_report("comprehensive_interpretability_report.json")
    
    print(f"\nComprehensive report saved to 'comprehensive_interpretability_report.json'")
    print(f"Report contains {report['summary']['total_analyses']} analyses")
    print(f"Available methods: {', '.join(report['summary']['available_methods'])}")
    
    # Print recommendations
    if 'recommendations' in report:
        print(f"\nRecommendations:")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"  {i}. {rec}")
    
    return report


def main():
    """
    Main function to run the complete interpretability demo.
    """
    print("="*80)
    print("NEURAL NETWORK INTERPRETABILITY AND EXPLAINABILITY DEMO")
    print("="*80)
    print("This demo provides comprehensive examples of neural network")
    print("interpretability techniques with educational explanations.")
    print("="*80)
    
    # Choose dataset
    print("\nChoose dataset:")
    print("1. Synthetic dataset (with known feature importance)")
    print("2. Breast cancer dataset (real-world data)")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        # Create synthetic dataset
        X, y, feature_names, true_importance = create_synthetic_dataset()
        dataset_name = "synthetic"
    else:
        # Use breast cancer dataset
        X, y, feature_names = create_breast_cancer_dataset()
        true_importance = None
        dataset_name = "breast_cancer"
    
    # Prepare data
    train_data, test_data, scaler = prepare_data(X, y)
    
    # Create model
    input_size = X.shape[1]
    hidden_sizes = [64, 32, 16]
    num_classes = len(np.unique(y))
    
    model = SimpleNeuralNetwork(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        num_classes=num_classes,
        dropout_rate=0.2
    )
    
    print(f"\nModel architecture:")
    print(f"  Input size: {input_size}")
    print(f"  Hidden sizes: {hidden_sizes}")
    print(f"  Output classes: {num_classes}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Train model
    training_results = train_model(model, train_data, test_data, epochs=50)
    
    # Run interpretability demos
    print(f"\n{'='*80}")
    print("STARTING INTERPRETABILITY ANALYSIS")
    print(f"{'='*80}")
    
    # 1. Feature importance analysis
    attribution_results = demo_feature_importance_analysis(
        model, test_data, feature_names, true_importance
    )
    
    # 2. Layer activation analysis
    demo_layer_activation_analysis(model, test_data)
    
    # 3. Decision boundary analysis
    demo_decision_boundary_analysis(model, test_data)
    
    # 4. Bias and fairness analysis
    demo_bias_and_fairness_analysis(model, test_data)
    
    # 5. Explanation generation
    demo_explanation_generation(model, test_data, feature_names)
    
    # 6. Comprehensive report
    comprehensive_report = create_comprehensive_report(
        model, test_data, feature_names, attribution_results
    )
    
    # Final summary
    print(f"\n{'='*80}")
    print("DEMO COMPLETED SUCCESSFULLY")
    print(f"{'='*80}")
    print("Generated files:")
    print("  - attribution_analysis.png")
    print("  - robustness_analysis.png") 
    print("  - fairness_analysis.png")
    print("  - comprehensive_interpretability_report.json")
    print("\nNext steps:")
    print("  1. Review the generated visualizations")
    print("  2. Examine the comprehensive report")
    print("  3. Try the guiding questions below")
    print("  4. Experiment with different models and datasets")
    
    # Guiding questions
    print(f"\n{'='*80}")
    print("GUIDING QUESTIONS FOR DEEPER UNDERSTANDING")
    print(f"{'='*80}")
    
    questions = [
        "1. Which attribution methods gave the most consistent results? Why might this be?",
        "2. How do the layer activations reveal the model's learning patterns?",
        "3. What does the robustness analysis tell you about model reliability?",
        "4. How would you use the bias analysis to improve model fairness?",
        "5. Which explanation method would you choose for different use cases?",
        "6. How could you use these insights to improve model performance?",
        "7. What additional analyses would you perform for your specific domain?",
        "8. How would you communicate these findings to non-technical stakeholders?"
    ]
    
    for question in questions:
        print(f"  {question}")
    
    print(f"\n{'='*80}")
    print("COMPLEMENTARY TASKS FOR HANDS-ON LEARNING")
    print(f"{'='*80}")
    
    tasks = [
        "1. Modify the model architecture and observe how it affects interpretability",
        "2. Try different datasets and compare interpretability patterns",
        "3. Implement custom attribution methods for your specific use case",
        "4. Create an interactive dashboard combining TensorBoard and interpretability",
        "5. Develop automated bias detection and mitigation pipelines",
        "6. Design experiments to validate attribution method reliability",
        "7. Build explanation systems for different stakeholder groups",
        "8. Integrate interpretability analysis into your model development workflow"
    ]
    
    for task in tasks:
        print(f"  {task}")
    
    print(f"\n{'='*80}")
    print("Thank you for exploring neural network interpretability!")
    print("Continue experimenting and learning to build more trustworthy AI systems.")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
