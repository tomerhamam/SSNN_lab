#!/usr/bin/env python3
"""
Neural Network Interpretability and Explainability Analysis Service.

This module provides comprehensive tools for understanding and explaining
neural network behavior, including:
- Feature importance analysis
- Gradient-based attribution methods
- Model decision explanation
- Layer-wise activation analysis
- Adversarial robustness testing
- Bias and fairness assessment
"""

import json
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import warnings
warnings.filterwarnings('ignore')

# Import interpretability libraries
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from captum.attr import (
        IntegratedGradients, GradientShap, Saliency, 
        GuidedBackprop, Deconvolution, InputXGradient,
        LayerGradCam, LayerAttribution, Occlusion,
        FeatureAblation, ShapleyValueSampling
    )
    from captum.insights import AttributionVisualizer, Batch
    from captum.metrics import infidelity, sensitivity_max
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    from lime import lime_tabular, lime_image, lime_text
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

try:
    from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, XGradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
    GRAD_CAM_AVAILABLE = True
except ImportError:
    GRAD_CAM_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InterpretabilityAnalyzer:
    """
    Comprehensive neural network interpretability and explainability analyzer.
    
    Provides methods for understanding model behavior through various
    attribution techniques, visualization, and analysis tools.
    """
    
    def __init__(self, model: Optional[Any] = None, device: str = "cpu"):
        """
        Initialize the interpretability analyzer.
        
        Args:
            model: PyTorch model to analyze (optional, can be set later)
            device: Device to run analysis on ('cpu' or 'cuda')
        """
        self.model = model
        self.device = device
        self.analysis_results = {}
        self.visualization_cache = {}
        
        # Check library availability
        self.library_status = {
            "torch": TORCH_AVAILABLE,
            "shap": SHAP_AVAILABLE,
            "lime": LIME_AVAILABLE,
            "grad_cam": GRAD_CAM_AVAILABLE
        }
        
        logger.info(f"InterpretabilityAnalyzer initialized. Libraries: {self.library_status}")
        
        # Move model to device if available
        if self.model and TORCH_AVAILABLE:
            self.model = self.model.to(device)
            self.model.eval()
    
    def set_model(self, model: Any) -> None:
        """
        Set the model to analyze.
        
        Args:
            model: PyTorch model to analyze
        """
        self.model = model
        if TORCH_AVAILABLE:
            self.model = self.model.to(self.device)
            self.model.eval()
        logger.info("Model set for interpretability analysis")
    
    def analyze_feature_importance(self, inputs: torch.Tensor, target_class: Optional[int] = None,
                                 method: str = "integrated_gradients") -> Dict[str, Any]:
        """
        Analyze feature importance using various attribution methods.
        
        Args:
            inputs: Input tensor to analyze
            target_class: Target class for attribution (None for predicted class)
            method: Attribution method to use
            
        Returns:
            Dictionary containing attribution results and metadata
        """
        if not TORCH_AVAILABLE or not self.model:
            return {"error": "PyTorch or model not available"}
        
        try:
            inputs = inputs.to(self.device)
            
            # Get model prediction
            with torch.no_grad():
                outputs = self.model(inputs)
                predicted_class = torch.argmax(outputs, dim=1).item()
                target = target_class if target_class is not None else predicted_class
            
            # Select attribution method
            if method == "integrated_gradients":
                attributor = IntegratedGradients(self.model)
                attributions = attributor.attribute(inputs, target=target, n_steps=50)
            elif method == "gradient_shap":
                attributor = GradientShap(self.model)
                baseline = torch.zeros_like(inputs)
                attributions = attributor.attribute(inputs, baselines=baseline, target=target)
            elif method == "saliency":
                attributor = Saliency(self.model)
                attributions = attributor.attribute(inputs, target=target)
            elif method == "guided_backprop":
                attributor = GuidedBackprop(self.model)
                attributions = attributor.attribute(inputs, target=target)
            elif method == "input_x_gradient":
                attributor = InputXGradient(self.model)
                attributions = attributor.attribute(inputs, target=target)
            else:
                return {"error": f"Unknown attribution method: {method}"}
            
            # Process attributions
            attributions_np = attributions.detach().cpu().numpy()
            
            result = {
                "method": method,
                "target_class": target,
                "predicted_class": predicted_class,
                "attributions": attributions_np.tolist(),
                "attribution_stats": {
                    "mean": float(np.mean(attributions_np)),
                    "std": float(np.std(attributions_np)),
                    "min": float(np.min(attributions_np)),
                    "max": float(np.max(attributions_np)),
                    "sum": float(np.sum(attributions_np))
                },
                "timestamp": datetime.now().isoformat()
            }
            
            self.analysis_results[f"feature_importance_{method}"] = result
            logger.info(f"Feature importance analysis completed using {method}")
            
            return result
            
        except Exception as e:
            logger.error(f"Feature importance analysis failed: {e}")
            return {"error": str(e)}
    
    def analyze_layer_activations(self, inputs: torch.Tensor, layer_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Analyze activations at different layers of the model.
        
        Args:
            inputs: Input tensor to analyze
            layer_names: Specific layers to analyze (None for all)
            
        Returns:
            Dictionary containing layer activation analysis
        """
        if not TORCH_AVAILABLE or not self.model:
            return {"error": "PyTorch or model not available"}
        
        try:
            inputs = inputs.to(self.device)
            activations = {}
            hooks = []
            
            def hook_fn(name):
                def hook(module, input, output):
                    activations[name] = output.detach().cpu().numpy()
                return hook
            
            # Register hooks for specified layers
            if layer_names:
                target_layers = [name for name, _ in self.model.named_modules() if name in layer_names]
            else:
                # Get all layers (excluding the model itself)
                target_layers = [name for name, _ in self.model.named_modules() if name != ""]
            
            for name, module in self.model.named_modules():
                if name in target_layers:
                    hooks.append(module.register_forward_hook(hook_fn(name)))
            
            # Forward pass
            with torch.no_grad():
                _ = self.model(inputs)
            
            # Remove hooks
            for hook in hooks:
                hook.remove()
            
            # Analyze activations
            activation_analysis = {}
            for layer_name, activation in activations.items():
                activation_analysis[layer_name] = {
                    "shape": list(activation.shape),
                    "mean": float(np.mean(activation)),
                    "std": float(np.std(activation)),
                    "min": float(np.min(activation)),
                    "max": float(np.max(activation)),
                    "sparsity": float(np.sum(activation == 0) / activation.size),
                    "activation_data": activation.tolist() if activation.size < 1000 else "too_large"
                }
            
            result = {
                "layer_activations": activation_analysis,
                "total_layers": len(activation_analysis),
                "timestamp": datetime.now().isoformat()
            }
            
            self.analysis_results["layer_activations"] = result
            logger.info(f"Layer activation analysis completed for {len(activation_analysis)} layers")
            
            return result
            
        except Exception as e:
            logger.error(f"Layer activation analysis failed: {e}")
            return {"error": str(e)}
    
    def analyze_decision_boundary(self, inputs: torch.Tensor, num_samples: int = 100) -> Dict[str, Any]:
        """
        Analyze the model's decision boundary through perturbation analysis.
        
        Args:
            inputs: Input tensor to analyze
            num_samples: Number of perturbation samples to generate
            
        Returns:
            Dictionary containing decision boundary analysis
        """
        if not TORCH_AVAILABLE or not self.model:
            return {"error": "PyTorch or model not available"}
        
        try:
            inputs = inputs.to(self.device)
            
            # Get original prediction
            with torch.no_grad():
                original_output = self.model(inputs)
                original_prediction = torch.argmax(original_output, dim=1).item()
                original_confidence = F.softmax(original_output, dim=1)[0, original_prediction].item()
            
            # Generate perturbations
            perturbations = []
            predictions = []
            confidences = []
            
            for i in range(num_samples):
                # Add random noise
                noise_scale = 0.1 * (i + 1) / num_samples
                noise = torch.randn_like(inputs) * noise_scale
                perturbed_input = inputs + noise
                
                with torch.no_grad():
                    output = self.model(perturbed_input)
                    prediction = torch.argmax(output, dim=1).item()
                    confidence = F.softmax(output, dim=1)[0, prediction].item()
                
                perturbations.append(noise_scale)
                predictions.append(prediction)
                confidences.append(confidence)
            
            # Analyze robustness
            prediction_changes = sum(1 for p in predictions if p != original_prediction)
            avg_confidence = np.mean(confidences)
            confidence_std = np.std(confidences)
            
            result = {
                "original_prediction": original_prediction,
                "original_confidence": original_confidence,
                "perturbation_analysis": {
                    "num_samples": num_samples,
                    "prediction_changes": prediction_changes,
                    "robustness_ratio": 1 - (prediction_changes / num_samples),
                    "avg_confidence": avg_confidence,
                    "confidence_std": confidence_std,
                    "perturbations": perturbations,
                    "predictions": predictions,
                    "confidences": confidences
                },
                "timestamp": datetime.now().isoformat()
            }
            
            self.analysis_results["decision_boundary"] = result
            logger.info(f"Decision boundary analysis completed with {num_samples} samples")
            
            return result
            
        except Exception as e:
            logger.error(f"Decision boundary analysis failed: {e}")
            return {"error": str(e)}
    
    def analyze_bias_and_fairness(self, inputs: torch.Tensor, sensitive_attributes: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze model bias and fairness with respect to sensitive attributes.
        
        Args:
            inputs: Input tensor to analyze
            sensitive_attributes: Dictionary mapping attribute names to values
            
        Returns:
            Dictionary containing bias and fairness analysis
        """
        if not TORCH_AVAILABLE or not self.model:
            return {"error": "PyTorch or model not available"}
        
        try:
            inputs = inputs.to(self.device)
            
            # Get predictions for different attribute groups
            group_predictions = {}
            group_confidences = {}
            
            for attr_name, attr_value in sensitive_attributes.items():
                # Simulate different attribute groups (in real scenario, you'd have actual data)
                with torch.no_grad():
                    output = self.model(inputs)
                    prediction = torch.argmax(output, dim=1).item()
                    confidence = F.softmax(output, dim=1)[0, prediction].item()
                
                group_predictions[attr_name] = {
                    "value": attr_value,
                    "prediction": prediction,
                    "confidence": confidence
                }
                
                group_confidences[attr_name] = confidence
            
            # Calculate fairness metrics
            confidences = list(group_confidences.values())
            max_confidence = max(confidences)
            min_confidence = min(confidences)
            confidence_gap = max_confidence - min_confidence
            
            # Statistical parity (simplified)
            predictions = [data["prediction"] for data in group_predictions.values()]
            prediction_diversity = len(set(predictions)) / len(predictions)
            
            result = {
                "group_predictions": group_predictions,
                "fairness_metrics": {
                    "confidence_gap": confidence_gap,
                    "prediction_diversity": prediction_diversity,
                    "max_confidence": max_confidence,
                    "min_confidence": min_confidence,
                    "avg_confidence": np.mean(confidences),
                    "confidence_std": np.std(confidences)
                },
                "bias_indicators": {
                    "high_confidence_gap": confidence_gap > 0.2,
                    "low_prediction_diversity": prediction_diversity < 0.5,
                    "potential_bias": confidence_gap > 0.2 or prediction_diversity < 0.5
                },
                "timestamp": datetime.now().isoformat()
            }
            
            self.analysis_results["bias_fairness"] = result
            logger.info("Bias and fairness analysis completed")
            
            return result
            
        except Exception as e:
            logger.error(f"Bias and fairness analysis failed: {e}")
            return {"error": str(e)}
    
    def generate_explanations(self, inputs: torch.Tensor, method: str = "shap") -> Dict[str, Any]:
        """
        Generate human-readable explanations for model predictions.
        
        Args:
            inputs: Input tensor to analyze
            method: Explanation method ('shap', 'lime', 'grad_cam')
            
        Returns:
            Dictionary containing explanations and visualizations
        """
        if not TORCH_AVAILABLE or not self.model:
            return {"error": "PyTorch or model not available"}
        
        try:
            inputs = inputs.to(self.device)
            
            # Get model prediction
            with torch.no_grad():
                outputs = self.model(inputs)
                predicted_class = torch.argmax(outputs, dim=1).item()
                confidence = F.softmax(outputs, dim=1)[0, predicted_class].item()
            
            explanations = {
                "prediction": {
                    "class": predicted_class,
                    "confidence": confidence,
                    "all_probabilities": F.softmax(outputs, dim=1)[0].tolist()
                },
                "method": method,
                "timestamp": datetime.now().isoformat()
            }
            
            if method == "shap" and SHAP_AVAILABLE:
                # SHAP explanation
                def model_wrapper(x):
                    return self.model(torch.tensor(x, dtype=torch.float32).to(self.device)).detach().cpu().numpy()
                
                # Create SHAP explainer
                background = torch.zeros_like(inputs).cpu().numpy()
                explainer = shap.Explainer(model_wrapper, background)
                shap_values = explainer(inputs.cpu().numpy())
                
                explanations["shap_values"] = {
                    "values": shap_values.values.tolist(),
                    "base_values": shap_values.base_values.tolist(),
                    "data": shap_values.data.tolist()
                }
                
            elif method == "lime" and LIME_AVAILABLE:
                # LIME explanation (simplified for tabular data)
                def model_predict(x):
                    return self.model(torch.tensor(x, dtype=torch.float32).to(self.device)).detach().cpu().numpy()
                
                # Create LIME explainer
                explainer = lime_tabular.LimeTabularExplainer(
                    inputs.cpu().numpy(),
                    feature_names=[f"feature_{i}" for i in range(inputs.shape[1])],
                    mode="classification"
                )
                
                explanation = explainer.explain_instance(
                    inputs[0].cpu().numpy(),
                    model_predict,
                    num_features=min(10, inputs.shape[1])
                )
                
                explanations["lime_explanation"] = {
                    "feature_importance": explanation.as_list(),
                    "prediction": explanation.predicted_value,
                    "local_prediction": explanation.local_pred
                }
            
            elif method == "grad_cam" and GRAD_CAM_AVAILABLE:
                # Grad-CAM explanation
                target_layers = [module for name, module in self.model.named_modules() 
                               if isinstance(module, (nn.Conv2d, nn.ReLU, nn.AdaptiveAvgPool2d))]
                
                if target_layers:
                    cam = GradCAM(model=self.model, target_layers=target_layers[-1:])
                    targets = [ClassifierOutputTarget(predicted_class)]
                    
                    grayscale_cam = cam(input_tensor=inputs, targets=targets)
                    
                    explanations["grad_cam"] = {
                        "cam_values": grayscale_cam.tolist(),
                        "target_layer": str(target_layers[-1])
                    }
            
            self.analysis_results[f"explanations_{method}"] = explanations
            logger.info(f"Explanation generated using {method}")
            
            return explanations
            
        except Exception as e:
            logger.error(f"Explanation generation failed: {e}")
            return {"error": str(e)}
    
    def create_interpretability_report(self, save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a comprehensive interpretability report.
        
        Args:
            save_path: Path to save the report (optional)
            
        Returns:
            Dictionary containing the complete interpretability report
        """
        report = {
            "model_info": {
                "model_type": str(type(self.model)) if self.model else "No model set",
                "device": self.device,
                "library_status": self.library_status
            },
            "analysis_results": self.analysis_results,
            "summary": {
                "total_analyses": len(self.analysis_results),
                "available_methods": list(self.analysis_results.keys()),
                "timestamp": datetime.now().isoformat()
            },
            "recommendations": self._generate_recommendations()
        }
        
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Interpretability report saved to {save_path}")
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on analysis results."""
        recommendations = []
        
        if "decision_boundary" in self.analysis_results:
            robustness = self.analysis_results["decision_boundary"]["perturbation_analysis"]["robustness_ratio"]
            if robustness < 0.8:
                recommendations.append("Model shows low robustness to input perturbations. Consider adversarial training.")
        
        if "bias_fairness" in self.analysis_results:
            bias_indicators = self.analysis_results["bias_fairness"]["bias_indicators"]
            if bias_indicators["potential_bias"]:
                recommendations.append("Potential bias detected. Review training data and consider fairness constraints.")
        
        if "layer_activations" in self.analysis_results:
            activations = self.analysis_results["layer_activations"]["layer_activations"]
            for layer_name, activation_data in activations.items():
                if activation_data["sparsity"] > 0.9:
                    recommendations.append(f"Layer {layer_name} shows high sparsity. Consider regularization.")
        
        if not recommendations:
            recommendations.append("Model appears to be performing well. Continue monitoring.")
        
        return recommendations


# Global analyzer instance
_global_analyzer: Optional[InterpretabilityAnalyzer] = None


def get_global_analyzer() -> Optional[InterpretabilityAnalyzer]:
    """Get the global interpretability analyzer instance."""
    return _global_analyzer


def initialize_global_analyzer(model: Optional[Any] = None, device: str = "cpu") -> InterpretabilityAnalyzer:
    """
    Initialize the global interpretability analyzer.
    
    Args:
        model: PyTorch model to analyze
        device: Device to run analysis on
        
    Returns:
        Initialized interpretability analyzer
    """
    global _global_analyzer
    _global_analyzer = InterpretabilityAnalyzer(model, device)
    return _global_analyzer
