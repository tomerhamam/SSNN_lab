"""
Modular SSL Framework with Deep Networks and Dataset Agnostic Design.

Key design principles:
1. Dataset-agnostic core network architecture
2. Pluggable pretext tasks  
3. Deeper, more powerful networks
4. Clean separation of concerns
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Optional, List
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# DATA INTERFACE
# ============================================================================

@dataclass
class DatasetConfig:
    """Configuration for dataset loading."""
    name: str
    input_shape: Tuple[int, ...]  # (height, width) or (height, width, channels)
    num_classes: int
    train_samples: Optional[int] = None
    test_samples: Optional[int] = None

class DatasetInterface(ABC):
    """Abstract interface for dataset loading."""
    
    @abstractmethod
    def load_data(self, config: DatasetConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load dataset.
        
        Returns:
            X_train, y_train, X_test, y_test
        """
        pass
    
    @abstractmethod
    def get_config(self) -> DatasetConfig:
        """Get dataset configuration."""
        pass

# ============================================================================
# PRETEXT TASK INTERFACE  
# ============================================================================

class PretextTask(ABC):
    """Abstract base class for pretext tasks."""
    
    @abstractmethod
    def create_dataset(self, X: np.ndarray, config: DatasetConfig) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create pretext task dataset from input data.
        
        Args:
            X: Input images of shape (N, H, W) or (N, H, W, C)
            config: Dataset configuration
            
        Returns:
            pretext_X, pretext_y
        """
        pass
    
    @abstractmethod
    def get_num_classes(self) -> int:
        """Get number of classes for this pretext task."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get name of the pretext task."""
        pass

# ============================================================================
# DEEP NEURAL NETWORK ARCHITECTURE
# ============================================================================

class DeepSSLNetwork:
    """
    Deep neural network for SSL with configurable architecture.
    Designed to be dataset-agnostic.
    """
    
    def __init__(self, 
                 input_dim: int,
                 architecture: List[int],
                 output_dim: int,
                 learning_rate: float = 0.01,
                 dropout_rate: float = 0.1,
                 use_batch_norm: bool = True):
        """
        Initialize deep network.
        
        Args:
            input_dim: Input dimensionality (flattened)
            architecture: List of hidden layer sizes [512, 256, 128]
            output_dim: Output classes for pretext task
            learning_rate: Learning rate
            dropout_rate: Dropout probability
            use_batch_norm: Whether to use batch normalization
        """
        self.input_dim = input_dim
        self.architecture = architecture
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        """Initialize network parameters using Xavier initialization."""
        rng = np.random.default_rng(42)
        
        # Build layer dimensions
        layer_dims = [self.input_dim] + self.architecture + [self.output_dim]
        
        self.weights = []
        self.biases = []
        self.bn_params = []  # Batch norm parameters
        
        for i in range(len(layer_dims) - 1):
            # Xavier initialization
            fan_in, fan_out = layer_dims[i], layer_dims[i + 1]
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            
            W = rng.uniform(-limit, limit, (fan_in, fan_out)).astype(np.float32)
            b = np.zeros(fan_out, dtype=np.float32)
            
            self.weights.append(W)
            self.biases.append(b)
            
            # Batch norm parameters (gamma, beta, running mean, running var)
            if self.use_batch_norm and i < len(layer_dims) - 2:  # Not for output layer
                gamma = np.ones(fan_out, dtype=np.float32)
                beta = np.zeros(fan_out, dtype=np.float32)
                running_mean = np.zeros(fan_out, dtype=np.float32)
                running_var = np.ones(fan_out, dtype=np.float32)
                self.bn_params.append({'gamma': gamma, 'beta': beta, 
                                     'running_mean': running_mean, 'running_var': running_var})
            else:
                self.bn_params.append(None)
    
    def _activation(self, x: np.ndarray, activation: str = 'relu') -> np.ndarray:
        """Apply activation function."""
        if activation == 'relu':
            return np.maximum(0, x)
        elif activation == 'tanh':
            return np.tanh(x)
        elif activation == 'leaky_relu':
            return np.where(x > 0, x, 0.01 * x)
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def _activation_derivative(self, x: np.ndarray, activation: str = 'relu') -> np.ndarray:
        """Compute activation derivative."""
        if activation == 'relu':
            return (x > 0).astype(np.float32)
        elif activation == 'tanh':
            return 1.0 - np.tanh(x)**2
        elif activation == 'leaky_relu':
            return np.where(x > 0, 1.0, 0.01)
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def _batch_norm_forward(self, x: np.ndarray, bn_param: Dict, training: bool = True) -> np.ndarray:
        """Forward pass through batch normalization."""
        if bn_param is None:
            return x
        
        gamma, beta = bn_param['gamma'], bn_param['beta']
        running_mean, running_var = bn_param['running_mean'], bn_param['running_var']
        
        if training:
            # Compute batch statistics
            batch_mean = np.mean(x, axis=0)
            batch_var = np.var(x, axis=0)
            
            # Update running statistics
            momentum = 0.9
            bn_param['running_mean'] = momentum * running_mean + (1 - momentum) * batch_mean
            bn_param['running_var'] = momentum * running_var + (1 - momentum) * batch_var
            
            # Normalize
            x_norm = (x - batch_mean) / np.sqrt(batch_var + 1e-8)
        else:
            # Use running statistics
            x_norm = (x - running_mean) / np.sqrt(running_var + 1e-8)
        
        # Scale and shift
        return gamma * x_norm + beta
    
    def forward(self, X: np.ndarray, training: bool = True) -> Tuple[np.ndarray, Dict]:
        """
        Forward pass through the network.
        
        Args:
            X: Input batch of shape (batch_size, input_dim)
            training: Whether in training mode
            
        Returns:
            output probabilities, cache for backprop
        """
        cache = {'activations': [X], 'pre_activations': [], 'dropout_masks': []}
        current = X
        
        # Forward through hidden layers
        for i, (W, b, bn_param) in enumerate(zip(self.weights[:-1], self.biases[:-1], self.bn_params[:-1])):
            # Linear transformation
            z = current @ W + b
            cache['pre_activations'].append(z)
            
            # Batch normalization
            if self.use_batch_norm and bn_param is not None:
                z = self._batch_norm_forward(z, bn_param, training)
            
            # Activation
            a = self._activation(z, 'relu')
            
            # Dropout
            if training and self.dropout_rate > 0:
                dropout_mask = np.random.binomial(1, 1 - self.dropout_rate, a.shape) / (1 - self.dropout_rate)
                a = a * dropout_mask
                cache['dropout_masks'].append(dropout_mask)
            else:
                cache['dropout_masks'].append(np.ones_like(a))
            
            cache['activations'].append(a)
            current = a
        
        # Output layer (no activation, no dropout, no batch norm)
        z_out = current @ self.weights[-1] + self.biases[-1]
        cache['pre_activations'].append(z_out)
        
        # Softmax
        exp_scores = np.exp(z_out - np.max(z_out, axis=1, keepdims=True))
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
        cache['probs'] = probs
        return probs, cache
    
    def backward(self, cache: Dict, y_true: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Backward pass through the network.
        
        Args:
            cache: Cache from forward pass
            y_true: True labels
            
        Returns:
            List of (weight_grad, bias_grad) tuples
        """
        batch_size = len(y_true)
        
        # Convert labels to one-hot
        one_hot = np.zeros_like(cache['probs'])
        one_hot[np.arange(batch_size), y_true] = 1
        
        # Output layer gradient
        dz = (cache['probs'] - one_hot) / batch_size
        
        gradients = []
        
        # Backward through layers
        for i in reversed(range(len(self.weights))):
            # Current layer activations
            a_prev = cache['activations'][i]
            
            # Gradients for current layer
            dW = a_prev.T @ dz
            db = np.sum(dz, axis=0)
            gradients.append((dW, db))
            
            # Don't compute gradients for input layer
            if i > 0:
                # Gradient w.r.t. previous activations
                da_prev = dz @ self.weights[i].T
                
                # Apply dropout mask
                da_prev = da_prev * cache['dropout_masks'][i-1]
                
                # Gradient through activation
                if i < len(self.weights):  # Hidden layers use ReLU
                    z_prev = cache['pre_activations'][i-1]
                    dz = da_prev * self._activation_derivative(z_prev, 'relu')
        
        # Reverse to match layer order
        gradients.reverse()
        return gradients
    
    def update_parameters(self, gradients: List[Tuple[np.ndarray, np.ndarray]]):
        """Update network parameters using gradients."""
        for i, (dW, db) in enumerate(gradients):
            self.weights[i] -= self.learning_rate * dW
            self.biases[i] -= self.learning_rate * db
    
    def train_epoch(self, X: np.ndarray, y: np.ndarray, batch_size: int = 128) -> float:
        """Train for one epoch."""
        n_samples = X.shape[0]
        total_loss = 0.0
        n_batches = 0
        
        # Shuffle data
        indices = np.random.permutation(n_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]
            
            # Forward pass
            probs, cache = self.forward(X_batch, training=True)
            
            # Compute loss
            loss = -np.mean(np.log(probs[np.arange(len(y_batch)), y_batch] + 1e-8))
            total_loss += loss
            n_batches += 1
            
            # Backward pass
            gradients = self.backward(cache, y_batch)
            
            # Update parameters
            self.update_parameters(gradients)
        
        return total_loss / n_batches
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        probs, _ = self.forward(X, training=False)
        return np.argmax(probs, axis=1)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """Evaluate accuracy."""
        predictions = self.predict(X)
        return np.mean(predictions == y)
    
    def get_representations(self, X: np.ndarray, layer_idx: int = -2) -> np.ndarray:
        """
        Extract representations from a specific layer.
        
        Args:
            X: Input data
            layer_idx: Layer index (-2 for second-to-last layer)
            
        Returns:
            Learned representations
        """
        _, cache = self.forward(X, training=False)
        return cache['activations'][layer_idx]

# ============================================================================
# DATASET IMPLEMENTATIONS
# ============================================================================

class CIFAR10Dataset(DatasetInterface):
    """CIFAR-10 dataset loader."""
    
    def load_data(self, config: DatasetConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load CIFAR-10 dataset."""
        try:
            import torchvision
            import torchvision.transforms as transforms
            
            # Load CIFAR-10
            transform = transforms.Compose([transforms.ToTensor()])
            
            train_dataset = torchvision.datasets.CIFAR10(
                root='./data', train=True, download=True, transform=transform
            )
            test_dataset = torchvision.datasets.CIFAR10(
                root='./data', train=False, download=True, transform=transform
            )
            
            # Extract data
            def extract_data(dataset, max_samples=None):
                X, y = [], []
                max_samples = max_samples or len(dataset)
                indices = np.random.choice(len(dataset), min(max_samples, len(dataset)), replace=False)
                
                for i in indices:
                    img, label = dataset[i]
                    # Convert to grayscale and flatten for simplicity
                    img_gray = np.mean(img.numpy(), axis=0)  # Average RGB channels
                    X.append(img_gray.flatten())
                    y.append(label)
                
                return np.array(X, dtype=np.float32), np.array(y)
            
            X_train, y_train = extract_data(train_dataset, config.train_samples)
            X_test, y_test = extract_data(test_dataset, config.test_samples)
            
            print(f"✅ Loaded CIFAR-10: train={X_train.shape}, test={X_test.shape}")
            return X_train, y_train, X_test, y_test
            
        except ImportError:
            # Fallback to synthetic CIFAR-like data
            print("⚠️  CIFAR-10 not available, generating synthetic 32x32 data...")
            return self._generate_synthetic_data(config)
    
    def _generate_synthetic_data(self, config: DatasetConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate synthetic CIFAR-like data."""
        np.random.seed(42)
        
        n_train = config.train_samples or 5000
        n_test = config.test_samples or 1000
        
        # Generate structured 32x32 images
        def generate_images(n_samples):
            X, y = [], []
            for i in range(n_samples):
                img = np.random.randn(32, 32) * 0.1
                
                # Add class-specific patterns
                class_id = i % config.num_classes
                
                if class_id == 0:  # Circles
                    center = np.random.randint(8, 24, 2)
                    radius = np.random.randint(4, 8)
                    y_coords, x_coords = np.ogrid[:32, :32]
                    mask = (x_coords - center[0])**2 + (y_coords - center[1])**2 <= radius**2
                    img[mask] += 1.0
                    
                elif class_id == 1:  # Horizontal lines
                    for _ in range(3):
                        y_pos = np.random.randint(5, 27)
                        x_start = np.random.randint(0, 16)
                        x_end = np.random.randint(16, 32)
                        img[y_pos:y_pos+2, x_start:x_end] += 1.0
                        
                # Add more patterns for other classes...
                else:
                    # Random geometric patterns
                    for _ in range(np.random.randint(2, 6)):
                        x, y_pos = np.random.randint(0, 30, 2)
                        size = np.random.randint(2, 6)
                        img[y_pos:y_pos+size, x:x+size] += 0.5
                
                img = np.clip(img, 0, 1)
                X.append(img.flatten())
                y.append(class_id)
            
            return np.array(X, dtype=np.float32), np.array(y)
        
        X_train, y_train = generate_images(n_train)
        X_test, y_test = generate_images(n_test)
        
        return X_train, y_train, X_test, y_test
    
    def get_config(self) -> DatasetConfig:
        """Get CIFAR-10 configuration."""
        return DatasetConfig(
            name="CIFAR-10",
            input_shape=(32, 32),
            num_classes=10,
            train_samples=5000,
            test_samples=1000
        )

class FashionMNISTDataset(DatasetInterface):
    """Fashion-MNIST dataset loader."""
    
    def load_data(self, config: DatasetConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load Fashion-MNIST dataset."""
        try:
            import torchvision
            import torchvision.transforms as transforms
            
            transform = transforms.Compose([transforms.ToTensor()])
            
            train_dataset = torchvision.datasets.FashionMNIST(
                root='./data', train=True, download=True, transform=transform
            )
            test_dataset = torchvision.datasets.FashionMNIST(
                root='./data', train=False, download=True, transform=transform
            )
            
            def extract_data(dataset, max_samples=None):
                X, y = [], []
                max_samples = max_samples or len(dataset)
                indices = np.random.choice(len(dataset), min(max_samples, len(dataset)), replace=False)
                
                for i in indices:
                    img, label = dataset[i]
                    X.append(img.numpy().flatten())
                    y.append(label)
                
                return np.array(X, dtype=np.float32), np.array(y)
            
            X_train, y_train = extract_data(train_dataset, config.train_samples)
            X_test, y_test = extract_data(test_dataset, config.test_samples)
            
            return X_train, y_train, X_test, y_test
            
        except ImportError:
            # Fallback to digits
            from sklearn.datasets import load_digits
            digits = load_digits()
            X = digits.data.astype(np.float32) / 16.0
            y = digits.target
            
            # Split
            from sklearn.model_selection import train_test_split
            return train_test_split(X, y, test_size=0.3, random_state=42)
    
    def get_config(self) -> DatasetConfig:
        """Get Fashion-MNIST configuration."""
        return DatasetConfig(
            name="Fashion-MNIST",
            input_shape=(28, 28),
            num_classes=10,
            train_samples=5000,
            test_samples=1000
        )

# ============================================================================
# PRETEXT TASK IMPLEMENTATIONS
# ============================================================================

class RotationTask(PretextTask):
    """Rotation prediction pretext task."""
    
    def __init__(self, num_rotations: int = 8):
        self.num_rotations = num_rotations
        self.angles = np.linspace(0, 360, num_rotations, endpoint=False)
    
    def create_dataset(self, X: np.ndarray, config: DatasetConfig) -> Tuple[np.ndarray, np.ndarray]:
        """Create rotation dataset."""
        h, w = config.input_shape
        images = X.reshape(-1, h, w)
        
        rot_images = []
        rot_labels = []
        
        for idx, angle in enumerate(self.angles):
            k = int(angle // 90) % 4  # Number of 90-degree rotations
            for img in images:
                rotated = np.rot90(img, k=k)
                rot_images.append(rotated.flatten())
                rot_labels.append(idx)
        
        return np.array(rot_images, dtype=np.float32), np.array(rot_labels, dtype=np.int64)
    
    def get_num_classes(self) -> int:
        return self.num_rotations
    
    def get_name(self) -> str:
        return f"Rotation-{self.num_rotations}"

class JigsawTask(PretextTask):
    """Jigsaw puzzle pretext task."""
    
    def __init__(self, grid_size: int = 2):
        self.grid_size = grid_size
        # Define permutations for 2x2 grid
        self.permutations = [
            [0, 1, 2, 3],  # Original
            [1, 0, 3, 2],  # Swap columns  
            [2, 3, 0, 1],  # Swap rows
            [3, 2, 1, 0],  # Diagonal flip
        ]
    
    def create_dataset(self, X: np.ndarray, config: DatasetConfig) -> Tuple[np.ndarray, np.ndarray]:
        """Create jigsaw dataset."""
        h, w = config.input_shape
        images = X.reshape(-1, h, w)
        
        jigsaw_images = []
        jigsaw_labels = []
        
        patch_h, patch_w = h // self.grid_size, w // self.grid_size
        
        for perm_idx, perm in enumerate(self.permutations):
            for img in images:
                # Extract patches
                patches = []
                for i in range(self.grid_size):
                    for j in range(self.grid_size):
                        patch = img[i*patch_h:(i+1)*patch_h, j*patch_w:(j+1)*patch_w]
                        patches.append(patch)
                
                # Apply permutation
                permuted_patches = [patches[i] for i in perm]
                
                # Reconstruct image
                reconstructed = np.zeros_like(img)
                idx = 0
                for i in range(self.grid_size):
                    for j in range(self.grid_size):
                        reconstructed[i*patch_h:(i+1)*patch_h, j*patch_w:(j+1)*patch_w] = permuted_patches[idx]
                        idx += 1
                
                jigsaw_images.append(reconstructed.flatten())
                jigsaw_labels.append(perm_idx)
        
        return np.array(jigsaw_images, dtype=np.float32), np.array(jigsaw_labels, dtype=np.int64)
    
    def get_num_classes(self) -> int:
        return len(self.permutations)
    
    def get_name(self) -> str:
        return f"Jigsaw-{self.grid_size}x{self.grid_size}"

class ContrastiveTask(PretextTask):
    """Simple contrastive learning task."""
    
    def __init__(self, num_augmentations: int = 4):
        self.num_augmentations = num_augmentations
    
    def create_dataset(self, X: np.ndarray, config: DatasetConfig) -> Tuple[np.ndarray, np.ndarray]:
        """Create contrastive dataset with augmentations."""
        h, w = config.input_shape
        images = X.reshape(-1, h, w)
        
        aug_images = []
        aug_labels = []
        
        for aug_idx in range(self.num_augmentations):
            for img in images:
                if aug_idx == 0:  # Original
                    augmented = img
                elif aug_idx == 1:  # Add noise
                    augmented = img + np.random.randn(*img.shape) * 0.1
                elif aug_idx == 2:  # Brightness
                    augmented = img * (0.8 + np.random.rand() * 0.4)
                elif aug_idx == 3:  # Slight rotation
                    augmented = np.rot90(img, k=np.random.randint(0, 4))
                
                augmented = np.clip(augmented, 0, 1)
                aug_images.append(augmented.flatten())
                aug_labels.append(aug_idx)
        
        return np.array(aug_images, dtype=np.float32), np.array(aug_labels, dtype=np.int64)
    
    def get_num_classes(self) -> int:
        return self.num_augmentations
    
    def get_name(self) -> str:
        return f"Contrastive-{self.num_augmentations}"

# ============================================================================
# SSL FRAMEWORK
# ============================================================================

class SSLFramework:
    """Main SSL framework that ties everything together."""
    
    def __init__(self, 
                 dataset: DatasetInterface,
                 pretext_task: PretextTask,
                 architecture: List[int] = [512, 256, 128],
                 learning_rate: float = 0.01,
                 dropout_rate: float = 0.1):
        """
        Initialize SSL framework.
        
        Args:
            dataset: Dataset interface
            pretext_task: Pretext task
            architecture: Hidden layer sizes
            learning_rate: Learning rate
            dropout_rate: Dropout rate
        """
        self.dataset = dataset
        self.pretext_task = pretext_task
        self.architecture = architecture
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        
        # Load dataset
        self.config = dataset.get_config()
        self.X_train, self.y_train, self.X_test, self.y_test = dataset.load_data(self.config)
        
        # Calculate input dimension
        self.input_dim = np.prod(self.config.input_shape)
        
        # Initialize network
        self.network = None
    
    def train_ssl(self, epochs: int = 30, batch_size: int = 128, verbose: bool = True) -> Dict[str, Any]:
        """Train the SSL model on pretext task."""
        
        if verbose:
            print(f"\n🎯 Training SSL Framework")
            print(f"Dataset: {self.config.name}")
            print(f"Pretext Task: {self.pretext_task.get_name()}")
            print(f"Architecture: {self.input_dim} → {' → '.join(map(str, self.architecture))} → {self.pretext_task.get_num_classes()}")
        
        # Create pretext dataset
        pretext_X, pretext_y = self.pretext_task.create_dataset(self.X_train, self.config)
        
        if verbose:
            print(f"Pretext dataset: {pretext_X.shape}")
        
        # Split pretext data
        from sklearn.model_selection import train_test_split
        X_pre_train, X_pre_val, y_pre_train, y_pre_val = train_test_split(
            pretext_X, pretext_y, test_size=0.2, random_state=42
        )
        
        # Initialize network
        self.network = DeepSSLNetwork(
            input_dim=self.input_dim,
            architecture=self.architecture,
            output_dim=self.pretext_task.get_num_classes(),
            learning_rate=self.learning_rate,
            dropout_rate=self.dropout_rate
        )
        
        # Training loop
        train_losses = []
        val_accuracies = []
        
        for epoch in range(epochs):
            # Train one epoch
            loss = self.network.train_epoch(X_pre_train, y_pre_train, batch_size)
            train_losses.append(loss)
            
            # Evaluate
            val_acc = self.network.evaluate(X_pre_val, y_pre_val)
            val_accuracies.append(val_acc)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:3d}: Loss = {loss:.4f}, Val Acc = {val_acc:.3f}")
        
        final_val_acc = val_accuracies[-1]
        if verbose:
            print(f"\n🎯 Final pretext validation accuracy: {final_val_acc:.3f}")
        
        return {
            'train_losses': train_losses,
            'val_accuracies': val_accuracies,
            'final_pretext_acc': final_val_acc
        }
    
    def evaluate_transfer(self, representation_layer: int = -2, verbose: bool = True) -> Dict[str, float]:
        """Evaluate transfer learning performance."""
        
        if self.network is None:
            raise ValueError("Must train SSL model first!")
        
        if verbose:
            print(f"\n🔄 Evaluating transfer learning...")
        
        # Extract representations
        ssl_train = self.network.get_representations(self.X_train, representation_layer)
        ssl_test = self.network.get_representations(self.X_test, representation_layer)
        
        if verbose:
            print(f"SSL features: {ssl_train.shape}")
        
        # Test different classifiers
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        
        results = {}
        
        # Logistic Regression
        clf_lr = LogisticRegression(max_iter=500, random_state=42)
        clf_lr.fit(ssl_train, self.y_train)
        results['ssl_logistic'] = clf_lr.score(ssl_test, self.y_test)
        
        # Random Forest
        clf_rf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf_rf.fit(ssl_train, self.y_train)
        results['ssl_rf'] = clf_rf.score(ssl_test, self.y_test)
        
        # Baseline (raw pixels)
        clf_baseline = LogisticRegression(max_iter=500, random_state=42)
        clf_baseline.fit(self.X_train, self.y_train)
        results['baseline'] = clf_baseline.score(self.X_test, self.y_test)
        
        # Best SSL result
        results['best_ssl'] = max(results['ssl_logistic'], results['ssl_rf'])
        results['improvement'] = results['best_ssl'] - results['baseline']
        
        if verbose:
            print(f"SSL (Logistic): {results['ssl_logistic']:.3f}")
            print(f"SSL (Random Forest): {results['ssl_rf']:.3f}")
            print(f"Baseline: {results['baseline']:.3f}")
            print(f"Best SSL: {results['best_ssl']:.3f}")
            print(f"Improvement: {results['improvement']*100:+.1f}%")
            
            if results['improvement'] > 0:
                print("✅ SSL beats baseline!")
            else:
                print("📊 SSL is competitive with baseline")
        
        return results

# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def run_ssl_experiment(dataset_name: str = 'cifar10', 
                      pretext_task: str = 'rotation',
                      architecture: List[int] = [512, 256, 128],
                      epochs: int = 30,
                      **kwargs) -> Dict[str, Any]:
    """
    Run a complete SSL experiment.
    
    Args:
        dataset_name: 'cifar10' or 'fashion_mnist'
        pretext_task: 'rotation', 'jigsaw', or 'contrastive'
        architecture: Network architecture
        epochs: Training epochs
        **kwargs: Additional arguments
    
    Returns:
        Complete results dictionary
    """
    # Create dataset
    if dataset_name.lower() == 'cifar10':
        dataset = CIFAR10Dataset()
    elif dataset_name.lower() == 'fashion_mnist':
        dataset = FashionMNISTDataset()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Create pretext task
    if pretext_task.lower() == 'rotation':
        task = RotationTask(kwargs.get('num_rotations', 8))
    elif pretext_task.lower() == 'jigsaw':
        task = JigsawTask(kwargs.get('grid_size', 2))
    elif pretext_task.lower() == 'contrastive':
        task = ContrastiveTask(kwargs.get('num_augmentations', 4))
    else:
        raise ValueError(f"Unknown pretext task: {pretext_task}")
    
    # Create framework
    framework = SSLFramework(
        dataset=dataset,
        pretext_task=task,
        architecture=architecture,
        learning_rate=kwargs.get('learning_rate', 0.01),
        dropout_rate=kwargs.get('dropout_rate', 0.1)
    )
    
    # Train and evaluate
    train_results = framework.train_ssl(epochs=epochs, verbose=kwargs.get('verbose', True))
    transfer_results = framework.evaluate_transfer(verbose=kwargs.get('verbose', True))
    
    # Combine results
    return {
        'dataset': dataset_name,
        'pretext_task': pretext_task,
        'architecture': architecture,
        'train_results': train_results,
        'transfer_results': transfer_results
    }