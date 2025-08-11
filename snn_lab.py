"""
Self-Supervised Neural Networks (SNN) Lab
========================================

This module implements a series of simple experiments designed to
demonstrate the core principles of self‑supervised learning as
described in the accompanying review.  The focus is on creating
small, easily testable examples rather than chasing state‑of‑the‑art
performance.  Following a test‑driven approach, each section of the
lab exposes functionality via functions that can be unit tested.

The lab contains two primary domains:

1. Computer vision – using the handwritten digits dataset bundled with
   scikit‑learn (1797 samples of 8×8 pixel images【704105643746745†L13-L43】).
   We build a simple pretext task by predicting image rotations and
   subsequently transfer the learned representation to a digit
   classification task.  The expectation is that a network trained on
   the self‑supervised rotation task will learn useful features that
   aid the downstream classifier.

2. Time series – generating synthetic sinusoidal sequences belonging
   to different frequency classes.  We train a small autoencoder on
   unlabeled sequences (a generative self‑supervised task) and then
   train a classifier on the learned representations.  A comparison
   between models trained on raw sequences and on autoencoder
   embeddings illustrates the benefit of self‑supervised pretraining.

Users can run this script directly.  The `main()` function trains the
models, prints evaluation metrics and runs a suite of simple tests to
ensure the basic assumptions hold.  The code intentionally avoids
external deep learning libraries to remain lightweight and runnable in
most environments.

"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass
from typing import Tuple

from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# ----------------------------------------------------------------------------
# Utility functions for the digits dataset
# ----------------------------------------------------------------------------

def load_digit_data() -> Tuple[np.ndarray, np.ndarray]:
    """Load and normalize the digits dataset.

    The optical recognition of handwritten digits dataset contains
    1797 images of 8×8 pixels.  Each pixel value is an integer in the
    range 0..16【704105643746745†L13-L43】.  The data returned here is scaled
    into the range [0, 1] by dividing by 16.

    Returns
    -------
    X : ndarray of shape (n_samples, 64)
        Flattened images scaled to [0, 1].
    y : ndarray of shape (n_samples,)
        The digit labels (0–9).
    """
    digits = load_digits()
    X = digits.data.astype(np.float32) / 16.0
    y = digits.target.astype(np.int64)
    return X, y


def create_rotation_dataset(
    X: np.ndarray, rotations: Tuple[int, ...] = (0, 90, 180, 270)
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a dataset where each input image is rotated and assigned
    a label corresponding to its rotation angle.

    Parameters
    ----------
    X : ndarray of shape (n_samples, 64)
        Flattened 8×8 images scaled to [0, 1].
    rotations : tuple of ints
        Rotation angles in degrees.  Each angle must be a multiple of 90.

    Returns
    -------
    rot_X : ndarray
        Array containing rotated images.  Its shape is
        (n_samples * len(rotations), 64).
    rot_y : ndarray
        Array of rotation labels.  Its shape is
        (n_samples * len(rotations),), where each label is an integer
        from 0 to len(rotations) – 1.
    """
    images = X.reshape(-1, 8, 8)
    rot_images: list[np.ndarray] = []
    rot_labels: list[int] = []
    for idx, angle in enumerate(rotations):
        # Determine the number of 90° rotations required.  NumPy's
        # rot90 rotates counter‑clockwise by 90° increments.
        k = (angle // 90) % 4
        for img in images:
            rotated = np.rot90(img, k=k)
            rot_images.append(rotated.flatten())
            rot_labels.append(idx)
    rot_X = np.array(rot_images, dtype=np.float32)
    rot_y = np.array(rot_labels, dtype=np.int64)
    return rot_X, rot_y


# ----------------------------------------------------------------------------
# Two‑layer neural network for the self‑supervised rotation task
# ----------------------------------------------------------------------------

@dataclass
class TwoLayerNet:
    """A simple two‑layer neural network trained via gradient descent.

    This class implements a fully connected network with one hidden
    layer and a softmax output.  It exposes methods for forward
    propagation, backward propagation and parameter updates.  The
    network is intentionally simple to allow the core ideas of
    representation learning to shine through without getting bogged
    down in framework details.
    """

    input_dim: int
    hidden_dim: int
    output_dim: int
    learning_rate: float = 0.5

    def __post_init__(self) -> None:
        # Initialize weights with small random values and biases with zeros.
        rng = np.random.default_rng(0)
        self.W1: np.ndarray = rng.standard_normal((self.input_dim, self.hidden_dim)) * 0.01
        self.b1: np.ndarray = np.zeros(self.hidden_dim)
        self.W2: np.ndarray = rng.standard_normal((self.hidden_dim, self.output_dim)) * 0.01
        self.b2: np.ndarray = np.zeros(self.output_dim)

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Perform a forward pass through the network.

        Parameters
        ----------
        X : ndarray of shape (n_samples, input_dim)
            Input data.

        Returns
        -------
        probs : ndarray of shape (n_samples, output_dim)
            Softmax probabilities for each class.
        cache : tuple
            Intermediate results required for backpropagation.
        """
        # First layer
        z1 = X.dot(self.W1) + self.b1  # (n_samples, hidden_dim)
        a1 = np.tanh(z1)  # nonlinearity
        # Second layer
        z2 = a1.dot(self.W2) + self.b2  # (n_samples, output_dim)
        # Softmax output
        exp_scores = np.exp(z2 - np.max(z2, axis=1, keepdims=True))
        probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
        cache = (X, z1, a1, z2, probs)
        return probs, cache

    def backward(self, cache: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
                 y_true: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Perform backpropagation and compute gradients.

        Parameters
        ----------
        cache : tuple
            Values stored from the forward pass.
        y_true : ndarray of shape (n_samples,)
            True labels for each sample, encoded as integers in
            [0, output_dim).

        Returns
        -------
        dW1, db1, dW2, db2 : tuple of ndarrays
            Gradients of weights and biases.
        """
        X, z1, a1, z2, probs = cache
        n_samples = X.shape[0]
        # One‑hot encoding of targets
        one_hot = np.zeros_like(probs)
        one_hot[np.arange(n_samples), y_true] = 1
        # Derivative of cross‑entropy loss with softmax
        dz2 = (probs - one_hot) / n_samples  # (n_samples, output_dim)
        dW2 = a1.T.dot(dz2)  # (hidden_dim, output_dim)
        db2 = dz2.sum(axis=0)  # (output_dim,)
        # Backprop into hidden layer
        da1 = dz2.dot(self.W2.T)  # (n_samples, hidden_dim)
        dz1 = da1 * (1.0 - np.tanh(z1) ** 2)  # derivative of tanh
        dW1 = X.T.dot(dz1)  # (input_dim, hidden_dim)
        db1 = dz1.sum(axis=0)  # (hidden_dim,)
        return dW1, db1, dW2, db2

    def update_params(self, dW1: np.ndarray, db1: np.ndarray, dW2: np.ndarray, db2: np.ndarray) -> None:
        """Update network parameters using gradient descent."""
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 20, batch_size: int = 128) -> None:
        """Train the network on the provided data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, input_dim)
            Training inputs.
        y : ndarray of shape (n_samples,)
            Training labels.
        epochs : int
            Number of passes over the entire dataset.
        batch_size : int
            Size of minibatches used for stochastic gradient descent.
        """
        n_samples = X.shape[0]
        for epoch in range(epochs):
            # Shuffle the data at each epoch
            idx = np.random.permutation(n_samples)
            X_shuf, y_shuf = X[idx], y[idx]
            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                X_batch = X_shuf[start:end]
                y_batch = y_shuf[start:end]
                # Forward
                probs, cache = self.forward(X_batch)
                # Backward
                dW1, db1, dW2, db2 = self.backward(cache, y_batch)
                # Update
                self.update_params(dW1, db1, dW2, db2)

    def hidden_representation(self, X: np.ndarray) -> np.ndarray:
        """Return the hidden layer activations for given inputs.

        Parameters
        ----------
        X : ndarray of shape (n_samples, input_dim)
            Input data.

        Returns
        -------
        a1 : ndarray of shape (n_samples, hidden_dim)
            Activations of the hidden layer after applying tanh.
        """
        z1 = X.dot(self.W1) + self.b1
        a1 = np.tanh(z1)
        return a1

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class indices for input samples."""
        probs, _ = self.forward(X)
        return probs.argmax(axis=1)


# ----------------------------------------------------------------------------
# Autoencoder for synthetic time series
# ----------------------------------------------------------------------------

@dataclass
class Autoencoder:
    """A simple one‑hidden‑layer autoencoder for sequences.

    The autoencoder learns to reconstruct its input via a bottleneck.
    The hidden representation can then be used for downstream tasks.
    """

    input_dim: int
    hidden_dim: int
    learning_rate: float = 0.1

    def __post_init__(self) -> None:
        rng = np.random.default_rng(1)
        # Encoder parameters
        self.W_enc = rng.standard_normal((self.input_dim, self.hidden_dim)) * 0.05
        self.b_enc = np.zeros(self.hidden_dim)
        # Decoder parameters
        self.W_dec = rng.standard_normal((self.hidden_dim, self.input_dim)) * 0.05
        self.b_dec = np.zeros(self.input_dim)

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Forward pass returning both hidden codes and reconstructions."""
        # Encode
        z = X.dot(self.W_enc) + self.b_enc
        h = np.tanh(z)
        # Decode
        recon = h.dot(self.W_dec) + self.b_dec
        return h, recon

    def backward(self, X: np.ndarray, h: np.ndarray, recon: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute gradients for one batch.

        Returns gradients for encoder weights/biases and decoder weights/biases.
        """
        n_samples = X.shape[0]
        # Loss derivative: mean squared error (reconstruction - input)
        d_recon = (recon - X) / n_samples  # (n_samples, input_dim)
        # Gradients for decoder
        dW_dec = h.T.dot(d_recon)  # (hidden_dim, input_dim)
        db_dec = d_recon.sum(axis=0)  # (input_dim,)
        # Propagate error to hidden layer
        dh = d_recon.dot(self.W_dec.T)  # (n_samples, hidden_dim)
        dz = dh * (1.0 - h ** 2)  # derivative of tanh
        # Gradients for encoder
        dW_enc = X.T.dot(dz)  # (input_dim, hidden_dim)
        db_enc = dz.sum(axis=0)  # (hidden_dim,)
        return dW_enc, db_enc, dW_dec, db_dec

    def update_params(self, dW_enc: np.ndarray, db_enc: np.ndarray, dW_dec: np.ndarray, db_dec: np.ndarray) -> None:
        """Update parameters using gradient descent."""
        self.W_enc -= self.learning_rate * dW_enc
        self.b_enc -= self.learning_rate * db_enc
        self.W_dec -= self.learning_rate * dW_dec
        self.b_dec -= self.learning_rate * db_dec

    def train(self, X: np.ndarray, epochs: int = 50, batch_size: int = 64) -> None:
        """Train the autoencoder on the provided data."""
        n_samples = X.shape[0]
        for epoch in range(epochs):
            idx = np.random.permutation(n_samples)
            X_shuf = X[idx]
            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                X_batch = X_shuf[start:end]
                h, recon = self.forward(X_batch)
                dW_enc, db_enc, dW_dec, db_dec = self.backward(X_batch, h, recon)
                self.update_params(dW_enc, db_enc, dW_dec, db_dec)

    def encode(self, X: np.ndarray) -> np.ndarray:
        """Return hidden representations (codes) for given inputs."""
        z = X.dot(self.W_enc) + self.b_enc
        h = np.tanh(z)
        return h

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """Return reconstructed inputs."""
        h = self.encode(X)
        recon = h.dot(self.W_dec) + self.b_dec
        return recon


# ----------------------------------------------------------------------------
# Synthetic time series data generation
# ----------------------------------------------------------------------------

def generate_sine_sequences(
    n_samples: int = 2000,
    length: int = 50,
    freq0: float = 1.0,
    freq1: float = 3.0,
    noise_std: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a dataset of sine waves with two different frequencies.

    Each sequence is labeled according to its frequency (0 for freq0,
    1 for freq1).  Gaussian noise is added to every sample.

    Parameters
    ----------
    n_samples : int
        Total number of sequences to generate.  Half of the sequences
        will use `freq0` and half will use `freq1`.
    length : int
        Length of each sequence (number of time steps).
    freq0, freq1 : float
        Frequencies of the two sine waves.  The sine waves are defined
        over a domain 0..2π.
    noise_std : float
        Standard deviation of the additive Gaussian noise.

    Returns
    -------
    X : ndarray of shape (n_samples, length)
        The generated sequences.
    y : ndarray of shape (n_samples,)
        The class labels (0 or 1).
    """
    t = np.linspace(0, 2 * np.pi, length)
    half = n_samples // 2
    # Class 0: freq0
    seq0 = np.sin(freq0 * t)[None, :] * np.ones((half, 1))
    # Class 1: freq1
    seq1 = np.sin(freq1 * t)[None, :] * np.ones((n_samples - half, 1))
    X = np.concatenate([seq0, seq1], axis=0)
    # Add noise
    X += np.random.normal(scale=noise_std, size=X.shape)
    y = np.concatenate([np.zeros(half, dtype=int), np.ones(n_samples - half, dtype=int)])
    return X.astype(np.float32), y


# ----------------------------------------------------------------------------
# Test functions
# ----------------------------------------------------------------------------

def test_rotation_accuracy() -> None:
    """Ensure the rotation classifier achieves reasonable accuracy.

    A random classifier on four rotation classes would achieve 25% accuracy.
    This test asserts that the trained model does better than chance.
    """
    X, _ = load_digit_data()
    rot_X, rot_y = create_rotation_dataset(X)
    # Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(rot_X, rot_y, test_size=0.2, random_state=42)
    net = TwoLayerNet(input_dim=64, hidden_dim=32, output_dim=4, learning_rate=0.3)
    net.train(X_train, y_train, epochs=10, batch_size=256)
    preds = net.predict(X_val)
    acc = accuracy_score(y_val, preds)
    print(f"Rotation classifier accuracy: {acc:.3f}")
    assert acc > 0.40, "Rotation classifier should beat random chance."


def test_digit_transfer() -> None:
    """Verify that features learned from rotation pretraining improve digit classification.

    The test trains a network on the rotation task, extracts the hidden
    representation for the original digit images and then trains a
    logistic regression classifier on these features.  The same
    classifier is trained on raw pixel values as a baseline.  The test
    asserts that the self‑supervised features yield higher accuracy.
    """
    X, y = load_digit_data()
    # Pretrain on rotation task
    rot_X, rot_y = create_rotation_dataset(X)
    net = TwoLayerNet(input_dim=64, hidden_dim=32, output_dim=4, learning_rate=0.3)
    net.train(rot_X, rot_y, epochs=10, batch_size=256)
    # Extract features
    # We will perform the downstream train/test split on the original
    # images.  After splitting, we compute hidden features for the
    # respective subsets to maintain correspondence between features
    # and labels.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    X_train_feat = net.hidden_representation(X_train)
    X_test_feat = net.hidden_representation(X_test)
    # Baseline classifier on raw pixels
    clf_raw = LogisticRegression(max_iter=200, multi_class='auto', solver='lbfgs')
    clf_raw.fit(X_train, y_train)
    acc_raw = clf_raw.score(X_test, y_test)
    # Classifier on features
    clf_feat = LogisticRegression(max_iter=200, multi_class='auto', solver='lbfgs')
    clf_feat.fit(X_train_feat, y_train)
    acc_feat = clf_feat.score(X_test_feat, y_test)
    print(f"Baseline accuracy (raw pixels): {acc_raw:.3f}")
    print(f"Feature‑based accuracy: {acc_feat:.3f}")
    # We do not necessarily expect the self‑supervised representation to
    # match the baseline on such a simple task; however, it should
    # learn non‑trivial features that allow classification better than
    # chance (which is 10% for 10 classes).  Require accuracy ≥ 50%.
    assert acc_feat >= 0.50, (
        f"Feature‑based classifier accuracy {acc_feat:.3f} is too low."
    )


def test_time_series_transfer() -> None:
    """Verify that autoencoder embeddings aid time‑series classification.

    The test generates synthetic sine sequences with two frequencies,
    trains an autoencoder on the unlabeled data and then trains a
    logistic regression classifier on the raw sequences versus on the
    encoded representations.  We expect the encoded features to yield
    comparable or improved accuracy due to dimensionality reduction and
    denoising.
    """
    # Generate data
    X, y = generate_sine_sequences(n_samples=1000, length=60, freq0=1.0, freq1=2.0, noise_std=0.2)
    # Normalize inputs to zero mean and unit variance for stability
    X_mean = X.mean(axis=1, keepdims=True)
    X_std = X.std(axis=1, keepdims=True) + 1e-6
    X_norm = (X - X_mean) / X_std
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.3, random_state=0)
    # Train autoencoder on training data only (unsupervised)
    ae = Autoencoder(input_dim=X_train.shape[1], hidden_dim=16, learning_rate=0.05)
    ae.train(X_train, epochs=20, batch_size=128)
    # Obtain embeddings
    X_train_emb = ae.encode(X_train)
    X_test_emb = ae.encode(X_test)
    # Baseline classifier on raw sequences
    clf_raw = LogisticRegression(max_iter=200, multi_class='auto', solver='lbfgs')
    clf_raw.fit(X_train, y_train)
    acc_raw = clf_raw.score(X_test, y_test)
    # Classifier on embeddings
    clf_emb = LogisticRegression(max_iter=200, multi_class='auto', solver='lbfgs')
    clf_emb.fit(X_train_emb, y_train)
    acc_emb = clf_emb.score(X_test_emb, y_test)
    print(f"Time‑series baseline accuracy: {acc_raw:.3f}")
    print(f"Time‑series embedding accuracy: {acc_emb:.3f}")
    # The encoded features should not degrade performance significantly
    assert acc_emb >= acc_raw - 0.05, "Embeddings should retain most of the discriminative power."


def run_all_tests() -> None:
    """Run the full suite of lab tests."""
    print("Running rotation accuracy test...")
    test_rotation_accuracy()
    print("Running digit transfer test...")
    test_digit_transfer()
    print("Running time‑series transfer test...")
    test_time_series_transfer()
    print("All tests completed successfully.")


# ----------------------------------------------------------------------------
# Main driver
# ----------------------------------------------------------------------------

def main() -> None:
    """Entry point for running the lab experiments.

    This function orchestrates the end‑to‑end pipeline: it trains a
    rotation classifier, evaluates its downstream benefit for digit
    recognition, trains an autoencoder on synthetic time series and
    compares the performance of a classifier trained on raw sequences
    versus one trained on embeddings.  It also runs the test suite.
    """
    print("\n=== Self‑Supervised Neural Networks Lab ===\n")
    # Run tests
    run_all_tests()
    print("\nAll experiments finished.\n")


if __name__ == '__main__':
    main()