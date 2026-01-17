"""
CNN Network Implementation from Scratch

Convolutional Neural Network using only NumPy.
Supports Conv2D, MaxPool2D, Flatten, and Dense layers.
"""

import numpy as np
from typing import Callable

from cnn_layers import Layer, Conv2D, MaxPool2D, Flatten, Dense, ReLU, Softmax


class CNNNetwork:
    """
    Convolutional Neural Network with backpropagation.

    Supports building networks layer-by-layer with:
    - Conv2D layers
    - MaxPool2D layers
    - Dense (fully connected) layers
    - ReLU and Softmax activations

    Example:
        >>> nn = CNNNetwork(input_shape=(8, 8, 1))
        >>> nn.add_conv2d(filters=4, kernel_size=3)
        >>> nn.add_maxpool2d(pool_size=2)
        >>> nn.add_flatten()
        >>> nn.add_dense(units=16)
        >>> nn.add_dense(units=3, activation='softmax')
        >>> nn.train(X, y, epochs=100, learning_rate=0.1)
    """

    def __init__(self, input_shape: tuple[int, int, int]):
        """
        Initialize CNN.

        Args:
            input_shape: (height, width, channels) of input images
        """
        self.input_shape = input_shape
        self.layers: list[Layer] = []
        self.layer_names: list[str] = []

        # Track current shape as we build the network
        self._current_shape = input_shape

        # Training history
        self.loss_history: list[float] = []
        self.accuracy_history: list[float] = []

    def add_conv2d(self, filters: int, kernel_size: int = 3,
                   activation: str = 'relu', padding: str = 'valid') -> None:
        """
        Add Conv2D layer with activation.

        Args:
            filters: Number of output filters
            kernel_size: Size of convolution kernel
            activation: 'relu' or None
            padding: 'valid' or 'same'
        """
        h, w, c = self._current_shape

        conv = Conv2D(
            num_filters=filters,
            kernel_size=kernel_size,
            in_channels=c,
            padding=padding
        )
        self.layers.append(conv)
        self.layer_names.append(f'conv2d_{filters}')

        # Update current shape
        if padding == 'valid':
            h = h - kernel_size + 1
            w = w - kernel_size + 1
        self._current_shape = (h, w, filters)

        # Add activation
        if activation == 'relu':
            self.layers.append(ReLU())
            self.layer_names.append('relu')

    def add_maxpool2d(self, pool_size: int = 2) -> None:
        """
        Add MaxPool2D layer.

        Args:
            pool_size: Size of pooling window
        """
        h, w, c = self._current_shape

        pool = MaxPool2D(pool_size=pool_size)
        self.layers.append(pool)
        self.layer_names.append(f'maxpool_{pool_size}')

        # Update current shape
        self._current_shape = (h // pool_size, w // pool_size, c)

    def add_flatten(self) -> None:
        """Add Flatten layer to convert 4D to 2D."""
        h, w, c = self._current_shape

        self.layers.append(Flatten())
        self.layer_names.append('flatten')

        # Update current shape (now 1D)
        self._current_shape = (h * w * c,)

    def add_dense(self, units: int, activation: str = 'relu') -> None:
        """
        Add Dense layer with activation.

        Args:
            units: Number of output units
            activation: 'relu', 'softmax', or None
        """
        if len(self._current_shape) == 3:
            in_features = self._current_shape[0] * self._current_shape[1] * self._current_shape[2]
        else:
            in_features = self._current_shape[0]

        dense = Dense(in_features=in_features, out_features=units)
        self.layers.append(dense)
        self.layer_names.append(f'dense_{units}')

        # Update current shape
        self._current_shape = (units,)

        # Add activation
        if activation == 'relu':
            self.layers.append(ReLU())
            self.layer_names.append('relu')
        elif activation == 'softmax':
            self.layers.append(Softmax())
            self.layer_names.append('softmax')

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        """
        Forward pass through all layers.

        Args:
            X: Input data of shape (batch, height, width, channels)

        Returns:
            output: Final predictions
            activations: List of activations for each layer
        """
        activations = [X]
        current = X

        for layer in self.layers:
            current = layer.forward(current)
            activations.append(current)

        return current, activations

    def backward(self, y_true: np.ndarray, activations: list[np.ndarray],
                 learning_rate: float) -> None:
        """
        Backward pass through all layers.

        Args:
            y_true: True labels (one-hot encoded for multi-class)
            activations: Activations from forward pass
            learning_rate: Learning rate for weight updates
        """
        # For softmax + cross-entropy, gradient is (output - target)
        grad = activations[-1] - y_true

        # Backpropagate through layers in reverse
        for layer in reversed(self.layers):
            grad = layer.backward(grad, learning_rate)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions for input data."""
        output, _ = self.forward(X)
        return output

    def predict_with_activations(self, X: np.ndarray) -> tuple[np.ndarray, list[list[float]]]:
        """
        Make predictions and return all layer activations for visualization.

        Returns:
            output: Predictions
            activations: List of activations per layer (flattened for JSON)
        """
        output, activations = self.forward(X)

        # Convert activations to list format
        activations_list = []
        for act in activations:
            if act.ndim <= 2:
                activations_list.append(act[0].tolist() if len(act) > 0 else [])
            else:
                # For 4D tensors, flatten spatial dimensions
                activations_list.append(act[0].tolist())

        return output, activations_list

    @staticmethod
    def cross_entropy_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Cross-entropy loss for classification."""
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return float(-np.mean(np.sum(y_true * np.log(y_pred), axis=1)))

    def calculate_accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate prediction accuracy."""
        predictions = self.predict(X)
        pred_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(y, axis=1)
        return float(np.mean(pred_classes == true_classes))

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int,
        learning_rate: float,
        verbose: bool = True,
        callback: Callable[[int, float, float], None] | None = None
    ) -> dict:
        """
        Train the network with static parameters.

        Args:
            X: Training data of shape (batch, height, width, channels)
            y: Labels (one-hot encoded)
            epochs: Number of training epochs
            learning_rate: Learning rate
            verbose: Print progress
            callback: Optional callback(epoch, loss, accuracy)

        Returns:
            Dictionary with training results
        """
        self.loss_history = []
        self.accuracy_history = []

        if verbose:
            print("-" * 60)
            print(f"Starting CNN training: {epochs} epochs, LR={learning_rate}")
            print(f"Input shape: {X.shape}")
            print("-" * 60)

        for epoch in range(epochs):
            # Forward pass
            output, activations = self.forward(X)

            # Calculate loss and accuracy
            loss = self.cross_entropy_loss(output, y)
            accuracy = self.calculate_accuracy(X, y)

            self.loss_history.append(loss)
            self.accuracy_history.append(accuracy)

            # Backward pass
            self.backward(y, activations, learning_rate)

            # Callback
            if callback:
                callback(epoch, loss, accuracy)

            # Print progress
            if verbose and (epoch % 50 == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch:4d}/{epochs} | Loss: {loss:.4f} | Accuracy: {accuracy*100:.1f}%")

        final_accuracy = self.accuracy_history[-1] if self.accuracy_history else 0

        if verbose:
            print("-" * 60)
            print(f"Training complete! Final accuracy: {final_accuracy*100:.1f}%")
            print("-" * 60)

        return {
            "epochs": epochs,
            "final_loss": self.loss_history[-1] if self.loss_history else 0,
            "final_accuracy": final_accuracy,
            "loss_history": self.loss_history,
            "accuracy_history": self.accuracy_history
        }

    def train_adaptive(
        self,
        X: np.ndarray,
        y: np.ndarray,
        target_accuracy: float = 0.95,
        max_epochs: int = 5000,
        verbose: bool = True,
        callback: Callable[[int, float, float], None] | None = None
    ) -> dict:
        """
        Adaptive training with automatic learning rate adjustment.

        Args:
            X: Training data
            y: Labels
            target_accuracy: Target accuracy to reach
            max_epochs: Maximum epochs
            verbose: Print progress
            callback: Optional callback

        Returns:
            Dictionary with training results
        """
        self.loss_history = []
        self.accuracy_history = []

        lr = 0.5  # Start with higher LR
        min_lr = 0.01
        lr_decay = 0.8
        patience = 200
        best_loss = float('inf')
        epochs_without_improvement = 0

        if verbose:
            print("-" * 60)
            print(f"Starting adaptive CNN training (target: {target_accuracy*100:.0f}%)")
            print(f"Input shape: {X.shape}")
            print("-" * 60)

        epoch = 0
        while epoch < max_epochs:
            # Forward pass
            output, activations = self.forward(X)

            # Calculate metrics
            loss = self.cross_entropy_loss(output, y)
            accuracy = self.calculate_accuracy(X, y)

            self.loss_history.append(loss)
            self.accuracy_history.append(accuracy)

            # Backward pass
            self.backward(y, activations, lr)

            # Check for improvement
            if loss < best_loss - 0.001:
                best_loss = loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            # Adjust learning rate
            if epochs_without_improvement >= patience:
                if lr > min_lr:
                    lr *= lr_decay
                    epochs_without_improvement = 0
                    if verbose:
                        print(f"  [Epoch {epoch}] Reducing LR to {lr:.4f}")

            # Callback
            if callback:
                callback(epoch, loss, accuracy)

            # Print progress
            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch:4d} | Loss: {loss:.4f} | Accuracy: {accuracy*100:.1f}% | LR: {lr:.4f}")

            # Check target
            if accuracy >= target_accuracy:
                if verbose:
                    print(f"\nTarget accuracy reached at epoch {epoch}!")
                break

            epoch += 1

        final_accuracy = self.accuracy_history[-1] if self.accuracy_history else 0

        if verbose:
            print("-" * 60)
            print(f"Adaptive training complete!")
            print(f"Final accuracy: {final_accuracy*100:.1f}%")
            print(f"Total epochs: {epoch}")
            print("-" * 60)

        return {
            "epochs": epoch,
            "final_loss": self.loss_history[-1] if self.loss_history else 0,
            "final_accuracy": final_accuracy,
            "loss_history": self.loss_history,
            "accuracy_history": self.accuracy_history,
            "target_reached": final_accuracy >= target_accuracy
        }

    def get_architecture(self) -> dict:
        """Get network architecture info."""
        return {
            "input_shape": self.input_shape,
            "layers": self.layer_names,
            "layer_count": len(self.layers),
            "type": "cnn"
        }

    def get_weights(self) -> list[dict]:
        """Get all layer weights for visualization."""
        weights = []
        for i, layer in enumerate(self.layers):
            layer_weights = layer.get_weights()
            if layer_weights:
                weights.append({
                    "layer": i,
                    "name": self.layer_names[i],
                    **layer_weights
                })
        return weights

    def get_feature_maps(self, X: np.ndarray) -> dict:
        """
        Get feature maps for visualization.

        Args:
            X: Single input image of shape (1, height, width, channels)

        Returns:
            Dictionary with conv and pool activations
        """
        _, activations = self.forward(X)

        feature_maps = {}
        for i, (name, act) in enumerate(zip(self.layer_names, activations[1:])):
            if 'conv' in name or 'pool' in name:
                # Convert to list for JSON serialization
                feature_maps[name] = act[0].tolist()  # Remove batch dimension

        return feature_maps

    def reset(self) -> None:
        """Reset network by reinitializing all layers."""
        # Store layer config
        layer_configs = []
        for name in self.layer_names:
            layer_configs.append(name)

        # Clear and rebuild
        self.layers = []
        self.layer_names = []
        self._current_shape = self.input_shape
        self.loss_history = []
        self.accuracy_history = []

        # Rebuild layers (simplified - would need full config in production)
        for name in layer_configs:
            if name.startswith('conv2d_'):
                filters = int(name.split('_')[1])
                self.add_conv2d(filters=filters, activation=None)
            elif name.startswith('maxpool_'):
                pool_size = int(name.split('_')[1])
                self.add_maxpool2d(pool_size=pool_size)
            elif name == 'flatten':
                self.add_flatten()
            elif name.startswith('dense_'):
                units = int(name.split('_')[1])
                self.add_dense(units=units, activation=None)
            elif name == 'relu':
                self.layers.append(ReLU())
                self.layer_names.append('relu')
            elif name == 'softmax':
                self.layers.append(Softmax())
                self.layer_names.append('softmax')


# -----------------------------------------------------------------------------
# Demo / Test
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("CNN Network Test")
    print("=" * 60)

    # Create simple CNN
    nn = CNNNetwork(input_shape=(8, 8, 1))
    nn.add_conv2d(filters=4, kernel_size=3)
    nn.add_maxpool2d(pool_size=2)
    nn.add_flatten()
    nn.add_dense(units=16)
    nn.add_dense(units=3, activation='softmax')

    print(f"\nArchitecture: {nn.get_architecture()}")

    # Generate synthetic data (3 classes)
    np.random.seed(42)
    n_samples = 150  # 50 per class

    X = np.random.randn(n_samples, 8, 8, 1) * 0.1
    y = np.zeros((n_samples, 3))

    # Class 0: high values in top-left
    for i in range(50):
        X[i, :4, :4, 0] += 0.8
        y[i] = [1, 0, 0]

    # Class 1: high values in center
    for i in range(50, 100):
        X[i, 2:6, 2:6, 0] += 0.8
        y[i] = [0, 1, 0]

    # Class 2: high values in bottom-right
    for i in range(100, 150):
        X[i, 4:, 4:, 0] += 0.8
        y[i] = [0, 0, 1]

    # Shuffle
    indices = np.random.permutation(n_samples)
    X = X[indices]
    y = y[indices]

    print(f"\nData shape: X={X.shape}, y={y.shape}")

    # Train
    result = nn.train(X, y, epochs=200, learning_rate=0.3, verbose=True)

    # Test accuracy
    accuracy = nn.calculate_accuracy(X, y)
    print(f"\nFinal accuracy: {accuracy*100:.1f}%")

    # Test feature maps
    feature_maps = nn.get_feature_maps(X[:1])
    print(f"\nFeature maps keys: {list(feature_maps.keys())}")

    print("\n" + "=" * 60)
    print("CNN Network test complete!")
    print("=" * 60)
