"""
Neural Network Implementation from Scratch

School Project: Maskininlärning - Projekt II (Variant 2)
Requirements: G1-G7, VG8-VG10

This module implements a feedforward neural network using only NumPy.
No external ML libraries (TensorFlow, PyTorch, etc.) are used.
"""

import numpy as np
from typing import Callable


class NeuralNetwork:
    """
    Feedforward neural network with backpropagation.

    Supports:
    - Arbitrary number of hidden layers (VG8)
    - Static training with user-defined parameters (G4)
    - Adaptive training to ~100% accuracy (VG9)
    - Multiple output activations: sigmoid (binary), softmax (multi-class)

    Example:
        >>> nn = NeuralNetwork([5, 12, 8, 1])  # 5 inputs, 2 hidden layers, 1 output
        >>> nn.train(X, y, epochs=1000, learning_rate=0.1)

        >>> nn = NeuralNetwork([8, 4, 3], output_activation='softmax')  # Multi-class
    """

    def __init__(self, layer_sizes: list[int], learning_rate: float = 0.1,
                 output_activation: str = 'sigmoid',
                 weight_init: str = 'xavier',
                 hidden_activation: str = 'relu',
                 use_biases: bool = True):
        """
        Initialize the neural network.

        Args:
            layer_sizes: List of neurons per layer, e.g., [5, 12, 8, 1]
                        First element is input size, last is output size.
            learning_rate: Step size for gradient descent (default: 0.1)
            output_activation: 'sigmoid' for binary/regression, 'softmax' for multi-class
            weight_init: Weight initialization method: 'xavier', 'he', 'random', 'zeros'
            hidden_activation: Activation for hidden layers: 'relu', 'sigmoid', 'tanh'
            use_biases: Whether to use bias terms (default: True)
        """
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.num_layers = len(layer_sizes)
        self.output_activation = output_activation
        self.weight_init = weight_init
        self.hidden_activation = hidden_activation
        self.use_biases = use_biases

        # Initialize weights and biases
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        self._initialize_weights()

        # Training history
        self.loss_history: list[float] = []
        self.accuracy_history: list[float] = []

    def _initialize_weights(self) -> None:
        """Initialize weights based on selected method."""
        self.weights = []
        self.biases = []

        for i in range(self.num_layers - 1):
            in_size = self.layer_sizes[i]
            out_size = self.layer_sizes[i + 1]

            # Weight initialization
            if self.weight_init == 'zeros':
                # Zeros - network won't learn (for demonstration)
                w = np.zeros((in_size, out_size))
            elif self.weight_init == 'random':
                # Random uniform [-1, 1] - can be unstable
                w = np.random.uniform(-1, 1, (in_size, out_size))
            elif self.weight_init == 'he':
                # He initialization - good for ReLU
                std = np.sqrt(2.0 / in_size)
                w = np.random.randn(in_size, out_size) * std
            else:  # xavier (default)
                # Xavier/Glorot - good general purpose
                limit = np.sqrt(6.0 / (in_size + out_size))
                w = np.random.uniform(-limit, limit, (in_size, out_size))

            # Biases
            if self.use_biases:
                b = np.zeros((1, out_size))
            else:
                b = np.zeros((1, out_size))  # Still create array but won't update

            self.weights.append(w)
            self.biases.append(b)

    # -------------------------------------------------------------------------
    # Activation Functions
    # -------------------------------------------------------------------------

    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        """ReLU activation: max(0, x)"""
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x: np.ndarray) -> np.ndarray:
        """Derivative of ReLU: 1 if x > 0, else 0"""
        return (x > 0).astype(float)

    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        """Sigmoid activation: 1 / (1 + e^-x)"""
        # Clip to avoid overflow
        x = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
        """Derivative of sigmoid: x * (1 - x) where x is already sigmoid(z)"""
        return x * (1.0 - x)

    @staticmethod
    def tanh(x: np.ndarray) -> np.ndarray:
        """Tanh activation: (e^x - e^-x) / (e^x + e^-x)"""
        return np.tanh(x)

    @staticmethod
    def tanh_derivative(x: np.ndarray) -> np.ndarray:
        """Derivative of tanh: 1 - x^2 where x is already tanh(z)"""
        return 1.0 - x ** 2

    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        """Softmax activation for multi-class classification."""
        # Subtract max for numerical stability
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def _get_hidden_activation(self, z: np.ndarray) -> np.ndarray:
        """Apply the configured hidden layer activation function."""
        if self.hidden_activation == 'sigmoid':
            return self.sigmoid(z)
        elif self.hidden_activation == 'tanh':
            return self.tanh(z)
        else:  # relu (default)
            return self.relu(z)

    def _get_hidden_activation_derivative(self, activated: np.ndarray) -> np.ndarray:
        """Get derivative of hidden activation (activated = already passed through activation)."""
        if self.hidden_activation == 'sigmoid':
            return self.sigmoid_derivative(activated)
        elif self.hidden_activation == 'tanh':
            return self.tanh_derivative(activated)
        else:  # relu
            return self.relu_derivative(activated)

    # -------------------------------------------------------------------------
    # Forward Propagation
    # -------------------------------------------------------------------------

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
        """
        Forward pass through the network.

        Args:
            X: Input data of shape (n_samples, n_features)

        Returns:
            output: Final prediction
            activations: List of activations for each layer
            z_values: List of pre-activation values for each layer
        """
        activations = [X]
        z_values = []

        current = X
        for i in range(self.num_layers - 1):
            # Linear transformation: z = X @ W + b
            z = np.dot(current, self.weights[i]) + self.biases[i]
            z_values.append(z)

            # Apply activation function
            if i < self.num_layers - 2:
                # Hidden layers: configurable activation
                current = self._get_hidden_activation(z)
            else:
                # Output layer: configurable activation
                if self.output_activation == 'softmax':
                    current = self.softmax(z)
                else:
                    current = self.sigmoid(z)

            activations.append(current)

        return current, activations, z_values

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions for input data.

        Args:
            X: Input data of shape (n_samples, n_features)

        Returns:
            Predictions of shape (n_samples, n_outputs)
        """
        output, _, _ = self.forward(X)
        return output

    def predict_with_activations(self, X: np.ndarray) -> tuple[np.ndarray, list[list[float]]]:
        """
        Make predictions and return all layer activations for visualization.

        Args:
            X: Input data of shape (n_samples, n_features)

        Returns:
            output: Predictions of shape (n_samples, n_outputs)
            activations: List of activations for each layer (flattened for single sample)
        """
        output, activations, _ = self.forward(X)

        # Convert activations to list format for JSON serialization
        # For single sample, extract first row of each activation
        activations_list = []
        for act in activations:
            if len(act.shape) == 1:
                activations_list.append(act.tolist())
            else:
                # Take first sample's activations
                activations_list.append(act[0].tolist())

        return output, activations_list

    # -------------------------------------------------------------------------
    # Backpropagation
    # -------------------------------------------------------------------------

    def backward(
        self,
        X: np.ndarray,
        y: np.ndarray,
        activations: list[np.ndarray],
        z_values: list[np.ndarray]
    ) -> None:
        """
        Backward pass: compute gradients and update weights.

        Args:
            X: Input data
            y: Target labels
            activations: Activations from forward pass
            z_values: Pre-activation values from forward pass
        """
        n_samples = X.shape[0]

        # Output layer error (gradient depends on activation)
        output = activations[-1]
        if self.output_activation == 'softmax':
            # Softmax + cross-entropy: gradient is simply (output - y)
            delta = output - y
        else:
            # Sigmoid + MSE: gradient is (output - y) * sigmoid_derivative
            error = output - y
            delta = error * self.sigmoid_derivative(output)

        # Store deltas for each layer (working backwards)
        deltas = [delta]

        # Backpropagate through hidden layers
        for i in range(self.num_layers - 2, 0, -1):
            delta = np.dot(deltas[0], self.weights[i].T) * self._get_hidden_activation_derivative(activations[i])
            deltas.insert(0, delta)

        # Update weights and biases
        for i in range(self.num_layers - 1):
            gradient_w = np.dot(activations[i].T, deltas[i]) / n_samples
            gradient_b = np.mean(deltas[i], axis=0, keepdims=True)

            self.weights[i] -= self.learning_rate * gradient_w
            if self.use_biases:
                self.biases[i] -= self.learning_rate * gradient_b

    # -------------------------------------------------------------------------
    # Loss Functions
    # -------------------------------------------------------------------------

    @staticmethod
    def mse_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Mean Squared Error loss."""
        return float(np.mean((y_pred - y_true) ** 2))

    @staticmethod
    def cross_entropy_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Cross-entropy loss for classification (works with both binary and multi-class)."""
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return float(-np.mean(y_true * np.log(y_pred)))

    def calculate_accuracy(self, X: np.ndarray, y: np.ndarray, tolerance: float = 0.1) -> float:
        """Calculate prediction accuracy (binary, multi-class, or regression)."""
        predictions = self.predict(X)
        if self.output_activation == 'softmax':
            # Multi-class: compare argmax
            pred_classes = np.argmax(predictions, axis=1)
            true_classes = np.argmax(y, axis=1)
            correct = np.sum(pred_classes == true_classes)
        else:
            # Check if targets are binary (all 0 or 1) or continuous (regression)
            is_binary = np.all((y == 0) | (y == 1))
            if is_binary:
                # Binary classification: threshold at 0.5
                predicted_classes = (predictions >= 0.5).astype(float)
                correct = np.sum(predicted_classes == y)
            else:
                # Regression: use tolerance-based accuracy
                correct = np.sum(np.abs(predictions - y) < tolerance)
        return float(correct / len(y))

    # -------------------------------------------------------------------------
    # Training Methods
    # -------------------------------------------------------------------------

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int,
        learning_rate: float,
        verbose: bool = True,
        callback: Callable[[int, float, float], None] | None = None,
        stop_check: Callable[[], bool] | None = None
    ) -> dict:
        """
        Static training with user-defined parameters (G4).

        Args:
            X: Training input data
            y: Training labels
            epochs: Number of training epochs
            learning_rate: Learning rate for gradient descent
            verbose: Print progress to terminal
            callback: Optional callback(epoch, loss, accuracy) for updates
            stop_check: Optional callback that returns True to stop training early

        Returns:
            Dictionary with training results
        """
        self.learning_rate = learning_rate
        self.loss_history = []
        self.accuracy_history = []
        stopped = False

        if verbose:
            print("-" * 80)
            print(f"Starting static training: {epochs} epochs, LR={learning_rate}")
            print(f"Network architecture: {self.layer_sizes}")
            print("-" * 80)

        for epoch in range(epochs):
            # Check for stop request
            if stop_check and stop_check():
                stopped = True
                if verbose:
                    print(f"\nTraining stopped by user at epoch {epoch}")
                break

            # Forward pass
            output, activations, z_values = self.forward(X)

            # Calculate loss and accuracy (use appropriate loss function)
            if self.output_activation == 'softmax':
                loss = self.cross_entropy_loss(output, y)
            else:
                loss = self.mse_loss(output, y)
            accuracy = self.calculate_accuracy(X, y)

            self.loss_history.append(loss)
            self.accuracy_history.append(accuracy)

            # Backward pass
            self.backward(X, y, activations, z_values)

            # Progress callback
            if callback:
                callback(epoch, loss, accuracy)

            # Print progress
            if verbose and (epoch % 100 == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch:5d}/{epochs} | Loss: {loss:.6f} | Accuracy: {accuracy*100:.1f}%")

        final_accuracy = self.accuracy_history[-1] if self.accuracy_history else 0
        actual_epochs = len(self.loss_history)

        if verbose and not stopped:
            print("-" * 80)
            print(f"Training complete! Final accuracy: {final_accuracy*100:.1f}%")
            print("-" * 80)

        return {
            "epochs": actual_epochs,
            "final_loss": self.loss_history[-1] if self.loss_history else 0,
            "final_accuracy": final_accuracy,
            "loss_history": self.loss_history,
            "accuracy_history": self.accuracy_history,
            "stopped": stopped
        }

    def train_adaptive(
        self,
        X: np.ndarray,
        y: np.ndarray,
        target_accuracy: float | Callable[[], float] = 0.99,
        max_epochs: int = 100000,
        verbose: bool = True,
        callback: Callable[[int, float, float], None] | None = None,
        stop_check: Callable[[], bool] | None = None
    ) -> dict:
        """
        Adaptive training to ~100% accuracy (VG9).

        The network automatically adjusts learning rate and trains until
        target accuracy is reached. User does NOT specify epochs or learning rate.

        Features:
        - Automatic learning rate adjustment
        - Restarts with new random weights if stuck
        - Continues until target accuracy reached
        - Target accuracy can be changed during training via callable

        Args:
            X: Training input data
            y: Training labels
            target_accuracy: Target accuracy to reach (float or callable returning float)
            max_epochs: Safety limit for epochs
            verbose: Print progress to terminal
            callback: Optional callback(epoch, loss, accuracy) for updates
            stop_check: Optional callback that returns True to stop training early

        Returns:
            Dictionary with training results
        """
        self.loss_history = []
        self.accuracy_history = []
        stopped = False

        # Helper to get current target (supports both float and callable)
        def get_target() -> float:
            if callable(target_accuracy):
                return target_accuracy()
            return target_accuracy

        # Adaptive parameters
        initial_lr = 1.0
        lr = initial_lr
        patience = 1000  # Epochs before adjusting
        lr_decay = 0.7
        min_lr = 0.01
        best_loss = float('inf')
        best_accuracy = 0.0
        epochs_without_improvement = 0
        restart_count = 0
        max_restarts = 10

        initial_target = get_target()
        if verbose:
            print("-" * 80)
            print(f"Starting adaptive training (target: {initial_target*100:.0f}% accuracy)")
            print(f"Network architecture: {self.layer_sizes}")
            print("-" * 80)

        epoch = 0
        while epoch < max_epochs:
            # Check for stop request
            if stop_check and stop_check():
                stopped = True
                if verbose:
                    print(f"\nTraining stopped by user at epoch {epoch}")
                break

            self.learning_rate = lr

            # Forward pass
            output, activations, z_values = self.forward(X)

            # Calculate loss and accuracy (use appropriate loss function)
            if self.output_activation == 'softmax':
                loss = self.cross_entropy_loss(output, y)
            else:
                loss = self.mse_loss(output, y)
            accuracy = self.calculate_accuracy(X, y)

            self.loss_history.append(loss)
            self.accuracy_history.append(accuracy)

            # Backward pass
            self.backward(X, y, activations, z_values)

            # Track best accuracy
            if accuracy > best_accuracy:
                best_accuracy = accuracy

            # Check for loss improvement
            if loss < best_loss - 0.0001:
                best_loss = loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            # Adaptive learning rate adjustment
            if epochs_without_improvement >= patience:
                if lr > min_lr:
                    lr *= lr_decay
                    epochs_without_improvement = 0
                    if verbose:
                        print(f"  [Epoch {epoch}] Reducing learning rate to {lr:.4f}")
                elif restart_count < max_restarts and accuracy < target_accuracy:
                    # Restart with new random weights
                    restart_count += 1
                    if verbose:
                        print(f"  [Epoch {epoch}] Restarting with new weights (attempt {restart_count})")
                    self._reinitialize_weights()
                    lr = initial_lr
                    best_loss = float('inf')
                    epochs_without_improvement = 0

            # Progress callback
            if callback:
                callback(epoch, loss, accuracy)

            # Print progress
            if verbose and (epoch % 1000 == 0):
                print(f"Epoch {epoch:5d} | Loss: {loss:.6f} | Accuracy: {accuracy*100:.1f}% | LR: {lr:.4f}")

            # Check if target reached (get current target in case it changed)
            current_target = get_target()
            if accuracy >= current_target:
                if verbose:
                    print(f"\nTarget accuracy ({current_target*100:.0f}%) reached at epoch {epoch}!")
                break

            epoch += 1

        final_accuracy = self.accuracy_history[-1] if self.accuracy_history else 0
        final_target = get_target()

        if verbose and not stopped:
            print("-" * 80)
            print(f"Adaptive training complete!")
            print(f"Final accuracy: {final_accuracy*100:.1f}%")
            print(f"Total epochs: {epoch}")
            print(f"Restarts: {restart_count}")
            print("-" * 80)

        return {
            "epochs": epoch,
            "final_loss": self.loss_history[-1] if self.loss_history else 0,
            "final_accuracy": final_accuracy,
            "loss_history": self.loss_history,
            "accuracy_history": self.accuracy_history,
            "target_reached": final_accuracy >= final_target,
            "restarts": restart_count,
            "stopped": stopped
        }

    def _reinitialize_weights(self) -> None:
        """Reinitialize weights using current initialization method."""
        self._initialize_weights()

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def get_weights(self) -> list[dict]:
        """Get all weights and biases for visualization."""
        return [
            {
                "layer": i,
                "weights": self.weights[i].tolist(),
                "biases": self.biases[i].tolist(),
                "input_size": self.layer_sizes[i],
                "output_size": self.layer_sizes[i + 1]
            }
            for i in range(len(self.weights))
        ]

    def get_architecture(self) -> dict:
        """Get network architecture info."""
        return {
            "layer_sizes": self.layer_sizes,
            "num_layers": self.num_layers,
            "num_weights": sum(w.size for w in self.weights),
            "num_biases": sum(b.size for b in self.biases),
            "weight_init": self.weight_init,
            "hidden_activation": self.hidden_activation,
            "use_biases": self.use_biases
        }

    def reset(self) -> None:
        """Reset weights to random initialization (preserves settings)."""
        self._initialize_weights()
        self.loss_history = []
        self.accuracy_history = []

    def save(self, filepath: str) -> None:
        """Save model weights to file."""
        np.savez(
            filepath,
            layer_sizes=self.layer_sizes,
            learning_rate=self.learning_rate,
            output_activation=self.output_activation,
            weights=[w for w in self.weights],
            biases=[b for b in self.biases]
        )

    def load(self, filepath: str) -> None:
        """Load model weights from file."""
        data = np.load(filepath, allow_pickle=True)
        self.layer_sizes = list(data['layer_sizes'])
        self.learning_rate = float(data['learning_rate'])
        self.output_activation = str(data.get('output_activation', 'sigmoid'))
        self.weights = list(data['weights'])
        self.biases = list(data['biases'])
        self.num_layers = len(self.layer_sizes)


# -----------------------------------------------------------------------------
# Training Data Generation
# -----------------------------------------------------------------------------

def generate_xor_data(bits: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate XOR training data for n-bit input.

    XOR rule: Output is 1 if odd number of inputs are 1, else 0.

    Args:
        bits: Number of input bits (default: 5 for VG10)

    Returns:
        X: Input data of shape (2^bits, bits)
        y: Labels of shape (2^bits, 1)
    """
    n_samples = 2 ** bits
    X = []
    y = []

    for i in range(n_samples):
        # Convert integer to binary representation
        inputs = [(i >> j) & 1 for j in range(bits)]
        # XOR: odd number of 1s = 1, even number = 0
        label = sum(inputs) % 2
        X.append(inputs)
        y.append([label])

    return np.array(X, dtype=float), np.array(y, dtype=float)


# -----------------------------------------------------------------------------
# Demo / Test
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 80)
    print("Neural Network XOR Demo")
    print("=" * 80)

    # Generate 5-bit XOR data (VG10)
    X, y = generate_xor_data(bits=5)
    print(f"\nTraining data: {len(X)} samples (5-bit XOR)")

    # Create network with 3 hidden layers (VG8, VG10)
    nn = NeuralNetwork([5, 12, 8, 4, 1])

    # Test adaptive training (VG9)
    print("\n--- Adaptive Training (VG9) ---")
    result = nn.train_adaptive(X, y, target_accuracy=0.99)

    # Print predictions in required format (G6)
    print("\n" + "-" * 80)
    print(f"Prediction accuracy: {result['final_accuracy']*100:.1f}%")

    predictions = nn.predict(X)
    for i in range(len(X)):
        pred_rounded = round(predictions[i][0])
        error = abs(pred_rounded - y[i][0])
        print(f"Input: {X[i].tolist()}, prediction: [{pred_rounded:.1f}], "
              f"reference: {y[i].tolist()}, error: {error:.1f}")

    print("-" * 80)
