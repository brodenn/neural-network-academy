"""
CNN Layer Implementations from Scratch

Pure NumPy implementation of convolutional neural network layers.
Uses im2col for efficient convolution operations.
"""

# Snygg användning av interfaces och underklasser här!

import numpy as np
from abc import ABC, abstractmethod


class Layer(ABC):
    """Abstract base class for all layers."""

    @abstractmethod
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Forward pass through the layer."""
        pass

    @abstractmethod
    def backward(self, grad_output: np.ndarray, learning_rate: float) -> np.ndarray:
        """Backward pass - compute gradients and update weights if applicable."""
        pass

    def get_weights(self) -> dict | None:
        """Return layer weights for visualization. Override if layer has weights."""
        return None


class Conv2D(Layer):
    """
    2D Convolutional Layer using im2col for efficient computation.

    Input shape: (batch, height, width, channels)
    Output shape: (batch, out_height, out_width, num_filters)
    """

    def __init__(self, num_filters: int, kernel_size: int, in_channels: int,
                 stride: int = 1, padding: str = 'valid'):
        """
        Initialize Conv2D layer.

        Args:
            num_filters: Number of output filters
            kernel_size: Size of square kernel (e.g., 3 for 3x3)
            in_channels: Number of input channels
            stride: Convolution stride (default: 1)
            padding: 'valid' (no padding) or 'same' (pad to maintain size)
        """
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.stride = stride
        self.padding = padding

        # Initialize filters with He initialization
        scale = np.sqrt(2.0 / (kernel_size * kernel_size * in_channels))
        self.filters = np.random.randn(
            kernel_size, kernel_size, in_channels, num_filters
        ) * scale
        self.biases = np.zeros(num_filters)

        # Cache for backward pass
        self._input_cache = None
        self._col_cache = None

    def _get_output_shape(self, input_shape: tuple) -> tuple:
        """Calculate output dimensions."""
        _, h, w, _ = input_shape
        if self.padding == 'same':
            out_h = int(np.ceil(h / self.stride))
            out_w = int(np.ceil(w / self.stride))
        else:  # valid
            out_h = (h - self.kernel_size) // self.stride + 1
            out_w = (w - self.kernel_size) // self.stride + 1
        return out_h, out_w

    def _im2col(self, input: np.ndarray) -> np.ndarray:
        """
        Convert input to column matrix for efficient convolution.

        Transforms sliding windows into columns for matrix multiplication.
        """
        batch, h, w, c = input.shape
        out_h, out_w = self._get_output_shape(input.shape)
        k = self.kernel_size

        # Apply padding if needed
        if self.padding == 'same':
            pad_h = max((out_h - 1) * self.stride + k - h, 0)
            pad_w = max((out_w - 1) * self.stride + k - w, 0)
            pad_top = pad_h // 2
            pad_left = pad_w // 2
            input = np.pad(input, ((0, 0), (pad_top, pad_h - pad_top),
                                   (pad_left, pad_w - pad_left), (0, 0)))
            h, w = input.shape[1], input.shape[2]

        # Extract windows using stride tricks for efficiency
        shape = (batch, out_h, out_w, k, k, c)
        strides = (
            input.strides[0],
            input.strides[1] * self.stride,
            input.strides[2] * self.stride,
            input.strides[1],
            input.strides[2],
            input.strides[3]
        )
        windows = np.lib.stride_tricks.as_strided(input, shape=shape, strides=strides)

        # Reshape to (batch * out_h * out_w, k * k * c)
        col = windows.reshape(batch * out_h * out_w, k * k * c)
        return col

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Forward pass using im2col.

        Args:
            inputs: Shape (batch, height, width, channels)

        Returns:
            Output of shape (batch, out_height, out_width, num_filters)
        """
        self._input_cache = inputs
        batch = inputs.shape[0]
        out_h, out_w = self._get_output_shape(inputs.shape)

        # Convert to columns
        col = self._im2col(inputs)
        self._col_cache = col

        # Reshape filters for matrix multiplication
        filters_flat = self.filters.reshape(-1, self.num_filters)

        # Convolve: (batch*out_h*out_w, k*k*c) @ (k*k*c, num_filters)
        output = col @ filters_flat + self.biases

        # Reshape to (batch, out_h, out_w, num_filters)
        output = output.reshape(batch, out_h, out_w, self.num_filters)
        return output

    def backward(self, grad_output: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        Backward pass - compute gradients and update weights.

        Args:
            grad_output: Gradient from next layer, shape (batch, out_h, out_w, num_filters)
            learning_rate: Learning rate for weight updates

        Returns:
            Gradient w.r.t. input for previous layer
        """
        batch, out_h, out_w, _ = grad_output.shape
        _, in_h, in_w, _ = self._input_cache.shape

        # Reshape gradient for matrix operations
        grad_flat = grad_output.reshape(-1, self.num_filters)

        # Gradient w.r.t. filters: col^T @ grad_flat
        grad_filters = self._col_cache.T @ grad_flat
        grad_filters = grad_filters.reshape(self.filters.shape)

        # Gradient w.r.t. biases: sum over batch and spatial dimensions
        grad_biases = np.sum(grad_flat, axis=0)

        # Gradient w.r.t. input using full convolution
        # Rotate filters 180 degrees
        filters_rotated = np.rot90(self.filters, 2, axes=(0, 1))

        # Pad grad_output for full convolution
        pad = self.kernel_size - 1
        grad_padded = np.pad(grad_output,
                             ((0, 0), (pad, pad), (pad, pad), (0, 0)))

        # Compute input gradient
        grad_input = np.zeros_like(self._input_cache)

        for b in range(batch):
            for c_in in range(self._input_cache.shape[3]):
                for c_out in range(self.num_filters):
                    for i in range(in_h):
                        for j in range(in_w):
                            grad_input[b, i, j, c_in] += np.sum(
                                grad_padded[b, i:i+self.kernel_size, j:j+self.kernel_size, c_out] *
                                filters_rotated[:, :, c_in, c_out]
                            )

        # Update weights
        self.filters -= learning_rate * grad_filters / batch
        self.biases -= learning_rate * grad_biases / batch

        return grad_input

    def get_weights(self) -> dict:
        """Return filter weights for visualization."""
        return {
            'filters': self.filters.tolist(),
            'biases': self.biases.tolist(),
            'shape': list(self.filters.shape)
        }


class MaxPool2D(Layer):
    """
    2D Max Pooling Layer.

    Input shape: (batch, height, width, channels)
    Output shape: (batch, out_height, out_width, channels)
    """

    def __init__(self, pool_size: int = 2, stride: int | None = None):
        """
        Initialize MaxPool2D layer.

        Args:
            pool_size: Size of pooling window (e.g., 2 for 2x2)
            stride: Pooling stride (default: same as pool_size)
        """
        self.pool_size = pool_size
        self.stride = stride if stride is not None else pool_size

        # Cache for backward pass
        self._input_cache = None
        self._max_indices = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Forward pass - take max in each pooling window.

        Args:
            inputs: Shape (batch, height, width, channels)

        Returns:
            Output of shape (batch, out_height, out_width, channels)
        """
        self._input_cache = inputs
        batch, h, w, c = inputs.shape

        out_h = (h - self.pool_size) // self.stride + 1
        out_w = (w - self.pool_size) // self.stride + 1

        output = np.zeros((batch, out_h, out_w, c))
        self._max_indices = np.zeros((batch, out_h, out_w, c, 2), dtype=int)

        for b in range(batch):
            for i in range(out_h):
                for j in range(out_w):
                    h_start = i * self.stride
                    w_start = j * self.stride
                    window = inputs[b, h_start:h_start+self.pool_size,
                                   w_start:w_start+self.pool_size, :]

                    # Get max value for each channel
                    for ch in range(c):
                        window_ch = window[:, :, ch]
                        max_idx = np.unravel_index(np.argmax(window_ch), window_ch.shape)
                        output[b, i, j, ch] = window_ch[max_idx]
                        self._max_indices[b, i, j, ch] = [
                            h_start + max_idx[0],
                            w_start + max_idx[1]
                        ]

        return output

    def backward(self, grad_output: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        Backward pass - route gradients to max positions only.

        Args:
            grad_output: Gradient from next layer
            learning_rate: Not used (no learnable parameters)

        Returns:
            Gradient w.r.t. input
        """
        grad_input = np.zeros_like(self._input_cache)
        batch, out_h, out_w, c = grad_output.shape

        for b in range(batch):
            for i in range(out_h):
                for j in range(out_w):
                    for ch in range(c):
                        max_h, max_w = self._max_indices[b, i, j, ch]
                        grad_input[b, max_h, max_w, ch] += grad_output[b, i, j, ch]

        return grad_input


class Flatten(Layer):
    """
    Flatten layer - converts 4D tensor to 2D.

    Input shape: (batch, height, width, channels)
    Output shape: (batch, height * width * channels)
    """

    def __init__(self):
        self._input_shape = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Flatten input to 2D."""
        self._input_shape = inputs.shape
        batch = inputs.shape[0]
        return inputs.reshape(batch, -1)

    def backward(self, grad_output: np.ndarray, learning_rate: float) -> np.ndarray:
        """Reshape gradient back to 4D."""
        return grad_output.reshape(self._input_shape)


class Dense(Layer):
    """
    Fully connected (dense) layer.

    Input shape: (batch, in_features)
    Output shape: (batch, out_features)
    """

    def __init__(self, in_features: int, out_features: int):
        """
        Initialize Dense layer.

        Args:
            in_features: Number of input features
            out_features: Number of output features
        """
        self.in_features = in_features
        self.out_features = out_features

        # Xavier initialization
        limit = np.sqrt(6.0 / (in_features + out_features))
        self.weights = np.random.uniform(-limit, limit, (in_features, out_features))
        self.biases = np.zeros((1, out_features))

        # Cache for backward pass
        self._input_cache = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Forward pass: output = input @ weights + biases."""
        self._input_cache = inputs
        return inputs @ self.weights + self.biases

    def backward(self, grad_output: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        Backward pass - compute gradients and update weights.

        Args:
            grad_output: Gradient from next layer, shape (batch, out_features)
            learning_rate: Learning rate for weight updates

        Returns:
            Gradient w.r.t. input for previous layer
        """
        batch = self._input_cache.shape[0]

        # Gradient w.r.t. weights
        grad_weights = self._input_cache.T @ grad_output / batch

        # Gradient w.r.t. biases
        grad_biases = np.mean(grad_output, axis=0, keepdims=True)

        # Gradient w.r.t. input
        grad_input = grad_output @ self.weights.T

        # Update weights
        self.weights -= learning_rate * grad_weights
        self.biases -= learning_rate * grad_biases

        return grad_input

    def get_weights(self) -> dict:
        """Return weights for visualization."""
        return {
            'weights': self.weights.tolist(),
            'biases': self.biases.tolist(),
            'shape': [self.in_features, self.out_features]
        }


class ReLU(Layer):
    """ReLU activation layer."""

    def __init__(self):
        self._input_cache = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Apply ReLU: max(0, x)."""
        self._input_cache = inputs
        return np.maximum(0, inputs)

    def backward(self, grad_output: np.ndarray, learning_rate: float) -> np.ndarray:
        """Gradient is 1 where input > 0, else 0."""
        return grad_output * (self._input_cache > 0).astype(float)


class Softmax(Layer):
    """Softmax activation layer for multi-class output."""

    def __init__(self):
        self._output_cache = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Apply softmax activation."""
        # Subtract max for numerical stability
        exp_x = np.exp(inputs - np.max(inputs, axis=-1, keepdims=True))
        self._output_cache = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        return self._output_cache

    def backward(self, grad_output: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        Backward pass for softmax.

        Note: When combined with cross-entropy loss, gradient simplifies to (output - target).
        This layer expects that simplified gradient as input.
        """
        return grad_output


# -----------------------------------------------------------------------------
# Test / Demo
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("CNN Layers Test")
    print("=" * 60)

    # Test Conv2D
    print("\n--- Conv2D Test ---")
    conv = Conv2D(num_filters=4, kernel_size=3, in_channels=1)
    x = np.random.randn(2, 8, 8, 1)  # batch=2, 8x8 image, 1 channel
    out = conv.forward(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    assert out.shape == (2, 6, 6, 4), f"Expected (2, 6, 6, 4), got {out.shape}"

    # Test backward
    grad = np.random.randn(*out.shape)
    grad_input = conv.backward(grad, learning_rate=0.01)
    print(f"Grad input shape: {grad_input.shape}")
    assert grad_input.shape == x.shape

    # Test MaxPool2D
    print("\n--- MaxPool2D Test ---")
    pool = MaxPool2D(pool_size=2)
    pool_out = pool.forward(out)
    print(f"Pool input: {out.shape}")
    print(f"Pool output: {pool_out.shape}")
    assert pool_out.shape == (2, 3, 3, 4), f"Expected (2, 3, 3, 4), got {pool_out.shape}"

    # Test Flatten
    print("\n--- Flatten Test ---")
    flatten = Flatten()
    flat_out = flatten.forward(pool_out)
    print(f"Flatten input: {pool_out.shape}")
    print(f"Flatten output: {flat_out.shape}")
    assert flat_out.shape == (2, 36), f"Expected (2, 36), got {flat_out.shape}"

    # Test Dense
    print("\n--- Dense Test ---")
    dense = Dense(in_features=36, out_features=3)
    dense_out = dense.forward(flat_out)
    print(f"Dense input: {flat_out.shape}")
    print(f"Dense output: {dense_out.shape}")
    assert dense_out.shape == (2, 3)

    print("\n" + "=" * 60)
    print("All layer tests passed!")
    print("=" * 60)
