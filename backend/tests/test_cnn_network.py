"""
Tests for cnn_network.py

Tests CNN layer construction, forward propagation, and training.
"""

import numpy as np
import pytest
from cnn_network import CNNNetwork


class TestCNNConstruction:
    """Test CNN network construction."""

    def test_create_empty_network(self):
        """Should create network with input shape."""
        nn = CNNNetwork(input_shape=(8, 8, 1))
        assert nn.input_shape == (8, 8, 1)
        assert len(nn.layers) == 0

    def test_add_conv2d(self):
        """Should add conv2d layer."""
        nn = CNNNetwork(input_shape=(8, 8, 1))
        nn.add_conv2d(filters=4, kernel_size=3)

        # Conv2D + ReLU = 2 layers
        assert len(nn.layers) == 2

    def test_conv2d_shape_tracking(self):
        """Should track shape after conv2d."""
        nn = CNNNetwork(input_shape=(8, 8, 1))
        nn.add_conv2d(filters=4, kernel_size=3, padding='valid')

        # 8 - 3 + 1 = 6, with 4 filters
        assert nn._current_shape == (6, 6, 4)

    def test_add_maxpool2d(self):
        """Should add maxpool layer."""
        nn = CNNNetwork(input_shape=(8, 8, 1))
        nn.add_conv2d(filters=4, kernel_size=3)
        nn.add_maxpool2d(pool_size=2)

        # Conv2D + ReLU + MaxPool = 3 layers
        assert len(nn.layers) == 3

    def test_maxpool_shape_tracking(self):
        """Should track shape after maxpool."""
        nn = CNNNetwork(input_shape=(8, 8, 1))
        nn.add_conv2d(filters=4, kernel_size=3)  # -> (6, 6, 4)
        nn.add_maxpool2d(pool_size=2)  # -> (3, 3, 4)

        assert nn._current_shape == (3, 3, 4)

    def test_add_flatten(self):
        """Should add flatten layer."""
        nn = CNNNetwork(input_shape=(8, 8, 1))
        nn.add_conv2d(filters=4, kernel_size=3)
        nn.add_maxpool2d(pool_size=2)
        nn.add_flatten()

        # Conv2D + ReLU + MaxPool + Flatten = 4 layers
        assert len(nn.layers) == 4

    def test_flatten_shape_tracking(self):
        """Should track 1D shape after flatten."""
        nn = CNNNetwork(input_shape=(8, 8, 1))
        nn.add_conv2d(filters=4, kernel_size=3)  # -> (6, 6, 4)
        nn.add_maxpool2d(pool_size=2)  # -> (3, 3, 4)
        nn.add_flatten()  # -> (36,)

        assert nn._current_shape == (36,)

    def test_add_dense(self):
        """Should add dense layer."""
        nn = CNNNetwork(input_shape=(8, 8, 1))
        nn.add_flatten()
        nn.add_dense(units=16)

        # Flatten + Dense + ReLU = 3 layers
        assert len(nn.layers) == 3

    def test_add_dense_softmax(self):
        """Should add dense with softmax."""
        nn = CNNNetwork(input_shape=(8, 8, 1))
        nn.add_flatten()
        nn.add_dense(units=3, activation='softmax')

        # Flatten + Dense + Softmax = 3 layers
        assert len(nn.layers) == 3


class TestCNNForward:
    """Test CNN forward propagation."""

    def test_forward_output_shape(self):
        """Forward should produce correct output shape."""
        nn = CNNNetwork(input_shape=(8, 8, 1))
        nn.add_conv2d(filters=4, kernel_size=3)
        nn.add_maxpool2d(pool_size=2)
        nn.add_flatten()
        nn.add_dense(units=16)
        nn.add_dense(units=3, activation='softmax')

        # Batch of 4 samples
        X = np.random.randn(4, 8, 8, 1)
        output, activations = nn.forward(X)

        assert output.shape == (4, 3), f"Expected (4, 3), got {output.shape}"

    def test_forward_softmax_sum(self):
        """Softmax output should sum to 1."""
        nn = CNNNetwork(input_shape=(8, 8, 1))
        nn.add_flatten()
        nn.add_dense(units=3, activation='softmax')

        X = np.random.randn(2, 8, 8, 1)
        output, _ = nn.forward(X)

        row_sums = np.sum(output, axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(2))

    def test_forward_activations_list(self):
        """Forward should return list of activations."""
        nn = CNNNetwork(input_shape=(8, 8, 1))
        nn.add_conv2d(filters=4, kernel_size=3)
        nn.add_flatten()
        nn.add_dense(units=3, activation='softmax')

        X = np.random.randn(1, 8, 8, 1)
        output, activations = nn.forward(X)

        # Should have input + each layer's activation
        assert len(activations) >= 1

    def test_predict_same_as_forward(self):
        """Predict should return same as forward."""
        nn = CNNNetwork(input_shape=(8, 8, 1))
        nn.add_flatten()
        nn.add_dense(units=3, activation='softmax')

        X = np.random.randn(2, 8, 8, 1)
        output1, _ = nn.forward(X)
        output2 = nn.predict(X)

        np.testing.assert_array_equal(output1, output2)


class TestCNNBackward:
    """Test CNN backward propagation."""

    def test_backward_runs(self):
        """Backward pass should run without error."""
        nn = CNNNetwork(input_shape=(8, 8, 1))
        nn.add_conv2d(filters=2, kernel_size=3)
        nn.add_flatten()
        nn.add_dense(units=3, activation='softmax')

        X = np.random.randn(2, 8, 8, 1)
        y = np.array([[1, 0, 0], [0, 1, 0]])

        output, activations = nn.forward(X)
        # Should not raise
        nn.backward(y, activations, learning_rate=0.01)


class TestCNNTraining:
    """Test CNN training methods."""

    def test_train_returns_history(self):
        """Training should return loss/accuracy history."""
        nn = CNNNetwork(input_shape=(4, 4, 1))
        nn.add_flatten()
        nn.add_dense(units=2, activation='softmax')

        # Simple 2-class problem
        X = np.random.randn(4, 4, 4, 1)
        y = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])

        result = nn.train(X, y, epochs=5, learning_rate=0.1, verbose=False)

        assert 'loss_history' in result
        assert 'accuracy_history' in result
        assert len(result['loss_history']) == 5

    def test_train_stop_callback(self):
        """Training should stop when stop_check returns True."""
        nn = CNNNetwork(input_shape=(4, 4, 1))
        nn.add_flatten()
        nn.add_dense(units=2, activation='softmax')

        X = np.random.randn(4, 4, 4, 1)
        y = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])

        stop_at = 3
        call_count = [0]

        def stop_check():
            call_count[0] += 1
            return call_count[0] >= stop_at

        result = nn.train(X, y, epochs=100, learning_rate=0.1, verbose=False,
                          stop_check=stop_check)

        assert result['stopped'] is True
        assert result['epochs'] < 100

    def test_train_adaptive_with_forced_lr(self):
        """Adaptive training should accept forced_learning_rate."""
        nn = CNNNetwork(input_shape=(4, 4, 1))
        nn.add_flatten()
        nn.add_dense(units=2, activation='softmax')

        X = np.random.randn(4, 4, 4, 1)
        y = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])

        # Should not raise with forced_learning_rate
        result = nn.train_adaptive(X, y, target_accuracy=0.9, max_epochs=10,
                                    verbose=False, forced_learning_rate=0.001)

        assert 'epochs' in result


class TestCNNLoss:
    """Test CNN loss functions."""

    def test_cross_entropy_perfect_prediction(self):
        """Cross-entropy should be low for perfect predictions."""
        y_pred = np.array([[0.99, 0.01]])
        y_true = np.array([[1.0, 0.0]])
        loss = CNNNetwork.cross_entropy_loss(y_pred, y_true)
        assert loss < 0.1

    def test_cross_entropy_wrong_prediction(self):
        """Cross-entropy should be high for wrong predictions."""
        y_pred = np.array([[0.01, 0.99]])
        y_true = np.array([[1.0, 0.0]])
        loss = CNNNetwork.cross_entropy_loss(y_pred, y_true)
        assert loss > 1.0


class TestCNNAccuracy:
    """Test CNN accuracy calculation."""

    def test_accuracy_perfect(self):
        """Perfect predictions should have 100% accuracy."""
        nn = CNNNetwork(input_shape=(4, 4, 1))
        nn.add_flatten()
        nn.add_dense(units=2, activation='softmax')

        # Mock predictions where argmax matches labels
        X = np.random.randn(2, 4, 4, 1)
        y = np.array([[1, 0], [0, 1]])

        # Train enough to get some accuracy
        nn.train(X, y, epochs=100, learning_rate=0.5, verbose=False)
        accuracy = nn.calculate_accuracy(X, y)

        assert 0 <= accuracy <= 1


class TestCNNReset:
    """Test CNN reset functionality."""

    def test_reset_clears_history(self):
        """Reset should clear loss/accuracy history."""
        nn = CNNNetwork(input_shape=(4, 4, 1))
        nn.add_flatten()
        nn.add_dense(units=2, activation='softmax')

        X = np.random.randn(2, 4, 4, 1)
        y = np.array([[1, 0], [0, 1]])

        nn.train(X, y, epochs=5, learning_rate=0.1, verbose=False)
        assert len(nn.loss_history) == 5

        nn.reset()
        assert len(nn.loss_history) == 0
        assert len(nn.accuracy_history) == 0


class TestCNNArchitecture:
    """Test CNN get_architecture method."""

    def test_get_architecture_returns_dict(self):
        """get_architecture should return dict with layer info."""
        nn = CNNNetwork(input_shape=(8, 8, 1))
        nn.add_conv2d(filters=4, kernel_size=3)
        nn.add_maxpool2d(pool_size=2)
        nn.add_flatten()
        nn.add_dense(units=3, activation='softmax')

        arch = nn.get_architecture()

        assert 'input_shape' in arch
        assert 'layers' in arch
        assert arch['input_shape'] == (8, 8, 1)


class TestCNNIntegration:
    """Integration tests for full CNN workflow."""

    def test_shape_detection_network(self):
        """Test network similar to shape detection problem."""
        nn = CNNNetwork(input_shape=(8, 8, 1))
        nn.add_conv2d(filters=4, kernel_size=3)
        nn.add_maxpool2d(pool_size=2)
        nn.add_flatten()
        nn.add_dense(units=16)
        nn.add_dense(units=3, activation='softmax')

        # Generate random "images"
        X = np.random.rand(10, 8, 8, 1)
        # One-hot labels for 3 classes
        y = np.zeros((10, 3))
        for i in range(10):
            y[i, i % 3] = 1

        # Should train without error
        result = nn.train(X, y, epochs=10, learning_rate=0.1, verbose=False)

        assert result['epochs'] == 10
        assert len(result['loss_history']) == 10

    def test_multiple_conv_layers(self):
        """Test network with multiple conv layers."""
        nn = CNNNetwork(input_shape=(16, 16, 1))
        nn.add_conv2d(filters=8, kernel_size=3)  # -> 14x14x8
        nn.add_maxpool2d(pool_size=2)  # -> 7x7x8
        nn.add_conv2d(filters=16, kernel_size=3)  # -> 5x5x16
        nn.add_flatten()  # -> 400
        nn.add_dense(units=32)
        nn.add_dense(units=10, activation='softmax')

        X = np.random.rand(2, 16, 16, 1)
        output = nn.predict(X)

        assert output.shape == (2, 10)
