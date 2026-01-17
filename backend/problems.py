"""
Embedded NN Learning Lab - Problem Definitions

This module defines various embedded systems problems that can be solved
with neural networks, serving as an interactive learning tool.
"""

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ProblemInfo:
    """Information about a problem for the frontend."""
    id: str
    name: str
    description: str
    category: str  # 'binary', 'regression', 'multi-class'
    default_architecture: list[int]
    input_labels: list[str]
    output_labels: list[str]
    output_activation: str  # 'sigmoid' or 'softmax'
    embedded_context: str  # Why this is relevant to embedded systems
    network_type: str = 'dense'  # 'dense' or 'cnn'
    input_shape: tuple[int, ...] | None = None  # For CNN: (height, width, channels)


class Problem(ABC):
    """Base class for all problems."""

    @property
    @abstractmethod
    def info(self) -> ProblemInfo:
        """Return problem metadata."""
        pass

    @abstractmethod
    def generate_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Generate full training dataset."""
        pass

    @abstractmethod
    def generate_sample(self, inputs: list[float] | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Generate a single sample (for interactive mode)."""
        pass


# -----------------------------------------------------------------------------
# Problem 1: XOR Gate (5-bit parity)
# -----------------------------------------------------------------------------

class XORProblem(Problem):
    """5-bit XOR (parity) classification - the original problem."""

    @property
    def info(self) -> ProblemInfo:
        return ProblemInfo(
            id='xor',
            name='XOR Gate (5-bit)',
            description='Classify if the number of HIGH inputs is odd (1) or even (0). '
                       'This is the classic XOR problem extended to 5 bits.',
            category='binary',
            default_architecture=[5, 12, 8, 4, 1],
            input_labels=['GPIO 0', 'GPIO 1', 'GPIO 2', 'GPIO 3', 'GPIO 4'],
            output_labels=['LED'],
            output_activation='sigmoid',
            embedded_context='Digital logic gates, parity checking in serial communications, '
                           'error detection in data transmission.'
        )

    def generate_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Generate all 32 combinations for 5-bit XOR."""
        n_samples = 32  # 2^5
        X = []
        y = []
        for i in range(n_samples):
            inputs = [(i >> j) & 1 for j in range(5)]
            label = sum(inputs) % 2
            X.append(inputs)
            y.append([label])
        return np.array(X, dtype=float), np.array(y, dtype=float)

    def generate_sample(self, inputs: list[float] | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Generate sample from button inputs or random."""
        if inputs is None:
            inputs = [float(np.random.randint(0, 2)) for _ in range(5)]
        X = np.array([inputs], dtype=float)
        y = np.array([[sum(int(x) for x in inputs) % 2]], dtype=float)
        return X, y


# -----------------------------------------------------------------------------
# Problem 2: Sensor Fusion (Temperature + Humidity -> Comfort)
# -----------------------------------------------------------------------------

class SensorFusionProblem(Problem):
    """Combine sensor readings to predict comfort level."""

    @property
    def info(self) -> ProblemInfo:
        return ProblemInfo(
            id='sensor_fusion',
            name='Sensor Fusion',
            description='Combine temperature and humidity readings to predict a comfort index. '
                       'Learn non-linear relationships between sensor inputs.',
            category='regression',
            default_architecture=[2, 8, 4, 1],
            input_labels=['Temperature', 'Humidity'],
            output_labels=['Comfort'],
            output_activation='sigmoid',
            embedded_context='HVAC control systems, smart thermostats, wearable health monitors, '
                           'greenhouse automation.'
        )

    def generate_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Generate comfort data based on temperature and humidity."""
        n_samples = 200
        X = np.random.rand(n_samples, 2)  # temp, humidity in [0, 1]

        # Comfort function: optimal around temp=0.5, humidity=0.4
        # Falls off quadratically from optimal
        temp = X[:, 0]
        humidity = X[:, 1]
        comfort = 1.0 - 2.0 * ((temp - 0.5) ** 2 + (humidity - 0.4) ** 2)
        comfort = np.clip(comfort, 0, 1)

        return X, comfort.reshape(-1, 1)

    def generate_sample(self, inputs: list[float] | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Generate sample from slider inputs."""
        if inputs is None:
            inputs = [np.random.rand(), np.random.rand()]
        X = np.array([inputs[:2]], dtype=float)
        temp, humidity = inputs[0], inputs[1]
        comfort = 1.0 - 2.0 * ((temp - 0.5) ** 2 + (humidity - 0.4) ** 2)
        comfort = np.clip(comfort, 0, 1)
        return X, np.array([[comfort]])


# -----------------------------------------------------------------------------
# Problem 3: Anomaly Detection
# -----------------------------------------------------------------------------

class AnomalyDetectionProblem(Problem):
    """Detect anomalous sensor patterns."""

    @property
    def info(self) -> ProblemInfo:
        return ProblemInfo(
            id='anomaly',
            name='Anomaly Detection',
            description='Detect abnormal patterns in sensor readings. '
                       'Training uses normal operation patterns vs injected faults.',
            category='binary',
            default_architecture=[3, 8, 4, 1],
            input_labels=['Voltage', 'Current', 'Temperature'],
            output_labels=['Anomaly'],
            output_activation='sigmoid',
            embedded_context='Predictive maintenance, fault detection in industrial systems, '
                           'battery health monitoring, power supply protection.'
        )

    def generate_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Generate normal and anomalous sensor patterns."""
        n_normal = 150
        n_anomaly = 50

        # Normal patterns: values cluster around center with correlation
        # (voltage ~ 0.5, current proportional, temp correlated)
        normal_voltage = 0.5 + 0.1 * np.random.randn(n_normal)
        normal_current = 0.4 + 0.2 * normal_voltage + 0.05 * np.random.randn(n_normal)
        normal_temp = 0.3 + 0.1 * normal_current + 0.05 * np.random.randn(n_normal)
        normal = np.stack([normal_voltage, normal_current, normal_temp], axis=1)
        normal = np.clip(normal, 0, 1)

        # Anomalies: break the correlations or extreme values
        anomaly_voltage = np.random.rand(n_anomaly)  # Random, not centered
        anomaly_current = np.random.rand(n_anomaly)  # Uncorrelated
        anomaly_temp = 0.7 + 0.3 * np.random.rand(n_anomaly)  # High temps
        anomaly = np.stack([anomaly_voltage, anomaly_current, anomaly_temp], axis=1)
        anomaly = np.clip(anomaly, 0, 1)

        X = np.vstack([normal, anomaly])
        y = np.vstack([np.zeros((n_normal, 1)), np.ones((n_anomaly, 1))])

        # Shuffle
        indices = np.random.permutation(len(X))
        return X[indices], y[indices]

    def generate_sample(self, inputs: list[float] | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Generate sample from sensor inputs."""
        if inputs is None:
            # Random sample, could be normal or anomaly
            inputs = [np.random.rand() for _ in range(3)]
        X = np.array([inputs[:3]], dtype=float)

        # Heuristic for expected label (matches training data pattern)
        v, c, t = inputs[0], inputs[1], inputs[2]
        expected_c = 0.4 + 0.2 * v
        expected_t = 0.3 + 0.1 * c
        deviation = abs(c - expected_c) + abs(t - expected_t) + (1 if t > 0.7 else 0)
        is_anomaly = 1 if deviation > 0.3 else 0

        return X, np.array([[is_anomaly]])


# -----------------------------------------------------------------------------
# Problem 5: Gesture Classification (Multi-class)
# -----------------------------------------------------------------------------

class GestureClassificationProblem(Problem):
    """Classify gesture patterns from accelerometer data."""

    @property
    def info(self) -> ProblemInfo:
        return ProblemInfo(
            id='gesture',
            name='Gesture Classification',
            description='Classify gestures from accelerometer data patterns. '
                       'Uses softmax for multi-class output (tap, swipe, shake).',
            category='multi-class',
            default_architecture=[8, 12, 8, 3],
            input_labels=['t0', 't1', 't2', 't3', 't4', 't5', 't6', 't7'],
            output_labels=['Tap', 'Swipe', 'Shake'],
            output_activation='softmax',
            embedded_context='Wearable devices, smart watches, gesture-controlled IoT, '
                           'motion-based user interfaces.'
        )

    def generate_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Generate gesture patterns: tap, swipe, shake."""
        n_per_class = 100
        samples = []
        labels = []

        # Tap: sharp spike then flat
        for _ in range(n_per_class):
            spike_pos = np.random.randint(2, 5)
            pattern = np.zeros(8)
            pattern[spike_pos] = 0.8 + 0.2 * np.random.rand()
            pattern += 0.05 * np.random.randn(8)
            samples.append(np.clip(pattern, 0, 1))
            labels.append([1, 0, 0])  # One-hot: tap

        # Swipe: gradual increase or decrease
        for _ in range(n_per_class):
            direction = np.random.choice([-1, 1])
            start = 0.3 if direction == 1 else 0.7
            pattern = np.linspace(start, start + 0.5 * direction, 8)
            pattern += 0.05 * np.random.randn(8)
            samples.append(np.clip(pattern, 0, 1))
            labels.append([0, 1, 0])  # One-hot: swipe

        # Shake: oscillating pattern
        for _ in range(n_per_class):
            freq = np.random.choice([2, 3])
            t = np.linspace(0, freq * np.pi, 8)
            pattern = 0.5 + 0.3 * np.sin(t)
            pattern += 0.05 * np.random.randn(8)
            samples.append(np.clip(pattern, 0, 1))
            labels.append([0, 0, 1])  # One-hot: shake

        X = np.array(samples)
        y = np.array(labels)

        # Shuffle
        indices = np.random.permutation(len(X))
        return X[indices], y[indices]

    def generate_sample(self, inputs: list[float] | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Generate sample from pattern inputs or random gesture."""
        if inputs is None:
            # Generate random gesture
            gesture_type = np.random.choice(['tap', 'swipe', 'shake'])
            if gesture_type == 'tap':
                spike_pos = np.random.randint(2, 5)
                inputs = [0.0] * 8
                inputs[spike_pos] = 0.9
            elif gesture_type == 'swipe':
                inputs = list(np.linspace(0.2, 0.8, 8))
            else:  # shake
                t = np.linspace(0, 2 * np.pi, 8)
                inputs = list(0.5 + 0.3 * np.sin(t))
        else:
            gesture_type = self._classify_pattern(inputs)

        X = np.array([inputs[:8]], dtype=float)

        # Determine expected class based on pattern analysis
        gesture_type = self._classify_pattern(inputs)
        if gesture_type == 'tap':
            y = np.array([[1, 0, 0]])
        elif gesture_type == 'swipe':
            y = np.array([[0, 1, 0]])
        else:
            y = np.array([[0, 0, 1]])

        return X, y

    def _classify_pattern(self, inputs: list[float]) -> str:
        """Heuristic classification of gesture pattern."""
        arr = np.array(inputs[:8])

        # Check for spike (tap)
        max_val = np.max(arr)
        if max_val > 0.7 and np.sum(arr > 0.5) <= 2:
            return 'tap'

        # Check for trend (swipe)
        diff = arr[-1] - arr[0]
        if abs(diff) > 0.3:
            return 'swipe'

        # Default to shake (oscillating)
        return 'shake'


# -----------------------------------------------------------------------------
# Problem 6: Shape Detection (CNN)
# -----------------------------------------------------------------------------

class ShapeDetectionProblem(Problem):
    """Classify shapes on 8x8 grid using CNN."""

    @property
    def info(self) -> ProblemInfo:
        return ProblemInfo(
            id='shape_detection',
            name='Shape Detection (CNN)',
            description='Classify 8×8 pixel images of circles, squares, and triangles. '
                       'Uses convolutional neural network for spatial feature learning.',
            category='multi-class',
            default_architecture=[64, 32, 16, 3],  # Fallback for dense, actual CNN defined in app
            input_labels=[f'p{i}' for i in range(64)],  # 8x8 = 64 pixels
            output_labels=['Circle', 'Square', 'Triangle'],
            output_activation='softmax',
            embedded_context='Computer vision for embedded systems, gesture recognition, '
                           'object detection, quality control inspection.',
            network_type='cnn',
            input_shape=(8, 8, 1)
        )

    def generate_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Generate 300 samples (100 per shape class)."""
        samples = []
        labels = []

        # Generate circles
        for _ in range(100):
            grid = self._generate_circle()
            samples.append(grid)
            labels.append([1, 0, 0])

        # Generate squares
        for _ in range(100):
            grid = self._generate_square()
            samples.append(grid)
            labels.append([0, 1, 0])

        # Generate triangles
        for _ in range(100):
            grid = self._generate_triangle()
            samples.append(grid)
            labels.append([0, 0, 1])

        X = np.array(samples).reshape(-1, 8, 8, 1)
        y = np.array(labels)

        # Shuffle
        indices = np.random.permutation(len(X))
        return X[indices], y[indices]

    def _generate_circle(self) -> np.ndarray:
        """Generate filled circle on 8x8 grid."""
        grid = np.zeros((8, 8))
        center = 3.5 + np.random.uniform(-0.3, 0.3)
        radius = 2.5 + np.random.uniform(-0.5, 0.5)

        for i in range(8):
            for j in range(8):
                dist = np.sqrt((i - center) ** 2 + (j - center) ** 2)
                if dist < radius:
                    grid[i, j] = 0.9 + np.random.uniform(-0.1, 0.1)

        # Add noise
        grid += np.random.uniform(-0.05, 0.05, (8, 8))
        return np.clip(grid, 0, 1)

    def _generate_square(self) -> np.ndarray:
        """Generate filled square on 8x8 grid."""
        grid = np.zeros((8, 8))
        margin = np.random.randint(1, 3)
        size_var = np.random.randint(0, 2)

        top = margin
        bottom = 8 - margin - size_var
        left = margin
        right = 8 - margin - size_var

        grid[top:bottom, left:right] = 0.9 + np.random.uniform(-0.1, 0.1)

        # Add noise
        grid += np.random.uniform(-0.05, 0.05, (8, 8))
        return np.clip(grid, 0, 1)

    def _generate_triangle(self) -> np.ndarray:
        """Generate filled triangle on 8x8 grid."""
        grid = np.zeros((8, 8))
        flip = np.random.choice([True, False])  # Upward or downward

        for row in range(8):
            if flip:
                # Upward triangle (apex at top)
                r = 7 - row
            else:
                # Downward triangle (apex at bottom)
                r = row

            # Width increases with row from apex
            half_width = r // 2 + 1
            center = 3.5
            left = max(0, int(center - half_width + 0.5))
            right = min(8, int(center + half_width + 0.5))

            if right > left:
                grid[row, left:right] = 0.9 + np.random.uniform(-0.1, 0.1)

        # Add noise
        grid += np.random.uniform(-0.05, 0.05, (8, 8))
        return np.clip(grid, 0, 1)

    def generate_sample(self, inputs: list[float] | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Generate sample from grid inputs or random shape."""
        if inputs is None:
            # Generate random shape
            shape_type = np.random.choice(['circle', 'square', 'triangle'])
            if shape_type == 'circle':
                grid = self._generate_circle()
                y = np.array([[1, 0, 0]])
            elif shape_type == 'square':
                grid = self._generate_square()
                y = np.array([[0, 1, 0]])
            else:
                grid = self._generate_triangle()
                y = np.array([[0, 0, 1]])
            X = grid.reshape(1, 8, 8, 1)
        else:
            # Use provided 64 values
            grid = np.array(inputs[:64]).reshape(8, 8)
            y = self._classify_grid(grid)
            X = grid.reshape(1, 8, 8, 1)

        return X, y

    def _classify_grid(self, grid: np.ndarray) -> np.ndarray:
        """Heuristic classification of drawn shape."""
        # Compute center of mass
        total = np.sum(grid) + 1e-10
        y_coords, x_coords = np.meshgrid(range(8), range(8))
        cx = np.sum(x_coords * grid) / total
        cy = np.sum(y_coords * grid) / total

        # Check for circular pattern (symmetry around center)
        # Compute variance from center
        distances = np.sqrt((x_coords - cx) ** 2 + (y_coords - cy) ** 2)
        active_distances = distances[grid > 0.5]
        if len(active_distances) > 0:
            dist_variance = np.var(active_distances)
        else:
            dist_variance = 10

        # Check for square pattern (rectangular bounding box fill)
        active = grid > 0.5
        if np.any(active):
            rows_active = np.any(active, axis=1)
            cols_active = np.any(active, axis=0)
            bbox_area = np.sum(rows_active) * np.sum(cols_active)
            fill_ratio = np.sum(active) / (bbox_area + 1e-10)
        else:
            fill_ratio = 0

        # Classify based on heuristics
        if dist_variance < 0.8 and fill_ratio > 0.6:
            return np.array([[1, 0, 0]])  # Circle
        elif fill_ratio > 0.8:
            return np.array([[0, 1, 0]])  # Square
        else:
            return np.array([[0, 0, 1]])  # Triangle

    def get_preset_shapes(self) -> dict[str, list[float]]:
        """Get preset shapes for UI buttons."""
        return {
            'circle': self._generate_circle().flatten().tolist(),
            'square': self._generate_square().flatten().tolist(),
            'triangle': self._generate_triangle().flatten().tolist()
        }


# -----------------------------------------------------------------------------
# Problem 7: Digit Recognition (CNN)
# -----------------------------------------------------------------------------

class DigitRecognitionProblem(Problem):
    """Classify handwritten digits 0-9 on 8x8 grid using CNN."""

    # Base digit patterns (8x8 grids) - simple stylized digits
    DIGIT_PATTERNS = {
        0: [
            [0,0,1,1,1,1,0,0],
            [0,1,1,0,0,1,1,0],
            [0,1,1,0,0,1,1,0],
            [0,1,1,0,0,1,1,0],
            [0,1,1,0,0,1,1,0],
            [0,1,1,0,0,1,1,0],
            [0,1,1,0,0,1,1,0],
            [0,0,1,1,1,1,0,0],
        ],
        1: [
            [0,0,0,1,1,0,0,0],
            [0,0,1,1,1,0,0,0],
            [0,0,0,1,1,0,0,0],
            [0,0,0,1,1,0,0,0],
            [0,0,0,1,1,0,0,0],
            [0,0,0,1,1,0,0,0],
            [0,0,0,1,1,0,0,0],
            [0,0,1,1,1,1,0,0],
        ],
        2: [
            [0,0,1,1,1,1,0,0],
            [0,1,1,0,0,1,1,0],
            [0,0,0,0,0,1,1,0],
            [0,0,0,0,1,1,0,0],
            [0,0,0,1,1,0,0,0],
            [0,0,1,1,0,0,0,0],
            [0,1,1,0,0,0,0,0],
            [0,1,1,1,1,1,1,0],
        ],
        3: [
            [0,0,1,1,1,1,0,0],
            [0,1,1,0,0,1,1,0],
            [0,0,0,0,0,1,1,0],
            [0,0,0,1,1,1,0,0],
            [0,0,0,0,0,1,1,0],
            [0,0,0,0,0,1,1,0],
            [0,1,1,0,0,1,1,0],
            [0,0,1,1,1,1,0,0],
        ],
        4: [
            [0,0,0,0,1,1,0,0],
            [0,0,0,1,1,1,0,0],
            [0,0,1,1,1,1,0,0],
            [0,1,1,0,1,1,0,0],
            [0,1,1,1,1,1,1,0],
            [0,0,0,0,1,1,0,0],
            [0,0,0,0,1,1,0,0],
            [0,0,0,0,1,1,0,0],
        ],
        5: [
            [0,1,1,1,1,1,1,0],
            [0,1,1,0,0,0,0,0],
            [0,1,1,0,0,0,0,0],
            [0,1,1,1,1,1,0,0],
            [0,0,0,0,0,1,1,0],
            [0,0,0,0,0,1,1,0],
            [0,1,1,0,0,1,1,0],
            [0,0,1,1,1,1,0,0],
        ],
        6: [
            [0,0,1,1,1,1,0,0],
            [0,1,1,0,0,0,0,0],
            [0,1,1,0,0,0,0,0],
            [0,1,1,1,1,1,0,0],
            [0,1,1,0,0,1,1,0],
            [0,1,1,0,0,1,1,0],
            [0,1,1,0,0,1,1,0],
            [0,0,1,1,1,1,0,0],
        ],
        7: [
            [0,1,1,1,1,1,1,0],
            [0,0,0,0,0,1,1,0],
            [0,0,0,0,1,1,0,0],
            [0,0,0,0,1,1,0,0],
            [0,0,0,1,1,0,0,0],
            [0,0,0,1,1,0,0,0],
            [0,0,0,1,1,0,0,0],
            [0,0,0,1,1,0,0,0],
        ],
        8: [
            [0,0,1,1,1,1,0,0],
            [0,1,1,0,0,1,1,0],
            [0,1,1,0,0,1,1,0],
            [0,0,1,1,1,1,0,0],
            [0,1,1,0,0,1,1,0],
            [0,1,1,0,0,1,1,0],
            [0,1,1,0,0,1,1,0],
            [0,0,1,1,1,1,0,0],
        ],
        9: [
            [0,0,1,1,1,1,0,0],
            [0,1,1,0,0,1,1,0],
            [0,1,1,0,0,1,1,0],
            [0,0,1,1,1,1,1,0],
            [0,0,0,0,0,1,1,0],
            [0,0,0,0,0,1,1,0],
            [0,0,0,0,1,1,0,0],
            [0,0,1,1,1,0,0,0],
        ],
    }

    @property
    def info(self) -> ProblemInfo:
        return ProblemInfo(
            id='digit_recognition',
            name='Digit Recognition (CNN)',
            description='Classify 8×8 pixel images of handwritten digits 0-9. '
                       'Uses convolutional neural network for pattern recognition.',
            category='multi-class',
            default_architecture=[64, 32, 16, 10],
            input_labels=[f'p{i}' for i in range(64)],
            output_labels=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
            output_activation='softmax',
            embedded_context='OCR for embedded displays, keypad digit recognition, '
                           'meter reading, industrial label scanning.',
            network_type='cnn',
            input_shape=(8, 8, 1)
        )

    def generate_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Generate 500 samples (50 per digit class)."""
        samples = []
        labels = []

        for digit in range(10):
            for _ in range(50):
                grid = self._generate_digit(digit)
                samples.append(grid)
                label = [0] * 10
                label[digit] = 1
                labels.append(label)

        X = np.array(samples).reshape(-1, 8, 8, 1)
        y = np.array(labels)

        indices = np.random.permutation(len(X))
        return X[indices], y[indices]

    def _generate_digit(self, digit: int) -> np.ndarray:
        """Generate a digit with variations."""
        base = np.array(self.DIGIT_PATTERNS[digit], dtype=float)

        # Apply random variations
        # 1. Small random shift (0 or 1 pixel)
        shift_x = np.random.choice([-1, 0, 0, 1])
        shift_y = np.random.choice([-1, 0, 0, 1])
        if shift_x != 0 or shift_y != 0:
            base = np.roll(base, shift_x, axis=1)
            base = np.roll(base, shift_y, axis=0)
            # Clear wrapped edges
            if shift_x > 0:
                base[:, :shift_x] = 0
            elif shift_x < 0:
                base[:, shift_x:] = 0
            if shift_y > 0:
                base[:shift_y, :] = 0
            elif shift_y < 0:
                base[shift_y:, :] = 0

        # 2. Intensity variation
        base = base * (0.8 + np.random.uniform(0, 0.2))

        # 3. Per-pixel noise
        base += np.random.uniform(-0.1, 0.1, (8, 8))

        # 4. Random pixel dropout (simulate imperfect writing)
        dropout_mask = np.random.random((8, 8)) > 0.05
        base = base * dropout_mask

        return np.clip(base, 0, 1)

    def generate_sample(self, inputs: list[float] | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Generate sample from grid inputs or random digit."""
        if inputs is None:
            # Generate random digit
            digit = np.random.randint(0, 10)
            grid = self._generate_digit(digit)
            y = np.zeros((1, 10))
            y[0, digit] = 1
            X = grid.reshape(1, 8, 8, 1)
        else:
            # Use provided 64 values (8x8 grid)
            if isinstance(inputs, list) and len(inputs) > 0:
                if isinstance(inputs[0], list):
                    # 2D grid input
                    grid = np.array(inputs, dtype=float)
                else:
                    # Flat input
                    grid = np.array(inputs[:64]).reshape(8, 8)
            else:
                grid = np.zeros((8, 8))

            # Heuristic classification based on pattern matching
            y = self._classify_grid(grid)
            X = grid.reshape(1, 8, 8, 1)

        return X, y

    def _classify_grid(self, grid: np.ndarray) -> np.ndarray:
        """Heuristic classification of drawn digit using pattern matching."""
        best_match = 0
        best_score = -np.inf

        for digit, pattern in self.DIGIT_PATTERNS.items():
            pattern_arr = np.array(pattern, dtype=float)
            # Normalize both to compare
            grid_norm = grid / (np.max(grid) + 1e-10)
            score = np.sum(grid_norm * pattern_arr) - 0.5 * np.sum((1 - grid_norm) * pattern_arr)
            if score > best_score:
                best_score = score
                best_match = digit

        y = np.zeros((1, 10))
        y[0, best_match] = 1
        return y

    def prepare_input(self, inputs: list) -> tuple[np.ndarray, np.ndarray]:
        """Prepare input for prediction."""
        if len(inputs) == 1 and isinstance(inputs[0], str):
            digit = int(inputs[0])
            grid = self._generate_digit(digit)
            y = np.zeros((1, 10))
            y[0, digit] = 1
            X = grid.reshape(1, 8, 8, 1)
        else:
            grid = np.array(inputs[:64]).reshape(8, 8)
            y = np.zeros((1, 10))  # Unknown expected
            X = grid.reshape(1, 8, 8, 1)
        return X, y

    def get_example_patterns(self) -> dict:
        """Get example digit patterns."""
        return {str(d): np.array(self.DIGIT_PATTERNS[d]).flatten().tolist()
                for d in range(10)}


# -----------------------------------------------------------------------------
# Problem 8: Arrow Direction (CNN)
# -----------------------------------------------------------------------------

class ArrowDirectionProblem(Problem):
    """Classify arrow directions on 8x8 grid using CNN."""

    # Arrow patterns (8x8 grids)
    ARROW_PATTERNS = {
        'up': [
            [0,0,0,1,1,0,0,0],
            [0,0,1,1,1,1,0,0],
            [0,1,1,1,1,1,1,0],
            [1,1,0,1,1,0,1,1],
            [0,0,0,1,1,0,0,0],
            [0,0,0,1,1,0,0,0],
            [0,0,0,1,1,0,0,0],
            [0,0,0,1,1,0,0,0],
        ],
        'down': [
            [0,0,0,1,1,0,0,0],
            [0,0,0,1,1,0,0,0],
            [0,0,0,1,1,0,0,0],
            [0,0,0,1,1,0,0,0],
            [1,1,0,1,1,0,1,1],
            [0,1,1,1,1,1,1,0],
            [0,0,1,1,1,1,0,0],
            [0,0,0,1,1,0,0,0],
        ],
        'left': [
            [0,0,0,1,0,0,0,0],
            [0,0,1,1,0,0,0,0],
            [0,1,1,1,1,1,1,1],
            [1,1,1,1,1,1,1,1],
            [1,1,1,1,1,1,1,1],
            [0,1,1,1,1,1,1,1],
            [0,0,1,1,0,0,0,0],
            [0,0,0,1,0,0,0,0],
        ],
        'right': [
            [0,0,0,0,1,0,0,0],
            [0,0,0,0,1,1,0,0],
            [1,1,1,1,1,1,1,0],
            [1,1,1,1,1,1,1,1],
            [1,1,1,1,1,1,1,1],
            [1,1,1,1,1,1,1,0],
            [0,0,0,0,1,1,0,0],
            [0,0,0,0,1,0,0,0],
        ],
    }

    @property
    def info(self) -> ProblemInfo:
        return ProblemInfo(
            id='arrow_direction',
            name='Arrow Direction (CNN)',
            description='Classify 8×8 pixel images of arrows pointing up, down, left, or right. '
                       'Simple CNN problem for quick training.',
            category='multi-class',
            default_architecture=[64, 32, 16, 4],
            input_labels=[f'p{i}' for i in range(64)],
            output_labels=['Up', 'Down', 'Left', 'Right'],
            output_activation='softmax',
            embedded_context='Gesture recognition, UI navigation, directional input detection.',
            network_type='cnn',
            input_shape=(8, 8, 1)
        )

    def generate_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Generate 200 samples (50 per direction)."""
        samples = []
        labels = []
        directions = ['up', 'down', 'left', 'right']

        for i, direction in enumerate(directions):
            for _ in range(50):
                grid = self._generate_arrow(direction)
                samples.append(grid)
                label = [0, 0, 0, 0]
                label[i] = 1
                labels.append(label)

        X = np.array(samples).reshape(-1, 8, 8, 1)
        y = np.array(labels)

        indices = np.random.permutation(len(X))
        return X[indices], y[indices]

    def _generate_arrow(self, direction: str) -> np.ndarray:
        """Generate arrow with variations."""
        base = np.array(self.ARROW_PATTERNS[direction], dtype=float)

        # Small random shift
        shift_x = np.random.choice([-1, 0, 0, 1])
        shift_y = np.random.choice([-1, 0, 0, 1])
        if shift_x != 0 or shift_y != 0:
            base = np.roll(base, shift_x, axis=1)
            base = np.roll(base, shift_y, axis=0)
            if shift_x > 0: base[:, :shift_x] = 0
            elif shift_x < 0: base[:, shift_x:] = 0
            if shift_y > 0: base[:shift_y, :] = 0
            elif shift_y < 0: base[shift_y:, :] = 0

        # Intensity variation
        base = base * (0.8 + np.random.uniform(0, 0.2))

        # Per-pixel noise
        base += np.random.uniform(-0.1, 0.1, (8, 8))

        return np.clip(base, 0, 1)

    def generate_sample(self, inputs: list[float] | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Generate sample from grid inputs or random arrow."""
        directions = ['up', 'down', 'left', 'right']
        if inputs is None:
            direction = np.random.choice(directions)
            grid = self._generate_arrow(direction)
            y = np.zeros((1, 4))
            y[0, directions.index(direction)] = 1
            X = grid.reshape(1, 8, 8, 1)
        else:
            if isinstance(inputs, list) and len(inputs) > 0:
                if isinstance(inputs[0], list):
                    grid = np.array(inputs, dtype=float)
                else:
                    grid = np.array(inputs[:64]).reshape(8, 8)
            else:
                grid = np.zeros((8, 8))
            y = self._classify_grid(grid)
            X = grid.reshape(1, 8, 8, 1)
        return X, y

    def _classify_grid(self, grid: np.ndarray) -> np.ndarray:
        """Classify arrow direction using pattern matching."""
        directions = ['up', 'down', 'left', 'right']
        best_match = 0
        best_score = -np.inf

        for i, direction in enumerate(directions):
            pattern = np.array(self.ARROW_PATTERNS[direction], dtype=float)
            grid_norm = grid / (np.max(grid) + 1e-10)
            score = np.sum(grid_norm * pattern)
            if score > best_score:
                best_score = score
                best_match = i

        y = np.zeros((1, 4))
        y[0, best_match] = 1
        return y


# -----------------------------------------------------------------------------
# Problem 9: Color Mixer
# -----------------------------------------------------------------------------

class ColorMixerProblem(Problem):
    """Classify colors from RGB values."""

    @property
    def info(self) -> ProblemInfo:
        return ProblemInfo(
            id='color_mixer',
            name='Color Mixer',
            description='Predict color category from RGB values. '
                       'Learn to classify Red, Green, Blue, Yellow, Cyan, Magenta.',
            category='multi-class',
            default_architecture=[3, 12, 8, 6],
            input_labels=['Red', 'Green', 'Blue'],
            output_labels=['Red', 'Green', 'Blue', 'Yellow', 'Cyan', 'Magenta'],
            output_activation='softmax',
            embedded_context='Color sensing, LED control, display calibration, '
                           'image processing for embedded vision.'
        )

    def generate_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Generate 300 samples (50 per color)."""
        samples = []
        labels = []

        # Color centers (R, G, B)
        colors = {
            0: (1.0, 0.2, 0.2),  # Red
            1: (0.2, 1.0, 0.2),  # Green
            2: (0.2, 0.2, 1.0),  # Blue
            3: (1.0, 1.0, 0.2),  # Yellow (R+G)
            4: (0.2, 1.0, 1.0),  # Cyan (G+B)
            5: (1.0, 0.2, 1.0),  # Magenta (R+B)
        }

        for color_idx, (r, g, b) in colors.items():
            for _ in range(50):
                # Add noise around color center
                noise = np.random.uniform(-0.15, 0.15, 3)
                rgb = np.clip([r + noise[0], g + noise[1], b + noise[2]], 0, 1)
                samples.append(rgb)
                label = [0] * 6
                label[color_idx] = 1
                labels.append(label)

        X = np.array(samples)
        y = np.array(labels)

        indices = np.random.permutation(len(X))
        return X[indices], y[indices]

    def generate_sample(self, inputs: list[float] | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Generate sample from RGB sliders."""
        if inputs is None:
            inputs = [np.random.rand() for _ in range(3)]

        X = np.array([inputs[:3]], dtype=float)
        y = self._classify_color(inputs[:3])
        return X, y

    def _classify_color(self, rgb: list[float]) -> np.ndarray:
        """Classify color based on RGB values."""
        r, g, b = rgb[0], rgb[1], rgb[2]

        # Determine dominant color(s)
        y = np.zeros((1, 6))

        # Primary colors
        if r > 0.6 and g < 0.5 and b < 0.5:
            y[0, 0] = 1  # Red
        elif g > 0.6 and r < 0.5 and b < 0.5:
            y[0, 1] = 1  # Green
        elif b > 0.6 and r < 0.5 and g < 0.5:
            y[0, 2] = 1  # Blue
        # Secondary colors
        elif r > 0.6 and g > 0.6 and b < 0.5:
            y[0, 3] = 1  # Yellow
        elif g > 0.6 and b > 0.6 and r < 0.5:
            y[0, 4] = 1  # Cyan
        elif r > 0.6 and b > 0.6 and g < 0.5:
            y[0, 5] = 1  # Magenta
        else:
            # Default to closest primary
            max_idx = np.argmax([r, g, b])
            y[0, max_idx] = 1

        return y


# -----------------------------------------------------------------------------
# Problem 10: Logic Gates
# -----------------------------------------------------------------------------

class LogicGatesProblem(Problem):
    """Classify logic gates from their complete truth table."""

    # Truth tables: [out(0,0), out(0,1), out(1,0), out(1,1)]
    GATE_TRUTH_TABLES = {
        0: [0, 0, 0, 1],  # AND
        1: [0, 1, 1, 1],  # OR
        2: [0, 1, 1, 0],  # XOR
        3: [1, 1, 1, 0],  # NAND
    }

    @property
    def info(self) -> ProblemInfo:
        return ProblemInfo(
            id='logic_gates',
            name='Logic Gates',
            description='Identify logic gates from their truth table. '
                       'Input is [out(0,0), out(0,1), out(1,0), out(1,1)] - the 4 outputs.',
            category='multi-class',
            default_architecture=[4, 8, 6, 4],
            input_labels=['0,0→', '0,1→', '1,0→', '1,1→'],
            output_labels=['AND', 'OR', 'XOR', 'NAND'],
            output_activation='softmax',
            embedded_context='Digital logic education, circuit testing, '
                           'programmable logic device configuration.'
        )

    def generate_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Generate samples for each gate type."""
        samples = []
        labels = []

        # Generate samples with noise for each gate
        for gate_idx, truth_table in self.GATE_TRUTH_TABLES.items():
            for _ in range(50):  # 50 samples per gate
                # Add noise to truth table values
                noisy_table = [
                    np.clip(float(v) + np.random.uniform(-0.1, 0.1), 0, 1)
                    for v in truth_table
                ]
                samples.append(noisy_table)

                label = [0, 0, 0, 0]
                label[gate_idx] = 1
                labels.append(label)

        X = np.array(samples)
        y = np.array(labels)

        indices = np.random.permutation(len(X))
        return X[indices], y[indices]

    def generate_sample(self, inputs: list[float] | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Generate sample from 4 toggle inputs."""
        if inputs is None:
            # Random gate
            gate_idx = np.random.randint(0, 4)
            inputs = [float(v) for v in self.GATE_TRUTH_TABLES[gate_idx]]

        X = np.array([inputs[:4]], dtype=float)
        y = self._classify_gate(inputs[:4])
        return X, y

    def _classify_gate(self, inputs: list[float]) -> np.ndarray:
        """Classify which gate matches the truth table."""
        # Round inputs to get binary values
        binary = [1 if v > 0.5 else 0 for v in inputs]

        y = np.zeros((1, 4))

        # Find matching gate
        for gate_idx, truth_table in self.GATE_TRUTH_TABLES.items():
            if binary == truth_table:
                y[0, gate_idx] = 1
                return y

        # No exact match - find closest
        best_match = 0
        best_score = -1
        for gate_idx, truth_table in self.GATE_TRUTH_TABLES.items():
            score = sum(1 for a, b in zip(binary, truth_table) if a == b)
            if score > best_score:
                best_score = score
                best_match = gate_idx

        y[0, best_match] = 1
        return y


# -----------------------------------------------------------------------------
# Problem Registry
# -----------------------------------------------------------------------------

PROBLEMS: dict[str, Problem] = {
    'xor': XORProblem(),
    'sensor_fusion': SensorFusionProblem(),
    'anomaly': AnomalyDetectionProblem(),
    'gesture': GestureClassificationProblem(),
    'shape_detection': ShapeDetectionProblem(),
    'digit_recognition': DigitRecognitionProblem(),
    'arrow_direction': ArrowDirectionProblem(),
    'color_mixer': ColorMixerProblem(),
}


def get_problem(problem_id: str) -> Problem:
    """Get a problem by ID."""
    if problem_id not in PROBLEMS:
        raise ValueError(f"Unknown problem: {problem_id}. Available: {list(PROBLEMS.keys())}")
    return PROBLEMS[problem_id]


def list_problems() -> list[dict]:
    """List all available problems with their info."""
    return [
        {
            'id': p.info.id,
            'name': p.info.name,
            'description': p.info.description,
            'category': p.info.category,
            'default_architecture': p.info.default_architecture,
            'input_labels': p.info.input_labels,
            'output_labels': p.info.output_labels,
            'output_activation': p.info.output_activation,
            'embedded_context': p.info.embedded_context,
        }
        for p in PROBLEMS.values()
    ]


# -----------------------------------------------------------------------------
# Demo
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Embedded NN Learning Lab - Problems Demo")
    print("=" * 60)

    for prob_id, prob in PROBLEMS.items():
        info = prob.info
        X, y = prob.generate_data()
        print(f"\n{info.name} ({prob_id})")
        print(f"  Category: {info.category}")
        print(f"  Data shape: X={X.shape}, y={y.shape}")
        print(f"  Architecture: {info.default_architecture}")
        print(f"  Activation: {info.output_activation}")
