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
# Problem 3: PWM Control Mapping
# -----------------------------------------------------------------------------

class PWMControlProblem(Problem):
    """Learn non-linear position to PWM duty cycle mapping."""

    @property
    def info(self) -> ProblemInfo:
        return ProblemInfo(
            id='pwm_control',
            name='PWM Control',
            description='Map position input to PWM duty cycle for motor control. '
                       'Learn smooth non-linear response curves.',
            category='regression',
            default_architecture=[1, 8, 4, 1],
            input_labels=['Position'],
            output_labels=['PWM Duty'],
            output_activation='sigmoid',
            embedded_context='DC motor speed control, servo positioning, LED brightness dimming, '
                           'fan speed control in thermal management.'
        )

    def generate_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Generate non-linear position to PWM mapping data."""
        n_samples = 100
        X = np.linspace(0, 1, n_samples).reshape(-1, 1)

        # S-curve response (common in motor control)
        # Slow start, fast middle, slow end
        y = 1 / (1 + np.exp(-10 * (X - 0.5)))

        return X, y

    def generate_sample(self, inputs: list[float] | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Generate sample from position slider."""
        if inputs is None:
            inputs = [np.random.rand()]
        X = np.array([[inputs[0]]], dtype=float)
        y = 1 / (1 + np.exp(-10 * (inputs[0] - 0.5)))
        return X, np.array([[y]])


# -----------------------------------------------------------------------------
# Problem 4: Anomaly Detection
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
# Problem Registry
# -----------------------------------------------------------------------------

PROBLEMS: dict[str, Problem] = {
    'xor': XORProblem(),
    'sensor_fusion': SensorFusionProblem(),
    'pwm_control': PWMControlProblem(),
    'anomaly': AnomalyDetectionProblem(),
    'gesture': GestureClassificationProblem(),
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
