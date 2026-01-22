"""
Neural Network Learning Lab - Educational Problem Definitions

This module defines progressive learning problems that teach neural network
concepts through hands-on experimentation.

Problems are organized by learning level:
  Level 1: Single Neuron (AND, OR, NOT, NAND) - Linear separability
  Level 2: Hidden Layers (XOR, XNOR, 5-bit parity) - Why hidden layers matter
  Level 3: Decision Boundaries (blobs, moons, circle, donut, spiral) - Visual 2D
  Level 4: Regression (line, sine, polynomial, surface) - Function approximation
  Level 5: FAILURE CASES - Learn from intentional failures!
  Level 6: Multi-Class (quadrants, blobs, colors, patterns) - Softmax
  Level 7: CNN (shapes, digits) - Convolutional networks
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
    # Educational fields
    difficulty: int  # 1-5 (1=beginner, 5=advanced)
    concept: str  # What this problem teaches
    learning_goal: str  # What the student should learn
    tips: list[str]  # Hints for solving
    network_type: str = 'dense'  # 'dense' or 'cnn'
    input_shape: tuple[int, ...] | None = None  # For CNN: (height, width, channels)
    # Level tracking
    level: int = 1  # Explicit level number (1-7)
    # Failure case fields
    is_failure_case: bool = False  # If True, expected to fail
    failure_reason: str | None = None  # Why it fails
    fix_suggestion: str | None = None  # How to fix it
    # Locked parameters (for failure cases)
    locked_architecture: bool = False  # Cannot change arch
    forced_weight_init: str | None = None  # Force init type
    forced_learning_rate: float | None = None  # Force LR


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


# =============================================================================
# LEVEL 1: Single Neuron Problems
# These problems CAN be solved with a single neuron (no hidden layers)
# =============================================================================

class ANDGateProblem(Problem):
    """AND gate - solvable with a single neuron."""

    @property
    def info(self) -> ProblemInfo:
        return ProblemInfo(
            id='and_gate',
            name='AND Gate',
            description='Output 1 only when BOTH inputs are 1. '
                       'This is linearly separable - a single neuron can solve it!',
            category='binary',
            default_architecture=[2, 1],  # Just input → output!
            input_labels=['Input A', 'Input B'],
            output_labels=['Output'],
            output_activation='sigmoid',
            difficulty=1,
            concept='Linear Separability',
            learning_goal='Understand that some problems can be solved with just one neuron. '
                         'The AND function is linearly separable.',
            tips=[
                'Try with NO hidden layers first [2, 1]',
                'A single line can separate 1s from 0s',
                'Watch how the decision boundary forms'
            ],
            level=1
        )

    def generate_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Generate all 4 combinations for 2-input AND."""
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        y = np.array([[0], [0], [0], [1]], dtype=float)  # Only 1,1 → 1
        return X, y

    def generate_sample(self, inputs: list[float] | None = None) -> tuple[np.ndarray, np.ndarray]:
        if inputs is None:
            inputs = [float(np.random.randint(0, 2)) for _ in range(2)]
        X = np.array([inputs[:2]], dtype=float)
        y = np.array([[1 if inputs[0] > 0.5 and inputs[1] > 0.5 else 0]], dtype=float)
        return X, y


class ORGateProblem(Problem):
    """OR gate - solvable with a single neuron."""

    @property
    def info(self) -> ProblemInfo:
        return ProblemInfo(
            id='or_gate',
            name='OR Gate',
            description='Output 1 when ANY input is 1. '
                       'Also linearly separable - one neuron is enough!',
            category='binary',
            default_architecture=[2, 1],
            input_labels=['Input A', 'Input B'],
            output_labels=['Output'],
            output_activation='sigmoid',
            difficulty=1,
            concept='Linear Separability',
            learning_goal='Reinforce that linearly separable problems need only one neuron. '
                         'Compare the decision boundary with AND.',
            tips=[
                'Same architecture as AND works here',
                'Notice the decision boundary is in a different position',
                'Only 0,0 maps to 0'
            ],
            level=1
        )

    def generate_data(self) -> tuple[np.ndarray, np.ndarray]:
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        y = np.array([[0], [1], [1], [1]], dtype=float)  # Only 0,0 → 0
        return X, y

    def generate_sample(self, inputs: list[float] | None = None) -> tuple[np.ndarray, np.ndarray]:
        if inputs is None:
            inputs = [float(np.random.randint(0, 2)) for _ in range(2)]
        X = np.array([inputs[:2]], dtype=float)
        y = np.array([[1 if inputs[0] > 0.5 or inputs[1] > 0.5 else 0]], dtype=float)
        return X, y


class NOTGateProblem(Problem):
    """NOT gate - single input, shows how negative weights work."""

    @property
    def info(self) -> ProblemInfo:
        return ProblemInfo(
            id='not_gate',
            name='NOT Gate',
            description='Output the opposite of the input. '
                       'Teaches how negative weights invert signals!',
            category='binary',
            default_architecture=[1, 1],  # Single input, single output
            input_labels=['Input'],
            output_labels=['Output'],
            output_activation='sigmoid',
            difficulty=1,
            concept='Negative Weights',
            learning_goal='Negative weights flip the output. '
                         'The neuron learns a negative weight to invert the input.',
            tips=[
                'Only 2 training examples (0→1, 1→0)',
                'Watch the weight become negative',
                'Bias shifts the sigmoid curve'
            ],
            level=1
        )

    def generate_data(self) -> tuple[np.ndarray, np.ndarray]:
        X = np.array([[0], [1]], dtype=float)
        y = np.array([[1], [0]], dtype=float)  # Inverted
        return X, y

    def generate_sample(self, inputs: list[float] | None = None) -> tuple[np.ndarray, np.ndarray]:
        if inputs is None:
            inputs = [float(np.random.randint(0, 2))]
        X = np.array([[inputs[0]]], dtype=float)
        y = np.array([[0 if inputs[0] > 0.5 else 1]], dtype=float)
        return X, y


class NANDGateProblem(Problem):
    """NAND gate - still linearly separable, basis of all logic."""

    @property
    def info(self) -> ProblemInfo:
        return ProblemInfo(
            id='nand_gate',
            name='NAND Gate',
            description='NOT-AND: Output 0 only when BOTH inputs are 1. '
                       'NAND is universal - you can build any logic circuit from it!',
            category='binary',
            default_architecture=[2, 1],
            input_labels=['Input A', 'Input B'],
            output_labels=['Output'],
            output_activation='sigmoid',
            difficulty=1,
            concept='Linear Separability',
            learning_goal='NAND is linearly separable (still just one line). '
                         'Fun fact: NAND gates are the building blocks of all computers!',
            tips=[
                'Output is opposite of AND',
                'Only 1,1 produces 0',
                'Still solvable with one neuron'
            ],
            level=1
        )

    def generate_data(self) -> tuple[np.ndarray, np.ndarray]:
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        y = np.array([[1], [1], [1], [0]], dtype=float)  # NAND: NOT(AND)
        return X, y

    def generate_sample(self, inputs: list[float] | None = None) -> tuple[np.ndarray, np.ndarray]:
        if inputs is None:
            inputs = [float(np.random.randint(0, 2)) for _ in range(2)]
        X = np.array([inputs[:2]], dtype=float)
        # NAND: 0 only when both are 1
        y = np.array([[0 if (inputs[0] > 0.5 and inputs[1] > 0.5) else 1]], dtype=float)
        return X, y


# =============================================================================
# LEVEL 2: The XOR Problem - Why Hidden Layers Matter
# This is THE classic problem that shows why we need hidden layers
# =============================================================================

class XORProblem(Problem):
    """XOR gate - the classic problem that REQUIRES hidden layers."""

    @property
    def info(self) -> ProblemInfo:
        return ProblemInfo(
            id='xor',
            name='XOR Gate',
            description='Output 1 when inputs are DIFFERENT. '
                       'NOT linearly separable - you NEED hidden layers!',
            category='binary',
            default_architecture=[2, 4, 1],  # Need hidden layer!
            input_labels=['Input A', 'Input B'],
            output_labels=['Output'],
            output_activation='sigmoid',
            difficulty=2,
            concept='Non-Linear Separability',
            learning_goal='The famous XOR problem! A single neuron CANNOT solve this. '
                         'This is why neural networks need hidden layers.',
            tips=[
                'First try [2, 1] and watch it FAIL to converge',
                'Then add a hidden layer [2, 4, 1]',
                'The hidden layer creates a new representation',
                'This is why deep learning works!'
            ],
            level=2
        )

    def generate_data(self) -> tuple[np.ndarray, np.ndarray]:
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        y = np.array([[0], [1], [1], [0]], dtype=float)  # Different → 1
        return X, y

    def generate_sample(self, inputs: list[float] | None = None) -> tuple[np.ndarray, np.ndarray]:
        if inputs is None:
            inputs = [float(np.random.randint(0, 2)) for _ in range(2)]
        X = np.array([inputs[:2]], dtype=float)
        # XOR: output 1 if inputs are different
        a, b = inputs[0] > 0.5, inputs[1] > 0.5
        y = np.array([[1 if a != b else 0]], dtype=float)
        return X, y


class XNORProblem(Problem):
    """XNOR gate - same geometry as XOR, just inverted."""

    @property
    def info(self) -> ProblemInfo:
        return ProblemInfo(
            id='xnor',
            name='XNOR Gate',
            description='Output 1 when inputs are SAME. '
                       'Inverse of XOR - same non-linear problem!',
            category='binary',
            default_architecture=[2, 4, 1],
            input_labels=['Input A', 'Input B'],
            output_labels=['Output'],
            output_activation='sigmoid',
            difficulty=2,
            concept='Non-Linear Separability',
            learning_goal='XNOR is XOR inverted. Still not linearly separable! '
                         'Notice the network learns the same structure, just with different output weights.',
            tips=[
                'Same difficulty as XOR',
                'Output is opposite of XOR',
                'The hidden layer learns the same features'
            ],
            level=2
        )

    def generate_data(self) -> tuple[np.ndarray, np.ndarray]:
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        y = np.array([[1], [0], [0], [1]], dtype=float)  # Same → 1 (opposite of XOR)
        return X, y

    def generate_sample(self, inputs: list[float] | None = None) -> tuple[np.ndarray, np.ndarray]:
        if inputs is None:
            inputs = [float(np.random.randint(0, 2)) for _ in range(2)]
        X = np.array([inputs[:2]], dtype=float)
        # XNOR: output 1 if inputs are same
        a, b = inputs[0] > 0.5, inputs[1] > 0.5
        y = np.array([[1 if a == b else 0]], dtype=float)
        return X, y


class XOR5BitProblem(Problem):
    """5-bit XOR (parity) - extended version for more challenge."""

    @property
    def info(self) -> ProblemInfo:
        return ProblemInfo(
            id='xor_5bit',
            name='5-Bit Parity',
            description='Output 1 if odd number of inputs are HIGH. '
                       'Extended XOR - tests if the network can learn parity.',
            category='binary',
            default_architecture=[5, 12, 8, 1],
            input_labels=['Bit 0', 'Bit 1', 'Bit 2', 'Bit 3', 'Bit 4'],
            output_labels=['Parity'],
            output_activation='sigmoid',
            difficulty=3,
            concept='Parity & Deeper Networks',
            learning_goal='Parity is hard! Requires the network to count modulo 2. '
                         'Tests if deeper networks can learn more complex patterns.',
            tips=[
                'More bits = harder problem',
                'Try different architectures',
                'Deeper networks often work better here',
                'Watch for vanishing gradients'
            ],
            level=2
        )

    def generate_data(self) -> tuple[np.ndarray, np.ndarray]:
        n_samples = 32  # 2^5
        X = []
        y = []
        for i in range(n_samples):
            inputs = [(i >> j) & 1 for j in range(5)]
            label = sum(inputs) % 2  # Parity
            X.append(inputs)
            y.append([label])
        return np.array(X, dtype=float), np.array(y, dtype=float)

    def generate_sample(self, inputs: list[float] | None = None) -> tuple[np.ndarray, np.ndarray]:
        if inputs is None:
            inputs = [float(np.random.randint(0, 2)) for _ in range(5)]
        X = np.array([inputs[:5]], dtype=float)
        y = np.array([[sum(int(x > 0.5) for x in inputs) % 2]], dtype=float)
        return X, y


# =============================================================================
# LEVEL 3: 2D Decision Boundaries
# Visual problems that show how networks carve up space
# =============================================================================

class TwoBlobsProblem(Problem):
    """Two blobs - almost linearly separable, easy starting point."""

    @property
    def info(self) -> ProblemInfo:
        return ProblemInfo(
            id='two_blobs',
            name='Two Blobs',
            description='Classify two clusters of points. '
                       'Almost linearly separable - easy warm-up for 2D visualization!',
            category='binary',
            default_architecture=[2, 4, 1],
            input_labels=['X', 'Y'],
            output_labels=['Class'],
            output_activation='sigmoid',
            difficulty=1,
            concept='Simple Classification',
            learning_goal='Start with something easy! Two well-separated clusters. '
                         'Great for understanding the decision boundary concept.',
            tips=[
                'This should train very quickly',
                'Even a small network works well',
                'Watch the decision boundary form between clusters'
            ],
            level=3
        )

    def generate_data(self) -> tuple[np.ndarray, np.ndarray]:
        n_per_class = 100
        # Blob 1: centered at (-0.5, -0.5)
        X1 = np.random.randn(n_per_class, 2) * 0.25 + np.array([-0.5, -0.5])
        # Blob 2: centered at (0.5, 0.5)
        X2 = np.random.randn(n_per_class, 2) * 0.25 + np.array([0.5, 0.5])

        X = np.vstack([X1, X2])
        y = np.vstack([np.zeros((n_per_class, 1)), np.ones((n_per_class, 1))])

        indices = np.random.permutation(len(X))
        return X[indices], y[indices]

    def generate_sample(self, inputs: list[float] | None = None) -> tuple[np.ndarray, np.ndarray]:
        if inputs is None:
            inputs = [np.random.uniform(-1, 1) for _ in range(2)]
        X = np.array([inputs[:2]], dtype=float)
        # Simple: class 1 if x + y > 0
        y = np.array([[1 if inputs[0] + inputs[1] > 0 else 0]], dtype=float)
        return X, y


class MoonsProblem(Problem):
    """Half moons - classic sklearn dataset."""

    @property
    def info(self) -> ProblemInfo:
        return ProblemInfo(
            id='moons',
            name='Half Moons',
            description='Classify two interleaved half-moon shapes. '
                       'A classic non-linear classification benchmark!',
            category='binary',
            default_architecture=[2, 8, 4, 1],
            input_labels=['X', 'Y'],
            output_labels=['Class'],
            output_activation='sigmoid',
            difficulty=2,
            concept='Non-Linear Boundaries',
            learning_goal='The boundary between moons is curved. '
                         'Networks learn to bend their decision surfaces.',
            tips=[
                'Needs at least one hidden layer',
                'Watch the curved boundary emerge',
                'Classic machine learning benchmark'
            ],
            level=3
        )

    def generate_data(self) -> tuple[np.ndarray, np.ndarray]:
        n_per_class = 150
        noise = 0.1

        # Moon 1 (upper)
        theta1 = np.linspace(0, np.pi, n_per_class)
        x1 = np.cos(theta1) + np.random.randn(n_per_class) * noise
        y1 = np.sin(theta1) + np.random.randn(n_per_class) * noise

        # Moon 2 (lower, shifted)
        theta2 = np.linspace(0, np.pi, n_per_class)
        x2 = 1 - np.cos(theta2) + np.random.randn(n_per_class) * noise
        y2 = 0.5 - np.sin(theta2) + np.random.randn(n_per_class) * noise

        X = np.vstack([
            np.column_stack([x1, y1]),
            np.column_stack([x2, y2])
        ])
        # Normalize to [-1, 1]
        X = (X - X.mean(axis=0)) / X.std(axis=0) * 0.5

        y = np.vstack([np.zeros((n_per_class, 1)), np.ones((n_per_class, 1))])

        indices = np.random.permutation(len(X))
        return X[indices], y[indices]

    def generate_sample(self, inputs: list[float] | None = None) -> tuple[np.ndarray, np.ndarray]:
        if inputs is None:
            inputs = [np.random.uniform(-1, 1) for _ in range(2)]
        X = np.array([inputs[:2]], dtype=float)
        # Heuristic: upper moon vs lower moon
        y = np.array([[1 if inputs[1] < 0.2 - 0.3 * np.sin(inputs[0] * np.pi) else 0]], dtype=float)
        return X, y


class CircleProblem(Problem):
    """Points inside vs outside a circle - beautiful decision boundary."""

    @property
    def info(self) -> ProblemInfo:
        return ProblemInfo(
            id='circle',
            name='Circle Classification',
            description='Classify points as inside or outside a circle. '
                       'The decision boundary should form a ring!',
            category='binary',
            default_architecture=[2, 8, 4, 1],
            input_labels=['X', 'Y'],
            output_labels=['Inside'],
            output_activation='sigmoid',
            difficulty=2,
            concept='Non-Linear Decision Boundaries',
            learning_goal='See how neural networks create curved decision boundaries. '
                         'A linear classifier would fail here!',
            tips=[
                'Visualize the decision boundary',
                'The network learns a circular region',
                'More neurons = smoother boundary',
                'Try with just 1 hidden layer first'
            ],
            level=3
        )

    def generate_data(self) -> tuple[np.ndarray, np.ndarray]:
        n_samples = 300
        # Random points in [-1, 1] x [-1, 1]
        X = np.random.uniform(-1, 1, (n_samples, 2))
        # Label based on distance from origin
        distances = np.sqrt(X[:, 0]**2 + X[:, 1]**2)
        y = (distances < 0.6).astype(float).reshape(-1, 1)
        return X, y

    def generate_sample(self, inputs: list[float] | None = None) -> tuple[np.ndarray, np.ndarray]:
        if inputs is None:
            inputs = [np.random.uniform(-1, 1) for _ in range(2)]
        X = np.array([inputs[:2]], dtype=float)
        distance = np.sqrt(inputs[0]**2 + inputs[1]**2)
        y = np.array([[1 if distance < 0.6 else 0]], dtype=float)
        return X, y


class DonutProblem(Problem):
    """Ring/donut shaped classification region."""

    @property
    def info(self) -> ProblemInfo:
        return ProblemInfo(
            id='donut',
            name='Donut (Ring)',
            description='Classify points in a ring/donut shape. '
                       'Inside hole = 0, in ring = 1, outside = 0.',
            category='binary',
            default_architecture=[2, 8, 8, 1],
            input_labels=['X', 'Y'],
            output_labels=['In Ring'],
            output_activation='sigmoid',
            difficulty=3,
            concept='Multi-Region Boundaries',
            learning_goal='The network must learn TWO boundaries - inner and outer circles. '
                         'Shows how networks can create complex regions.',
            tips=[
                'There are two decision boundaries',
                'Inner radius and outer radius',
                'The network learns both simultaneously'
            ],
            level=3
        )

    def generate_data(self) -> tuple[np.ndarray, np.ndarray]:
        n_samples = 400
        X = np.random.uniform(-1.5, 1.5, (n_samples, 2))
        distances = np.sqrt(X[:, 0]**2 + X[:, 1]**2)
        # Ring: 0.4 < distance < 0.9
        y = ((distances > 0.4) & (distances < 0.9)).astype(float).reshape(-1, 1)
        return X, y

    def generate_sample(self, inputs: list[float] | None = None) -> tuple[np.ndarray, np.ndarray]:
        if inputs is None:
            inputs = [np.random.uniform(-1.5, 1.5) for _ in range(2)]
        X = np.array([inputs[:2]], dtype=float)
        distance = np.sqrt(inputs[0]**2 + inputs[1]**2)
        in_ring = 0.4 < distance < 0.9
        y = np.array([[1 if in_ring else 0]], dtype=float)
        return X, y


class SpiralProblem(Problem):
    """Two interleaved spirals - the classic hard problem."""

    @property
    def info(self) -> ProblemInfo:
        return ProblemInfo(
            id='spiral',
            name='Spiral Dataset',
            description='Classify two interleaved spirals. '
                       'A classic ML challenge - requires deeper networks!',
            category='binary',
            default_architecture=[2, 16, 16, 8, 1],
            input_labels=['X', 'Y'],
            output_labels=['Class'],
            output_activation='sigmoid',
            difficulty=4,
            concept='Deep Networks & Complex Boundaries',
            learning_goal='The spiral problem is HARD. It demonstrates why deeper networks '
                         'can learn more complex functions.',
            tips=[
                'Shallow networks will struggle',
                'Try adding more layers',
                'Lower learning rate helps',
                'Be patient - it takes many epochs!'
            ],
            level=3
        )

    def generate_data(self) -> tuple[np.ndarray, np.ndarray]:
        n_points = 200
        noise = 0.15

        # Spiral 1
        theta1 = np.linspace(0, 4 * np.pi, n_points)
        r1 = theta1 / (4 * np.pi)
        x1 = r1 * np.cos(theta1) + np.random.randn(n_points) * noise * 0.1
        y1 = r1 * np.sin(theta1) + np.random.randn(n_points) * noise * 0.1

        # Spiral 2 (rotated 180°)
        theta2 = np.linspace(0, 4 * np.pi, n_points)
        r2 = theta2 / (4 * np.pi)
        x2 = -r2 * np.cos(theta2) + np.random.randn(n_points) * noise * 0.1
        y2 = -r2 * np.sin(theta2) + np.random.randn(n_points) * noise * 0.1

        X = np.vstack([
            np.column_stack([x1, y1]),
            np.column_stack([x2, y2])
        ])
        y = np.vstack([
            np.zeros((n_points, 1)),
            np.ones((n_points, 1))
        ])

        # Shuffle
        indices = np.random.permutation(len(X))
        return X[indices], y[indices]

    def generate_sample(self, inputs: list[float] | None = None) -> tuple[np.ndarray, np.ndarray]:
        if inputs is None:
            inputs = [np.random.uniform(-1, 1) for _ in range(2)]
        X = np.array([inputs[:2]], dtype=float)
        # Heuristic: use angle to guess class
        angle = np.arctan2(inputs[1], inputs[0])
        r = np.sqrt(inputs[0]**2 + inputs[1]**2)
        # Approximate spiral membership
        expected_angle = r * 4 * np.pi
        diff = abs((angle - expected_angle) % (2 * np.pi))
        y = np.array([[0 if diff < np.pi/2 else 1]], dtype=float)
        return X, y


# =============================================================================
# LEVEL 4: Regression Problems
# Continuous output - function approximation
# =============================================================================

class LinearProblem(Problem):
    """Simple line - trivial regression baseline."""

    @property
    def info(self) -> ProblemInfo:
        return ProblemInfo(
            id='linear',
            name='Line (y=x)',
            description='Approximate a simple line: y = x. '
                       'The simplest regression problem - perfect baseline!',
            category='regression',
            default_architecture=[1, 4, 1],
            input_labels=['x'],
            output_labels=['y'],
            output_activation='sigmoid',
            difficulty=1,
            concept='Basic Regression',
            learning_goal='The simplest function approximation. '
                         'A neural network should learn this instantly.',
            tips=[
                'Should reach ~100% accuracy quickly',
                'Output is normalized to [0, 1]',
                'Try with even smaller networks'
            ],
            level=4
        )

    def generate_data(self) -> tuple[np.ndarray, np.ndarray]:
        n_samples = 100
        X = np.linspace(0, 1, n_samples).reshape(-1, 1)
        y = X.copy()  # y = x
        return X, y

    def generate_sample(self, inputs: list[float] | None = None) -> tuple[np.ndarray, np.ndarray]:
        if inputs is None:
            inputs = [np.random.rand()]
        X = np.array([[inputs[0]]], dtype=float)
        y = np.array([[inputs[0]]], dtype=float)  # y = x
        return X, y


class SineWaveProblem(Problem):
    """Learn to approximate a sine wave - universal approximation demo."""

    @property
    def info(self) -> ProblemInfo:
        return ProblemInfo(
            id='sine_wave',
            name='Sine Wave',
            description='Approximate the sine function. '
                       'Demonstrates that neural networks can learn any function!',
            category='regression',
            default_architecture=[1, 16, 8, 1],
            input_labels=['x'],
            output_labels=['sin(x)'],
            output_activation='sigmoid',
            difficulty=2,
            concept='Universal Approximation',
            learning_goal='Neural networks can approximate ANY continuous function! '
                         'This is the Universal Approximation Theorem in action.',
            tips=[
                'Output is normalized to [0, 1]',
                'More neurons = better approximation',
                'Watch for overfitting on the edges',
                'Regression uses MSE loss'
            ],
            level=4
        )

    def generate_data(self) -> tuple[np.ndarray, np.ndarray]:
        n_samples = 200
        X = np.linspace(0, 1, n_samples).reshape(-1, 1)
        # Sine wave normalized to [0, 1]
        y = (np.sin(X * 2 * np.pi) + 1) / 2
        return X, y

    def generate_sample(self, inputs: list[float] | None = None) -> tuple[np.ndarray, np.ndarray]:
        if inputs is None:
            inputs = [np.random.rand()]
        X = np.array([[inputs[0]]], dtype=float)
        y = np.array([[(np.sin(inputs[0] * 2 * np.pi) + 1) / 2]], dtype=float)
        return X, y


class PolynomialProblem(Problem):
    """Learn a polynomial function."""

    @property
    def info(self) -> ProblemInfo:
        return ProblemInfo(
            id='polynomial',
            name='Polynomial Function',
            description='Approximate y = x³ - x (an S-curve). '
                       'Tests if the network can learn smooth curves.',
            category='regression',
            default_architecture=[1, 12, 8, 1],
            input_labels=['x'],
            output_labels=['y'],
            output_activation='sigmoid',
            difficulty=2,
            concept='Function Approximation',
            learning_goal='Networks can learn polynomial relationships. '
                         'Compare to sine wave - different shapes, same principle.',
            tips=[
                'Output normalized to [0, 1]',
                'The function has an inflection point',
                'Watch the learned curve take shape'
            ],
            level=4
        )

    def generate_data(self) -> tuple[np.ndarray, np.ndarray]:
        n_samples = 200
        X = np.linspace(0, 1, n_samples).reshape(-1, 1)
        # x³ - x normalized
        raw = (X - 0.5)**3 - (X - 0.5) * 0.5
        y = (raw - raw.min()) / (raw.max() - raw.min())
        return X, y

    def generate_sample(self, inputs: list[float] | None = None) -> tuple[np.ndarray, np.ndarray]:
        if inputs is None:
            inputs = [np.random.rand()]
        x = inputs[0]
        X = np.array([[x]], dtype=float)
        raw = (x - 0.5)**3 - (x - 0.5) * 0.5
        # Approximate normalization
        y = np.clip((raw + 0.2) / 0.4, 0, 1)
        return X, np.array([[y]], dtype=float)


class TwoInputRegressionProblem(Problem):
    """2D regression - learning a surface."""

    @property
    def info(self) -> ProblemInfo:
        return ProblemInfo(
            id='surface',
            name='2D Surface',
            description='Learn a 2D function: peaks and valleys. '
                       'Combines two inputs to produce one output.',
            category='regression',
            default_architecture=[2, 12, 8, 1],
            input_labels=['x', 'y'],
            output_labels=['z'],
            output_activation='sigmoid',
            difficulty=3,
            concept='Multi-Input Regression',
            learning_goal='Neural networks can learn functions of multiple variables. '
                         'The output is a surface in 3D space.',
            tips=[
                'Think of it as learning a height map',
                'The function has a peak in the center',
                'More data points = better surface learning'
            ],
            level=4
        )

    def generate_data(self) -> tuple[np.ndarray, np.ndarray]:
        n_samples = 400
        X = np.random.rand(n_samples, 2)
        # Gaussian bump centered at (0.5, 0.5)
        x, y = X[:, 0], X[:, 1]
        z = np.exp(-((x - 0.5)**2 + (y - 0.5)**2) / 0.1)
        return X, z.reshape(-1, 1)

    def generate_sample(self, inputs: list[float] | None = None) -> tuple[np.ndarray, np.ndarray]:
        if inputs is None:
            inputs = [np.random.rand(), np.random.rand()]
        X = np.array([inputs[:2]], dtype=float)
        x, y = inputs[0], inputs[1]
        z = np.exp(-((x - 0.5)**2 + (y - 0.5)**2) / 0.1)
        return X, np.array([[z]], dtype=float)


# =============================================================================
# LEVEL 5: FAILURE CASES - Learn What NOT to Do!
# These problems are designed to FAIL and teach important lessons
# =============================================================================

class FailXORNoHiddenProblem(Problem):
    """XOR with no hidden layer - intentionally fails to teach linear limits."""

    @property
    def info(self) -> ProblemInfo:
        return ProblemInfo(
            id='fail_xor_no_hidden',
            name='XOR (No Hidden Layer)',
            description='Try to solve XOR with just input→output. '
                       'THIS WILL FAIL! Learn why hidden layers are essential.',
            category='binary',
            default_architecture=[2, 1],  # No hidden layer!
            input_labels=['Input A', 'Input B'],
            output_labels=['Output'],
            output_activation='sigmoid',
            difficulty=2,
            concept='Linear Limits',
            learning_goal='Experience failure! A single neuron cannot solve XOR because '
                         'XOR is not linearly separable. No straight line can separate the classes.',
            tips=[
                'Watch the accuracy get stuck at ~50%',
                'The loss will plateau, not converge',
                'After you see the failure, try adding a hidden layer!'
            ],
            level=5,
            is_failure_case=True,
            failure_reason='XOR is not linearly separable. A single neuron can only draw '
                          'a straight line decision boundary, but XOR needs a curved boundary.',
            fix_suggestion='Add a hidden layer! Try architecture [2, 4, 1] or [2, 2, 1].',
            locked_architecture=True  # Cannot change - that would defeat the lesson
        )

    def generate_data(self) -> tuple[np.ndarray, np.ndarray]:
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        y = np.array([[0], [1], [1], [0]], dtype=float)
        return X, y

    def generate_sample(self, inputs: list[float] | None = None) -> tuple[np.ndarray, np.ndarray]:
        if inputs is None:
            inputs = [float(np.random.randint(0, 2)) for _ in range(2)]
        X = np.array([inputs[:2]], dtype=float)
        a, b = inputs[0] > 0.5, inputs[1] > 0.5
        y = np.array([[1 if a != b else 0]], dtype=float)
        return X, y


class FailZeroInitProblem(Problem):
    """Zero weight initialization - neurons become identical."""

    @property
    def info(self) -> ProblemInfo:
        return ProblemInfo(
            id='fail_zero_init',
            name='Zero Init Trap',
            description='XOR with all weights initialized to zero. '
                       'THIS WILL FAIL! Learn why weight initialization matters.',
            category='binary',
            default_architecture=[2, 4, 1],
            input_labels=['Input A', 'Input B'],
            output_labels=['Output'],
            output_activation='sigmoid',
            difficulty=2,
            concept='Symmetry Breaking',
            learning_goal='All neurons start identical → they stay identical forever! '
                         'This is called the "symmetry problem". Random initialization breaks it.',
            tips=[
                'All hidden neurons will have identical outputs',
                'Gradients will be identical too',
                'The network has hidden capacity it cannot use'
            ],
            level=5,
            is_failure_case=True,
            failure_reason='When all weights are zero, all hidden neurons compute the same thing. '
                          'Backprop gives them identical gradients, so they stay identical forever. '
                          'It is as if you have only 1 hidden neuron, not 4!',
            fix_suggestion='Use random initialization (Xavier or He). This breaks symmetry so each neuron can learn something different.',
            locked_architecture=True,
            forced_weight_init='zeros'  # Force zero initialization
        )

    def generate_data(self) -> tuple[np.ndarray, np.ndarray]:
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        y = np.array([[0], [1], [1], [0]], dtype=float)
        return X, y

    def generate_sample(self, inputs: list[float] | None = None) -> tuple[np.ndarray, np.ndarray]:
        if inputs is None:
            inputs = [float(np.random.randint(0, 2)) for _ in range(2)]
        X = np.array([inputs[:2]], dtype=float)
        a, b = inputs[0] > 0.5, inputs[1] > 0.5
        y = np.array([[1 if a != b else 0]], dtype=float)
        return X, y


class FailHighLRProblem(Problem):
    """Learning rate too high - loss explodes!"""

    @property
    def info(self) -> ProblemInfo:
        return ProblemInfo(
            id='fail_lr_high',
            name='LR Explosion',
            description='XOR with learning rate = 10 (way too high!). '
                       'Watch the loss EXPLODE!',
            category='binary',
            default_architecture=[2, 4, 1],
            input_labels=['Input A', 'Input B'],
            output_labels=['Output'],
            output_activation='sigmoid',
            difficulty=2,
            concept='Learning Rate Selection',
            learning_goal='Too high a learning rate overshoots the optimal weights. '
                         'The gradient steps are so big they make things worse!',
            tips=[
                'Watch the loss graph go CRAZY',
                'Values may become NaN (not a number)',
                'This is why learning rate tuning is important'
            ],
            level=5,
            is_failure_case=True,
            failure_reason='Learning rate of 10 is far too high. Each gradient step massively '
                          'overshoots the minimum, causing oscillation or divergence. '
                          'Loss explodes instead of decreasing!',
            fix_suggestion='Use a learning rate around 0.1-1.0. Start with 0.5 and adjust based on how training goes.',
            locked_architecture=True,
            forced_learning_rate=10.0  # Force high LR
        )

    def generate_data(self) -> tuple[np.ndarray, np.ndarray]:
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        y = np.array([[0], [1], [1], [0]], dtype=float)
        return X, y

    def generate_sample(self, inputs: list[float] | None = None) -> tuple[np.ndarray, np.ndarray]:
        if inputs is None:
            inputs = [float(np.random.randint(0, 2)) for _ in range(2)]
        X = np.array([inputs[:2]], dtype=float)
        a, b = inputs[0] > 0.5, inputs[1] > 0.5
        y = np.array([[1 if a != b else 0]], dtype=float)
        return X, y


class FailLowLRProblem(Problem):
    """Learning rate too low - training takes forever!"""

    @property
    def info(self) -> ProblemInfo:
        return ProblemInfo(
            id='fail_lr_low',
            name='LR Stagnation',
            description='XOR with learning rate = 0.0001 (way too low!). '
                       'Training will be painfully slow.',
            category='binary',
            default_architecture=[2, 4, 1],
            input_labels=['Input A', 'Input B'],
            output_labels=['Output'],
            output_activation='sigmoid',
            difficulty=2,
            concept='Learning Rate Selection',
            learning_goal='Too low a learning rate = glacially slow training. '
                         'The network CAN learn, but it takes forever.',
            tips=[
                'Loss decreases, but very very slowly',
                'You would need millions of epochs',
                'Compare this to normal learning rate'
            ],
            level=5,
            is_failure_case=True,
            failure_reason='Learning rate of 0.0001 means tiny gradient steps. '
                          'The network slowly inches toward the solution but would take '
                          'thousands of epochs to converge.',
            fix_suggestion='Use a learning rate around 0.1-1.0. Adaptive training automatically finds good rates.',
            locked_architecture=True,
            forced_learning_rate=0.0001  # Force low LR
        )

    def generate_data(self) -> tuple[np.ndarray, np.ndarray]:
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        y = np.array([[0], [1], [1], [0]], dtype=float)
        return X, y

    def generate_sample(self, inputs: list[float] | None = None) -> tuple[np.ndarray, np.ndarray]:
        if inputs is None:
            inputs = [float(np.random.randint(0, 2)) for _ in range(2)]
        X = np.array([inputs[:2]], dtype=float)
        a, b = inputs[0] > 0.5, inputs[1] > 0.5
        y = np.array([[1 if a != b else 0]], dtype=float)
        return X, y


class FailVanishingGradientProblem(Problem):
    """Deep sigmoid network - vanishing gradients!"""

    @property
    def info(self) -> ProblemInfo:
        return ProblemInfo(
            id='fail_vanishing',
            name='Vanishing Gradients',
            description='Deep network with sigmoid everywhere. '
                       'Early layers stop learning - gradients vanish!',
            category='binary',
            default_architecture=[2, 8, 8, 8, 8, 1],  # Deep!
            input_labels=['Input A', 'Input B'],
            output_labels=['Output'],
            output_activation='sigmoid',
            difficulty=3,
            concept='Vanishing Gradients',
            learning_goal='In deep networks with sigmoid activation, gradients become tiny '
                         'as they backpropagate. Early layers barely learn!',
            tips=[
                'First layers will hardly change',
                'Later layers do most of the learning',
                'This is why ReLU activation was invented'
            ],
            level=5,
            is_failure_case=True,
            failure_reason='Sigmoid squashes values to [0,1]. Its gradient is at most 0.25. '
                          'Through 4 layers: 0.25^4 = 0.004. Gradients become negligible!',
            fix_suggestion='Use ReLU activation for hidden layers. ReLU gradient is 0 or 1, so gradients flow better.',
            locked_architecture=True
            # Note: Would also need to force sigmoid hidden activation, but our current impl doesn't support that
        )

    def generate_data(self) -> tuple[np.ndarray, np.ndarray]:
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        y = np.array([[0], [1], [1], [0]], dtype=float)
        return X, y

    def generate_sample(self, inputs: list[float] | None = None) -> tuple[np.ndarray, np.ndarray]:
        if inputs is None:
            inputs = [float(np.random.randint(0, 2)) for _ in range(2)]
        X = np.array([inputs[:2]], dtype=float)
        a, b = inputs[0] > 0.5, inputs[1] > 0.5
        y = np.array([[1 if a != b else 0]], dtype=float)
        return X, y


class FailUnderfitProblem(Problem):
    """Too few neurons for spiral - can't learn complex boundary."""

    @property
    def info(self) -> ProblemInfo:
        return ProblemInfo(
            id='fail_underfit',
            name='Underfitting',
            description='Spiral problem with only 2 hidden neurons. '
                       'Way too simple to learn the complex boundary!',
            category='binary',
            default_architecture=[2, 2, 1],  # Too simple!
            input_labels=['X', 'Y'],
            output_labels=['Class'],
            output_activation='sigmoid',
            difficulty=2,
            concept='Model Capacity',
            learning_goal='A network that is too small cannot learn complex patterns. '
                         'This is called underfitting - the model lacks capacity.',
            tips=[
                'Accuracy will plateau around 50-60%',
                'The decision boundary will be too simple',
                'Compare with a larger network on the same problem'
            ],
            level=5,
            is_failure_case=True,
            failure_reason='2 hidden neurons can only create 2 "features" for the output to combine. '
                          'The spiral needs many more features to carve out its complex boundary.',
            fix_suggestion='Use more hidden neurons [2, 16, 16, 8, 1] for the spiral problem.',
            locked_architecture=True
        )

    def generate_data(self) -> tuple[np.ndarray, np.ndarray]:
        # Same as spiral problem
        n_points = 200
        noise = 0.15

        theta1 = np.linspace(0, 4 * np.pi, n_points)
        r1 = theta1 / (4 * np.pi)
        x1 = r1 * np.cos(theta1) + np.random.randn(n_points) * noise * 0.1
        y1 = r1 * np.sin(theta1) + np.random.randn(n_points) * noise * 0.1

        theta2 = np.linspace(0, 4 * np.pi, n_points)
        r2 = theta2 / (4 * np.pi)
        x2 = -r2 * np.cos(theta2) + np.random.randn(n_points) * noise * 0.1
        y2 = -r2 * np.sin(theta2) + np.random.randn(n_points) * noise * 0.1

        X = np.vstack([np.column_stack([x1, y1]), np.column_stack([x2, y2])])
        y = np.vstack([np.zeros((n_points, 1)), np.ones((n_points, 1))])

        indices = np.random.permutation(len(X))
        return X[indices], y[indices]

    def generate_sample(self, inputs: list[float] | None = None) -> tuple[np.ndarray, np.ndarray]:
        if inputs is None:
            inputs = [np.random.uniform(-1, 1) for _ in range(2)]
        X = np.array([inputs[:2]], dtype=float)
        angle = np.arctan2(inputs[1], inputs[0])
        r = np.sqrt(inputs[0]**2 + inputs[1]**2)
        expected_angle = r * 4 * np.pi
        diff = abs((angle - expected_angle) % (2 * np.pi))
        y = np.array([[0 if diff < np.pi/2 else 1]], dtype=float)
        return X, y


# =============================================================================
# LEVEL 6: Multi-Class Classification
# More than 2 classes - requires softmax
# =============================================================================

class QuadrantProblem(Problem):
    """Classify points by which quadrant they're in."""

    @property
    def info(self) -> ProblemInfo:
        return ProblemInfo(
            id='quadrants',
            name='Quadrant Classification',
            description='Classify 2D points into 4 quadrants. '
                       'Simple multi-class problem with clear boundaries.',
            category='multi-class',
            default_architecture=[2, 8, 4],
            input_labels=['X', 'Y'],
            output_labels=['Q1 (+,+)', 'Q2 (-,+)', 'Q3 (-,-)', 'Q4 (+,-)'],
            output_activation='softmax',
            difficulty=2,
            concept='Multi-Class Boundaries',
            learning_goal='A simple 4-class problem. The decision boundaries are '
                         'axis-aligned lines crossing at the origin.',
            tips=[
                'The boundaries are straight lines',
                'Should train quickly',
                'Good warm-up for multi-class'
            ],
            level=6
        )

    def generate_data(self) -> tuple[np.ndarray, np.ndarray]:
        n_samples = 400
        X = np.random.uniform(-1, 1, (n_samples, 2))
        y = np.zeros((n_samples, 4))

        for i, (x, yc) in enumerate(X):
            if x >= 0 and yc >= 0:
                y[i, 0] = 1  # Q1
            elif x < 0 and yc >= 0:
                y[i, 1] = 1  # Q2
            elif x < 0 and yc < 0:
                y[i, 2] = 1  # Q3
            else:
                y[i, 3] = 1  # Q4

        return X, y

    def generate_sample(self, inputs: list[float] | None = None) -> tuple[np.ndarray, np.ndarray]:
        if inputs is None:
            inputs = [np.random.uniform(-1, 1) for _ in range(2)]

        X = np.array([inputs[:2]], dtype=float)
        x, yc = inputs[0], inputs[1]
        y = np.zeros((1, 4))

        if x >= 0 and yc >= 0:
            y[0, 0] = 1
        elif x < 0 and yc >= 0:
            y[0, 1] = 1
        elif x < 0 and yc < 0:
            y[0, 2] = 1
        else:
            y[0, 3] = 1

        return X, y


class GaussianBlobsProblem(Problem):
    """5 Gaussian blobs for multi-class classification."""

    @property
    def info(self) -> ProblemInfo:
        return ProblemInfo(
            id='blobs',
            name='Gaussian Blobs',
            description='Classify points into 5 different clusters. '
                       'A classic multi-class dataset with clear separation.',
            category='multi-class',
            default_architecture=[2, 8, 5],
            input_labels=['X', 'Y'],
            output_labels=['Blob 1', 'Blob 2', 'Blob 3', 'Blob 4', 'Blob 5'],
            output_activation='softmax',
            difficulty=2,
            concept='Multi-Class Classification',
            learning_goal='5-class classification problem. Watch how softmax assigns '
                         'probabilities to each class based on position.',
            tips=[
                'Each blob is a Gaussian cluster',
                'Blobs are well-separated',
                'Good for understanding softmax output'
            ],
            level=6
        )

    def generate_data(self) -> tuple[np.ndarray, np.ndarray]:
        n_per_class = 80
        centers = [
            [0.0, 0.0],    # Center
            [-0.7, 0.7],   # Top-left
            [0.7, 0.7],    # Top-right
            [-0.7, -0.7],  # Bottom-left
            [0.7, -0.7],   # Bottom-right
        ]

        samples = []
        labels = []

        for class_idx, center in enumerate(centers):
            points = np.random.randn(n_per_class, 2) * 0.15 + np.array(center)
            samples.append(points)
            label = np.zeros((n_per_class, 5))
            label[:, class_idx] = 1
            labels.append(label)

        X = np.vstack(samples)
        y = np.vstack(labels)

        indices = np.random.permutation(len(X))
        return X[indices], y[indices]

    def generate_sample(self, inputs: list[float] | None = None) -> tuple[np.ndarray, np.ndarray]:
        if inputs is None:
            inputs = [np.random.uniform(-1, 1) for _ in range(2)]

        X = np.array([inputs[:2]], dtype=float)

        # Find nearest blob center
        centers = [[0.0, 0.0], [-0.7, 0.7], [0.7, 0.7], [-0.7, -0.7], [0.7, -0.7]]
        distances = [np.sqrt((inputs[0] - c[0])**2 + (inputs[1] - c[1])**2) for c in centers]
        nearest = np.argmin(distances)

        y = np.zeros((1, 5))
        y[0, nearest] = 1
        return X, y


class ColorClassificationProblem(Problem):
    """Classify RGB colors into categories."""

    @property
    def info(self) -> ProblemInfo:
        return ProblemInfo(
            id='colors',
            name='Color Classification',
            description='Classify RGB values into color names: '
                       'Red, Green, Blue, Yellow, Cyan, Magenta.',
            category='multi-class',
            default_architecture=[3, 12, 8, 6],
            input_labels=['Red', 'Green', 'Blue'],
            output_labels=['Red', 'Green', 'Blue', 'Yellow', 'Cyan', 'Magenta'],
            output_activation='softmax',
            difficulty=3,
            concept='Multi-Class with Softmax',
            learning_goal='Multi-class classification uses softmax activation. '
                         'Each output neuron represents the probability of a class.',
            tips=[
                'Softmax makes outputs sum to 1',
                'Watch the probability distribution',
                'Yellow = Red + Green, Cyan = Green + Blue, etc.',
                'The network learns color mixing!'
            ],
            level=6
        )

    def generate_data(self) -> tuple[np.ndarray, np.ndarray]:
        samples = []
        labels = []

        # Color centers (R, G, B) and their class index
        colors = [
            (1.0, 0.2, 0.2),  # Red
            (0.2, 1.0, 0.2),  # Green
            (0.2, 0.2, 1.0),  # Blue
            (1.0, 1.0, 0.2),  # Yellow
            (0.2, 1.0, 1.0),  # Cyan
            (1.0, 0.2, 1.0),  # Magenta
        ]

        for color_idx, (r, g, b) in enumerate(colors):
            for _ in range(50):
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
        if inputs is None:
            inputs = [np.random.rand() for _ in range(3)]

        X = np.array([inputs[:3]], dtype=float)
        r, g, b = inputs[0], inputs[1], inputs[2]

        y = np.zeros((1, 6))
        # Classify based on dominant colors
        if r > 0.6 and g < 0.5 and b < 0.5:
            y[0, 0] = 1  # Red
        elif g > 0.6 and r < 0.5 and b < 0.5:
            y[0, 1] = 1  # Green
        elif b > 0.6 and r < 0.5 and g < 0.5:
            y[0, 2] = 1  # Blue
        elif r > 0.6 and g > 0.6:
            y[0, 3] = 1  # Yellow
        elif g > 0.6 and b > 0.6:
            y[0, 4] = 1  # Cyan
        elif r > 0.6 and b > 0.6:
            y[0, 5] = 1  # Magenta
        else:
            y[0, np.argmax([r, g, b])] = 1

        return X, y


class PatternClassificationProblem(Problem):
    """Classify simple signal patterns: flat, rising, falling, spike."""

    @property
    def info(self) -> ProblemInfo:
        return ProblemInfo(
            id='patterns',
            name='Signal Patterns',
            description='Classify signal patterns: Flat, Rising, Falling, Spike. '
                       'Learn to recognize temporal patterns in data.',
            category='multi-class',
            default_architecture=[8, 12, 8, 4],
            input_labels=['t0', 't1', 't2', 't3', 't4', 't5', 't6', 't7'],
            output_labels=['Flat', 'Rising', 'Falling', 'Spike'],
            output_activation='softmax',
            difficulty=3,
            concept='Sequence Pattern Recognition',
            learning_goal='Neural networks can recognize patterns in sequential data. '
                         'This is a precursor to understanding RNNs.',
            tips=[
                'Each pattern has a distinct shape',
                'The network learns shape features',
                'Try drawing your own patterns!'
            ],
            level=6
        )

    def generate_data(self) -> tuple[np.ndarray, np.ndarray]:
        samples = []
        labels = []

        n_per_class = 100

        # Flat: constant with noise
        for _ in range(n_per_class):
            level = np.random.uniform(0.3, 0.7)
            pattern = np.full(8, level) + np.random.randn(8) * 0.05
            samples.append(np.clip(pattern, 0, 1))
            labels.append([1, 0, 0, 0])

        # Rising: increasing trend
        for _ in range(n_per_class):
            start = np.random.uniform(0.1, 0.3)
            pattern = np.linspace(start, start + 0.5, 8) + np.random.randn(8) * 0.05
            samples.append(np.clip(pattern, 0, 1))
            labels.append([0, 1, 0, 0])

        # Falling: decreasing trend
        for _ in range(n_per_class):
            start = np.random.uniform(0.7, 0.9)
            pattern = np.linspace(start, start - 0.5, 8) + np.random.randn(8) * 0.05
            samples.append(np.clip(pattern, 0, 1))
            labels.append([0, 0, 1, 0])

        # Spike: sharp peak in middle
        for _ in range(n_per_class):
            base = np.random.uniform(0.2, 0.4)
            pattern = np.full(8, base)
            spike_pos = np.random.randint(2, 6)
            pattern[spike_pos] = 0.9
            pattern = pattern + np.random.randn(8) * 0.03
            samples.append(np.clip(pattern, 0, 1))
            labels.append([0, 0, 0, 1])

        X = np.array(samples)
        y = np.array(labels)

        indices = np.random.permutation(len(X))
        return X[indices], y[indices]

    def generate_sample(self, inputs: list[float] | None = None) -> tuple[np.ndarray, np.ndarray]:
        if inputs is None:
            # Generate random pattern
            pattern_type = np.random.choice(['flat', 'rising', 'falling', 'spike'])
            if pattern_type == 'flat':
                inputs = [0.5] * 8
            elif pattern_type == 'rising':
                inputs = list(np.linspace(0.2, 0.8, 8))
            elif pattern_type == 'falling':
                inputs = list(np.linspace(0.8, 0.2, 8))
            else:
                inputs = [0.3] * 8
                inputs[4] = 0.9

        X = np.array([inputs[:8]], dtype=float)
        y = self._classify_pattern(inputs[:8])
        return X, y

    def _classify_pattern(self, inputs: list[float]) -> np.ndarray:
        arr = np.array(inputs)
        y = np.zeros((1, 4))

        # Check for spike
        max_val = np.max(arr)
        if max_val > 0.7 and np.sum(arr > 0.6) <= 2:
            y[0, 3] = 1
            return y

        # Check for trend
        diff = arr[-1] - arr[0]
        variance = np.var(arr)

        if diff > 0.3:
            y[0, 1] = 1  # Rising
        elif diff < -0.3:
            y[0, 2] = 1  # Falling
        else:
            y[0, 0] = 1  # Flat

        return y


# =============================================================================
# LEVEL 7: CNN Problems (Images)
# =============================================================================

class ShapeDetectionProblem(Problem):
    """Classify shapes on 8x8 grid using CNN."""

    @property
    def info(self) -> ProblemInfo:
        return ProblemInfo(
            id='shapes',
            name='Shape Detection (CNN)',
            description='Classify 8×8 images of circles, squares, and triangles. '
                       'Introduction to Convolutional Neural Networks!',
            category='multi-class',
            default_architecture=[64, 32, 16, 3],
            input_labels=[f'p{i}' for i in range(64)],
            output_labels=['Circle', 'Square', 'Triangle'],
            output_activation='softmax',
            difficulty=4,
            concept='Convolutional Neural Networks',
            learning_goal='CNNs use filters to detect spatial patterns. '
                         'They learn edges, shapes, and hierarchical features.',
            tips=[
                'CNN filters detect edges first',
                'Pooling reduces size, keeps features',
                'Try the "Learn How It Works" button!',
                'Watch the feature maps evolve'
            ],
            network_type='cnn',
            input_shape=(8, 8, 1),
            level=7
        )

    def generate_data(self) -> tuple[np.ndarray, np.ndarray]:
        samples = []
        labels = []

        for _ in range(100):
            samples.append(self._generate_circle())
            labels.append([1, 0, 0])

        for _ in range(100):
            samples.append(self._generate_square())
            labels.append([0, 1, 0])

        for _ in range(100):
            samples.append(self._generate_triangle())
            labels.append([0, 0, 1])

        X = np.array(samples).reshape(-1, 8, 8, 1)
        y = np.array(labels)

        indices = np.random.permutation(len(X))
        return X[indices], y[indices]

    def _generate_circle(self) -> np.ndarray:
        grid = np.zeros((8, 8))
        center = 3.5 + np.random.uniform(-0.3, 0.3)
        radius = 2.5 + np.random.uniform(-0.5, 0.5)

        for i in range(8):
            for j in range(8):
                dist = np.sqrt((i - center)**2 + (j - center)**2)
                if dist < radius:
                    grid[i, j] = 0.9 + np.random.uniform(-0.1, 0.1)

        grid += np.random.uniform(-0.05, 0.05, (8, 8))
        return np.clip(grid, 0, 1)

    def _generate_square(self) -> np.ndarray:
        grid = np.zeros((8, 8))
        margin = np.random.randint(1, 3)
        size_var = np.random.randint(0, 2)

        top, left = margin, margin
        bottom = 8 - margin - size_var
        right = 8 - margin - size_var

        grid[top:bottom, left:right] = 0.9 + np.random.uniform(-0.1, 0.1)
        grid += np.random.uniform(-0.05, 0.05, (8, 8))
        return np.clip(grid, 0, 1)

    def _generate_triangle(self) -> np.ndarray:
        grid = np.zeros((8, 8))
        flip = np.random.choice([True, False])
        center_x = 3.5 + np.random.uniform(-0.5, 0.5)
        top_row = 1 + np.random.randint(0, 2)
        bottom_row = 6 + np.random.randint(0, 2)
        height = bottom_row - top_row

        for row in range(8):
            if row < top_row or row > bottom_row:
                continue

            if flip:
                progress = (row - top_row) / max(height, 1)
            else:
                progress = (bottom_row - row) / max(height, 1)

            half_width = 0.5 + progress * 2.5
            left = max(0, int(center_x - half_width + 0.5))
            right = min(8, int(center_x + half_width + 0.5))

            if right > left:
                grid[row, left:right] = 0.9 + np.random.uniform(-0.1, 0.1)

        grid += np.random.uniform(-0.03, 0.03, (8, 8))
        return np.clip(grid, 0, 1)

    def generate_sample(self, inputs: list[float] | None = None) -> tuple[np.ndarray, np.ndarray]:
        if inputs is None:
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
            grid = np.array(inputs[:64]).reshape(8, 8)
            y = self._classify_grid(grid)
            X = grid.reshape(1, 8, 8, 1)

        return X, y

    def _classify_grid(self, grid: np.ndarray) -> np.ndarray:
        """Classify user-drawn shape based on geometric properties."""
        active = grid > 0.5
        if not np.any(active):
            return np.array([[1, 0, 0]])  # Default to circle if empty

        # Find bounding box
        rows_active = np.any(active, axis=1)
        cols_active = np.any(active, axis=0)
        row_indices = np.where(rows_active)[0]
        col_indices = np.where(cols_active)[0]

        if len(row_indices) == 0 or len(col_indices) == 0:
            return np.array([[1, 0, 0]])  # Default

        top, bottom = row_indices[0], row_indices[-1]
        left, right = col_indices[0], col_indices[-1]
        height = bottom - top + 1
        width = right - left + 1
        bbox_area = height * width
        active_count = np.sum(active)
        fill_ratio = active_count / (bbox_area + 1e-10)

        # Check if roughly square-shaped bounding box
        aspect_ratio = width / (height + 1e-10)
        is_roughly_square_bbox = 0.7 < aspect_ratio < 1.4

        # For squares: high fill ratio (>85%) and roughly square bounding box
        if fill_ratio > 0.85 and is_roughly_square_bbox:
            return np.array([[0, 1, 0]])  # Square

        # For circles: check if edges are curved (corners are empty)
        # Extract the bounding box region
        bbox = active[top:bottom+1, left:right+1]
        if bbox.size > 0 and height >= 3 and width >= 3:
            # Check corners - circles have empty corners, squares don't
            corners = [
                bbox[0, 0], bbox[0, -1],  # top corners
                bbox[-1, 0], bbox[-1, -1]  # bottom corners
            ]
            empty_corners = sum(1 for c in corners if not c)

            # Circles typically have 3-4 empty corners, squares have 0
            if empty_corners >= 2 and fill_ratio > 0.5:
                return np.array([[1, 0, 0]])  # Circle

        # Triangles: lower fill ratio due to sloped edges
        if fill_ratio < 0.75:
            return np.array([[0, 0, 1]])  # Triangle

        # Default based on fill ratio
        if fill_ratio > 0.8:
            return np.array([[0, 1, 0]])  # Square
        else:
            return np.array([[0, 0, 1]])  # Triangle


class DigitRecognitionProblem(Problem):
    """Classify digits 0-9 on 8x8 grid."""

    DIGIT_PATTERNS = {
        0: [[0,0,1,1,1,1,0,0],[0,1,1,0,0,1,1,0],[0,1,1,0,0,1,1,0],[0,1,1,0,0,1,1,0],
            [0,1,1,0,0,1,1,0],[0,1,1,0,0,1,1,0],[0,1,1,0,0,1,1,0],[0,0,1,1,1,1,0,0]],
        1: [[0,0,0,1,1,0,0,0],[0,0,1,1,1,0,0,0],[0,0,0,1,1,0,0,0],[0,0,0,1,1,0,0,0],
            [0,0,0,1,1,0,0,0],[0,0,0,1,1,0,0,0],[0,0,0,1,1,0,0,0],[0,0,1,1,1,1,0,0]],
        2: [[0,0,1,1,1,1,0,0],[0,1,1,0,0,1,1,0],[0,0,0,0,0,1,1,0],[0,0,0,0,1,1,0,0],
            [0,0,0,1,1,0,0,0],[0,0,1,1,0,0,0,0],[0,1,1,0,0,0,0,0],[0,1,1,1,1,1,1,0]],
        3: [[0,0,1,1,1,1,0,0],[0,1,1,0,0,1,1,0],[0,0,0,0,0,1,1,0],[0,0,0,1,1,1,0,0],
            [0,0,0,0,0,1,1,0],[0,0,0,0,0,1,1,0],[0,1,1,0,0,1,1,0],[0,0,1,1,1,1,0,0]],
        4: [[0,0,0,0,1,1,0,0],[0,0,0,1,1,1,0,0],[0,0,1,1,1,1,0,0],[0,1,1,0,1,1,0,0],
            [0,1,1,1,1,1,1,0],[0,0,0,0,1,1,0,0],[0,0,0,0,1,1,0,0],[0,0,0,0,1,1,0,0]],
        5: [[0,1,1,1,1,1,1,0],[0,1,1,0,0,0,0,0],[0,1,1,0,0,0,0,0],[0,1,1,1,1,1,0,0],
            [0,0,0,0,0,1,1,0],[0,0,0,0,0,1,1,0],[0,1,1,0,0,1,1,0],[0,0,1,1,1,1,0,0]],
        6: [[0,0,1,1,1,1,0,0],[0,1,1,0,0,0,0,0],[0,1,1,0,0,0,0,0],[0,1,1,1,1,1,0,0],
            [0,1,1,0,0,1,1,0],[0,1,1,0,0,1,1,0],[0,1,1,0,0,1,1,0],[0,0,1,1,1,1,0,0]],
        7: [[0,1,1,1,1,1,1,0],[0,0,0,0,0,1,1,0],[0,0,0,0,1,1,0,0],[0,0,0,0,1,1,0,0],
            [0,0,0,1,1,0,0,0],[0,0,0,1,1,0,0,0],[0,0,0,1,1,0,0,0],[0,0,0,1,1,0,0,0]],
        8: [[0,0,1,1,1,1,0,0],[0,1,1,0,0,1,1,0],[0,1,1,0,0,1,1,0],[0,0,1,1,1,1,0,0],
            [0,1,1,0,0,1,1,0],[0,1,1,0,0,1,1,0],[0,1,1,0,0,1,1,0],[0,0,1,1,1,1,0,0]],
        9: [[0,0,1,1,1,1,0,0],[0,1,1,0,0,1,1,0],[0,1,1,0,0,1,1,0],[0,0,1,1,1,1,1,0],
            [0,0,0,0,0,1,1,0],[0,0,0,0,0,1,1,0],[0,0,0,0,1,1,0,0],[0,0,1,1,1,0,0,0]],
    }

    @property
    def info(self) -> ProblemInfo:
        return ProblemInfo(
            id='digits',
            name='Digit Recognition (CNN)',
            description='Classify 8×8 images of digits 0-9. '
                       'The classic MNIST-style problem on a small scale.',
            category='multi-class',
            default_architecture=[64, 32, 16, 10],
            input_labels=[f'p{i}' for i in range(64)],
            output_labels=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
            output_activation='softmax',
            difficulty=5,
            concept='Image Classification',
            learning_goal='Digit recognition is a classic benchmark. '
                         'CNNs learn to recognize strokes and curves.',
            tips=[
                '10 classes is more challenging',
                'Some digits look similar (3 vs 8)',
                'More training data helps',
                'Try drawing your own digits!'
            ],
            network_type='cnn',
            input_shape=(8, 8, 1),
            level=7
        )

    def generate_data(self) -> tuple[np.ndarray, np.ndarray]:
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
        base = np.array(self.DIGIT_PATTERNS[digit], dtype=float)

        # Random shift
        shift_x = np.random.choice([-1, 0, 0, 1])
        shift_y = np.random.choice([-1, 0, 0, 1])
        if shift_x != 0 or shift_y != 0:
            base = np.roll(base, shift_x, axis=1)
            base = np.roll(base, shift_y, axis=0)
            if shift_x > 0: base[:, :shift_x] = 0
            elif shift_x < 0: base[:, shift_x:] = 0
            if shift_y > 0: base[:shift_y, :] = 0
            elif shift_y < 0: base[shift_y:, :] = 0

        base = base * (0.8 + np.random.uniform(0, 0.2))
        base += np.random.uniform(-0.1, 0.1, (8, 8))
        dropout_mask = np.random.random((8, 8)) > 0.05
        base = base * dropout_mask

        return np.clip(base, 0, 1)

    def generate_sample(self, inputs: list[float] | None = None) -> tuple[np.ndarray, np.ndarray]:
        if inputs is None:
            digit = np.random.randint(0, 10)
            grid = self._generate_digit(digit)
            y = np.zeros((1, 10))
            y[0, digit] = 1
            X = grid.reshape(1, 8, 8, 1)
        else:
            grid = np.array(inputs[:64]).reshape(8, 8)
            y = self._classify_grid(grid)
            X = grid.reshape(1, 8, 8, 1)

        return X, y

    def _classify_grid(self, grid: np.ndarray) -> np.ndarray:
        best_match = 0
        best_score = -np.inf

        for digit, pattern in self.DIGIT_PATTERNS.items():
            pattern_arr = np.array(pattern, dtype=float)
            grid_norm = grid / (np.max(grid) + 1e-10)
            score = np.sum(grid_norm * pattern_arr)
            if score > best_score:
                best_score = score
                best_match = digit

        y = np.zeros((1, 10))
        y[0, best_match] = 1
        return y


# =============================================================================
# Problem Registry - Organized by Learning Level
# =============================================================================

PROBLEMS: dict[str, Problem] = {
    # Level 1: Single Neuron
    'and_gate': ANDGateProblem(),
    'or_gate': ORGateProblem(),
    'not_gate': NOTGateProblem(),
    'nand_gate': NANDGateProblem(),

    # Level 2: XOR - Why Hidden Layers
    'xor': XORProblem(),
    'xnor': XNORProblem(),
    'xor_5bit': XOR5BitProblem(),

    # Level 3: 2D Decision Boundaries
    'two_blobs': TwoBlobsProblem(),
    'moons': MoonsProblem(),
    'circle': CircleProblem(),
    'donut': DonutProblem(),
    'spiral': SpiralProblem(),

    # Level 4: Regression
    'linear': LinearProblem(),
    'sine_wave': SineWaveProblem(),
    'polynomial': PolynomialProblem(),
    'surface': TwoInputRegressionProblem(),

    # Level 5: FAILURE CASES
    'fail_xor_no_hidden': FailXORNoHiddenProblem(),
    'fail_zero_init': FailZeroInitProblem(),
    'fail_lr_high': FailHighLRProblem(),
    'fail_lr_low': FailLowLRProblem(),
    'fail_vanishing': FailVanishingGradientProblem(),
    'fail_underfit': FailUnderfitProblem(),

    # Level 6: Multi-Class
    'quadrants': QuadrantProblem(),
    'blobs': GaussianBlobsProblem(),
    'colors': ColorClassificationProblem(),
    'patterns': PatternClassificationProblem(),

    # Level 7: CNN
    'shapes': ShapeDetectionProblem(),
    'digits': DigitRecognitionProblem(),
}


def get_problem(problem_id: str) -> Problem:
    """Get a problem by ID."""
    if problem_id not in PROBLEMS:
        raise ValueError(f"Unknown problem: {problem_id}. Available: {list(PROBLEMS.keys())}")
    return PROBLEMS[problem_id]


def list_problems() -> list[dict]:
    """List all available problems with their info."""
    results = []
    for p in PROBLEMS.values():
        info = p.info
        results.append({
            'id': info.id,
            'name': info.name,
            'description': info.description,
            'category': info.category,
            'default_architecture': info.default_architecture,
            'input_labels': info.input_labels,
            'output_labels': info.output_labels,
            'output_activation': info.output_activation,
            'difficulty': info.difficulty,
            'concept': info.concept,
            'learning_goal': info.learning_goal,
            'tips': info.tips,
            'network_type': info.network_type,
            'input_shape': info.input_shape,
            # New fields
            'level': info.level,
            'is_failure_case': info.is_failure_case,
            'failure_reason': info.failure_reason,
            'fix_suggestion': info.fix_suggestion,
            'locked_architecture': info.locked_architecture,
            'forced_weight_init': info.forced_weight_init,
            'forced_learning_rate': info.forced_learning_rate,
        })
    return results


def list_problems_by_level() -> dict[str, list[dict]]:
    """List problems organized by difficulty level."""
    levels = {
        'Level 1: Single Neuron': [],
        'Level 2: Hidden Layers': [],
        'Level 3: Decision Boundaries': [],
        'Level 4: Regression': [],
        'Level 5: Failure Cases': [],
        'Level 6: Multi-Class': [],
        'Level 7: CNN': [],
    }

    level_mapping = {
        1: 'Level 1: Single Neuron',
        2: 'Level 2: Hidden Layers',
        3: 'Level 3: Decision Boundaries',
        4: 'Level 4: Regression',
        5: 'Level 5: Failure Cases',
        6: 'Level 6: Multi-Class',
        7: 'Level 7: CNN',
    }

    for p in PROBLEMS.values():
        info = p.info
        problem_data = {
            'id': info.id,
            'name': info.name,
            'description': info.description,
            'difficulty': info.difficulty,
            'concept': info.concept,
            'is_failure_case': info.is_failure_case,
            'failure_reason': info.failure_reason,
        }

        level_key = level_mapping.get(info.level, f'Level {info.level}')
        if level_key in levels:
            levels[level_key].append(problem_data)

    return levels


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Neural Network Learning Lab - Problems Demo")
    print("=" * 60)

    for level, problems in list_problems_by_level().items():
        print(f"\n{level}")
        print("-" * 40)
        for p in problems:
            marker = " [FAILURE CASE]" if p.get('is_failure_case') else ""
            print(f"  {p['name']}: {p['concept']}{marker}")

    print("\n" + "=" * 60)
    print("Problem Statistics")
    print("=" * 60)
    print(f"Total problems: {len(PROBLEMS)}")

    failure_cases = [p for p in PROBLEMS.values() if p.info.is_failure_case]
    print(f"Failure cases: {len(failure_cases)}")

    for prob_id, prob in PROBLEMS.items():
        info = prob.info
        X, y = prob.generate_data()
        print(f"\n{info.name} ({prob_id})")
        print(f"  Level: {info.level}")
        print(f"  Data shape: X={X.shape}, y={y.shape}")
        if info.is_failure_case:
            print(f"  [FAILURE] {info.failure_reason[:50]}...")
