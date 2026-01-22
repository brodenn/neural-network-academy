"""
Tests for problems.py

Tests problem data generation, labels, and attributes for all 28 problems.
"""

import numpy as np
import pytest
from problems import (
    PROBLEMS, get_problem, list_problems,
    ANDGateProblem, ORGateProblem, NOTGateProblem, NANDGateProblem,
    XORProblem, XNORProblem, XOR5BitProblem,
)


class TestProblemRegistry:
    """Test the PROBLEMS registry."""

    def test_all_problems_registered(self):
        """All 28 problems should be registered."""
        assert len(PROBLEMS) == 28

    def test_get_problem_valid(self):
        """get_problem should return correct problem."""
        prob = get_problem('xor')
        assert prob.info.id == 'xor'

    def test_get_problem_invalid(self):
        """get_problem should raise for unknown ID."""
        with pytest.raises(ValueError):
            get_problem('nonexistent_problem')

    def test_list_problems_format(self):
        """list_problems should return list of dicts."""
        problems = list_problems()
        assert isinstance(problems, list)
        assert len(problems) == 28
        for p in problems:
            assert 'id' in p
            assert 'name' in p
            assert 'category' in p


class TestProblemInfo:
    """Test that all problems have complete info."""

    @pytest.mark.parametrize("problem_id", list(PROBLEMS.keys()))
    def test_problem_has_required_fields(self, problem_id):
        """Each problem should have all required info fields."""
        prob = get_problem(problem_id)
        info = prob.info

        # Required fields
        assert info.id, f"{problem_id} missing id"
        assert info.name, f"{problem_id} missing name"
        assert info.description, f"{problem_id} missing description"
        assert info.category in ['binary', 'regression', 'multi-class'], \
            f"{problem_id} has invalid category: {info.category}"
        assert info.default_architecture, f"{problem_id} missing architecture"
        assert info.input_labels, f"{problem_id} missing input_labels"
        assert info.output_labels, f"{problem_id} missing output_labels"
        assert info.output_activation in ['sigmoid', 'softmax'], \
            f"{problem_id} has invalid activation: {info.output_activation}"
        assert 1 <= info.difficulty <= 5, f"{problem_id} difficulty out of range"
        assert info.concept, f"{problem_id} missing concept"
        assert info.learning_goal, f"{problem_id} missing learning_goal"

    @pytest.mark.parametrize("problem_id", list(PROBLEMS.keys()))
    def test_problem_architecture_matches_labels(self, problem_id):
        """Architecture input/output should match label counts."""
        prob = get_problem(problem_id)
        info = prob.info

        # Skip CNN problems (architecture is different)
        if info.network_type == 'cnn':
            return

        # Input layer should match input_labels
        assert info.default_architecture[0] == len(info.input_labels), \
            f"{problem_id}: arch input {info.default_architecture[0]} != labels {len(info.input_labels)}"

        # Output layer should match output_labels
        assert info.default_architecture[-1] == len(info.output_labels), \
            f"{problem_id}: arch output {info.default_architecture[-1]} != labels {len(info.output_labels)}"


class TestDataGeneration:
    """Test data generation for all problems."""

    @pytest.mark.parametrize("problem_id", list(PROBLEMS.keys()))
    def test_generate_data_returns_arrays(self, problem_id):
        """generate_data should return numpy arrays."""
        prob = get_problem(problem_id)
        X, y = prob.generate_data()

        assert isinstance(X, np.ndarray), f"{problem_id} X is not ndarray"
        assert isinstance(y, np.ndarray), f"{problem_id} y is not ndarray"

    @pytest.mark.parametrize("problem_id", list(PROBLEMS.keys()))
    def test_generate_data_shapes_match(self, problem_id):
        """X and y should have matching first dimension."""
        prob = get_problem(problem_id)
        X, y = prob.generate_data()

        assert X.shape[0] == y.shape[0], \
            f"{problem_id}: X samples {X.shape[0]} != y samples {y.shape[0]}"

    @pytest.mark.parametrize("problem_id", list(PROBLEMS.keys()))
    def test_generate_data_has_samples(self, problem_id):
        """Should generate at least 1 sample."""
        prob = get_problem(problem_id)
        X, y = prob.generate_data()

        assert X.shape[0] >= 1, f"{problem_id} generates no samples"

    @pytest.mark.parametrize("problem_id", list(PROBLEMS.keys()))
    def test_generate_data_no_nan(self, problem_id):
        """Data should not contain NaN values."""
        prob = get_problem(problem_id)
        X, y = prob.generate_data()

        assert not np.any(np.isnan(X)), f"{problem_id} X contains NaN"
        assert not np.any(np.isnan(y)), f"{problem_id} y contains NaN"


class TestSampleGeneration:
    """Test single sample generation."""

    @pytest.mark.parametrize("problem_id", list(PROBLEMS.keys()))
    def test_generate_sample_returns_arrays(self, problem_id):
        """generate_sample should return numpy arrays."""
        prob = get_problem(problem_id)
        X, y = prob.generate_sample()

        assert isinstance(X, np.ndarray), f"{problem_id} X is not ndarray"
        assert isinstance(y, np.ndarray), f"{problem_id} y is not ndarray"

    @pytest.mark.parametrize("problem_id", list(PROBLEMS.keys()))
    def test_generate_sample_single_row(self, problem_id):
        """generate_sample should return single sample."""
        prob = get_problem(problem_id)
        X, y = prob.generate_sample()

        assert X.shape[0] == 1, f"{problem_id} X should be single sample"
        assert y.shape[0] == 1, f"{problem_id} y should be single sample"


class TestLevel1Problems:
    """Test Level 1: Single Neuron problems."""

    def test_and_gate_truth_table(self):
        """AND gate should produce correct truth table."""
        prob = ANDGateProblem()
        X, y = prob.generate_data()

        # Verify all 4 combinations
        expected = {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 1}
        for i in range(len(X)):
            key = (int(X[i, 0]), int(X[i, 1]))
            assert y[i, 0] == expected[key], f"AND({key}) should be {expected[key]}"

    def test_or_gate_truth_table(self):
        """OR gate should produce correct truth table."""
        prob = ORGateProblem()
        X, y = prob.generate_data()

        expected = {(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 1}
        for i in range(len(X)):
            key = (int(X[i, 0]), int(X[i, 1]))
            assert y[i, 0] == expected[key], f"OR({key}) should be {expected[key]}"

    def test_not_gate_truth_table(self):
        """NOT gate should produce correct truth table."""
        prob = NOTGateProblem()
        X, y = prob.generate_data()

        assert len(X) == 2
        # NOT(0) = 1, NOT(1) = 0
        for i in range(len(X)):
            assert y[i, 0] == 1 - X[i, 0], f"NOT({X[i, 0]}) should be {1 - X[i, 0]}"

    def test_nand_gate_truth_table(self):
        """NAND gate should produce correct truth table."""
        prob = NANDGateProblem()
        X, y = prob.generate_data()

        expected = {(0, 0): 1, (0, 1): 1, (1, 0): 1, (1, 1): 0}
        for i in range(len(X)):
            key = (int(X[i, 0]), int(X[i, 1]))
            assert y[i, 0] == expected[key], f"NAND({key}) should be {expected[key]}"


class TestLevel2Problems:
    """Test Level 2: XOR problems."""

    def test_xor_truth_table(self):
        """XOR should produce correct truth table."""
        prob = XORProblem()
        X, y = prob.generate_data()

        expected = {(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 0}
        for i in range(len(X)):
            key = (int(X[i, 0]), int(X[i, 1]))
            assert y[i, 0] == expected[key], f"XOR({key}) should be {expected[key]}"

    def test_xnor_truth_table(self):
        """XNOR should produce correct truth table."""
        prob = XNORProblem()
        X, y = prob.generate_data()

        expected = {(0, 0): 1, (0, 1): 0, (1, 0): 0, (1, 1): 1}
        for i in range(len(X)):
            key = (int(X[i, 0]), int(X[i, 1]))
            assert y[i, 0] == expected[key], f"XNOR({key}) should be {expected[key]}"

    def test_xor_5bit_parity(self):
        """5-bit XOR should follow parity rule."""
        prob = XOR5BitProblem()
        X, y = prob.generate_data()

        # Should have 32 samples (2^5)
        assert len(X) == 32

        # Verify parity rule: odd number of 1s = 1
        for i in range(len(X)):
            expected = int(sum(X[i])) % 2
            assert y[i, 0] == expected, \
                f"5-bit XOR({X[i].tolist()}) should be {expected}"


class TestLevel3Problems:
    """Test Level 3: Decision Boundary problems."""

    def test_two_blobs_has_both_classes(self):
        """Two blobs should have samples from both classes."""
        prob = get_problem('two_blobs')
        X, y = prob.generate_data()

        class_0 = np.sum(y == 0)
        class_1 = np.sum(y == 1)
        assert class_0 > 0, "two_blobs missing class 0"
        assert class_1 > 0, "two_blobs missing class 1"

    def test_circle_has_both_classes(self):
        """Circle problem should have inside/outside classes."""
        prob = get_problem('circle')
        X, y = prob.generate_data()

        class_0 = np.sum(y == 0)
        class_1 = np.sum(y == 1)
        assert class_0 > 0, "circle missing class 0"
        assert class_1 > 0, "circle missing class 1"

    def test_moons_coordinates_in_range(self):
        """Moons data should be in reasonable range."""
        prob = get_problem('moons')
        X, y = prob.generate_data()

        # Data should roughly be in [-2, 3] range
        assert X.min() > -5, "moons X has extreme negative values"
        assert X.max() < 5, "moons X has extreme positive values"


class TestLevel4Problems:
    """Test Level 4: Regression problems."""

    def test_linear_regression_output_range(self):
        """Linear regression output should be in [0, 1] for sigmoid."""
        prob = get_problem('linear')
        X, y = prob.generate_data()

        # Labels should be normalized for sigmoid
        assert y.min() >= 0, "linear y has values < 0"
        assert y.max() <= 1, "linear y has values > 1"

    def test_sine_wave_output_range(self):
        """Sine wave output should be normalized."""
        prob = get_problem('sine_wave')
        X, y = prob.generate_data()

        assert y.min() >= 0, "sine_wave y has values < 0"
        assert y.max() <= 1, "sine_wave y has values > 1"

    def test_polynomial_has_enough_samples(self):
        """Polynomial should have enough samples for learning."""
        prob = get_problem('polynomial')
        X, y = prob.generate_data()

        assert len(X) >= 50, "polynomial has too few samples"


class TestLevel5FailureCases:
    """Test Level 5: Failure Case problems."""

    failure_cases = [
        'fail_xor_no_hidden',
        'fail_zero_init',
        'fail_lr_high',
        'fail_lr_low',
        'fail_vanishing',
        'fail_underfit',
    ]

    @pytest.mark.parametrize("problem_id", failure_cases)
    def test_failure_case_flag(self, problem_id):
        """Failure cases should have is_failure_case=True."""
        prob = get_problem(problem_id)
        assert prob.info.is_failure_case is True, \
            f"{problem_id} should have is_failure_case=True"

    @pytest.mark.parametrize("problem_id", failure_cases)
    def test_failure_case_has_reason(self, problem_id):
        """Failure cases should explain why they fail."""
        prob = get_problem(problem_id)
        assert prob.info.failure_reason, \
            f"{problem_id} missing failure_reason"

    @pytest.mark.parametrize("problem_id", failure_cases)
    def test_failure_case_has_fix(self, problem_id):
        """Failure cases should suggest how to fix."""
        prob = get_problem(problem_id)
        assert prob.info.fix_suggestion, \
            f"{problem_id} missing fix_suggestion"

    def test_xor_no_hidden_has_locked_arch(self):
        """XOR no hidden should have locked architecture."""
        prob = get_problem('fail_xor_no_hidden')
        assert prob.info.locked_architecture is True

    def test_zero_init_has_forced_init(self):
        """Zero init should force zeros initialization."""
        prob = get_problem('fail_zero_init')
        assert prob.info.forced_weight_init == 'zeros'

    def test_high_lr_has_forced_lr(self):
        """High LR should force high learning rate."""
        prob = get_problem('fail_lr_high')
        assert prob.info.forced_learning_rate is not None
        assert prob.info.forced_learning_rate >= 10.0  # Should be very high


class TestLevel6MultiClass:
    """Test Level 6: Multi-class problems."""

    multi_class_problems = ['quadrants', 'blobs', 'colors', 'patterns']

    @pytest.mark.parametrize("problem_id", multi_class_problems)
    def test_softmax_activation(self, problem_id):
        """Multi-class problems should use softmax."""
        prob = get_problem(problem_id)
        assert prob.info.output_activation == 'softmax', \
            f"{problem_id} should use softmax"

    @pytest.mark.parametrize("problem_id", multi_class_problems)
    def test_one_hot_encoding(self, problem_id):
        """Multi-class labels should be one-hot encoded."""
        prob = get_problem(problem_id)
        X, y = prob.generate_data()

        # Each row should sum to 1 (one-hot)
        row_sums = np.sum(y, axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(len(y)), decimal=5,
            err_msg=f"{problem_id} labels not one-hot encoded")

    def test_quadrants_has_4_classes(self):
        """Quadrants should have 4 classes."""
        prob = get_problem('quadrants')
        assert prob.info.default_architecture[-1] == 4


class TestLevel7CNN:
    """Test Level 7: CNN problems."""

    cnn_problems = ['shapes', 'digits']

    @pytest.mark.parametrize("problem_id", cnn_problems)
    def test_cnn_network_type(self, problem_id):
        """CNN problems should have network_type='cnn'."""
        prob = get_problem(problem_id)
        assert prob.info.network_type == 'cnn', \
            f"{problem_id} should have network_type='cnn'"

    @pytest.mark.parametrize("problem_id", cnn_problems)
    def test_cnn_has_input_shape(self, problem_id):
        """CNN problems should specify input_shape."""
        prob = get_problem(problem_id)
        assert prob.info.input_shape is not None, \
            f"{problem_id} missing input_shape"

    def test_shapes_is_8x8(self):
        """Shape detection should use 8x8 grid with 1 channel."""
        prob = get_problem('shapes')
        # CNN input includes channels: (height, width, channels)
        assert prob.info.input_shape == (8, 8, 1), \
            f"shapes input_shape should be (8, 8, 1), got {prob.info.input_shape}"

    def test_shapes_has_3_classes(self):
        """Shape detection should classify 3 shapes."""
        prob = get_problem('shapes')
        # Check output_labels
        assert len(prob.info.output_labels) == 3


class TestProblemLevels:
    """Test problem level assignments."""

    def test_all_problems_have_level(self):
        """All problems should have a level 1-7."""
        for prob_id, prob in PROBLEMS.items():
            level = prob.info.level
            assert 1 <= level <= 7, \
                f"{prob_id} has invalid level {level}"

    def test_level_1_problems(self):
        """Level 1 should have 4 single neuron problems."""
        level_1 = [p for p in PROBLEMS.values() if p.info.level == 1]
        assert len(level_1) == 4

    def test_level_5_failure_cases(self):
        """Level 5 should be all failure cases."""
        level_5 = [p for p in PROBLEMS.values() if p.info.level == 5]
        for p in level_5:
            assert p.info.is_failure_case is True


class TestDifficulty:
    """Test difficulty ratings make sense."""

    def test_level_1_is_easy(self):
        """Level 1 problems should have low difficulty."""
        for p in PROBLEMS.values():
            if p.info.level == 1:
                assert p.info.difficulty <= 2, \
                    f"{p.info.id} (level 1) has high difficulty {p.info.difficulty}"

    def test_spiral_is_hard(self):
        """Spiral problem should be high difficulty."""
        prob = get_problem('spiral')
        assert prob.info.difficulty >= 4


class TestCategories:
    """Test problem categories are correct."""

    def test_level_1_2_are_binary(self):
        """Level 1-2 should be binary classification."""
        for p in PROBLEMS.values():
            if p.info.level in [1, 2]:
                assert p.info.category == 'binary', \
                    f"{p.info.id} should be binary"

    def test_level_4_is_regression(self):
        """Level 4 should be regression."""
        for p in PROBLEMS.values():
            if p.info.level == 4:
                assert p.info.category == 'regression', \
                    f"{p.info.id} should be regression"

    def test_level_6_is_multiclass(self):
        """Level 6 should be multi-class."""
        for p in PROBLEMS.values():
            if p.info.level == 6:
                assert p.info.category == 'multi-class', \
                    f"{p.info.id} should be multi-class"
