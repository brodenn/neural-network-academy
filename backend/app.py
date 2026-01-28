"""
Neural Network Learning Lab - Flask Backend

Interactive neural network learning tool with progressive problems.
Learn neural network concepts from basic logic gates to CNNs.

Run with: python app.py
"""

import threading
import time
from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import numpy as np

from neural_network import NeuralNetwork
from cnn_network import CNNNetwork
from problems import get_problem, list_problems


# -----------------------------------------------------------------------------
# Flask App Setup
# -----------------------------------------------------------------------------

app = Flask(__name__)
CORS(app, origins=["http://localhost:5173", "http://localhost:5174", "http://localhost:5175", "http://localhost:5176", "http://localhost:5177", "http://127.0.0.1:5173"])
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# -----------------------------------------------------------------------------
# Global State
# -----------------------------------------------------------------------------

# Thread locks for shared state (prevents race conditions)
state_lock = threading.Lock()  # Protects system_state dictionary
nn_lock = threading.Lock()      # Protects neural network during training/prediction

# Current problem selection
current_problem_id = 'xor'
current_problem = get_problem(current_problem_id)

# Network type ('dense' for standard NN, 'cnn' for convolutional)
current_network_type = current_problem.info.network_type

# Neural network (initialized from current problem's default architecture)
nn = NeuralNetwork(
    current_problem.info.default_architecture,
    output_activation=current_problem.info.output_activation
)


# Training data (generated from current problem)
X_train, y_train = current_problem.generate_data()

# Current input state (for interactive prediction)
current_inputs: list[float] = [0.0] * len(current_problem.info.input_labels)

# System state
system_state = {
    "training_complete": False,
    "training_in_progress": False,
    "stop_requested": False,
    "target_accuracy": 0.99,  # Can be changed during training
    "current_epoch": 0,
    "current_loss": 0.0,
    "current_accuracy": 0.0,
    "prediction_count": 0
}


# -----------------------------------------------------------------------------
# Prediction and Terminal Output (G6)
# -----------------------------------------------------------------------------

def make_prediction(inputs: list[float] | list[list[list[float]]]) -> dict:
    """
    Make prediction for given inputs.

    Args:
        inputs: List of input values (1D for dense) or 2D grid (for CNN)

    Returns:
        Prediction result dictionary with activations for visualization
    """
    global current_inputs

    with state_lock:
        if not system_state["training_complete"]:
            return {"error": "Training not complete"}

    info = current_problem.info

    # Handle CNN vs Dense input format
    if current_network_type == 'cnn':
        # CNN expects 4D input: (batch, height, width, channels)
        if isinstance(inputs, list) and isinstance(inputs[0], list):
            # 2D grid input - convert to 4D
            grid = np.array(inputs, dtype=float)
            if grid.ndim == 2:
                grid = grid[:, :, np.newaxis]  # Add channel dimension
            X = grid[np.newaxis, :, :, :]  # Add batch dimension
        else:
            # Flat input - reshape to grid
            expected_shape = info.input_shape
            flat = np.array(inputs, dtype=float)
            X = flat.reshape(1, *expected_shape)
        current_inputs = inputs
    else:
        # Dense network - 1D input
        expected_len = len(info.input_labels)
        if len(inputs) != expected_len:
            inputs = inputs[:expected_len] if len(inputs) > expected_len else inputs + [0.0] * (expected_len - len(inputs))
        current_inputs = inputs
        X = np.array([inputs], dtype=float)

    # Predict with activations for visualization
    output, activations = nn.predict_with_activations(X)

    # Handle different output types
    if info.output_activation == 'softmax':
        # Multi-class: return probabilities and predicted class
        probs = output[0].tolist()
        predicted_class = int(np.argmax(probs))
        prediction_value = probs
        led_on = predicted_class > 0  # For LED, any non-zero class turns it on

        # For CNN, we don't have generate_sample with grid inputs
        if current_network_type == 'cnn':
            expected = None
            is_correct = None
        else:
            # Get expected class from problem
            _, expected_y = current_problem.generate_sample(inputs)
            expected_class = int(np.argmax(expected_y[0]))
            expected = expected_y[0].tolist()
            is_correct = predicted_class == expected_class
    else:
        # Binary/regression: single value
        prediction_value = float(output[0][0])
        led_on = prediction_value >= 0.5
        predicted_class = int(led_on)

        # Get expected from problem
        _, expected_y = current_problem.generate_sample(inputs)
        expected = float(expected_y[0][0])
        if info.category == 'binary':
            is_correct = predicted_class == int(expected >= 0.5)
        else:
            is_correct = abs(prediction_value - expected) < 0.1

    # Terminal output (G6 requirement)
    with state_lock:
        system_state["prediction_count"] += 1
    print(f"\nInput change detected!")
    print(f"Problem: {info.name}")
    if current_network_type == 'cnn':
        print(f"CNN Input grid, Prediction: {prediction_value}, LED: {'ON' if led_on else 'OFF'}")
    else:
        print(f"Input: {[f'{x:.2f}' for x in inputs]}, Prediction: {prediction_value if isinstance(prediction_value, list) else f'{prediction_value:.2f}'}, "
              f"LED: {'ON' if led_on else 'OFF'}")

    # Build response
    result = {
        "inputs": inputs,
        "prediction": prediction_value,
        "prediction_rounded": predicted_class,
        "led_state": led_on,
        "expected": expected,
        "correct": is_correct,
        "activations": activations,
        "problem_id": current_problem_id,
        "output_labels": info.output_labels,
        "network_type": current_network_type
    }

    # Add feature maps for CNN
    if current_network_type == 'cnn':
        result["feature_maps"] = nn.get_feature_maps(X)

    return result


# -----------------------------------------------------------------------------
# Training Functions
# -----------------------------------------------------------------------------

def training_callback(epoch: int, loss: float, accuracy: float):
    """Callback during training to emit progress."""
    with state_lock:
        system_state["current_epoch"] = epoch
        system_state["current_loss"] = loss
        system_state["current_accuracy"] = accuracy

    # Emit progress to frontend (throttled, but always emit first few epochs)
    if epoch < 5 or epoch % 10 == 0:
        progress_data = {
            "epoch": epoch,
            "loss": loss,
            "accuracy": accuracy
        }

        # Include gradient info for visualization (only for dense networks)
        if current_network_type == 'dense' and hasattr(nn, 'last_gradients') and nn.last_gradients:
            progress_data["gradients"] = nn.last_gradients

        socketio.emit('training_progress', progress_data)


def stop_check() -> bool:
    """Check if training stop has been requested."""
    with state_lock:
        return system_state["stop_requested"]


def get_target_accuracy() -> float:
    """Get current target accuracy (can be changed during training)."""
    with state_lock:
        return system_state["target_accuracy"]


def run_training(mode: str, epochs: int = 1000, learning_rate: float = 0.1, target_accuracy: float | None = None, forced_lr: float | None = None):
    """Run training in background thread."""
    # Initialize training state with lock
    with state_lock:
        system_state["training_in_progress"] = True
        system_state["training_complete"] = False
        system_state["stop_requested"] = False

        # Set initial target accuracy
        if target_accuracy is None:
            # CNN can realistically reach ~95%, dense networks can reach 99%
            system_state["target_accuracy"] = 0.95 if current_network_type == 'cnn' else 0.99
        else:
            system_state["target_accuracy"] = target_accuracy

        emit_data = {"mode": mode, "target_accuracy": system_state["target_accuracy"]}
    if forced_lr is not None:
        emit_data["forced_learning_rate"] = forced_lr
    socketio.emit('training_started', emit_data)

    try:
        if mode == "static":
            result = nn.train(
                X_train, y_train,
                epochs=epochs,
                learning_rate=learning_rate,
                verbose=True,
                callback=training_callback,
                stop_check=stop_check
            )
        else:  # adaptive
            # Pass the getter function so target can be changed during training
            # Also pass forced_lr if set (for failure case problems)
            result = nn.train_adaptive(
                X_train, y_train,
                target_accuracy=get_target_accuracy,  # Pass function, not value
                verbose=True,
                callback=training_callback,
                stop_check=stop_check,
                forced_learning_rate=forced_lr
            )

        # Mark as complete even if stopped early (allows predictions)
        with state_lock:
            system_state["training_complete"] = True

        # Check if stopped by user
        if result.get("stopped"):
            print("\nTraining stopped by user")
            socketio.emit('training_stopped', result)
        else:
            # Print all predictions in required format (G6)
            print_all_predictions()
            socketio.emit('training_complete', result)

        return result

    except Exception as e:
        print(f"\nTraining error: {e}")
        socketio.emit('training_error', {"error": str(e)})
        return {"error": str(e)}

    finally:
        # Always reset training_in_progress, even on error
        with state_lock:
            system_state["training_in_progress"] = False
            system_state["stop_requested"] = False


def print_all_predictions():
    """Print predictions in format matching Bilaga A (G6 requirement)."""
    print("\n" + "-" * 80)
    accuracy = nn.calculate_accuracy(X_train, y_train)
    print(f"Prediction accuracy: {accuracy * 100:.1f}%")

    predictions = nn.predict(X_train)

    if current_network_type == 'cnn':
        # For CNN, print summary (not all samples)
        pred_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(y_train, axis=1)
        info = current_problem.info
        print(f"\nCNN Shape Detection Summary:")
        for class_idx, label in enumerate(info.output_labels):
            mask = true_classes == class_idx
            class_acc = np.mean(pred_classes[mask] == true_classes[mask]) if np.any(mask) else 0
            print(f"  {label}: {class_acc*100:.1f}% accuracy ({np.sum(mask)} samples)")
    else:
        # Original dense network output
        for i in range(len(X_train)):
            pred_rounded = round(predictions[i][0])
            error = abs(pred_rounded - y_train[i][0])
            print(f"Input: {X_train[i].tolist()}, prediction: [{pred_rounded:.1f}], "
                  f"reference: {y_train[i].tolist()}, error: {error:.1f}")

    print("-" * 80)


# -----------------------------------------------------------------------------
# Input Validation Helpers
# -----------------------------------------------------------------------------

def validate_int(value, name: str, min_val: int = None, max_val: int = None) -> tuple[int | None, str | None]:
    """Validate an integer value. Returns (value, error_message)."""
    if value is None:
        return None, None
    try:
        value = int(value)
        if min_val is not None and value < min_val:
            return None, f"{name} must be at least {min_val}"
        if max_val is not None and value > max_val:
            return None, f"{name} must be at most {max_val}"
        return value, None
    except (TypeError, ValueError):
        return None, f"{name} must be an integer"


def validate_float(value, name: str, min_val: float = None, max_val: float = None) -> tuple[float | None, str | None]:
    """Validate a float value. Returns (value, error_message)."""
    if value is None:
        return None, None
    try:
        value = float(value)
        if min_val is not None and value < min_val:
            return None, f"{name} must be at least {min_val}"
        if max_val is not None and value > max_val:
            return None, f"{name} must be at most {max_val}"
        return value, None
    except (TypeError, ValueError):
        return None, f"{name} must be a number"


def get_json_body() -> tuple[dict | None, str | None]:
    """Safely get JSON body from request. Returns (data, error_message)."""
    try:
        data = request.get_json(silent=True)
        if data is None:
            return {}, None  # Empty body is OK for optional fields
        if not isinstance(data, dict):
            return None, "Request body must be a JSON object"
        return data, None
    except Exception as e:
        return None, f"Invalid JSON: {str(e)}"


# -----------------------------------------------------------------------------
# Problem Routes
# -----------------------------------------------------------------------------

@app.route('/api/problems', methods=['GET'])
def get_problems():
    """List all available problems."""
    return jsonify(list_problems())


@app.route('/api/problems/<problem_id>/select', methods=['POST'])
def select_problem(problem_id: str):
    """Switch to a different problem."""
    global current_problem_id, current_problem, nn, X_train, y_train, current_inputs, current_network_type

    # Force reset training state when switching problems
    # This allows recovery from stuck training states
    with state_lock:
        if system_state["training_in_progress"]:
            print("\nWarning: Forcing training stop due to problem switch")
            system_state["training_in_progress"] = False
            system_state["stop_requested"] = True

    try:
        current_problem = get_problem(problem_id)
        current_problem_id = problem_id
        info = current_problem.info
        current_network_type = info.network_type

        # Determine weight initialization (may be forced for failure cases)
        weight_init = info.forced_weight_init if info.forced_weight_init else 'xavier'

        # Create new network based on problem type
        if info.network_type == 'cnn':
            # CNN network for shape detection
            nn = CNNNetwork(input_shape=info.input_shape)
            nn.add_conv2d(filters=4, kernel_size=3, activation='relu')
            nn.add_maxpool2d(pool_size=2)
            nn.add_flatten()
            nn.add_dense(units=16, activation='relu')
            nn.add_dense(units=len(info.output_labels), activation='softmax')
        else:
            # Standard dense network
            nn = NeuralNetwork(
                info.default_architecture,
                output_activation=info.output_activation,
                weight_init=weight_init
            )

        # Generate training data
        X_train, y_train = current_problem.generate_data()

        # Reset input state
        if info.network_type == 'cnn':
            # For CNN, init with zeros grid
            h, w, c = info.input_shape
            current_inputs = [[0.0] * w for _ in range(h)]
        else:
            current_inputs = [0.0] * len(info.input_labels)

        # Reset training state completely
        with state_lock:
            system_state["training_complete"] = False
            system_state["training_in_progress"] = False
            system_state["stop_requested"] = False
            system_state["current_epoch"] = 0
            system_state["current_loss"] = 0.0
            system_state["current_accuracy"] = 0.0

        print(f"\n{'='*60}")
        print(f"Switched to problem: {info.name}")
        print(f"Network type: {info.network_type}")
        if info.network_type == 'cnn':
            print(f"Input shape: {info.input_shape}")
            print(f"CNN Architecture: {nn.get_architecture()}")
        else:
            print(f"Architecture: {info.default_architecture}")
        print(f"Category: {info.category}")
        print(f"{'='*60}")

        # Notify frontend
        socketio.emit('problem_changed', {
            'problem_id': problem_id,
            'info': {
                'id': info.id,
                'name': info.name,
                'description': info.description,
                'category': info.category,
                'default_architecture': info.default_architecture,
                'input_labels': info.input_labels,
                'output_labels': info.output_labels,
                'output_activation': info.output_activation,
                'network_type': info.network_type,
                'input_shape': info.input_shape,
                # Educational fields
                'difficulty': info.difficulty,
                'concept': info.concept,
                'learning_goal': info.learning_goal,
                'tips': info.tips,
                # Level and failure case fields
                'level': info.level,
                'is_failure_case': info.is_failure_case,
                'failure_reason': info.failure_reason,
                'fix_suggestion': info.fix_suggestion,
                'locked_architecture': info.locked_architecture,
                'forced_weight_init': info.forced_weight_init,
                'forced_learning_rate': info.forced_learning_rate,
            }
        })

        return jsonify({
            "success": True,
            "problem_id": problem_id,
            "network_type": info.network_type,
            "architecture": nn.get_architecture()
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 404


# Learning Paths API
@app.route('/api/paths', methods=['GET'])
def get_learning_paths():
    """Get all learning paths."""
    from learning_paths import get_all_paths
    try:
        paths = get_all_paths()
        return jsonify(paths)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/paths/<path_id>', methods=['GET'])
def get_learning_path(path_id: str):
    """Get specific learning path with detailed steps."""
    from learning_paths import get_path
    try:
        path = get_path(path_id)
        if path is None:
            return jsonify({"error": "Path not found"}), 404
        return jsonify(path)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/paths/<path_id>/reset', methods=['POST'])
def reset_learning_path(path_id: str):
    """Reset learning path progress. Returns confirmation for frontend to clear localStorage."""
    from learning_paths import get_path
    try:
        path = get_path(path_id)
        if path is None:
            return jsonify({"error": "Path not found"}), 404
        # Progress is stored client-side in localStorage
        # This endpoint validates the path exists and confirms reset is allowed
        return jsonify({
            "success": True,
            "message": f"Path '{path['name']}' can be reset",
            "pathId": path_id
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/network', methods=['GET'])
def get_network():
    """Get neural network architecture and weights."""
    # Sample history to max 500 points for performance while keeping full range
    loss_hist = nn.loss_history if nn.loss_history else []
    acc_hist = nn.accuracy_history if nn.accuracy_history else []
    total_epochs = len(loss_hist)

    if len(loss_hist) > 500:
        step = len(loss_hist) // 500
        loss_hist = [loss_hist[i] for i in range(0, len(loss_hist), step)]
        acc_hist = [acc_hist[i] for i in range(0, len(acc_hist), step)]

    return jsonify({
        "architecture": nn.get_architecture(),
        "weights": nn.get_weights(),
        "loss_history": loss_hist,
        "accuracy_history": acc_hist,
        "total_epochs": total_epochs,
        "network_type": current_network_type
    })


@app.route('/api/network/architecture', methods=['POST'])
def set_architecture():
    """Set new network architecture and/or settings."""
    global nn

    # Validate request body
    data, err = get_json_body()
    if err:
        return jsonify({"error": err}), 400

    info = current_problem.info
    default_arch = info.default_architecture

    # Check if architecture is locked (failure case problems)
    if info.locked_architecture:
        return jsonify({
            "error": f"Architecture is locked for this problem. {info.failure_reason or 'This is an intentional failure case to demonstrate a concept.'}"
        }), 400

    # Get architecture (required or use current)
    layer_sizes = data.get('layer_sizes', nn.layer_sizes if nn else default_arch)

    # Get optional settings (use current values as defaults)
    weight_init = data.get('weight_init', nn.weight_init if nn else 'xavier')
    hidden_activation = data.get('hidden_activation', nn.hidden_activation if nn else 'relu')
    use_biases = data.get('use_biases', nn.use_biases if nn else True)

    # Apply forced settings for failure case problems
    if info.forced_weight_init:
        weight_init = info.forced_weight_init

    # Validate against current problem
    expected_input = len(info.input_labels)
    expected_output = len(info.output_labels)

    if len(layer_sizes) < 2:
        return jsonify({"error": "Need at least input and output layers"}), 400
    if layer_sizes[0] != expected_input:
        return jsonify({"error": f"Input layer must have {expected_input} neurons for {info.name}"}), 400
    if layer_sizes[-1] != expected_output:
        return jsonify({"error": f"Output layer must have {expected_output} neurons for {info.name}"}), 400

    # Validate settings
    valid_inits = ['xavier', 'he', 'random', 'zeros']
    valid_activations = ['relu', 'sigmoid', 'tanh']

    if weight_init not in valid_inits:
        return jsonify({"error": f"weight_init must be one of: {valid_inits}"}), 400
    if hidden_activation not in valid_activations:
        return jsonify({"error": f"hidden_activation must be one of: {valid_activations}"}), 400

    nn = NeuralNetwork(
        layer_sizes,
        output_activation=info.output_activation,
        weight_init=weight_init,
        hidden_activation=hidden_activation,
        use_biases=use_biases
    )
    with state_lock:
        system_state["training_complete"] = False

    print(f"\nNetwork configuration changed:")
    print(f"  Architecture: {layer_sizes}")
    print(f"  Weight init: {weight_init}")
    print(f"  Hidden activation: {hidden_activation}")
    print(f"  Use biases: {use_biases}")

    return jsonify({
        "success": True,
        "architecture": nn.get_architecture()
    })


@app.route('/api/train', methods=['POST'])
def start_training():
    """Start static training (G4)."""
    with state_lock:
        if system_state["training_in_progress"]:
            return jsonify({"error": "Training already in progress"}), 400

    # Validate request body
    data, err = get_json_body()
    if err:
        return jsonify({"error": err}), 400

    # Validate epochs
    epochs, err = validate_int(data.get('epochs', 1000), 'epochs', min_val=1, max_val=100000)
    if err:
        return jsonify({"error": err}), 400

    # Validate learning rate
    learning_rate, err = validate_float(data.get('learning_rate', 0.1), 'learning_rate', min_val=0.0001, max_val=10.0)
    if err:
        return jsonify({"error": err}), 400

    # Apply forced learning rate for failure case problems
    info = current_problem.info
    forced_lr = None
    if info.forced_learning_rate is not None:
        forced_lr = info.forced_learning_rate
        learning_rate = forced_lr

    # Run in background thread
    thread = threading.Thread(
        target=run_training,
        args=("static", epochs, learning_rate)
    )
    thread.start()

    response = {
        "success": True,
        "mode": "static",
        "epochs": epochs,
        "learning_rate": learning_rate
    }
    if forced_lr is not None:
        response["forced_learning_rate"] = True
        response["reason"] = info.failure_reason

    return jsonify(response)


@app.route('/api/train/adaptive', methods=['POST'])
def start_adaptive_training():
    """Start adaptive training (VG9)."""
    with state_lock:
        if system_state["training_in_progress"]:
            return jsonify({"error": "Training already in progress"}), 400

    # Validate request body
    data, err = get_json_body()
    if err:
        return jsonify({"error": err}), 400

    # Validate target_accuracy if provided
    target_accuracy, err = validate_float(data.get('target_accuracy'), 'target_accuracy', min_val=0.5, max_val=1.0)
    if err:
        return jsonify({"error": err}), 400

    # Check for forced learning rate (failure case problems)
    info = current_problem.info
    forced_lr = None
    if info.forced_learning_rate is not None:
        forced_lr = info.forced_learning_rate

    # Run in background thread
    thread = threading.Thread(
        target=run_training,
        args=("adaptive",),
        kwargs={"target_accuracy": target_accuracy, "forced_lr": forced_lr}
    )
    thread.start()

    response = {
        "success": True,
        "mode": "adaptive",
        "target_accuracy": target_accuracy
    }
    if forced_lr is not None:
        response["forced_learning_rate"] = forced_lr
        response["reason"] = info.failure_reason

    return jsonify(response)


@app.route('/api/train/stop', methods=['POST'])
def stop_training():
    """Stop training early (user satisfied with current accuracy)."""
    with state_lock:
        if not system_state["training_in_progress"]:
            return jsonify({"error": "No training in progress"}), 400
        system_state["stop_requested"] = True
    print("\nStop training requested by user...")

    return jsonify({
        "success": True,
        "message": "Stop requested, training will stop after current epoch"
    })


@app.route('/api/train/target', methods=['POST'])
def update_target_accuracy():
    """Update target accuracy during training."""
    with state_lock:
        if not system_state["training_in_progress"]:
            return jsonify({"error": "No training in progress"}), 400

    # Validate request body
    data, err = get_json_body()
    if err:
        return jsonify({"error": err}), 400

    # Validate target_accuracy (required)
    raw_target = data.get('target_accuracy')
    if raw_target is None:
        return jsonify({"error": "target_accuracy is required"}), 400

    new_target, err = validate_float(raw_target, 'target_accuracy', min_val=0.5, max_val=1.0)
    if err:
        return jsonify({"error": err}), 400

    with state_lock:
        old_target = system_state["target_accuracy"]
        system_state["target_accuracy"] = new_target

    print(f"\nTarget accuracy changed: {old_target*100:.0f}% -> {new_target*100:.0f}%")

    return jsonify({
        "success": True,
        "old_target": old_target,
        "new_target": new_target
    })


@app.route('/api/train/step', methods=['POST'])
def train_step():
    """Run a single training epoch (for step-by-step learning)."""
    with state_lock:
        if system_state["training_in_progress"]:
            return jsonify({"error": "Training already in progress"}), 400

    # Validate request body
    data, err = get_json_body()
    if err:
        return jsonify({"error": err}), 400

    # Validate learning rate
    learning_rate, err = validate_float(data.get('learning_rate', 0.1), 'learning_rate', min_val=0.0001, max_val=10.0)
    if err:
        return jsonify({"error": err}), 400

    # Validate batch_size if provided
    batch_size, err = validate_int(data.get('batch_size'), 'batch_size', min_val=1)
    if err:
        return jsonify({"error": err}), 400

    # Apply forced learning rate for failure case problems
    info = current_problem.info
    if info.forced_learning_rate is not None:
        learning_rate = info.forced_learning_rate

    # Run single epoch synchronously (fast enough)
    nn.learning_rate = learning_rate

    # Get batch
    if batch_size and batch_size < len(X_train):
        indices = np.random.choice(len(X_train), batch_size, replace=False)
        X_batch = X_train[indices]
        y_batch = y_train[indices]
    else:
        X_batch = X_train
        y_batch = y_train

    # Forward pass
    output, activations, z_values = nn.forward(X_batch)

    # Calculate loss and accuracy
    if nn.output_activation == 'softmax':
        loss = nn.cross_entropy_loss(output, y_batch)
    else:
        loss = nn.mse_loss(output, y_batch)
    accuracy = nn.calculate_accuracy(X_train, y_train)  # Full dataset accuracy

    # Record history
    nn.loss_history.append(loss)
    nn.accuracy_history.append(accuracy)

    # Backward pass
    nn.backward(X_batch, y_batch, activations, z_values)

    # Update state
    with state_lock:
        system_state["current_epoch"] = len(nn.loss_history)
        system_state["current_loss"] = loss
        system_state["current_accuracy"] = accuracy
        system_state["training_complete"] = True  # Allow predictions after any training

    # Emit progress
    socketio.emit('training_progress', {
        "epoch": len(nn.loss_history),
        "loss": loss,
        "accuracy": accuracy
    })

    return jsonify({
        "success": True,
        "epoch": len(nn.loss_history),
        "loss": loss,
        "accuracy": accuracy,
        "batch_size": batch_size or len(X_train)
    })


@app.route('/api/network/reset', methods=['POST'])
def reset_network():
    """Reset network weights and training state."""
    nn.reset()
    with state_lock:
        system_state["training_complete"] = False
        system_state["training_in_progress"] = False  # Force reset stuck training
        system_state["current_epoch"] = 0
        system_state["current_loss"] = 0.0
        system_state["current_accuracy"] = 0.0

    print("\nNetwork weights reset")

    # Notify clients of reset - this resets the training progress display
    socketio.emit('network_reset', {
        "training_complete": False,
        "training_in_progress": False,
        "current_problem": current_problem_id,
        "network_type": current_network_type
    })

    return jsonify({"success": True})


@app.route('/api/decision-boundary', methods=['GET'])
def get_decision_boundary():
    """
    Get decision boundary visualization data for 2D problems.
    Returns a grid of predictions for visualizing how the network classifies 2D space.
    """
    with state_lock:
        if not system_state["training_complete"]:
            return jsonify({"error": "Training not complete"}), 400

    info = current_problem.info

    # Only works for 2D input problems
    if len(info.input_labels) != 2:
        return jsonify({"error": "Decision boundary only available for 2D input problems"}), 400

    # Get grid resolution from query params (default 50x50)
    resolution = request.args.get('resolution', 50, type=int)
    resolution = min(max(resolution, 10), 100)  # Clamp between 10 and 100

    # Get range from query params (default -1 to 1)
    x_min = request.args.get('x_min', -1.0, type=float)
    x_max = request.args.get('x_max', 1.0, type=float)
    y_min = request.args.get('y_min', -1.0, type=float)
    y_max = request.args.get('y_max', 1.0, type=float)

    # Create grid of points
    x_range = np.linspace(x_min, x_max, resolution)
    y_range = np.linspace(y_min, y_max, resolution)

    # Generate all (x, y) combinations
    grid_points = []
    for y in y_range:
        for x in x_range:
            grid_points.append([x, y])

    X_grid = np.array(grid_points)

    # Get predictions for all points
    predictions = nn.predict(X_grid)

    # Reshape predictions to grid format
    if info.output_activation == 'softmax':
        # Multi-class: get the predicted class for each point
        pred_grid = np.argmax(predictions, axis=1).reshape(resolution, resolution)
        # Also get confidence (max probability)
        confidence_grid = np.max(predictions, axis=1).reshape(resolution, resolution)
    else:
        # Binary: get probability
        pred_grid = predictions[:, 0].reshape(resolution, resolution)
        confidence_grid = pred_grid  # For binary, prediction IS confidence

    return jsonify({
        "predictions": pred_grid.tolist(),
        "confidence": confidence_grid.tolist(),
        "x_range": x_range.tolist(),
        "y_range": y_range.tolist(),
        "resolution": resolution,
        "problem_id": current_problem_id,
        "category": info.category,
        "output_labels": info.output_labels,
        "training_data": {
            "inputs": X_train.tolist(),
            "labels": y_train.tolist()
        }
    })


# -----------------------------------------------------------------------------
# Visualization Endpoints
# -----------------------------------------------------------------------------

@app.route('/api/loss-landscape', methods=['POST'])
def get_loss_landscape():
    """
    Sample actual loss values along 2 random directions in weight space.
    Returns a grid of loss values for 3D visualization of the loss landscape.
    """
    if current_network_type != 'dense':
        return jsonify({"error": "Loss landscape only available for dense networks"}), 400

    # Validate request body
    data, err = get_json_body()
    if err:
        return jsonify({"error": err}), 400

    resolution = data.get('resolution', 20)
    range_val = data.get('range', 1.0)

    # Clamp values
    resolution = min(max(resolution, 5), 40)
    range_val = min(max(range_val, 0.1), 5.0)

    # Get current weights as center point
    center_weights = nn.get_flat_weights()

    # Generate 2 random directions (normalized)
    np.random.seed(42)  # For reproducibility in same session
    dir1 = np.random.randn(len(center_weights))
    dir1 = dir1 / np.linalg.norm(dir1)
    dir2 = np.random.randn(len(center_weights))
    # Orthogonalize dir2 with respect to dir1
    dir2 = dir2 - np.dot(dir2, dir1) * dir1
    dir2 = dir2 / np.linalg.norm(dir2)

    # Sample grid
    losses = []
    alphas = []
    betas = []

    for i in range(resolution):
        row = []
        alpha = (i / (resolution - 1) - 0.5) * 2 * range_val
        alphas.append(alpha)
        for j in range(resolution):
            beta = (j / (resolution - 1) - 0.5) * 2 * range_val
            if i == 0:
                betas.append(beta)
            test_weights = center_weights + alpha * dir1 + beta * dir2
            nn.set_flat_weights(test_weights)
            loss = nn.compute_loss(X_train, y_train)
            row.append(float(loss))
        losses.append(row)

    # Restore original weights
    nn.set_flat_weights(center_weights)

    center_idx = resolution // 2
    return jsonify({
        'losses': losses,
        'resolution': resolution,
        'range': range_val,
        'center_loss': losses[center_idx][center_idx],
        'alphas': alphas,
        'betas': betas
    })


@app.route('/api/gradcam', methods=['POST'])
def get_gradcam():
    """
    Compute Grad-CAM heatmap for CNN input.
    Returns a heatmap showing which parts of the input were most important.
    """
    if current_network_type != 'cnn':
        return jsonify({"error": "Grad-CAM only available for CNN networks"}), 400

    with state_lock:
        if not system_state["training_complete"]:
            return jsonify({"error": "Training not complete"}), 400

    # Validate request body
    data, err = get_json_body()
    if err:
        return jsonify({"error": err}), 400

    inputs = data.get('inputs')
    target_class = data.get('target_class')

    if inputs is None:
        return jsonify({"error": "inputs is required"}), 400

    # Convert to numpy array
    info = current_problem.info
    grid = np.array(inputs, dtype=float)
    if grid.ndim == 2:
        grid = grid[:, :, np.newaxis]  # Add channel dimension
    X = grid[np.newaxis, :, :, :]  # Add batch dimension

    # Compute Grad-CAM
    heatmap = nn.compute_gradcam(X, target_class)

    if heatmap is None:
        return jsonify({"error": "Could not compute Grad-CAM (no conv layers found)"}), 400

    # Get prediction for context
    output = nn.predict(X)
    predicted_class = int(np.argmax(output[0]))
    probabilities = output[0].tolist()

    return jsonify({
        'heatmap': heatmap,
        'input_shape': list(X.shape[1:3]),
        'target_class': target_class if target_class is not None else predicted_class,
        'predicted_class': predicted_class,
        'probabilities': probabilities,
        'output_labels': info.output_labels
    })


@app.route('/api/decision-surface', methods=['GET'])
def get_decision_surface():
    """
    Get 3D decision surface data for 2D input problems.
    Returns prediction probabilities as a height map.
    """
    with state_lock:
        if not system_state["training_complete"]:
            return jsonify({"error": "Training not complete"}), 400

    info = current_problem.info

    # Only works for 2D input problems
    if len(info.input_labels) != 2:
        return jsonify({"error": "Decision surface only available for 2D input problems"}), 400

    resolution = request.args.get('resolution', 30, type=int)
    resolution = min(max(resolution, 10), 60)

    # Generate grid
    x = np.linspace(-1.5, 1.5, resolution)
    y = np.linspace(-1.5, 1.5, resolution)
    X_grid = np.array([[xi, yi] for yi in y for xi in x])

    predictions = nn.predict(X_grid)

    # For binary: probability of class 1
    # For multi-class: max probability
    if info.output_activation == 'softmax':
        surface = np.max(predictions, axis=1)
        pred_classes = np.argmax(predictions, axis=1).reshape(resolution, resolution)
    else:
        surface = predictions[:, 0]
        pred_classes = (surface >= 0.5).astype(int).reshape(resolution, resolution)

    surface = surface.reshape(resolution, resolution)

    return jsonify({
        'surface': surface.tolist(),
        'classes': pred_classes.tolist(),
        'x_range': [-1.5, 1.5],
        'y_range': [-1.5, 1.5],
        'resolution': resolution,
        'problem_id': current_problem_id,
        'category': info.category,
        'output_labels': info.output_labels,
        'training_data': {
            'inputs': X_train[:, :2].tolist(),
            'labels': y_train.tolist()
        }
    })


# -----------------------------------------------------------------------------
# WebSocket Events
# -----------------------------------------------------------------------------

@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    print(f"Client connected")
    info = current_problem.info
    with state_lock:
        emit('status', {
            "connected": True,
            "training_complete": system_state["training_complete"],
            "training_in_progress": system_state["training_in_progress"],
            "current_problem": current_problem_id,
            "network_type": current_network_type
        })
    emit('problem_info', {
        'id': info.id,
        'name': info.name,
        'description': info.description,
        'category': info.category,
        'default_architecture': info.default_architecture,
        'input_labels': info.input_labels,
        'output_labels': info.output_labels,
        'output_activation': info.output_activation,
        'network_type': info.network_type,
        'input_shape': info.input_shape,
        # Educational fields
        'difficulty': info.difficulty,
        'concept': info.concept,
        'learning_goal': info.learning_goal,
        'tips': info.tips,
        # Level and failure case fields
        'level': info.level,
        'is_failure_case': info.is_failure_case,
        'failure_reason': info.failure_reason,
        'fix_suggestion': info.fix_suggestion,
        'locked_architecture': info.locked_architecture,
        'forced_weight_init': info.forced_weight_init,
        'forced_learning_rate': info.forced_learning_rate,
    })


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    print(f"Client disconnected")


@socketio.on('set_inputs')
def handle_set_inputs(data):
    """Handle input change from frontend (generic for all problems)."""
    inputs = data.get('inputs', [])
    result = make_prediction(inputs)
    emit('prediction', result, broadcast=True)


# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------

def startup_training():
    """Run training on startup (as per assignment requirement)."""
    print("\n" + "=" * 80)
    print("Neural Network Learning Lab - Starting Up")
    print("=" * 80)
    info = current_problem.info
    print(f"\nCurrent problem: {info.name}")
    print(f"Category: {info.category}")
    print(f"Architecture: {info.default_architecture}")
    print("\nTraining starts automatically on system boot...")
    print("No predictions until training is complete.\n")

    # Wait a bit for Flask to start
    time.sleep(1)

    # Run adaptive training on startup
    run_training("adaptive")


if __name__ == '__main__':
    # Start training in background thread
    training_thread = threading.Thread(target=startup_training)
    training_thread.daemon = True
    training_thread.start()

    # Run Flask with SocketIO
    print("\nStarting Flask server on http://localhost:5000")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)
