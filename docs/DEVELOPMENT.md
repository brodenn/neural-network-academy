# Development Guide

Developer documentation for Neural Network Academy.

## Project Overview

Neural Network Academy is a comprehensive neural network education platform featuring **32 progressive learning problems** across **7 difficulty levels**. It implements feedforward and convolutional neural networks from scratch using only NumPy (no ML frameworks) and provides an interactive web dashboard with real-time visualization.

**School Project**: Maskininlärning - Projekt II (Variant 2)

**Key Features:**
- 32 problems from basic gates to CNNs
- Pure NumPy implementation (educational first principles)
- Interactive visualization (network diagrams, decision boundaries, 3D loss landscapes)
- Adaptive training with automatic LR adjustment
- Failure case demonstrations (teaches common pitfalls)
- Comprehensive testing (Pytest + Playwright E2E)

## Development Commands

### Backend (Flask + WebSocket)
```bash
cd backend
pip install -r requirements.txt
python app.py                    # Starts on http://localhost:5000
python neural_network.py         # Run standalone NN demo
python problems.py               # Demo all problem types
```

### Frontend (React + Vite + Tailwind)
```bash
cd frontend
npm install
npm run dev                      # Starts on http://localhost:5173
npm run build                    # Production build
npm run lint                     # ESLint check
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend (React)                          │
│  App.tsx orchestrates all components (38 total)                  │
│  ├── ProblemSelector - Navigate 32 problems across 7 levels     │
│  ├── InputPanel - Adaptive controls (1D/2D/CNN canvas)          │
│  ├── NetworkVisualization - SVG network diagram (dense + CNN)   │
│  ├── CNNEducationalViz - CNN feature map visualization          │
│  ├── TrainingPanel - Training controls (static/adaptive/step)   │
│  ├── LossCurve - Loss/accuracy charts (Recharts)                │
│  ├── DecisionBoundaryViz - 2D decision boundary plotting        │
│  ├── LossLandscape3D - 3D loss surface (Three.js)               │
│  └── OutputDisplay - Prediction results                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                   WebSocket (socket.io-client)
                              │
┌─────────────────────────────────────────────────────────────────┐
│                        Backend (Flask)                           │
│  app.py - REST API + WebSocket server                            │
│  ├── neural_network.py - Pure NumPy dense NN                    │
│  │     ├── Forward propagation with ReLU/Sigmoid/Softmax        │
│  │     ├── Backpropagation with gradient descent                │
│  │     ├── Static training (user-defined epochs/LR)             │
│  │     └── Adaptive training (auto LR + restarts)               │
│  ├── cnn_network.py - Pure NumPy CNN implementation             │
│  │     ├── Conv2D, MaxPool2D, Flatten layers                    │
│  │     └── Feature map extraction for visualization             │
│  ├── problems.py - 32 problem types (7 difficulty levels)       │
│  │     ├── Level 1-2: Logic gates, XOR, parity                  │
│  │     ├── Level 3-4: 2D boundaries, regression                 │
│  │     ├── Level 5: Failure cases (educational)                 │
│  │     ├── Level 6: Multi-class classification                  │
│  │     └── Level 7: CNNs (shapes, digits)                       │
└─────────────────────────────────────────────────────────────────┘
```

## Key Design Decisions

### Neural Network (neural_network.py)
- **Pure NumPy**: No TensorFlow/PyTorch - implements backpropagation from first principles
- **Xavier/He initialization**: Supports multiple weight init strategies
- **Activation functions**: ReLU (hidden), Sigmoid/Softmax (output)
- **Adaptive training**: Auto-adjusts learning rate and restarts if stuck

### Problem System (problems.py)
**32 progressive learning problems** organized into 7 educational levels:

**Level 1 - Single Neuron (Linear Separability):**
- AND, OR, NOT, NAND gates - teaches what single neurons can/cannot do

**Level 2 - Hidden Layers Required:**
- XOR, XNOR, 5-bit parity - demonstrates why hidden layers are necessary

**Level 3 - 2D Decision Boundaries:**
- Two Blobs, Moons, Circle, Donut, Spiral - visualizing non-linear boundaries

**Level 4 - Regression:**
- Linear, Sine Wave, Polynomial, 2D Surface - continuous function approximation

**Level 5 - Failure Cases (Educational):**
- XOR with no hidden layer, zero init, bad LR (high/low), vanishing gradients, underfitting
- **Intentionally designed to fail** to teach common pitfalls

**Level 6 - Multi-Class Classification:**
- Quadrant (4 classes), Gaussian Blobs (5 classes), Color RGB (6 classes), Signal Patterns

**Level 7 - CNNs:**
- Shape Detection (circles, squares, triangles), Digit Recognition (0-9 on 8×8 grids)

Each problem includes educational metadata: `difficulty`, `concept`, `learning_goal`, `tips`, and for failure cases: `failure_reason`, `fix_suggestion`.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/problems` | GET | List all problems |
| `/api/problems/<id>/select` | POST | Switch active problem |
| `/api/network` | GET | Get architecture + weights |
| `/api/network/architecture` | POST | Change network config |
| `/api/train` | POST | Static training (epochs, LR) |
| `/api/train/adaptive` | POST | Adaptive training (~99% accuracy) |
| `/api/train/step` | POST | Single epoch (step-by-step mode) |

## WebSocket Events

| Event | Direction | Purpose |
|-------|-----------|---------|
| `training_progress` | Server→Client | Epoch updates (loss, accuracy) |
| `training_complete` | Server→Client | Training finished |
| `prediction` | Server→Client | Real-time prediction result |
| `problem_changed` | Server→Client | Problem switch notification |
| `set_inputs` | Client→Server | Update input values |

## Frontend Component State Flow

```
App.tsx manages:
├── currentProblem: ProblemInfo
├── networkData: architecture + weights + history
├── trainingState: {inProgress, complete, epoch, loss, accuracy}
└── prediction: {inputs, output, activations, correct}

Socket events update state automatically via useSocket hook
```

## Extending the Project

### Adding a New Problem Type
1. Create new class in `problems.py` extending `Problem`
2. Implement `info`, `generate_data()`, `generate_sample()`
3. Add to `PROBLEMS` registry at bottom of file

### Custom Network Configuration
POST to `/api/network/architecture`:
```json
{
  "layer_sizes": [5, 16, 8, 1],
  "weight_init": "he",           // xavier, he, random, zeros
  "hidden_activation": "relu",   // relu, sigmoid, tanh
  "use_biases": true
}
```
