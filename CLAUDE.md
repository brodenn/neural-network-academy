# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

XOR Learning Lab is a neural network learning tool for embedded systems education. It implements a feedforward neural network from scratch using only NumPy (no ML libraries) and provides an interactive web dashboard for visualization and training.

**School Project**: Maskininlärning - Projekt II (Variant 2)

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
│  App.tsx orchestrates all components                             │
│  ├── ProblemSelector - Switch between 5 problem types           │
│  ├── InputPanel - Problem-specific input controls               │
│  ├── NetworkVisualization - SVG network diagram                 │
│  ├── TrainingPanel - Training controls (static/adaptive)        │
│  ├── LossCurve - Loss/accuracy charts (Recharts)                │
│  └── OutputDisplay - Prediction results                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                   WebSocket (socket.io-client)
                              │
┌─────────────────────────────────────────────────────────────────┐
│                        Backend (Flask)                           │
│  app.py - REST API + WebSocket server                            │
│  ├── neural_network.py - Pure NumPy NN implementation           │
│  │     ├── Forward propagation with ReLU/Sigmoid/Softmax        │
│  │     ├── Backpropagation with gradient descent                │
│  │     ├── Static training (user-defined epochs/LR)             │
│  │     └── Adaptive training (auto LR + restarts)               │
│  ├── problems.py - 5 embedded systems problem types             │
│  └── gpio_simulator.py - Virtual Raspberry Pi GPIO              │
└─────────────────────────────────────────────────────────────────┘
```

## Key Design Decisions

### Neural Network (neural_network.py)
- **Pure NumPy**: No TensorFlow/PyTorch - implements backpropagation from first principles
- **Xavier/He initialization**: Supports multiple weight init strategies
- **Activation functions**: ReLU (hidden), Sigmoid/Softmax (output)
- **Adaptive training**: Auto-adjusts learning rate and restarts if stuck

### Problem System (problems.py)
Five embedded systems problems with distinct characteristics:
| Problem | Category | Architecture | Output |
|---------|----------|--------------|--------|
| XOR (5-bit) | binary | [5,12,8,4,1] | sigmoid |
| Sensor Fusion | regression | [2,8,4,1] | sigmoid |
| PWM Control | regression | [1,8,4,1] | sigmoid |
| Anomaly Detection | binary | [3,8,4,1] | sigmoid |
| Gesture Classification | multi-class | [8,12,8,3] | softmax |

### GPIO Simulation (gpio_simulator.py)
- `GPIOSimulator`: Virtual buttons/LED for development
- `GPIOHardware`: Real Raspberry Pi 5 implementation (requires lgpio)

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
| `/api/network/export/c` | GET | Export as C header file |
| `/api/input` | POST | Set inputs + get prediction |

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
