# Neural Network Academy

An interactive educational platform for learning neural networks from first principles. Features **32 progressive problems**, **7 guided learning paths**, and hands-on challenges - all implemented from scratch using only NumPy.

**School Project:** Maskininlärning - Projekt II (Variant 2)

---

## Prerequisites

- **Python 3.8+** with pip
- **Node.js 18+** with npm

---

## Quick Start

```bash
# Terminal 1: Backend
cd backend
pip install -r requirements.txt
python app.py

# Terminal 2: Frontend
cd frontend
npm install
npm run dev
```

Open http://localhost:5173

---

## Features

| Feature | Description |
|---------|-------------|
| **Interactive Challenges** | Build networks with drag-and-drop, predict outcomes, debug broken configs |
| **7 Learning Paths** | Guided journeys from beginner to advanced with progress tracking |
| **32 Problems** | Logic gates → XOR → 2D boundaries → regression → CNNs |
| **Real-time Visualization** | Network diagrams, decision boundaries, loss curves, 3D landscapes |
| **Failure Cases** | Learn from intentional failures (bad LR, zero init, vanishing gradients) |
| **Achievement System** | Badges, milestones, confetti celebrations |
| **Pure NumPy** | No ML frameworks - educational from first principles |

---

## Learning Paths

| Path | Level | Steps | What You'll Learn |
|------|-------|-------|-------------------|
| **Interactive Fundamentals** | Beginner | 7 | Build, predict, debug - learn by doing |
| **Foundations** | Beginner | 7 | Single neurons, XOR, linear separability |
| **Deep Learning Basics** | Intermediate | 10 | Training, initialization, hyperparameters |
| **Multi-Class Mastery** | Intermediate | 4 | Softmax, one-hot encoding, multi-class |
| **Pitfall Prevention** | Intermediate | 6 | Common mistakes and how to avoid them |
| **Convolutional Vision** | Advanced | 3 | CNNs, shape detection, digit recognition |
| **Research Frontier** | Advanced | 4 | Complex 2D problems (spirals, donuts) |

---

## Problem Levels

1. **Level 1: Single Neuron** - AND, OR, NOT, NAND gates
2. **Level 2: Hidden Layers** - XOR, XNOR, 5-bit parity
3. **Level 3: 2D Boundaries** - Blobs, moons, circles, spirals
4. **Level 4: Regression** - Linear, sine, polynomial, surfaces
5. **Level 5: Failure Cases** - Bad LR, zero init, vanishing gradients
6. **Level 6: Multi-Class** - Quadrants, blobs, colors, patterns
7. **Level 7: CNNs** - Shape detection, digit recognition

---

## Interactive Challenge Types

**Build Challenges** - Design network architectures with drag-and-drop
- Drag hidden layers into place
- Configure neuron counts
- Learn which architectures solve which problems

**Prediction Quizzes** - Predict outcomes before training
- "What happens with no hidden layer on XOR?"
- "What if learning rate is 10?"
- Build intuition through prediction

**Debug Challenges** - Diagnose broken configurations
- See symptoms: "Loss explodes to NaN"
- Find the bug from multiple choices
- Apply the fix and verify

---

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Space` | Start training |
| `Escape` | Stop training |
| `R` | Reset network |
| `S` | Single training step |

---

## Project Structure

```
neural-network-academy/
├── backend/
│   ├── app.py                 # Flask API + WebSocket
│   ├── neural_network.py      # Pure NumPy neural network
│   ├── cnn_network.py         # Pure NumPy CNN
│   ├── problems.py            # 32 problem definitions
│   ├── learning_paths.py      # 7 learning paths
│   └── tests/                 # 360 pytest tests
│
├── frontend/
│   ├── src/
│   │   ├── components/        # 38 React components
│   │   ├── stores/            # Zustand state management
│   │   └── hooks/             # Custom React hooks
│   └── tests/                 # Playwright E2E tests
│
└── README.md
```

---

## API

### REST Endpoints
```
GET  /api/paths                # List learning paths
GET  /api/paths/<id>           # Get path with steps
GET  /api/problems             # List all problems
POST /api/problems/<id>/select # Switch problem
POST /api/train                # Start training
POST /api/train/adaptive       # Adaptive training
GET  /api/network              # Get network state
```

### WebSocket Events
```
training_progress    # Epoch updates (loss, accuracy)
training_complete    # Training finished
problem_changed      # Problem switched
prediction           # Real-time prediction
```

---

## Testing

```bash
# Backend (360 tests)
cd backend && pytest

# Frontend E2E
cd frontend && npx playwright test
```

---

## Tech Stack

- **Backend:** Python, Flask, Flask-SocketIO, NumPy
- **Frontend:** React, TypeScript, Vite, Tailwind CSS
- **Visualization:** Recharts, Three.js, Framer Motion
- **State:** Zustand
- **Drag & Drop:** @dnd-kit
- **Testing:** Pytest, Playwright

---

## Why NumPy Only?

This project implements neural networks from scratch without TensorFlow, PyTorch, or other ML frameworks. Every forward pass, backpropagation step, and gradient update is written in plain NumPy. This makes the learning experience transparent - you can trace exactly how the network learns.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend (React)                          │
│  ├── Network Visualization    Real-time SVG network diagram     │
│  ├── Decision Boundary        2D classification visualization   │
│  ├── Loss Landscape 3D        Three.js loss surface             │
│  ├── CNN Pipeline Viewer      Step-by-step conv/pool/flatten    │
│  └── Training Controls        Static, adaptive, step-by-step    │
└─────────────────────────────────────────────────────────────────┘
                              │
                    WebSocket (real-time updates)
                              │
┌─────────────────────────────────────────────────────────────────┐
│                        Backend (Flask)                           │
│  ├── neural_network.py        Dense networks (NumPy)            │
│  ├── cnn_network.py           Conv2D, MaxPool, Flatten (NumPy)  │
│  ├── problems.py              32 problem definitions            │
│  └── learning_paths.py        7 guided learning paths           │
└─────────────────────────────────────────────────────────────────┘
```

---

## Visualizations

| Visualization | Description |
|---------------|-------------|
| **Network Diagram** | Interactive SVG showing neurons, connections, and weight magnitudes |
| **Decision Boundary** | 2D plot showing how the network divides the input space |
| **Loss Curve** | Real-time loss and accuracy over training epochs |
| **3D Loss Landscape** | Three.js surface showing loss as a function of two weights |
| **CNN Pipeline** | Step-by-step view: padding → convolution → ReLU → pooling → flatten → dense |
| **Feature Maps** | Visual representation of what CNN filters detect |
| **Weight Histogram** | Distribution of weights across layers |

---

## Training Modes

**Static Training**
- Fixed epochs and learning rate
- Full control over hyperparameters
- Good for understanding the effect of each parameter

**Adaptive Training**
- Automatically adjusts learning rate
- Restarts with new weights if stuck in local minima
- Targets ~99% accuracy or converged loss

**Step-by-Step**
- Single epoch at a time
- Watch the network learn incrementally
- Perfect for understanding the training process

---

## Network Configuration

Configure network architecture via the UI or API:

```json
{
  "layer_sizes": [2, 8, 4, 1],
  "weight_init": "he",
  "hidden_activation": "relu",
  "use_biases": true
}
```

| Option | Values | Description |
|--------|--------|-------------|
| `layer_sizes` | Array of integers | Neurons per layer (input → hidden → output) |
| `weight_init` | `xavier`, `he`, `random`, `zeros` | Weight initialization strategy |
| `hidden_activation` | `relu`, `sigmoid`, `tanh` | Activation function for hidden layers |
| `use_biases` | `true`, `false` | Whether to include bias terms |

---

## Educational Philosophy

This platform teaches neural networks through **progressive complexity** and **learning from failure**:

1. **Start Simple** - Single neurons on logic gates (AND, OR) build intuition
2. **Hit Walls** - XOR fails with no hidden layer, teaching *why* depth matters
3. **See It Work** - Visualizations make abstract concepts concrete
4. **Break Things** - Failure cases (Level 5) show what happens with bad hyperparameters
5. **Build Intuition** - Prediction quizzes test understanding before training

---

## License

MIT
