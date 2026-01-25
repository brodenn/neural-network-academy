# Neural Network Academy

An interactive educational platform for learning neural networks from first principles. Features **32 progressive problems**, **7 guided learning paths**, and hands-on challenges - all implemented from scratch using only NumPy.

**Try it:** Clone, run, and start learning in under 2 minutes.

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
│   └── tests/                 # 358 pytest tests
│
├── frontend/
│   ├── src/
│   │   ├── components/        # 41 React components
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
# Backend (358 tests)
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

## License

MIT
