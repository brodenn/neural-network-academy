# Neural Network Academy

**School Project:** Maskininlärning - Projekt II (Variant 2)

An educational neural network platform with **32 progressive learning problems** (from basic logic gates to CNNs), implemented from scratch using only NumPy. Features **guided learning paths**, interactive web-based visualization, adaptive training, real-time progress tracking, and embedded systems deployment for Raspberry Pi.

---

## Quick Start

```bash
# Terminal 1: Start backend
cd backend
pip install -r requirements.txt
python app.py

# Terminal 2: Start frontend
cd frontend
npm install
npm run dev
```

Open http://localhost:5173 in your browser.

---

## 🆕 What's New

**Interactive Learning Challenges (Latest):**
- 🎮 **Build Challenges** - Drag-and-drop network architecture builder
- 🤔 **Prediction Quizzes** - Predict outcomes before training to build intuition
- 🐛 **Debug Challenges** - Diagnose broken configurations from symptoms
- 🎯 **New "Interactive Fundamentals" Path** - Learn by DOING, not just watching

**Learning Paths Feature:**
- 🎓 7 guided learning paths with step-by-step journeys
- 📊 Real-time progress tracking with localStorage persistence
- 💡 Progressive hint system (unlocks based on attempts)
- 🎉 Celebration modal with animated badges and confetti
- 🔄 Resume from where you left off

**Enhanced Training Experience:**
- 📢 Training Narrator - Real-time insights during training
- 💥 Failure Dramatization - Visual effects for failure cases
- 📈 Enhanced Loss Curve - Tooltips and annotations
- 🏆 Achievement System - Earn badges for milestones

---

## ✨ Key Features

- 🎮 **Interactive Challenges** - Build networks, predict outcomes, debug problems
- 🎓 **7 Learning Paths** - Guided journeys from beginner to advanced
- 🧠 **32 Progressive Problems** - From basic gates to CNNs (7 difficulty levels)
- 🎨 **Interactive Visualization** - Real-time network diagram, decision boundaries, 3D loss landscapes
- 🎯 **Adaptive Training** - Auto-adjusts learning rate to reach ~99% accuracy
- 💡 **Failure Case Education** - Learn from intentional failures (bad LR, zero init, vanishing gradients)
- 📊 **Live Training Insights** - Real-time narrator explains what's happening
- ⌨️ **Keyboard Shortcuts** - Space (train), Escape (stop), R (reset), S (step)
- 🏆 **Achievement System** - Badges, milestones, progress tracking
- 📱 **Responsive Design** - Works on desktop and mobile
- 🔧 **Pure NumPy** - No ML frameworks, educational from first principles
- 🎮 **Embedded Support** - Runs on Raspberry Pi with GPIO

---

## Learning Paths 🎓

The platform now features **guided learning paths** that provide structured, step-by-step journeys through neural network concepts.

### Available Paths

**Interactive Fundamentals** (Beginner) ⭐ NEW
- 7 interactive steps: build, predict, debug, train
- Learn by DOING - not just watching
- Drag-and-drop architecture builder
- Prediction quizzes and debug challenges
- Badge: 🎮 "Active Learner"

**Foundations** (Beginner)
- 7 steps covering single neurons and XOR
- Learn linear separability and why hidden layers matter
- Badge: 🏆 "Foundation Scholar"

**Deep Learning Basics** (Intermediate)
- 10 steps on training, initialization, hyperparameters
- Includes failure case demonstrations
- Badge: 🧠 "Neural Navigator"

**Multi-Class Mastery** (Intermediate)
- 4 steps exploring multi-class classification
- Softmax, one-hot encoding, probability outputs
- Badge: 🎨 "Classifier Champion"

**Pitfall Prevention** (Intermediate)
- 6 steps teaching what NOT to do
- Learn from intentional failures
- Badge: 🛡️ "Error Expert"

**Convolutional Vision** (Advanced)
- 3 steps on CNNs for image data
- Shape detection and digit recognition
- Badge: 👁️ "Vision Virtuoso"

**Research Frontier** (Advanced)
- 4 steps tackling challenging problems
- Spirals, donuts, complex surfaces
- Badge: 🚀 "Research Pioneer"

### Screenshots

![Learning Path Selector](docs/screenshots/path-selector.png)
*Choose from curated learning paths with progress tracking*

![Path Detail View](docs/screenshots/path-detail.png)
*Step-by-step interface with hints and progress visualization*

![Completion Modal](docs/screenshots/completion.png)
*Celebrate your achievement with animated badge and confetti*

### How It Works

1. **Select a Path** - Click "Learning Paths" in the header
2. **Step-by-Step Progress** - Complete problems in sequence
3. **Auto-Unlock** - Next step unlocks when you reach required accuracy
4. **Hint System** - Unlock hints after multiple attempts (1 hint per 2 attempts)
5. **Progress Tracking** - Your progress persists in localStorage
6. **Celebration** - Complete all steps to earn your badge with confetti animation! 🎉

### Features

- ✅ **Progress Persistence** - Pick up where you left off
- ✅ **Visual Progress Bar** - See your journey at a glance
- ✅ **Smart Hints** - Get help when you need it
- ✅ **Completion Detection** - Auto-advances on success
- ✅ **Achievement Badges** - Earn rewards for completion
- ✅ **Failure Case Support** - Some steps teach by failing intentionally

---

## Project Overview

### What it does

This platform offers **32 learning problems** organized into **7 progressive difficulty levels**:

1. **Level 1: Single Neuron** - AND, OR, NOT, NAND gates
2. **Level 2: Hidden Layers Required** - XOR, XNOR, 5-bit parity
3. **Level 3: 2D Decision Boundaries** - Blobs, moons, circles, spirals
4. **Level 4: Regression** - Linear, sine wave, polynomial, 2D surfaces
5. **Level 5: Failure Cases** - Demonstrates common pitfalls (bad LR, zero init, vanishing gradients)
6. **Level 6: Multi-Class** - Quadrant classification, Gaussian blobs, color recognition
7. **Level 7: CNNs** - Shape detection, digit recognition on 8×8 grids

Each problem teaches specific neural network concepts through interactive visualization.

### System Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────┐
│  5 Buttons  │ ──▶ │  Neural Network  │ ──▶ │    LED      │
│  (GPIO in)  │     │  [5,12,8,4,1]    │     │  (GPIO out) │
└─────────────┘     └──────────────────┘     └─────────────┘
                            │
                            ▼
                    ┌──────────────────┐
                    │   Web Dashboard  │
                    │  (Visualization) │
                    └──────────────────┘
```

---

## Requirements Coverage

### G-Level (Pass)

| # | Requirement | Implementation |
|---|-------------|----------------|
| **G1** | NN from scratch (no ML libs) | `neural_network.py` - Pure NumPy |
| **G2** | Clean code, documented | Type hints, docstrings throughout |
| **G3** | 1 hidden layer, user-configurable | `NeuralNetwork([5, 8, 1])` |
| **G4** | Static training (user params) | `train(epochs=1000, lr=0.5)` |
| **G5** | Train 4-bit XOR | We do 5-bit for VG10 |
| **G6** | Terminal output on change | Prints prediction on button press |
| **G7** | GitHub + live demo | This repo |

### VG-Level (Higher Grade)

| # | Requirement | Implementation |
|---|-------------|----------------|
| **VG8** | Arbitrary hidden layers | `NeuralNetwork([5, 12, 8, 4, 1])` |
| **VG9** | Adaptive training to ~100% | `train_adaptive()` - auto LR adjustment + restarts |
| **VG10** | 5-bit XOR, 3-5 hidden layers | Default architecture has 3 hidden layers |

---

## Neural Network Theory

### Network Structure

```
Input Layer      Hidden Layers           Output Layer
(5 neurons)      (configurable)          (1 neuron)

    ○              ○    ○    ○              ○
    ○ ─────────▶   ○    ○    ○  ─────────▶  │
    ○              ○    ○    ○              LED ON/OFF
    ○              ○    ○    ○
    ○              ○    ○
                   ○    ○

  Buttons        ReLU  ReLU  ReLU         Sigmoid
```

### Key Concepts

1. **Forward Propagation**: Data flows through layers via matrix multiplication
2. **Activation Functions**: ReLU (hidden), Sigmoid (output)
3. **Backpropagation**: Compute gradients using chain rule
4. **Gradient Descent**: Update weights to minimize loss

### Why XOR needs hidden layers

XOR is not linearly separable - you cannot draw a straight line to separate the classes. This is why a single-layer perceptron cannot learn XOR, proving the need for hidden layers.

```
Standard XOR (2 inputs):     Linear attempt (fails):

  0,1 ● ─ ─ ─ ● 1,1            0,1 ●       ● 1,1
      │       │                    │ ╲   ╱ │
      │  XOR  │                    │  ╲ ╱  │  ← No single line works!
      │       │                    │ ╱ ╲   │
  0,0 ● ─ ─ ─ ● 1,0            0,0 ●       ● 1,0
```

---

## Project Structure

```
neural-network-academy/
├── backend/
│   ├── app.py                 # Flask API + WebSocket server
│   ├── neural_network.py      # Pure NumPy dense network
│   ├── cnn_network.py         # Pure NumPy CNN implementation
│   ├── cnn_layers.py          # Conv2D, MaxPool2D, Flatten layers
│   ├── problems.py            # 32 problem definitions (7 levels)
│   ├── learning_paths.py      # Learning path definitions
│   ├── gpio_simulator.py      # Raspberry Pi GPIO simulation
│   ├── requirements.txt
│   └── tests/                 # Comprehensive pytest suite
│
├── frontend/
│   ├── src/
│   │   ├── App.tsx            # Main orchestrator
│   │   ├── components/        # 28 specialized components
│   │   │   ├── ProblemSelector.tsx          # Level/problem navigation
│   │   │   ├── LearningPathSelector.tsx     # Path selection grid
│   │   │   ├── LearningPathCard.tsx         # Path card with progress
│   │   │   ├── PathDetailView.tsx           # Step-by-step path interface
│   │   │   ├── PathProgressBar.tsx          # Visual step progress
│   │   │   ├── PathStepCard.tsx             # Step info display
│   │   │   ├── StepHintPanel.tsx            # Progressive hint reveal
│   │   │   ├── PathCompletionModal.tsx      # Celebration screen
│   │   │   ├── InputPanel.tsx               # Adaptive input controls
│   │   │   ├── NetworkVisualization.tsx     # SVG network diagram
│   │   │   ├── CNNEducationalViz.tsx        # CNN feature maps
│   │   │   ├── TrainingPanel.tsx            # Training controls
│   │   │   ├── LossCurve.tsx                # Recharts integration
│   │   │   ├── DecisionBoundaryViz.tsx      # 2D boundary plotting
│   │   │   ├── LossLandscape3D.tsx          # 3D loss surface (Three.js)
│   │   │   └── ... (13 more components)
│   │   └── hooks/
│   │       ├── useSocket.ts                 # WebSocket management
│   │       ├── usePathProgress.ts           # Progress tracking
│   │       └── useKeyboardShortcuts.ts      # Keyboard controls
│   ├── tests/                 # Playwright E2E tests (109 tests)
│   └── package.json
│
└── README.md
```

---

## API Endpoints

### Core Endpoints
```
GET  /api/status              # System status
GET  /api/network             # Network architecture + weights
POST /api/network/architecture # Change architecture
POST /api/train               # Start static training (G4)
POST /api/train/adaptive      # Start adaptive training (VG9)
POST /api/train/stop          # Stop training
POST /api/train/step          # Single epoch (step-by-step)
POST /api/network/reset       # Reset weights
GET  /api/gpio                # GPIO state
POST /api/gpio/button/<id>    # Toggle button
GET  /api/predict             # Current prediction
```

### Learning Paths API
```
GET  /api/paths               # List all learning paths
GET  /api/paths/<id>          # Get path details with steps
GET  /api/problems            # List all 32 problems
GET  /api/problems/<id>       # Get problem info
POST /api/problems/<id>/select # Switch to problem
```

### WebSocket Events
```
training_started              # Training begins
training_progress             # Epoch update (loss, accuracy)
training_complete             # Training finished
problem_changed               # Problem switched
prediction                    # Real-time prediction
```

---

## Testing

### Backend Tests (Pytest)
```bash
cd backend
pytest                        # Run all tests
pytest -v                     # Verbose output
pytest tests/test_network.py  # Specific test file
```

### Frontend Tests (Playwright E2E)
```bash
cd frontend
npm run test                  # Run E2E tests
npm run test:ui              # Interactive test UI
```

**Test Coverage:**
- ✅ Backend: 358 tests passing
- ✅ Frontend E2E: Interactive challenges (14/14)
- ✅ Frontend E2E: New features (9/9)
- ✅ Frontend E2E: Learning paths verified

---

## Running on Raspberry Pi

To run on actual Raspberry Pi hardware:

1. Install lgpio: `pip install lgpio`
2. Connect buttons to GPIO pins 17, 27, 22, 23, 24
3. Connect LED to GPIO pin 18
4. In `app.py`, replace `GPIOSimulator` with `GPIOHardware`

---

## Evaluation (Utvärdering)

### 1. Vad lärde ni er av projektet?

- Deep understanding of backpropagation by implementing it manually
- How matrix operations make neural networks efficient
- Why weight initialization (Xavier) matters for training convergence
- The importance of learning rate - too high causes instability, too low is slow
- Why XOR requires hidden layers (not linearly separable)
- How to design educational user experiences with guided learning paths
- Progressive enhancement - building complex features on solid foundations

### 2. Vad var lätt/svårt?

**Easy:**
- Setting up Flask API and React frontend
- Understanding forward propagation (just matrix multiplication)
- GPIO simulation

**Difficult:**
- Getting backpropagation correct (chain rule application)
- Debugging when the network wouldn't converge to 100%
- Understanding why different weight initializations give different results

### 3. Vad hade ni velat ha lärt er mer innan projektet?

- More about calculus and the chain rule
- Different optimization algorithms (Adam, momentum)
- How to visualize what the network is learning

### 4. Övriga kommentarer?

Building from scratch without ML libraries was challenging but educational. The interactive web frontend made it much easier to understand what's happening during training. Adaptive training with automatic restarts was key to reliably reaching 100% accuracy.

The addition of guided learning paths transforms this from a demonstration into a complete educational platform. Progress tracking, hint systems, and celebration animations make learning neural networks engaging and rewarding.

---

## License

MIT
