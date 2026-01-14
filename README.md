# XOR Learning Lab

**School Project:** Maskininlärning - Projekt II (Variant 2)

A neural network implemented from scratch using only NumPy, trained to recognize 5-bit XOR patterns on a Raspberry Pi with GPIO simulation.

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

## Project Overview

### What it does

The neural network learns the XOR function for 5 binary inputs:

```
Input: [0, 0, 0, 0, 0] → Output: 0 (even number of 1s)
Input: [0, 0, 0, 0, 1] → Output: 1 (odd number of 1s)
Input: [1, 1, 0, 0, 1] → Output: 1 (odd number of 1s)
Input: [1, 1, 1, 1, 1] → Output: 1 (odd number of 1s)
```

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
xor-learning-lab/
├── backend/
│   ├── app.py                 # Flask API + WebSocket
│   ├── neural_network.py      # Core NN implementation
│   ├── gpio_simulator.py      # Simulated Raspberry Pi GPIO
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   │   ├── App.tsx            # Main application
│   │   ├── components/
│   │   │   ├── ButtonSimulator.tsx    # 5 virtual buttons
│   │   │   ├── LedIndicator.tsx       # LED visualization
│   │   │   ├── TrainingPanel.tsx      # Training controls
│   │   │   ├── LossCurve.tsx          # Loss/accuracy chart
│   │   │   ├── NetworkVisualization.tsx
│   │   │   └── TerminalOutput.tsx     # Prediction log
│   │   └── hooks/
│   │       └── useSocket.ts           # WebSocket hook
│   └── package.json
│
└── README.md
```

---

## API Endpoints

```
GET  /api/status              # System status
GET  /api/network             # Network architecture + weights
POST /api/network/architecture # Change architecture
POST /api/train               # Start static training (G4)
POST /api/train/adaptive      # Start adaptive training (VG9)
POST /api/network/reset       # Reset weights
GET  /api/gpio                # GPIO state
POST /api/gpio/button/<id>    # Toggle button
GET  /api/predict             # Current prediction
```

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

---

## License

MIT
