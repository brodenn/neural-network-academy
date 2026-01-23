# Implementation Guide: Neural Network Academy Features

**Date**: 2026-01-22
**Status**: Ready for Implementation
**Research**: Context7 + WebSearch verified

This guide provides concrete implementation steps with real code examples from Motion.dev and Recharts documentation.

---

## Quick Start: Which Feature First?

**Recommendation**: Start with **Learning Path Progress Visualization** (easiest, highest impact)

**Timeline**:
- **Day 1-2**: Learning path progress rings + path selector
- **Day 3-4**: Weight heatmap animation (Canvas)
- **Day 5-6**: Achievement system + challenges

---

## 1. Learning Path Progress Ring

### Implementation (Recharts PieChart)

Based on Context7 example from Recharts official docs:

```typescript
// components/PathProgressRing.tsx
import { PieChart, Pie, Cell, ResponsiveContainer } from 'recharts';

interface PathProgressRingProps {
  completed: number;
  total: number;
  size?: number;
}

export const PathProgressRing = ({
  completed,
  total,
  size = 120
}: PathProgressRingProps) => {
  const percentage = (completed / total) * 100;

  const data = [
    { name: 'Completed', value: completed, fill: '#10B981' }, // Green
    { name: 'Remaining', value: total - completed, fill: '#E5E7EB' } // Gray
  ];

  // Custom label showing percentage
  const renderLabel = ({ cx, cy }: any) => {
    return (
      <text
        x={cx}
        y={cy}
        fill="#1F2937"
        textAnchor="middle"
        dominantBaseline="central"
        className="font-bold text-2xl"
      >
        {Math.round(percentage)}%
      </text>
    );
  };

  return (
    <ResponsiveContainer width={size} height={size}>
      <PieChart>
        <Pie
          data={data}
          cx="50%"
          cy="50%"
          innerRadius={size * 0.6} // Creates the ring effect
          outerRadius={size * 0.8}
          startAngle={90}
          endAngle={-270} // Clockwise from top
          dataKey="value"
          label={renderLabel}
          labelLine={false}
        >
          {data.map((entry, index) => (
            <Cell key={`cell-${index}`} fill={entry.fill} />
          ))}
        </Pie>
      </PieChart>
    </ResponsiveContainer>
  );
};
```

**Usage**:
```tsx
<PathProgressRing completed={4} total={7} size={100} />
```

**Alternative: Lightweight CSS-Only** (no library):
```typescript
// components/CSSProgressRing.tsx
export const CSSProgressRing = ({
  completed,
  total
}: { completed: number; total: number }) => {
  const percentage = (completed / total) * 100;

  return (
    <div className="relative w-24 h-24">
      <div
        className="w-full h-full rounded-full"
        style={{
          background: `conic-gradient(
            #10B981 0% ${percentage}%,
            #E5E7EB ${percentage}% 100%
          )`
        }}
      >
        <div className="absolute inset-3 bg-white rounded-full flex items-center justify-center">
          <span className="text-xl font-bold">{Math.round(percentage)}%</span>
        </div>
      </div>
    </div>
  );
};
```

---

## 2. Weight Heatmap Animation

### Implementation (Motion.dev Spring Values)

Based on Context7 examples from Motion.dev official docs:

```typescript
// components/WeightHeatmapCanvas.tsx
import { useEffect, useRef, useState } from 'react';
import { springValue } from 'motion';

interface WeightHeatmapCanvasProps {
  weights: number[][][]; // [layer][from][to]
  width: number;
  height: number;
  updateInterval?: number; // ms between updates
}

export const WeightHeatmapCanvas = ({
  weights,
  width,
  height,
  updateInterval = 100
}: WeightHeatmapCanvasProps) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const springValuesRef = useRef<Map<string, any>>(new Map());

  // Initialize spring values for each weight
  useEffect(() => {
    weights.forEach((layer, l) => {
      layer.forEach((neuronWeights, i) => {
        neuronWeights.forEach((weight, j) => {
          const key = `${l}-${i}-${j}`;
          if (!springValuesRef.current.has(key)) {
            // Create spring value for this weight
            const spring = springValue(weight, {
              stiffness: 100,
              damping: 30,
              mass: 0.5
            });
            springValuesRef.current.set(key, spring);
          }
        });
      });
    });
  }, []);

  // Update spring values when weights change
  useEffect(() => {
    weights.forEach((layer, l) => {
      layer.forEach((neuronWeights, i) => {
        neuronWeights.forEach((weight, j) => {
          const key = `${l}-${i}-${j}`;
          const spring = springValuesRef.current.get(key);
          if (spring) {
            // Animate to new weight value
            spring.set(weight);
          }
        });
      });
    });
  }, [weights]);

  // Render loop
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let animationFrameId: number;

    const render = () => {
      ctx.clearRect(0, 0, width, height);

      const cellSize = 20;
      const padding = 2;

      weights.forEach((layer, l) => {
        const layerX = l * (layer[0].length * (cellSize + padding) + 40);

        layer.forEach((neuronWeights, i) => {
          neuronWeights.forEach((_, j) => {
            const key = `${l}-${i}-${j}`;
            const spring = springValuesRef.current.get(key);

            if (spring) {
              const value = spring.get(); // Get current animated value

              // Map weight to color (-1 to 1 -> blue to red)
              const color = getWeightColor(value);

              const x = layerX + j * (cellSize + padding);
              const y = i * (cellSize + padding);

              ctx.fillStyle = color;
              ctx.fillRect(x, y, cellSize, cellSize);

              // Optional: Add text for exact value on hover
              ctx.fillStyle = '#fff';
              ctx.font = '10px monospace';
              ctx.fillText(
                value.toFixed(2),
                x + 2,
                y + cellSize / 2 + 4
              );
            }
          });
        });

        // Layer label
        ctx.fillStyle = '#000';
        ctx.font = '12px sans-serif';
        ctx.fillText(
          `Layer ${l}→${l + 1}`,
          layerX,
          -10
        );
      });

      animationFrameId = requestAnimationFrame(render);
    };

    render();

    return () => {
      cancelAnimationFrame(animationFrameId);
    };
  }, [weights, width, height]);

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      className="border border-gray-300 rounded"
    />
  );
};

// Color mapping function
function getWeightColor(weight: number): string {
  // Normalize weight to 0-1 range (assuming weights are -1 to 1)
  const normalized = (weight + 1) / 2;

  if (normalized < 0.5) {
    // Blue (negative weights)
    const intensity = Math.floor((1 - normalized * 2) * 255);
    return `rgb(${intensity}, ${intensity}, 255)`;
  } else {
    // Red (positive weights)
    const intensity = Math.floor((normalized - 0.5) * 2 * 255);
    return `rgb(255, ${255 - intensity}, ${255 - intensity})`;
  }
}
```

**Usage**:
```tsx
const [weights, setWeights] = useState<number[][][]>([
  [[0.5, -0.2], [0.1, 0.8]], // Layer 1→2
  [[0.3, -0.5], [0.9, 0.1]]  // Layer 2→3
]);

// Update weights from WebSocket
socket.on('weight_update', (data) => {
  setWeights(data.weights);
});

<WeightHeatmapCanvas weights={weights} width={600} height={400} />
```

### Simpler Version: React Components with Framer Motion

```typescript
// components/WeightHeatmapSVG.tsx (for smaller networks)
import { motion } from 'framer-motion';

interface WeightCellProps {
  value: number;
  x: number;
  y: number;
  size: number;
}

const WeightCell = ({ value, x, y, size }: WeightCellProps) => {
  const color = getWeightColor(value);

  return (
    <motion.rect
      x={x}
      y={y}
      width={size}
      height={size}
      fill={color}
      initial={{ opacity: 0, scale: 0 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{
        type: 'spring',
        stiffness: 100,
        damping: 30
      }}
    />
  );
};

export const WeightHeatmapSVG = ({ weights }: { weights: number[][][] }) => {
  const cellSize = 20;
  const padding = 2;

  return (
    <svg width={800} height={400}>
      {weights.map((layer, l) =>
        layer.map((neuronWeights, i) =>
          neuronWeights.map((weight, j) => (
            <WeightCell
              key={`${l}-${i}-${j}`}
              value={weight}
              x={l * 200 + j * (cellSize + padding)}
              y={i * (cellSize + padding)}
              size={cellSize}
            />
          ))
        )
      )}
    </svg>
  );
};
```

---

## 3. Achievement System

### Implementation (react-achievements)

```bash
npm install react-achievements
```

```typescript
// contexts/AchievementContext.tsx
import React, { createContext, useContext } from 'react';
import { AchievementsProvider, useSimpleAchievements } from 'react-achievements';

const achievementsList = [
  {
    id: 'first-train',
    name: 'First Steps',
    description: 'Train your first neural network',
    icon: '🎓',
    points: 10
  },
  {
    id: 'xor-master',
    name: 'XOR Master',
    description: 'Achieve 99% accuracy on XOR',
    icon: '🧠',
    points: 50
  },
  {
    id: 'path-foundations',
    name: 'Foundation Scholar',
    description: 'Complete the Foundations learning path',
    icon: '🏆',
    points: 100
  },
  {
    id: 'perfect-score',
    name: 'Perfectionist',
    description: 'Achieve 100% accuracy',
    icon: '💎',
    points: 200
  },
  {
    id: 'speed-demon',
    name: 'Speed Demon',
    description: 'Train to 99% in under 100 epochs',
    icon: '⚡',
    points: 150
  }
];

export const AchievementWrapper = ({ children }: { children: React.ReactNode }) => {
  return (
    <AchievementsProvider achievements={achievementsList}>
      {children}
    </AchievementsProvider>
  );
};

// Hook to use in components
export const useAchievements = () => {
  const { achievements, unlock, isUnlocked } = useSimpleAchievements();
  return { achievements, unlock, isUnlocked };
};
```

**Usage in Training Component**:
```typescript
// components/TrainingPanel.tsx
import { useAchievements } from '@/contexts/AchievementContext';
import { useEffect } from 'react';

export const TrainingPanel = () => {
  const { unlock, isUnlocked } = useAchievements();
  const [trainingStats, setTrainingStats] = useState({
    firstTraining: false,
    accuracy: 0,
    epochs: 0,
    problemId: ''
  });

  useEffect(() => {
    // Check for achievement unlocks
    if (!trainingStats.firstTraining && !isUnlocked('first-train')) {
      unlock('first-train');
      setTrainingStats(prev => ({ ...prev, firstTraining: true }));
    }

    if (trainingStats.accuracy >= 0.99 &&
        trainingStats.problemId === 'xor' &&
        !isUnlocked('xor-master')) {
      unlock('xor-master');
    }

    if (trainingStats.accuracy === 1.0 && !isUnlocked('perfect-score')) {
      unlock('perfect-score');
    }

    if (trainingStats.accuracy >= 0.99 &&
        trainingStats.epochs < 100 &&
        !isUnlocked('speed-demon')) {
      unlock('speed-demon');
    }
  }, [trainingStats, unlock, isUnlocked]);

  // ... rest of training logic
};
```

### Custom Achievement Toast

```typescript
// components/AchievementToast.tsx
import { motion, AnimatePresence } from 'framer-motion';
import { useEffect } from 'react';
import confetti from 'canvas-confetti';

interface Achievement {
  id: string;
  name: string;
  description: string;
  icon: string;
  points: number;
}

interface AchievementToastProps {
  achievement: Achievement;
  onDismiss: () => void;
}

export const AchievementToast = ({ achievement, onDismiss }: AchievementToastProps) => {
  useEffect(() => {
    // Trigger confetti
    confetti({
      particleCount: 100,
      spread: 70,
      origin: { y: 0.6 }
    });

    // Auto-dismiss after 5 seconds
    const timer = setTimeout(onDismiss, 5000);
    return () => clearTimeout(timer);
  }, [onDismiss]);

  return (
    <motion.div
      initial={{ x: 400, opacity: 0 }}
      animate={{ x: 0, opacity: 1 }}
      exit={{ x: 400, opacity: 0 }}
      transition={{
        type: 'spring',
        stiffness: 120,
        damping: 25
      }}
      className="fixed right-4 top-20 bg-white shadow-2xl rounded-lg p-4 z-50 min-w-[300px]"
    >
      <div className="flex items-center gap-3">
        <motion.span
          className="text-5xl"
          initial={{ scale: 0, rotate: -180 }}
          animate={{ scale: 1, rotate: 0 }}
          transition={{
            type: 'spring',
            stiffness: 200,
            damping: 20,
            delay: 0.2
          }}
        >
          {achievement.icon}
        </motion.span>
        <div className="flex-1">
          <h3 className="font-bold text-lg text-gray-900">
            Achievement Unlocked!
          </h3>
          <p className="font-semibold text-gray-700">{achievement.name}</p>
          <p className="text-sm text-gray-600">{achievement.description}</p>
          <p className="text-xs text-green-600 mt-1 font-medium">
            +{achievement.points} points
          </p>
        </div>
        <button
          onClick={onDismiss}
          className="text-gray-400 hover:text-gray-600"
        >
          ✕
        </button>
      </div>
    </motion.div>
  );
};

// Usage in App.tsx
export const App = () => {
  const [currentAchievement, setCurrentAchievement] = useState<Achievement | null>(null);

  // Listen for achievement unlocks
  useEffect(() => {
    const handleAchievement = (achievement: Achievement) => {
      setCurrentAchievement(achievement);
    };

    // Subscribe to achievement events
    window.addEventListener('achievement-unlocked', (e: any) => {
      handleAchievement(e.detail);
    });

    return () => {
      window.removeEventListener('achievement-unlocked', handleAchievement);
    };
  }, []);

  return (
    <div>
      {/* Main app content */}

      <AnimatePresence>
        {currentAchievement && (
          <AchievementToast
            achievement={currentAchievement}
            onDismiss={() => setCurrentAchievement(null)}
          />
        )}
      </AnimatePresence>
    </div>
  );
};
```

---

## 4. Learning Path Selector

### Implementation

```typescript
// components/LearningPathCard.tsx
import { motion } from 'framer-motion';
import { PathProgressRing } from './PathProgressRing';

interface LearningPath {
  id: string;
  name: string;
  description: string;
  difficulty: 'beginner' | 'intermediate' | 'advanced';
  steps: number;
  estimatedTime: string;
  icon: string;
  badge: {
    icon: string;
    title: string;
  };
}

interface LearningPathCardProps {
  path: LearningPath;
  completed: number;
  onSelect: () => void;
}

const difficultyColors = {
  beginner: 'bg-blue-100 text-blue-800 border-blue-300',
  intermediate: 'bg-purple-100 text-purple-800 border-purple-300',
  advanced: 'bg-orange-100 text-orange-800 border-orange-300'
};

export const LearningPathCard = ({
  path,
  completed,
  onSelect
}: LearningPathCardProps) => {
  const isComplete = completed === path.steps;

  return (
    <motion.div
      whileHover={{ scale: 1.02, y: -4 }}
      whileTap={{ scale: 0.98 }}
      className="bg-white rounded-lg shadow-md p-6 cursor-pointer border-2 border-gray-200 hover:border-blue-400 transition-colors"
      onClick={onSelect}
    >
      <div className="flex items-start justify-between mb-4">
        <div className="flex items-center gap-3">
          <span className="text-4xl">{path.icon}</span>
          <div>
            <h3 className="font-bold text-lg text-gray-900">{path.name}</h3>
            <span className={`text-xs px-2 py-1 rounded-full ${difficultyColors[path.difficulty]}`}>
              {path.difficulty}
            </span>
          </div>
        </div>
        {isComplete && (
          <motion.span
            initial={{ scale: 0, rotate: -180 }}
            animate={{ scale: 1, rotate: 0 }}
            transition={{ type: 'spring', stiffness: 200 }}
            className="text-3xl"
          >
            {path.badge.icon}
          </motion.span>
        )}
      </div>

      <p className="text-gray-600 text-sm mb-4">{path.description}</p>

      <div className="flex items-center justify-between mb-4">
        <div className="text-sm text-gray-500">
          <span className="font-medium">{path.steps} steps</span>
          <span className="mx-2">·</span>
          <span>{path.estimatedTime}</span>
        </div>
        <PathProgressRing completed={completed} total={path.steps} size={60} />
      </div>

      <motion.button
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
        className={`w-full py-2 rounded-lg font-medium transition-colors ${
          completed === 0
            ? 'bg-blue-500 text-white hover:bg-blue-600'
            : isComplete
            ? 'bg-green-500 text-white hover:bg-green-600'
            : 'bg-purple-500 text-white hover:bg-purple-600'
        }`}
      >
        {completed === 0 ? 'Start Path' : isComplete ? 'Review' : 'Continue'}
      </motion.button>
    </motion.div>
  );
};

// components/LearningPathSelector.tsx
export const LearningPathSelector = () => {
  const paths: LearningPath[] = [
    {
      id: 'foundations',
      name: 'Foundations',
      description: 'Master basic neural network concepts',
      difficulty: 'beginner',
      steps: 7,
      estimatedTime: '2-3 hours',
      icon: '🎓',
      badge: { icon: '🏆', title: 'Foundation Scholar' }
    },
    {
      id: 'deep-learning',
      name: 'Deep Learning Basics',
      description: 'Training, initialization, and hyperparameters',
      difficulty: 'intermediate',
      steps: 8,
      estimatedTime: '3-4 hours',
      icon: '🧠',
      badge: { icon: '🎯', title: 'Neural Navigator' }
    },
    // ... more paths
  ];

  const [progress, setProgress] = useState<Map<string, number>>(new Map());

  return (
    <div className="container mx-auto p-6">
      <h1 className="text-3xl font-bold mb-2">Learning Paths</h1>
      <p className="text-gray-600 mb-8">
        Choose your journey through neural network fundamentals
      </p>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {paths.map(path => (
          <LearningPathCard
            key={path.id}
            path={path}
            completed={progress.get(path.id) || 0}
            onSelect={() => {/* Navigate to path */}}
          />
        ))}
      </div>
    </div>
  );
};
```

---

## 5. Installation & Dependencies

```bash
# Install new dependencies
cd frontend
npm install react-achievements canvas-confetti
npm install --save-dev @types/canvas-confetti

# Already installed (verify versions)
npm list framer-motion recharts  # Should show framer-motion@12.27.1, recharts@3.6.0
```

---

## 6. Integration with Existing App

### Update App.tsx Structure

```typescript
// App.tsx
import { AchievementWrapper } from './contexts/AchievementContext';
import { LearningPathSelector } from './components/LearningPathSelector';
import { useState } from 'react';

type View = 'paths' | 'problems' | 'challenges';

export const App = () => {
  const [currentView, setCurrentView] = useState<View>('paths');

  return (
    <AchievementWrapper>
      <div className="min-h-screen bg-gray-50">
        {/* Navigation */}
        <nav className="bg-white shadow-sm border-b">
          <div className="container mx-auto px-6 py-4">
            <div className="flex items-center justify-between">
              <h1 className="text-2xl font-bold">Neural Network Academy</h1>
              <div className="flex gap-4">
                <button
                  onClick={() => setCurrentView('paths')}
                  className={`px-4 py-2 rounded ${
                    currentView === 'paths' ? 'bg-blue-500 text-white' : 'text-gray-600'
                  }`}
                >
                  Learning Paths
                </button>
                <button
                  onClick={() => setCurrentView('problems')}
                  className={`px-4 py-2 rounded ${
                    currentView === 'problems' ? 'bg-blue-500 text-white' : 'text-gray-600'
                  }`}
                >
                  All Problems
                </button>
                <button
                  onClick={() => setCurrentView('challenges')}
                  className={`px-4 py-2 rounded ${
                    currentView === 'challenges' ? 'bg-blue-500 text-white' : 'text-gray-600'
                  }`}
                >
                  Challenges
                </button>
              </div>
            </div>
          </div>
        </nav>

        {/* Main Content */}
        <main>
          {currentView === 'paths' && <LearningPathSelector />}
          {currentView === 'problems' && <ProblemView />} {/* Existing */}
          {currentView === 'challenges' && <ChallengeHub />} {/* Coming soon */}
        </main>
      </div>
    </AchievementWrapper>
  );
};
```

---

## 7. Backend Updates

### Add Learning Paths Endpoint

```python
# backend/learning_paths.py
from dataclasses import dataclass
from typing import List

@dataclass
class PathStep:
    step_number: int
    problem_id: str
    title: str
    learning_objectives: List[str]
    required_accuracy: float = 0.95

@dataclass
class LearningPath:
    id: str
    name: str
    description: str
    difficulty: str
    steps: List[PathStep]
    estimated_time: str
    badge_icon: str
    badge_title: str

LEARNING_PATHS = {
    'foundations': LearningPath(
        id='foundations',
        name='Foundations',
        description='Master basic neural network concepts',
        difficulty='beginner',
        estimated_time='2-3 hours',
        badge_icon='🏆',
        badge_title='Foundation Scholar',
        steps=[
            PathStep(
                step_number=1,
                problem_id='and',
                title='AND Gate - Single Neuron Capability',
                learning_objectives=[
                    'Understand what a single neuron can learn',
                    'See linear separability in action'
                ]
            ),
            PathStep(
                step_number=2,
                problem_id='or',
                title='OR Gate - Linear Classification',
                learning_objectives=[
                    'Recognize another linearly separable problem',
                    'Compare with AND gate behavior'
                ]
            ),
            PathStep(
                step_number=3,
                problem_id='not',
                title='NOT Gate - Simple Transformation',
                learning_objectives=[
                    'See how neurons invert signals',
                    'Understand bias importance'
                ]
            ),
            PathStep(
                step_number=4,
                problem_id='fail_xor_no_hidden',
                title='FAIL: XOR with No Hidden Layer',
                learning_objectives=[
                    'Understand why XOR is not linearly separable',
                    'See the necessity of hidden layers'
                ],
                required_accuracy=0.0  # Failure case - expects to fail
            ),
            PathStep(
                step_number=5,
                problem_id='xor',
                title='XOR - Hidden Layers to the Rescue',
                learning_objectives=[
                    'Solve XOR with hidden layers',
                    'Understand how hidden layers create non-linear boundaries'
                ]
            ),
            PathStep(
                step_number=6,
                problem_id='xnor',
                title='XNOR - Applying the Concept',
                learning_objectives=[
                    'Apply hidden layer knowledge to similar problem',
                    'Recognize pattern similarity to XOR'
                ]
            ),
            PathStep(
                step_number=7,
                problem_id='two_blobs',
                title='Two Blobs - Visualizing Decision Boundaries',
                learning_objectives=[
                    'See decision boundaries in 2D space',
                    'Understand how networks separate classes'
                ]
            )
        ]
    ),
    # Add more paths...
}
```

### Add API Endpoints

```python
# backend/app.py (add these routes)

@app.route('/api/paths', methods=['GET'])
def get_learning_paths():
    """Get all learning paths"""
    from learning_paths import LEARNING_PATHS

    paths = []
    for path_id, path in LEARNING_PATHS.items():
        paths.append({
            'id': path.id,
            'name': path.name,
            'description': path.description,
            'difficulty': path.difficulty,
            'steps': len(path.steps),
            'estimatedTime': path.estimated_time,
            'badge': {
                'icon': path.badge_icon,
                'title': path.badge_title
            }
        })

    return jsonify(paths)

@app.route('/api/paths/<path_id>', methods=['GET'])
def get_learning_path(path_id):
    """Get specific learning path details"""
    from learning_paths import LEARNING_PATHS

    if path_id not in LEARNING_PATHS:
        return jsonify({'error': 'Path not found'}), 404

    path = LEARNING_PATHS[path_id]

    return jsonify({
        'id': path.id,
        'name': path.name,
        'description': path.description,
        'difficulty': path.difficulty,
        'estimatedTime': path.estimated_time,
        'badge': {
            'icon': path.badge_icon,
            'title': path.badge_title
        },
        'steps': [
            {
                'stepNumber': step.step_number,
                'problemId': step.problem_id,
                'title': step.title,
                'learningObjectives': step.learning_objectives,
                'requiredAccuracy': step.required_accuracy
            }
            for step in path.steps
        ]
    })

@app.route('/api/paths/<path_id>/progress', methods=['GET', 'POST'])
def path_progress(path_id):
    """Get or update path progress (localStorage for now)"""
    # For MVP, this just echoes back - frontend uses localStorage
    # Future: Add user accounts and database storage

    if request.method == 'POST':
        progress = request.json
        # TODO: Save to database when user accounts are added
        return jsonify({'success': True, 'progress': progress})
    else:
        # TODO: Load from database when user accounts are added
        return jsonify({'completed': 0, 'total': len(LEARNING_PATHS[path_id].steps)})
```

---

## 8. Testing Strategy

### Component Tests

```typescript
// __tests__/PathProgressRing.test.tsx
import { render, screen } from '@testing-library/react';
import { PathProgressRing } from '@/components/PathProgressRing';

describe('PathProgressRing', () => {
  it('displays correct percentage', () => {
    render(<PathProgressRing completed={3} total={7} />);
    expect(screen.getByText('43%')).toBeInTheDocument();
  });

  it('shows 100% when complete', () => {
    render(<PathProgressRing completed={7} total={7} />);
    expect(screen.getByText('100%')).toBeInTheDocument();
  });
});
```

### E2E Tests (Playwright)

```typescript
// e2e/learning-paths.spec.ts
import { test, expect } from '@playwright/test';

test('can select and navigate learning path', async ({ page }) => {
  await page.goto('/');

  // Click on Learning Paths tab
  await page.click('text=Learning Paths');

  // Should see path cards
  await expect(page.locator('text=Foundations')).toBeVisible();
  await expect(page.locator('text=Deep Learning Basics')).toBeVisible();

  // Click on Foundations path
  await page.click('text=Start Path');

  // Should navigate to first step
  await expect(page.locator('text=Step 1: AND Gate')).toBeVisible();
});

test('progress ring updates correctly', async ({ page }) => {
  await page.goto('/paths/foundations');

  // Complete first step
  await page.click('text=Train Network');
  await page.waitForSelector('text=Accuracy: 99%');
  await page.click('text=Mark Complete');

  // Progress ring should update
  await expect(page.locator('text=14%')).toBeVisible(); // 1/7 = 14%
});
```

---

## 9. Performance Considerations

### Canvas Optimization

```typescript
// Use offscreen canvas for better performance
const offscreenCanvas = new OffscreenCanvas(width, height);
const offscreenCtx = offscreenCanvas.getContext('2d');

// Render to offscreen canvas
renderHeatmap(offscreenCtx, weights);

// Transfer to visible canvas
const visibleCtx = canvasRef.current.getContext('2d');
visibleCtx.drawImage(offscreenCanvas, 0, 0);
```

### Throttle WebSocket Updates

```typescript
import { throttle } from 'lodash';

const throttledWeightUpdate = throttle((weights) => {
  setWeights(weights);
}, 100); // Max 10 updates/second

socket.on('weight_update', throttledWeightUpdate);
```

### Lazy Load Components

```typescript
import { lazy, Suspense } from 'react';

const WeightHeatmapCanvas = lazy(() => import('./WeightHeatmapCanvas'));

<Suspense fallback={<div>Loading visualization...</div>}>
  <WeightHeatmapCanvas weights={weights} />
</Suspense>
```

---

## 10. Next Steps

### Week 1: Learning Paths
- [ ] Day 1: Create `PathProgressRing` component
- [ ] Day 2: Create `LearningPathCard` and `LearningPathSelector`
- [ ] Day 3: Add backend `learning_paths.py` with path definitions
- [ ] Day 4: Add API endpoints and integrate frontend
- [ ] Day 5: Add localStorage persistence for progress
- [ ] Day 6: Testing and bug fixes
- [ ] Day 7: Polish and documentation

### Week 2: Weight Animation
- [ ] Day 1: Create `WeightHeatmapCanvas` component
- [ ] Day 2: Integrate Motion.dev spring values
- [ ] Day 3: Add WebSocket weight updates
- [ ] Day 4: Add color mapping and legends
- [ ] Day 5: Performance optimization
- [ ] Day 6: Testing
- [ ] Day 7: Polish

### Week 3: Achievements & Challenges
- [ ] Day 1: Install and configure `react-achievements`
- [ ] Day 2: Create achievement definitions
- [ ] Day 3: Add unlock triggers throughout app
- [ ] Day 4: Create custom `AchievementToast` component
- [ ] Day 5: Add confetti effects
- [ ] Day 6: Testing
- [ ] Day 7: Challenge system planning

---

## Summary

**What We Have**:
- ✅ Real code examples from Motion.dev and Recharts (via Context7)
- ✅ Complete component implementations
- ✅ Backend API endpoints designed
- ✅ Testing strategy defined
- ✅ Performance optimizations planned

**Ready to Start**: Pick Week 1 (Learning Paths) and begin implementation!
