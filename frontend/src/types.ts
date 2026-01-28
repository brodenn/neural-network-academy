// API Response Types

export type WeightInit = 'xavier' | 'he' | 'random' | 'zeros';
export type HiddenActivation = 'relu' | 'sigmoid' | 'tanh';
export type NetworkType = 'dense' | 'cnn';

export interface NetworkArchitecture {
  layer_sizes?: number[];
  num_layers?: number;
  num_weights?: number;
  num_biases?: number;
  weight_init?: WeightInit;
  hidden_activation?: HiddenActivation;
  use_biases?: boolean;
  // CNN specific
  type?: NetworkType;
  input_shape?: [number, number, number];
  layers?: string[];
  layer_count?: number;
}

export interface LayerWeights {
  layer: number;
  weights: number[][];
  biases: number[][];
  input_size: number;
  output_size: number;
}

export interface NetworkState {
  architecture: NetworkArchitecture;
  weights: LayerWeights[];
  loss_history: number[];
  accuracy_history: number[];
  total_epochs?: number;
  network_type?: NetworkType;
}

// CNN Feature Maps for visualization
export interface CNNFeatureMaps {
  [layerName: string]: number[][][];  // (height, width, filters) or filter data
}

export interface TrainingProgress {
  epoch: number;
  loss: number;
  accuracy: number;
  gradients?: number[][];  // Per-layer gradient magnitudes for visualization
}

export interface TrainingResult {
  epochs: number;
  final_loss: number;
  final_accuracy: number;
  loss_history: number[];
  accuracy_history: number[];
  target_reached?: boolean;
  restarts?: number;
}

export interface PredictionResult {
  inputs: number[] | number[][];  // 1D for dense, 2D grid for CNN
  prediction: number | number[];  // Single value or array for multi-class
  prediction_rounded: number;
  led_state: boolean;
  expected: number | number[] | null;  // Single value or one-hot array
  correct: boolean | null;
  activations?: number[][];
  problem_id?: string;
  output_labels?: string[];
  network_type?: NetworkType;
  feature_maps?: CNNFeatureMaps;
}

export interface TrainingData {
  inputs: number[][];
  labels: number[][];
  num_samples: number;
  problem_id?: string;
  input_labels?: string[];
  output_labels?: string[];
}

// Problem Types

export type ProblemCategory = 'binary' | 'regression' | 'multi-class';

export interface ProblemInfo {
  id: string;
  name: string;
  description: string;
  category: ProblemCategory;
  default_architecture: number[];
  input_labels: string[];
  output_labels: string[];
  output_activation: 'sigmoid' | 'softmax';
  // Educational fields
  difficulty: number;  // 1-5 (1=beginner, 5=advanced)
  concept: string;  // What this problem teaches
  learning_goal: string;  // What the student should learn
  tips: string[];  // Hints for solving
  // Level and failure case fields
  level: number;  // Explicit level number (1-7)
  is_failure_case: boolean;  // If true, expected to fail
  failure_reason?: string;  // Why it fails (for failure cases)
  fix_suggestion?: string;  // How to fix it (for failure cases)
  locked_architecture: boolean;  // Cannot change architecture
  forced_weight_init?: WeightInit;  // Force specific init type
  forced_learning_rate?: number;  // Force specific LR
  sample_count?: number;
  input_size?: number | [number, number, number];
  output_size?: number;
  network_type?: NetworkType;
  input_shape?: [number, number, number];
}

// Input Configuration for dynamic UI

export interface InputConfig {
  type: 'binary' | 'slider' | 'pattern' | 'grid';
  labels: string[];
  min?: number;
  max?: number;
  step?: number;
  gridSize?: number;  // For grid input type (CNN)
}

export function getInputConfigForProblem(problem: ProblemInfo): InputConfig {
  switch (problem.id) {
    // Level 1: Single Neuron - binary inputs
    case 'and_gate':
    case 'or_gate':
    case 'nand_gate':
      return { type: 'binary', labels: problem.input_labels };
    case 'not_gate':
      return { type: 'binary', labels: problem.input_labels };

    // Level 2: Hidden Layers - XOR problems
    case 'xor':
    case 'xnor':
      return { type: 'binary', labels: problem.input_labels };
    case 'xor_5bit':
      return { type: 'binary', labels: problem.input_labels };

    // Level 3: Decision Boundaries - slider for x,y coordinates
    case 'two_blobs':
    case 'moons':
    case 'circle':
    case 'donut':
    case 'spiral':
      return { type: 'slider', labels: problem.input_labels, min: -1, max: 1, step: 0.05 };

    // Level 4: Regression - slider inputs
    case 'linear':
    case 'sine_wave':
    case 'polynomial':
      return { type: 'slider', labels: problem.input_labels, min: 0, max: 1, step: 0.01 };
    case 'surface':
      return { type: 'slider', labels: problem.input_labels, min: 0, max: 1, step: 0.05 };

    // Level 5: Failure Cases - inherit from base problem type
    case 'fail_xor_no_hidden':
    case 'fail_zero_init':
    case 'fail_lr_high':
    case 'fail_lr_low':
    case 'fail_vanishing':
      // These are all XOR-based problems with binary inputs
      return { type: 'binary', labels: problem.input_labels };
    case 'fail_underfit':
      // Spiral-based problem with continuous 2D inputs
      return { type: 'slider', labels: problem.input_labels, min: -1, max: 1, step: 0.05 };

    // Level 6: Multi-class
    case 'colors':
      return { type: 'slider', labels: problem.input_labels, min: 0, max: 1, step: 0.01 };
    case 'quadrants':
    case 'blobs':
      return { type: 'slider', labels: problem.input_labels, min: -1, max: 1, step: 0.1 };
    case 'patterns':
      return { type: 'pattern', labels: problem.input_labels, min: 0, max: 1, step: 0.1 };

    // Level 7: CNN - grid inputs
    case 'shapes':
    case 'digits':
      return { type: 'grid', labels: problem.output_labels, gridSize: 8, min: 0, max: 1 };

    // Legacy problem IDs for backwards compatibility
    case 'sensor_fusion':
      return { type: 'slider', labels: problem.input_labels, min: 0, max: 1, step: 0.01 };
    case 'anomaly':
      return { type: 'slider', labels: problem.input_labels, min: 0, max: 1, step: 0.01 };
    case 'gesture':
      return { type: 'pattern', labels: problem.input_labels, min: 0, max: 1, step: 0.1 };
    case 'shape_detection':
      return { type: 'grid', labels: problem.output_labels, gridSize: 8, min: 0, max: 1 };
    case 'digit_recognition':
      return { type: 'grid', labels: problem.output_labels, gridSize: 8, min: 0, max: 1 };

    default:
      // Default based on input count and network type
      if (problem.network_type === 'cnn') {
        return { type: 'grid', labels: problem.output_labels, gridSize: 8, min: 0, max: 1 };
      }
      if (problem.input_labels.length <= 5 && problem.category === 'binary') {
        return { type: 'binary', labels: problem.input_labels };
      }
      return { type: 'slider', labels: problem.input_labels, min: 0, max: 1, step: 0.01 };
  }
}

// Learning Path Types

export type Difficulty = 'beginner' | 'intermediate' | 'advanced' | 'research';

// Step types for interactive challenges
export type StepType = 'training' | 'build_challenge' | 'prediction_quiz' | 'debug_challenge' | 'tuning_challenge';

// Challenge-specific data interfaces
export interface BuildChallengeData {
  minLayers: number;
  maxLayers: number;
  minHiddenNeurons: number;
  maxHiddenNeurons: number;
  mustHaveHidden: boolean;
  successMessage: string;
}

export interface PredictionQuizData {
  quizId: string;
  showResultAfter: boolean;
}

export interface DebugChallengeData {
  challengeId: string;
}

export interface TuningChallengeData {
  parameter: string;
  targetLoss: number;
  maxAttempts: number;
}

export interface PathStep {
  stepNumber: number;
  problemId: string;
  title: string;
  learningObjectives: string[];
  requiredAccuracy?: number;
  hints: string[];
  // New: step type for interactive challenges
  stepType?: StepType;
  // Challenge-specific data
  buildChallenge?: BuildChallengeData;
  predictionQuiz?: PredictionQuizData;
  debugChallenge?: DebugChallengeData;
  tuningChallenge?: TuningChallengeData;
  // Runtime progress fields
  unlocked: boolean;
  completed: boolean;
  completedAt?: Date;
  attempts: number;
  bestAccuracy: number;
}

export interface LearningPath {
  id: string;
  name: string;
  description: string;
  difficulty: Difficulty;
  estimatedTime: string;
  prerequisites: string[];
  steps: number; // Total number of steps
  badge: {
    icon: string;
    color: string;
    title: string;
  };
}

// Progress tracking types for localStorage persistence

export interface StepProgressData {
  stepNumber: number;
  problemId: string;
  unlocked: boolean;
  completed: boolean;
  completedAt?: string;
  attempts: number;
  bestAccuracy: number;
  hintsUsed: number;
}

export interface PathProgressData {
  pathId: string;
  currentStep: number;
  stepsCompleted: number;
  startedAt: string;
  lastActiveAt: string;
  steps: StepProgressData[];
}

// Note: Types are exported inline, no need for additional re-exports
