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

export interface GPIOState {
  buttons: number[];
  led: boolean;
  button_pins: number[];
  led_pin: number;
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

export interface SystemStatus {
  training_complete: boolean;
  training_in_progress: boolean;
  current_epoch: number;
  current_loss: number;
  current_accuracy: number;
  prediction_count: number;
  current_problem?: string;
  network_type?: NetworkType;
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
  embedded_context?: string;
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
    case 'xor':
      return { type: 'binary', labels: problem.input_labels };
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
    case 'arrow_direction':
      return { type: 'grid', labels: problem.output_labels, gridSize: 8, min: 0, max: 1 };
    case 'color_mixer':
      return { type: 'slider', labels: problem.input_labels, min: 0, max: 1, step: 0.01 };
    default:
      return { type: 'slider', labels: problem.input_labels, min: 0, max: 1, step: 0.01 };
  }
}
