// =============================================================================
// DEBUG CHALLENGES DATA
// =============================================================================

export interface DebugOption {
  id: string;
  text: string;
  isCorrect: boolean;
  explanation: string;
}

export interface BrokenConfig {
  architecture: number[];
  learningRate: number;
  epochs: number;
  weightInit?: string;
  hiddenActivation?: string;
}

export interface DebugChallengeData {
  title: string;
  description: string;
  problem: string;
  config: BrokenConfig;
  symptoms: string[];
  options: DebugOption[];
}

export const DEBUG_CHALLENGES: Record<string, DebugChallengeData> = {
  xor_no_hidden: {
    title: "XOR Won't Learn",
    description: "A student is trying to train a neural network on XOR but it's stuck at 50% accuracy.",
    problem: "XOR Gate",
    config: { architecture: [2, 1], learningRate: 0.5, epochs: 1000 },
    symptoms: [
      "Accuracy stays at exactly 50%",
      "Loss decreases initially then plateaus",
      "Decision boundary is a straight line",
    ],
    options: [
      { id: 'lr', text: "Learning rate is wrong", isCorrect: false, explanation: "Learning rate is fine at 0.5. The problem is architectural." },
      { id: 'epochs', text: "Need more epochs", isCorrect: false, explanation: "No amount of training will help - the architecture can't solve this problem." },
      { id: 'hidden', text: "Missing hidden layer", isCorrect: true, explanation: "XOR is not linearly separable! A network with only [2,1] can only draw a straight line. XOR needs a hidden layer." },
      { id: 'init', text: "Bad weight initialization", isCorrect: false, explanation: "Weight init affects training speed but wouldn't cause this specific symptom." },
    ],
  },
  zero_init_bug: {
    title: "Hidden Neurons Not Helping",
    description: "The network has 4 hidden neurons but behaves like it only has 1.",
    problem: "XOR Gate",
    config: { architecture: [2, 4, 1], learningRate: 0.5, epochs: 500, weightInit: "zeros" },
    symptoms: [
      "All hidden neurons have identical activations",
      "Adding more hidden neurons doesn't help",
      "Accuracy stuck around 50%",
    ],
    options: [
      { id: 'arch', text: "Need different architecture", isCorrect: false, explanation: "[2,4,1] should be able to solve XOR easily." },
      { id: 'zeros', text: "Weights initialized to zeros", isCorrect: true, explanation: "When all weights are zero, every hidden neuron computes the same thing! This is the 'symmetry problem'." },
      { id: 'activation', text: "Wrong activation function", isCorrect: false, explanation: "Activation function affects learning but wouldn't cause identical neurons." },
      { id: 'lr', text: "Learning rate too low", isCorrect: false, explanation: "LR affects speed, not whether neurons differentiate." },
    ],
  },
  exploding_loss: {
    title: "Loss Goes to Infinity",
    description: "Training starts but after a few epochs the loss explodes to NaN!",
    problem: "XOR Gate",
    config: { architecture: [2, 4, 1], learningRate: 10, epochs: 100 },
    symptoms: [
      "Loss spikes wildly in first few epochs",
      "Loss becomes NaN (Not a Number)",
      "Weights become extremely large",
    ],
    options: [
      { id: 'arch', text: "Architecture is wrong", isCorrect: false, explanation: "Architecture is fine - the problem is the hyperparameters." },
      { id: 'lr', text: "Learning rate is way too high", isCorrect: true, explanation: "With LR=10, gradient descent overshoots and bounces around until values overflow to NaN." },
      { id: 'epochs', text: "Training too long", isCorrect: false, explanation: "The explosion happens in the first few epochs." },
      { id: 'data', text: "Training data is corrupted", isCorrect: false, explanation: "The data is fine - this is a classic high learning rate symptom." },
    ],
  },
  glacial_learning: {
    title: "Learning is Glacially Slow",
    description: "The network seems to be learning but it's taking FOREVER.",
    problem: "XOR Gate",
    config: { architecture: [2, 4, 1], learningRate: 0.0001, epochs: 1000 },
    symptoms: [
      "Loss decreases but very slowly",
      "Would need 100,000+ epochs to converge",
      "Network IS learning, just extremely slowly",
    ],
    options: [
      { id: 'arch', text: "Need more neurons", isCorrect: false, explanation: "The architecture is sufficient." },
      { id: 'lr', text: "Learning rate is too low", isCorrect: true, explanation: "LR=0.0001 means tiny baby steps. Try LR=0.5 for reasonable speed." },
      { id: 'init', text: "Bad initialization", isCorrect: false, explanation: "Initialization affects early training but wouldn't cause this." },
      { id: 'activation', text: "Wrong activation function", isCorrect: false, explanation: "Activation choice wouldn't cause uniformly slow learning." },
    ],
  },
};

export type DebugChallengeId = keyof typeof DEBUG_CHALLENGES;

// =============================================================================
// PREDICTION QUIZZES DATA
// =============================================================================

export interface PredictionOption {
  id: string;
  text: string;
  isCorrect: boolean;
  explanation: string;
}

export interface PredictionQuizData {
  question: string;
  context: {
    architecture: number[];
    learningRate: number;
    epochs: number;
    problem: string;
  };
  options: PredictionOption[];
}

export const PREDICTION_QUIZZES: Record<string, PredictionQuizData> = {
  xor_no_hidden: {
    question: 'What will happen when we train XOR with NO hidden layer?',
    context: { architecture: [2, 1], learningRate: 0.5, epochs: 500, problem: 'XOR Gate' },
    options: [
      { id: 'converge', text: 'It will converge to 100% accuracy', isCorrect: false, explanation: 'Without a hidden layer, the network can only learn linear boundaries.' },
      { id: 'stuck', text: 'Accuracy will get stuck around 50%', isCorrect: true, explanation: 'Correct! XOR is not linearly separable. A single neuron cannot separate XOR outputs.' },
      { id: 'slow', text: 'It will learn slowly but eventually succeed', isCorrect: false, explanation: 'No amount of training will help - the architecture fundamentally cannot solve this.' },
      { id: 'explode', text: 'The loss will explode to infinity', isCorrect: false, explanation: 'Loss explosion typically happens with too high learning rate.' },
    ],
  },
  high_lr: {
    question: 'What will happen with a learning rate of 10?',
    context: { architecture: [2, 4, 1], learningRate: 10, epochs: 100, problem: 'XOR Gate' },
    options: [
      { id: 'fast', text: 'It will learn faster than normal', isCorrect: false, explanation: 'Higher LR means bigger steps, but too big means overshooting.' },
      { id: 'oscillate', text: 'Loss will oscillate wildly or explode', isCorrect: true, explanation: 'Correct! With LR=10, the network overshoots the minimum, bouncing around or exploding to NaN.' },
      { id: 'same', text: "Same as normal, LR doesn't matter much", isCorrect: false, explanation: 'Learning rate is one of the most critical hyperparameters!' },
      { id: 'plateau', text: 'It will plateau early', isCorrect: false, explanation: 'Plateaus happen with too LOW learning rate, not too high.' },
    ],
  },
  zero_init: {
    question: 'What happens when all weights start at zero?',
    context: { architecture: [2, 4, 1], learningRate: 0.5, epochs: 500, problem: 'XOR Gate' },
    options: [
      { id: 'normal', text: 'Training proceeds normally', isCorrect: false, explanation: 'Zero initialization breaks the network in a subtle but fundamental way.' },
      { id: 'symmetry', text: 'All hidden neurons will learn the same thing', isCorrect: true, explanation: "Correct! With zero weights, all hidden neurons compute identical values and receive identical gradients - the 'symmetry problem'." },
      { id: 'no_learn', text: "Network won't learn at all (0% accuracy)", isCorrect: false, explanation: "The network does learn something, but it's as if you only have 1 hidden neuron." },
      { id: 'faster', text: 'It will learn faster from a clean slate', isCorrect: false, explanation: 'Random initialization is crucial for breaking symmetry.' },
    ],
  },
  and_simple: {
    question: 'Does the AND gate need a hidden layer?',
    context: { architecture: [2, 1], learningRate: 0.5, epochs: 100, problem: 'AND Gate' },
    options: [
      { id: 'needs_hidden', text: 'Yes, all problems need hidden layers', isCorrect: false, explanation: 'Not all problems need hidden layers - it depends on linear separability.' },
      { id: 'no_hidden', text: 'No, AND is linearly separable', isCorrect: true, explanation: 'Correct! AND outputs 1 only for (1,1). A single line can separate this.' },
      { id: 'maybe', text: 'It depends on the learning rate', isCorrect: false, explanation: "The need for hidden layers depends on the problem's geometry, not hyperparameters." },
      { id: 'more_neurons', text: 'Yes, need at least 4 neurons', isCorrect: false, explanation: 'AND is one of the simplest problems - just 1 output neuron is sufficient.' },
    ],
  },
  deep_vs_shallow: {
    question: 'For the spiral problem, which architecture will work better?',
    context: { architecture: [2, 4, 1], learningRate: 0.5, epochs: 1000, problem: 'Spiral' },
    options: [
      { id: 'shallow_wide', text: '[2, 16, 1] - shallow but wide', isCorrect: false, explanation: 'Width helps but the spiral needs the compositional power of depth.' },
      { id: 'deep_narrow', text: '[2, 8, 8, 8, 1] - deep and narrow', isCorrect: true, explanation: "Correct! Deeper networks can learn more complex, hierarchical features." },
      { id: 'minimal', text: '[2, 4, 1] - minimal', isCorrect: false, explanation: 'This might work eventually but will struggle with the complex spiral.' },
      { id: 'same', text: "They're all equivalent", isCorrect: false, explanation: 'Architecture significantly impacts what a network can learn!' },
    ],
  },
};

export type QuizId = keyof typeof PREDICTION_QUIZZES;
