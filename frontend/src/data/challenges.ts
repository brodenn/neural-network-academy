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
  vanishing_gradients: {
    title: "Deep Network Barely Learns",
    description: "A 6-layer sigmoid network is supposed to learn XOR but accuracy barely improves.",
    problem: "XOR Gate",
    config: { architecture: [2, 8, 8, 8, 8, 1], learningRate: 0.5, epochs: 1000, hiddenActivation: 'sigmoid' },
    symptoms: [
      "Loss decreases extremely slowly despite many epochs",
      "Early layers' weights barely change",
      "Deeper layers learn slightly but shallow layers are stuck",
    ],
    options: [
      { id: 'lr', text: "Learning rate is too low", isCorrect: false, explanation: "LR=0.5 is reasonable. The problem is how gradients flow through layers." },
      { id: 'activation', text: "Sigmoid activation causes vanishing gradients", isCorrect: true, explanation: "Correct! Sigmoid squashes values to 0-1, and its gradient is at most 0.25. Through 6 layers, gradients shrink exponentially. Switch to ReLU!" },
      { id: 'arch', text: "Too many layers - simplify", isCorrect: false, explanation: "Deep networks can work, but they need the right activation function." },
      { id: 'init', text: "Bad weight initialization", isCorrect: false, explanation: "Initialization matters, but the core issue is gradient flow through sigmoid layers." },
    ],
  },
  wrong_activation: {
    title: "Sine Wave Stuck in 0-1 Range",
    description: "The network should learn a sine wave (outputs from -1 to 1) but all outputs are between 0 and 1.",
    problem: "Sine Wave",
    config: { architecture: [1, 16, 8, 1], learningRate: 0.5, epochs: 500, hiddenActivation: 'sigmoid' },
    symptoms: [
      "Output values are always between 0 and 1",
      "The network can't produce negative values",
      "Loss plateaus at a high value",
    ],
    options: [
      { id: 'arch', text: "Need more neurons", isCorrect: false, explanation: "The architecture has plenty of neurons. The problem is the activation function." },
      { id: 'activation', text: "Sigmoid can't output negative values - use ReLU", isCorrect: true, explanation: "Correct! Sigmoid squashes everything to (0, 1). For regression that needs negative outputs, use ReLU in hidden layers so the network can learn arbitrary mappings." },
      { id: 'lr', text: "Learning rate needs adjustment", isCorrect: false, explanation: "The LR is fine. No amount of LR tuning will let sigmoid output negative values." },
      { id: 'epochs', text: "Need more training epochs", isCorrect: false, explanation: "More epochs won't help - the activation function fundamentally limits the output range." },
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
  regression_intro: {
    question: 'Can a neural network learn a straight line (y = 2x + 1)?',
    context: { architecture: [1, 1], learningRate: 0.1, epochs: 200, problem: 'Linear Regression' },
    options: [
      { id: 'no_hidden', text: 'Yes, a single neuron can learn any linear function', isCorrect: true, explanation: 'Correct! A single neuron computes y = wx + b, which is exactly a linear function. No hidden layers needed!' },
      { id: 'needs_hidden', text: 'No, it needs hidden layers for any function', isCorrect: false, explanation: 'Linear functions are the simplest case - a single neuron is a linear function itself.' },
      { id: 'only_class', text: 'Neural networks can only classify, not regress', isCorrect: false, explanation: 'Neural networks can do both classification and regression. Regression outputs continuous values.' },
      { id: 'maybe', text: 'Only if the learning rate is exactly right', isCorrect: false, explanation: 'Many learning rates will work. The architecture is what matters for capability.' },
    ],
  },
  boundary_shape: {
    question: 'The moons dataset has two interlocking crescents. What shape boundary is needed to separate them?',
    context: { architecture: [2, 8, 1], learningRate: 0.5, epochs: 500, problem: 'Moons' },
    options: [
      { id: 'line', text: 'A straight line', isCorrect: false, explanation: 'A straight line would cut through both crescents. The boundary needs to curve.' },
      { id: 'curve', text: 'A curved, non-linear boundary', isCorrect: true, explanation: 'Correct! The interlocking crescents need a curved boundary that wraps around them. This is why hidden layers are essential.' },
      { id: 'circle', text: 'A perfect circle', isCorrect: false, explanation: 'A circle would work for concentric patterns, but moons need a more complex curve.' },
      { id: 'two_lines', text: 'Two parallel lines', isCorrect: false, explanation: 'Parallel lines create a band, which doesn\'t match the crescent shape.' },
    ],
  },
  multiclass_output: {
    question: 'For 4-class quadrant classification, how should the output layer look?',
    context: { architecture: [2, 8, 4], learningRate: 0.5, epochs: 500, problem: 'Quadrants' },
    options: [
      { id: 'one_neuron', text: '1 output neuron (outputs 0, 1, 2, or 3)', isCorrect: false, explanation: 'A single neuron can\'t cleanly represent discrete classes. Each class needs its own neuron.' },
      { id: 'four_neurons', text: '4 output neurons with softmax', isCorrect: true, explanation: 'Correct! One neuron per class, with softmax ensuring the outputs sum to 1 as probabilities.' },
      { id: 'two_neurons', text: '2 output neurons (binary for each axis)', isCorrect: false, explanation: 'While clever, this encoding makes training harder. Standard practice is one neuron per class.' },
      { id: 'eight_neurons', text: '8 output neurons for more precision', isCorrect: false, explanation: 'More outputs than classes adds unnecessary complexity. Match outputs to class count.' },
    ],
  },
  color_classes: {
    question: 'How many output neurons does a network need to classify 6 different colors?',
    context: { architecture: [3, 8, 6], learningRate: 0.5, epochs: 500, problem: 'Color Classification' },
    options: [
      { id: 'three', text: '3 (one per RGB channel)', isCorrect: false, explanation: 'RGB channels are inputs, not outputs. Outputs represent classes.' },
      { id: 'six', text: '6 (one per color class)', isCorrect: true, explanation: 'Correct! Each output neuron represents one color class. Softmax converts them to probabilities.' },
      { id: 'one', text: '1 (output a color index 0-5)', isCorrect: false, explanation: 'A single neuron outputting discrete indices doesn\'t train well. One-hot encoding is standard.' },
      { id: 'twelve', text: '12 (two per class for confidence)', isCorrect: false, explanation: 'Softmax already provides confidence. One neuron per class is sufficient.' },
    ],
  },
  cnn_advantage: {
    question: 'Why are CNNs better than dense networks for image classification?',
    context: { architecture: [2, 8, 3], learningRate: 0.5, epochs: 500, problem: 'Shape Detection' },
    options: [
      { id: 'more_params', text: 'They have more parameters', isCorrect: false, explanation: 'CNNs actually have fewer parameters due to weight sharing. That\'s a feature, not a limitation.' },
      { id: 'spatial', text: 'They understand spatial patterns through weight sharing', isCorrect: true, explanation: 'Correct! Convolutional filters detect the same pattern anywhere in the image. A vertical edge detector works whether the edge is on the left or right.' },
      { id: 'faster', text: 'They always train faster', isCorrect: false, explanation: 'Speed depends on many factors. The key advantage is spatial understanding.' },
      { id: 'deeper', text: 'They are always deeper', isCorrect: false, explanation: 'Depth is a choice, not a requirement. The key innovation is convolutional weight sharing.' },
    ],
  },
  cnn_vs_dense: {
    question: 'A CNN was trained on shapes. Can the same CNN architecture work well on non-spatial 1D signal patterns?',
    context: { architecture: [2, 8, 3], learningRate: 0.5, epochs: 500, problem: 'Pattern Classification' },
    options: [
      { id: 'yes_always', text: 'Yes, CNNs are always superior', isCorrect: false, explanation: 'CNNs excel at spatial data but aren\'t always the best choice for non-spatial patterns.' },
      { id: 'yes_local', text: 'Yes, if the signal has local patterns worth detecting', isCorrect: true, explanation: 'Correct! CNNs can detect local patterns in 1D signals too (like peaks or edges). They work when local features matter.' },
      { id: 'never', text: 'No, CNNs only work on 2D images', isCorrect: false, explanation: 'CNNs work on 1D data too! Think of audio processing or time-series analysis.' },
      { id: 'random', text: 'It depends entirely on luck', isCorrect: false, explanation: 'Architecture choice is principled, not random. Match the architecture to the data structure.' },
    ],
  },
  slow_vs_fast_lr: {
    question: 'What happens if you train XOR with learning rate = 0.0001?',
    context: { architecture: [2, 4, 1], learningRate: 0.0001, epochs: 1000, problem: 'XOR Gate' },
    options: [
      { id: 'precise', text: 'It will find a more precise solution', isCorrect: false, explanation: 'Small LR means slow convergence, not more precision. The solution quality depends on architecture.' },
      { id: 'glacial', text: 'Learning will be extremely slow - barely any progress in 1000 epochs', isCorrect: true, explanation: 'Correct! With LR=0.0001, each gradient step is tiny. The network would need 100,000+ epochs to converge on XOR.' },
      { id: 'same', text: 'Same result, just takes a bit longer', isCorrect: false, explanation: 'It\'s not "a bit" longer - it\'s orders of magnitude slower. With 1000 epochs, it will barely have moved.' },
      { id: 'fail', text: 'The network will completely fail to learn', isCorrect: false, explanation: 'It IS learning, just absurdly slowly. Given enough epochs it would converge.' },
    ],
  },
  digit_architecture: {
    question: 'Digit recognition has 10 classes (0-9) on an 8x8 grid. What makes this harder than 3-class shape recognition?',
    context: { architecture: [2, 8, 10], learningRate: 0.5, epochs: 500, problem: 'Digit Recognition' },
    options: [
      { id: 'more_classes', text: 'More classes need more output neurons and decision boundaries', isCorrect: true, explanation: 'Correct! 10 classes means 10 output neurons and far more complex decision boundaries. Some digits (like 1 vs 7, or 3 vs 8) are very similar.' },
      { id: 'same', text: 'It\'s not harder - just add more outputs', isCorrect: false, explanation: 'Adding outputs is easy, but distinguishing similar digits (3 vs 8, 1 vs 7) requires much more nuanced features.' },
      { id: 'impossible', text: 'It\'s impossible without a GPU', isCorrect: false, explanation: 'Small digit recognition works fine on CPU. Real MNIST-scale problems benefit from GPUs.' },
      { id: 'data', text: 'Only because we need more training data', isCorrect: false, explanation: 'Data helps, but the fundamental challenge is the increased complexity of distinguishing 10 similar patterns.' },
    ],
  },
  or_vs_and: {
    question: 'How is the OR gate different from AND in terms of what the network learns?',
    context: { architecture: [2, 1], learningRate: 0.5, epochs: 100, problem: 'OR Gate' },
    options: [
      { id: 'same', text: 'They are identical - same weights work for both', isCorrect: false, explanation: 'While both are linearly separable, they need different weight values.' },
      { id: 'boundary', text: 'OR needs a different decision boundary position', isCorrect: true, explanation: 'Correct! AND outputs 1 only for (1,1), while OR outputs 1 for (1,0), (0,1), and (1,1). The boundary line shifts to include more positive cases.' },
      { id: 'hidden', text: 'OR requires a hidden layer but AND does not', isCorrect: false, explanation: 'Both AND and OR are linearly separable - neither needs hidden layers.' },
      { id: 'harder', text: 'OR is much harder and needs more epochs', isCorrect: false, explanation: 'OR is equally simple - both converge quickly with a single neuron.' },
    ],
  },
  nand_universal: {
    question: 'NAND is called the "universal gate." Why is this significant?',
    context: { architecture: [2, 1], learningRate: 0.5, epochs: 100, problem: 'NAND Gate' },
    options: [
      { id: 'hardest', text: 'Because NAND is the hardest gate to learn', isCorrect: false, explanation: 'NAND is just as easy as AND or OR - still linearly separable.' },
      { id: 'universal', text: 'Any logic circuit can be built using only NAND gates', isCorrect: true, explanation: 'Correct! NAND is functionally complete - you can build AND, OR, NOT, XOR, and any other logic using only NAND. This is why it\'s "universal."' },
      { id: 'fastest', text: 'Because it trains faster than other gates', isCorrect: false, explanation: 'Training speed is similar to AND and OR. Universality is about computational power, not speed.' },
      { id: 'nonlinear', text: 'Because it requires non-linear computation', isCorrect: false, explanation: 'NAND is linearly separable like AND and OR. Its power comes from composition, not non-linearity.' },
    ],
  },
  blob_boundary: {
    question: 'Two blob clusters are clearly separated in 2D space. What type of decision boundary does the network learn?',
    context: { architecture: [2, 4, 1], learningRate: 0.5, epochs: 200, problem: 'Two Blobs' },
    options: [
      { id: 'circle', text: 'A circle around one cluster', isCorrect: false, explanation: 'Circles are needed for concentric patterns. Well-separated blobs need simpler boundaries.' },
      { id: 'line', text: 'A roughly straight line between the clusters', isCorrect: true, explanation: 'Correct! Well-separated clusters can be divided by a nearly linear boundary. The hidden layer helps fine-tune it, but the boundary is essentially a line.' },
      { id: 'complex', text: 'A complex wavy curve', isCorrect: false, explanation: 'Complex boundaries are for interleaved patterns like spirals. Blobs are much simpler.' },
      { id: 'none', text: 'No clear boundary - it memorizes each point', isCorrect: false, explanation: 'Neural networks learn generalizable boundaries, not individual point positions.' },
    ],
  },
  pattern_features: {
    question: 'The network must classify different signal patterns (sine, square, triangle waves). What does it learn to detect?',
    context: { architecture: [8, 16, 3], learningRate: 0.5, epochs: 500, problem: 'Signal Patterns' },
    options: [
      { id: 'amplitude', text: 'Only the amplitude (height) of the signal', isCorrect: false, explanation: 'Different wave types can have the same amplitude. Shape matters more than height.' },
      { id: 'features', text: 'Shape features like smoothness, sharp transitions, and symmetry', isCorrect: true, explanation: 'Correct! The hidden layers learn to detect features like sharp corners (square wave), smooth curves (sine), and linear slopes (triangle). These features distinguish the patterns.' },
      { id: 'frequency', text: 'Only the frequency of the signal', isCorrect: false, explanation: 'Different wave shapes can have the same frequency. The network needs to distinguish shape, not just frequency.' },
      { id: 'memorize', text: 'It memorizes exact waveform values', isCorrect: false, explanation: 'Generalization, not memorization, is key. It learns abstract features that work for new signals too.' },
    ],
  },
  cnn_digit_challenge: {
    question: 'For CNN digit recognition (0-9 on 8x8 grid), why is this harder than detecting 3 basic shapes?',
    context: { architecture: [2, 8, 10], learningRate: 0.5, epochs: 500, problem: 'Digit Recognition' },
    options: [
      { id: 'more_pixels', text: 'Digits use more pixels', isCorrect: false, explanation: 'Both use the same 8x8 grid. The challenge is in the pattern complexity, not resolution.' },
      { id: 'similarity', text: 'Many digits look similar (3 vs 8, 1 vs 7) and need fine-grained features', isCorrect: true, explanation: 'Correct! With 10 classes, many digits share strokes and curves. The network must learn subtle differences - like the gap in "3" vs the closed loops in "8".' },
      { id: 'more_data', text: 'Only because we need more training data', isCorrect: false, explanation: 'Data helps, but the architectural challenge is distinguishing 10 visually similar classes.' },
      { id: 'impossible', text: 'It\'s impossible at 8x8 resolution', isCorrect: false, explanation: '8x8 is enough to distinguish handwritten digits. Early digit recognition systems used similar resolutions.' },
    ],
  },
};

export type QuizId = keyof typeof PREDICTION_QUIZZES;
