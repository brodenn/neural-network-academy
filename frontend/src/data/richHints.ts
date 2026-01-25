import type { RichHint } from '../components/InteractiveHint';
import type { Experiment } from '../components/ExperimentCard';

// =============================================================================
// RICH HINTS FOR KEY PROBLEMS
// These demonstrate the enhanced hint system with:
// - Progressive disclosure (multiple layers)
// - Interactive experiments
// - Socratic questions
// - Visual concepts
// =============================================================================

export const RICH_HINTS: Record<string, RichHint[]> = {
  // ============================================================================
  // XOR GATE - The classic "why deep learning?" problem
  // ============================================================================
  xor_gate: [
    {
      id: 'xor-1',
      type: 'concept',
      icon: '🎯',
      title: 'The XOR Challenge',
      layers: [
        {
          content: 'XOR is special: it outputs 1 when inputs are DIFFERENT, 0 when they\'re the SAME.',
        },
        {
          content: 'This creates a diagonal pattern that a single line cannot separate.',
          details: 'This is called "non-linear separability" - one of the most important concepts in neural networks!',
        },
      ],
      question: {
        ask: 'Can you draw a single straight line that separates the 1s from the 0s in XOR?',
        options: ['Yes, easily!', 'No, impossible', 'Maybe with the right angle?'],
        answer: 'No, impossible',
        explanation: 'The outputs form a diagonal pattern: (0,0)→0, (1,1)→0 are on one diagonal, while (0,1)→1, (1,0)→1 are on the other. No single line can separate them!',
      },
    },
    {
      id: 'xor-2',
      type: 'experiment',
      icon: '🔬',
      title: 'See It Fail',
      layers: [
        {
          content: 'Try training with NO hidden layer [2, 1] and watch it struggle.',
        },
        {
          content: 'The network will plateau around 50% accuracy - basically random guessing!',
          details: 'This is the famous "XOR problem" that stumped AI researchers in the 1960s and led to the first "AI winter".',
        },
      ],
      experiment: {
        prompt: 'Set the architecture to [2, 1] (no hidden layer) and train.',
        action: 'Try It',
        expectedResult: 'The loss will decrease but accuracy will stay around 50%',
        successMessage: 'Notice how the decision boundary is just a straight line - it cannot curve to separate XOR!',
      },
    },
    {
      id: 'xor-3',
      type: 'insight',
      icon: '💡',
      title: 'The Hidden Layer Solution',
      layers: [
        {
          content: 'Add a hidden layer: [2, 4, 1]. Now the network can create a curved boundary!',
        },
        {
          content: 'The hidden layer transforms the input into a NEW space where the problem IS linearly separable.',
          details: 'This is the fundamental insight of deep learning: each layer creates new representations that make the problem easier to solve.',
        },
        {
          content: 'Watch the activations in the network visualization - each hidden neuron learns a different feature!',
        },
      ],
    },
  ],

  // ============================================================================
  // AND GATE - Simple, shows what ONE neuron can do
  // ============================================================================
  and_gate: [
    {
      id: 'and-1',
      type: 'concept',
      icon: '✨',
      title: 'Single Neuron Power',
      layers: [
        {
          content: 'AND gate outputs 1 only when BOTH inputs are 1. This is linearly separable!',
        },
        {
          content: 'A single line can separate the one "1" output from all the "0" outputs.',
        },
      ],
      question: {
        ask: 'Where would you draw a line to separate AND\'s outputs?',
        answer: 'Draw a line just below and left of the (1,1) point - it separates the single 1 from all the 0s!',
        explanation: 'Since only (1,1)→1 and all others →0, we need a line that puts (1,1) on one side and (0,0), (0,1), (1,0) on the other.',
      },
    },
    {
      id: 'and-2',
      type: 'experiment',
      icon: '🔬',
      title: 'Minimal Architecture',
      layers: [
        {
          content: 'Try with just [2, 1] - no hidden layer needed!',
        },
      ],
      experiment: {
        prompt: 'Train with architecture [2, 1] and watch how quickly it converges.',
        action: 'Train [2,1]',
        expectedResult: 'Should reach 100% accuracy in just a few epochs!',
        successMessage: 'A single neuron is all you need for linearly separable problems.',
      },
    },
  ],

  // ============================================================================
  // FAILURE: Zero Initialization
  // ============================================================================
  fail_zero_init: [
    {
      id: 'zero-1',
      type: 'warning',
      icon: '⚠️',
      title: 'The Symmetry Problem',
      layers: [
        {
          content: 'When all weights start at zero, every neuron computes the SAME thing.',
        },
        {
          content: 'During backpropagation, they all receive identical gradients and update identically.',
          details: 'This is called the "symmetry problem" - the network has hidden capacity it cannot use!',
        },
        {
          content: 'The hidden layer effectively becomes a single neuron, no matter how many you add.',
        },
      ],
    },
    {
      id: 'zero-2',
      type: 'experiment',
      icon: '🔬',
      title: 'Watch It Fail',
      layers: [
        {
          content: 'This problem intentionally uses zero initialization to demonstrate the issue.',
        },
      ],
      experiment: {
        prompt: 'Train and observe: the network cannot learn XOR despite having enough neurons!',
        action: 'Train',
        expectedResult: 'Accuracy stays around 50% - same as a single neuron',
        successMessage: 'All hidden neurons learned the same thing - symmetry was never broken!',
      },
    },
    {
      id: 'zero-3',
      type: 'insight',
      icon: '💡',
      title: 'The Fix',
      layers: [
        {
          content: 'Random initialization breaks symmetry - each neuron starts different and learns different features.',
        },
        {
          content: 'Modern initializations like Xavier and He are carefully designed to keep gradients healthy.',
          details: 'Xavier works well for sigmoid/tanh, He works better for ReLU activations.',
        },
      ],
    },
  ],

  // ============================================================================
  // FAILURE: Learning Rate Too High
  // ============================================================================
  fail_high_lr: [
    {
      id: 'lr-high-1',
      type: 'warning',
      icon: '📈',
      title: 'Explosive Learning',
      layers: [
        {
          content: 'Watch the loss graph - it\'s going to be chaotic!',
        },
        {
          content: 'With a learning rate that\'s too high, gradient descent overshoots the minimum.',
          details: 'Imagine trying to find the bottom of a valley by taking GIANT steps - you\'d bounce from hillside to hillside!',
        },
      ],
    },
    {
      id: 'lr-high-2',
      type: 'experiment',
      icon: '🔬',
      title: 'Watch the Chaos',
      layers: [
        {
          content: 'This problem uses a very high learning rate (10.0) to demonstrate the issue.',
        },
      ],
      experiment: {
        prompt: 'Train and watch the loss curve go wild!',
        action: 'Train',
        expectedResult: 'Loss will spike, oscillate wildly, or explode to NaN (Not a Number)',
        successMessage: 'See how unstable it is? The gradients are too large!',
      },
    },
    {
      id: 'lr-high-3',
      type: 'concept',
      icon: '🎯',
      title: 'Finding the Sweet Spot',
      layers: [
        {
          content: 'Good learning rates are typically between 0.001 and 0.1.',
        },
        {
          content: 'A common strategy: start with 0.01, reduce if unstable, increase if too slow.',
        },
      ],
      question: {
        ask: 'What happens if the learning rate is too LOW?',
        options: ['Network explodes', 'Training is very slow', 'Perfect training'],
        answer: 'Training is very slow',
        explanation: 'Too low and you\'ll need thousands of epochs. Too high and you overshoot. Finding the right balance is key!',
      },
    },
  ],

  // ============================================================================
  // SPIRAL - Deep network challenge
  // ============================================================================
  spiral: [
    {
      id: 'spiral-1',
      type: 'concept',
      icon: '🌀',
      title: 'The Spiral Challenge',
      layers: [
        {
          content: 'Two interleaving spirals - one of the hardest 2D classification problems!',
        },
        {
          content: 'Requires a very complex, winding decision boundary.',
        },
      ],
    },
    {
      id: 'spiral-2',
      type: 'experiment',
      icon: '🔬',
      title: 'Shallow vs Deep',
      layers: [
        {
          content: 'First try [2, 8, 1]. Then try [2, 8, 8, 1] or [2, 16, 16, 1].',
        },
      ],
      experiment: {
        prompt: 'Compare shallow vs deep architectures on this problem.',
        action: 'Try Different Depths',
        expectedResult: 'Deeper networks can capture the spiral pattern better',
        successMessage: 'More layers = more capacity for complex patterns!',
      },
    },
    {
      id: 'spiral-3',
      type: 'insight',
      icon: '⏳',
      title: 'Patience Required',
      layers: [
        {
          content: 'This problem needs many epochs (1000+) and careful tuning.',
        },
        {
          content: 'Try lower learning rate (0.1 or lower) if training is unstable.',
          details: 'Complex problems need both sufficient capacity AND enough training time.',
        },
      ],
    },
  ],
};

// =============================================================================
// EXPERIMENTS FOR KEY PROBLEMS
// Stand-alone guided experiments that walk through concepts step-by-step
// =============================================================================

export const EXPERIMENTS: Record<string, Experiment[]> = {
  xor_gate: [
    {
      id: 'xor-exp-1',
      title: 'The XOR Discovery',
      icon: '🔬',
      challenge: 'Discover why XOR is the gateway to deep learning!',
      steps: [
        {
          instruction: 'First, set the architecture to [2, 1] - just input and output, no hidden layer.',
          action: { type: 'set-architecture', value: [2, 1] },
        },
        {
          instruction: 'Now train the network and watch what happens.',
          action: { type: 'train' },
          observation: 'Notice the accuracy plateaus around 50% - no better than guessing!',
        },
        {
          instruction: 'Look at the decision boundary. What shape is it?',
          isQuestion: true,
          answer: 'It\'s a straight line - and no straight line can separate XOR\'s diagonal pattern!',
        },
        {
          instruction: 'Now change the architecture to [2, 4, 1] - adding 4 hidden neurons.',
          action: { type: 'set-architecture', value: [2, 4, 1] },
        },
        {
          instruction: 'Train again and observe the difference!',
          action: { type: 'train' },
          observation: 'The network can now reach 100% accuracy with a curved boundary!',
        },
      ],
      insight: 'Hidden layers enable networks to learn non-linear patterns. This is why we need "deep" learning!',
      celebrateOnComplete: true,
    },
  ],

  fail_zero_init: [
    {
      id: 'zero-exp-1',
      title: 'Breaking Symmetry',
      icon: '🔗',
      challenge: 'Understand why initialization matters for neural networks.',
      steps: [
        {
          instruction: 'This problem uses zero initialization. Train it and observe.',
          action: { type: 'train' },
          observation: 'Accuracy stays around 50% even though we have hidden neurons!',
        },
        {
          instruction: 'Why can\'t it learn? Think about what happens when all weights are zero...',
          isQuestion: true,
          answer: 'Every hidden neuron computes the exact same thing! They\'re all identical.',
        },
        {
          instruction: 'During backprop, each hidden neuron gets the SAME gradient.',
          observation: 'They update identically, so they STAY identical forever!',
        },
        {
          instruction: 'This is called the "symmetry problem" - the network can\'t differentiate its neurons.',
        },
      ],
      insight: 'Random initialization "breaks symmetry" - giving each neuron a unique starting point so they can learn different features.',
      celebrateOnComplete: true,
    },
  ],
};

// =============================================================================
// HELPER: Get rich hints for a problem
// Falls back to converting simple hints if no rich hints defined
// =============================================================================

export const getRichHintsForProblem = (problemId: string, fallbackHints?: string[]): RichHint[] => {
  if (RICH_HINTS[problemId]) {
    return RICH_HINTS[problemId];
  }

  // Convert simple hints to basic rich hints
  if (fallbackHints && fallbackHints.length > 0) {
    const defaultIcons = ['💡', '🔍', '🎯'];
    const defaultTypes: RichHint['type'][] = ['concept', 'insight', 'experiment'];

    return fallbackHints.map((hint, index) => ({
      id: `${problemId}-${index}`,
      type: defaultTypes[index % defaultTypes.length],
      icon: defaultIcons[index % defaultIcons.length],
      title: `Hint ${index + 1}`,
      layers: [{ content: hint }],
    }));
  }

  return [];
};

// =============================================================================
// HELPER: Get experiments for a problem
// =============================================================================

export const getExperimentsForProblem = (problemId: string): Experiment[] => {
  return EXPERIMENTS[problemId] || [];
};
