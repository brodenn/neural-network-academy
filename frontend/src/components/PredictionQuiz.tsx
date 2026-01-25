import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

// =============================================================================
// TYPES
// =============================================================================

export interface PredictionOption {
  id: string;
  text: string;
  isCorrect: boolean;
  explanation: string;
}

export interface PredictionQuizProps {
  question: string;
  context: {
    architecture: number[];
    learningRate: number;
    epochs: number;
    problem: string;
  };
  options: PredictionOption[];
  onAnswer: (correct: boolean, selectedId: string) => void;
  onRevealAndTrain: () => void;
}

// =============================================================================
// COMPONENT
// =============================================================================

export const PredictionQuiz = ({
  question,
  context,
  options,
  onAnswer,
  onRevealAndTrain,
}: PredictionQuizProps) => {
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [revealed, setRevealed] = useState(false);
  const [showExplanation, setShowExplanation] = useState(false);

  const selectedOption = options.find(o => o.id === selectedId);
  const correctOption = options.find(o => o.isCorrect);

  const handleSelect = (id: string) => {
    if (revealed) return;
    setSelectedId(id);
  };

  const handleReveal = () => {
    if (!selectedId) return;
    setRevealed(true);
    setShowExplanation(true);
    onAnswer(selectedOption?.isCorrect ?? false, selectedId);
  };

  const handleTrainAndSee = () => {
    onRevealAndTrain();
  };

  return (
    <div className="bg-gray-800 rounded-lg border border-gray-700 overflow-hidden">
      {/* Header */}
      <div className="bg-purple-900/30 border-b border-purple-600/30 px-4 py-3">
        <div className="flex items-center gap-2">
          <span className="text-2xl">🤔</span>
          <h3 className="text-lg font-semibold text-white">Predict the Outcome</h3>
        </div>
      </div>

      {/* Context */}
      <div className="px-4 py-3 bg-gray-900/50 border-b border-gray-700">
        <p className="text-sm text-gray-400 mb-2">Configuration:</p>
        <div className="flex flex-wrap gap-2 text-sm">
          <span className="px-2 py-1 bg-blue-900/50 rounded text-blue-300">
            Architecture: [{context.architecture.join(', ')}]
          </span>
          <span className="px-2 py-1 bg-amber-900/50 rounded text-amber-300">
            LR: {context.learningRate}
          </span>
          <span className="px-2 py-1 bg-green-900/50 rounded text-green-300">
            Epochs: {context.epochs}
          </span>
          <span className="px-2 py-1 bg-purple-900/50 rounded text-purple-300">
            Problem: {context.problem}
          </span>
        </div>
      </div>

      {/* Question */}
      <div className="px-4 py-4">
        <p className="text-white text-lg mb-4">{question}</p>

        {/* Options */}
        <div className="space-y-2">
          {options.map((option) => {
            const isSelected = selectedId === option.id;
            const isCorrect = option.isCorrect;

            let bgClass = 'bg-gray-700/50 hover:bg-gray-700 border-gray-600';
            let textClass = 'text-gray-200';

            if (revealed) {
              if (isCorrect) {
                bgClass = 'bg-green-900/50 border-green-500';
                textClass = 'text-green-300';
              } else if (isSelected && !isCorrect) {
                bgClass = 'bg-red-900/50 border-red-500';
                textClass = 'text-red-300';
              } else {
                bgClass = 'bg-gray-800/50 border-gray-700';
                textClass = 'text-gray-500';
              }
            } else if (isSelected) {
              bgClass = 'bg-purple-900/50 border-purple-500';
              textClass = 'text-purple-200';
            }

            return (
              <motion.button
                key={option.id}
                onClick={() => handleSelect(option.id)}
                disabled={revealed}
                whileHover={!revealed ? { scale: 1.01 } : {}}
                whileTap={!revealed ? { scale: 0.99 } : {}}
                className={`w-full text-left px-4 py-3 rounded-lg border-2 transition-colors ${bgClass}`}
              >
                <div className="flex items-center gap-3">
                  <div className={`w-6 h-6 rounded-full border-2 flex items-center justify-center ${
                    isSelected ? 'border-current bg-current/20' : 'border-gray-500'
                  }`}>
                    {revealed && isCorrect && <span className="text-green-400">✓</span>}
                    {revealed && isSelected && !isCorrect && <span className="text-red-400">✗</span>}
                    {!revealed && isSelected && <span className="text-purple-400">●</span>}
                  </div>
                  <span className={textClass}>{option.text}</span>
                </div>
              </motion.button>
            );
          })}
        </div>

        {/* Explanation */}
        <AnimatePresence>
          {showExplanation && selectedOption && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="mt-4"
            >
              <div className={`p-4 rounded-lg border ${
                selectedOption.isCorrect
                  ? 'bg-green-900/30 border-green-600/50'
                  : 'bg-amber-900/30 border-amber-600/50'
              }`}>
                <div className="flex items-center gap-2 mb-2">
                  {selectedOption.isCorrect ? (
                    <>
                      <span className="text-xl">🎉</span>
                      <span className="text-green-400 font-medium">Correct!</span>
                    </>
                  ) : (
                    <>
                      <span className="text-xl">💡</span>
                      <span className="text-amber-400 font-medium">Not quite!</span>
                    </>
                  )}
                </div>
                <p className="text-gray-200 text-sm">{selectedOption.explanation}</p>
                {!selectedOption.isCorrect && correctOption && (
                  <p className="text-green-300 text-sm mt-2">
                    <strong>The correct answer:</strong> {correctOption.text}
                  </p>
                )}
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Actions */}
        <div className="mt-4 flex gap-2">
          {!revealed ? (
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={handleReveal}
              disabled={!selectedId}
              className={`flex-1 py-2 rounded-lg font-medium transition-colors ${
                selectedId
                  ? 'bg-purple-600 hover:bg-purple-500 text-white'
                  : 'bg-gray-700 text-gray-500 cursor-not-allowed'
              }`}
            >
              Check My Prediction
            </motion.button>
          ) : (
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={handleTrainAndSee}
              className="flex-1 py-2 bg-green-600 hover:bg-green-500 rounded-lg text-white font-medium"
            >
              Train & See It Happen
            </motion.button>
          )}
        </div>
      </div>
    </div>
  );
};

// =============================================================================
// PRESET QUIZZES
// =============================================================================

export const PREDICTION_QUIZZES = {
  xor_no_hidden: {
    question: 'What will happen when we train XOR with NO hidden layer?',
    context: {
      architecture: [2, 1],
      learningRate: 0.5,
      epochs: 500,
      problem: 'XOR Gate',
    },
    options: [
      {
        id: 'converge',
        text: 'It will converge to 100% accuracy',
        isCorrect: false,
        explanation: 'Without a hidden layer, the network can only learn linear boundaries. XOR requires a non-linear boundary.',
      },
      {
        id: 'stuck',
        text: 'Accuracy will get stuck around 50%',
        isCorrect: true,
        explanation: 'Correct! XOR is not linearly separable. A single neuron can only draw a straight line, but XOR outputs form a diagonal pattern that no straight line can separate.',
      },
      {
        id: 'slow',
        text: 'It will learn slowly but eventually succeed',
        isCorrect: false,
        explanation: 'No amount of training will help - the architecture fundamentally cannot solve this problem.',
      },
      {
        id: 'explode',
        text: 'The loss will explode to infinity',
        isCorrect: false,
        explanation: 'Loss explosion typically happens with too high learning rate, not architectural limitations.',
      },
    ],
  },

  high_lr: {
    question: 'What will happen with a learning rate of 10?',
    context: {
      architecture: [2, 4, 1],
      learningRate: 10,
      epochs: 100,
      problem: 'XOR Gate',
    },
    options: [
      {
        id: 'fast',
        text: 'It will learn faster than normal',
        isCorrect: false,
        explanation: 'Higher LR means bigger steps, but too big means overshooting the optimum.',
      },
      {
        id: 'oscillate',
        text: 'Loss will oscillate wildly or explode',
        isCorrect: true,
        explanation: 'Correct! With LR=10, gradient steps are so large that the network overshoots the minimum, bouncing around or exploding to NaN.',
      },
      {
        id: 'same',
        text: 'Same as normal, LR doesn\'t matter much',
        isCorrect: false,
        explanation: 'Learning rate is one of the most critical hyperparameters!',
      },
      {
        id: 'plateau',
        text: 'It will plateau early',
        isCorrect: false,
        explanation: 'Plateaus typically happen with too LOW learning rate, not too high.',
      },
    ],
  },

  zero_init: {
    question: 'What happens when all weights start at zero?',
    context: {
      architecture: [2, 4, 1],
      learningRate: 0.5,
      epochs: 500,
      problem: 'XOR Gate',
    },
    options: [
      {
        id: 'normal',
        text: 'Training proceeds normally',
        isCorrect: false,
        explanation: 'Zero initialization breaks the network in a subtle but fundamental way.',
      },
      {
        id: 'symmetry',
        text: 'All hidden neurons will learn the same thing',
        isCorrect: true,
        explanation: 'Correct! With zero weights, all hidden neurons compute identical values and receive identical gradients. They stay identical forever - this is called the "symmetry problem".',
      },
      {
        id: 'no_learn',
        text: 'Network won\'t learn at all (0% accuracy)',
        isCorrect: false,
        explanation: 'The network does learn something, but it\'s as if you only have 1 hidden neuron instead of 4.',
      },
      {
        id: 'faster',
        text: 'It will learn faster from a clean slate',
        isCorrect: false,
        explanation: 'Random initialization is crucial for breaking symmetry and allowing neurons to specialize.',
      },
    ],
  },

  and_simple: {
    question: 'Does the AND gate need a hidden layer?',
    context: {
      architecture: [2, 1],
      learningRate: 0.5,
      epochs: 100,
      problem: 'AND Gate',
    },
    options: [
      {
        id: 'needs_hidden',
        text: 'Yes, all problems need hidden layers',
        isCorrect: false,
        explanation: 'Not all problems need hidden layers - it depends on whether the problem is linearly separable.',
      },
      {
        id: 'no_hidden',
        text: 'No, AND is linearly separable',
        isCorrect: true,
        explanation: 'Correct! AND outputs 1 only for (1,1). A single line can separate this from all other inputs. One neuron is enough!',
      },
      {
        id: 'maybe',
        text: 'It depends on the learning rate',
        isCorrect: false,
        explanation: 'The need for hidden layers depends on the problem\'s geometry, not hyperparameters.',
      },
      {
        id: 'more_neurons',
        text: 'Yes, need at least 4 neurons',
        isCorrect: false,
        explanation: 'AND is one of the simplest problems - just 1 output neuron is sufficient.',
      },
    ],
  },

  deep_vs_shallow: {
    question: 'For the spiral problem, which architecture will work better?',
    context: {
      architecture: [2, 4, 1],
      learningRate: 0.5,
      epochs: 1000,
      problem: 'Spiral',
    },
    options: [
      {
        id: 'shallow_wide',
        text: '[2, 16, 1] - shallow but wide',
        isCorrect: false,
        explanation: 'Width helps but the spiral needs the compositional power of depth.',
      },
      {
        id: 'deep_narrow',
        text: '[2, 8, 8, 8, 1] - deep and narrow',
        isCorrect: true,
        explanation: 'Correct! Deeper networks can learn more complex, hierarchical features. The spiral\'s winding boundary benefits from depth.',
      },
      {
        id: 'minimal',
        text: '[2, 4, 1] - minimal',
        isCorrect: false,
        explanation: 'This might work eventually but will struggle with the complex spiral pattern.',
      },
      {
        id: 'same',
        text: 'They\'re all equivalent',
        isCorrect: false,
        explanation: 'Architecture significantly impacts what a network can learn!',
      },
    ],
  },
};

export type QuizId = keyof typeof PREDICTION_QUIZZES;

export default PredictionQuiz;
