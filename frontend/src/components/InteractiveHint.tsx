import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import * as Accordion from '@radix-ui/react-accordion';

// =============================================================================
// TYPES
// =============================================================================

export type HintType = 'concept' | 'experiment' | 'question' | 'warning' | 'insight';

export interface HintLayer {
  content: string;
  details?: string;
  visual?: 'decision-boundary' | 'network-diagram' | 'loss-curve';
}

export interface RichHint {
  id: string;
  type: HintType;
  icon: string;
  title: string;
  layers: HintLayer[];
  experiment?: {
    prompt: string;
    action: string;
    expectedResult: string;
    successMessage: string;
  };
  question?: {
    ask: string;
    options?: string[];
    answer: string;
    explanation: string;
  };
}

interface InteractiveHintProps {
  hint: RichHint;
  isUnlocked: boolean;
  isRevealed: boolean;
  onReveal: () => void;
  attemptCount: number;
}

// =============================================================================
// STYLES
// =============================================================================

const typeStyles: Record<HintType, { bg: string; border: string; iconBg: string }> = {
  concept: {
    bg: 'bg-blue-900/30',
    border: 'border-blue-600/50',
    iconBg: 'bg-blue-600',
  },
  experiment: {
    bg: 'bg-purple-900/30',
    border: 'border-purple-600/50',
    iconBg: 'bg-purple-600',
  },
  question: {
    bg: 'bg-amber-900/30',
    border: 'border-amber-600/50',
    iconBg: 'bg-amber-600',
  },
  warning: {
    bg: 'bg-red-900/30',
    border: 'border-red-600/50',
    iconBg: 'bg-red-600',
  },
  insight: {
    bg: 'bg-green-900/30',
    border: 'border-green-600/50',
    iconBg: 'bg-green-600',
  },
};

// =============================================================================
// ANIMATION VARIANTS
// =============================================================================

const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.1,
      delayChildren: 0.1,
    },
  },
};

const itemVariants = {
  hidden: { opacity: 0, y: 10 },
  visible: {
    opacity: 1,
    y: 0,
    transition: {
      type: 'spring' as const,
      stiffness: 300,
      damping: 24,
    },
  },
};

// =============================================================================
// COMPONENT
// =============================================================================

export const InteractiveHint = ({
  hint,
  isUnlocked,
  isRevealed,
  onReveal,
  attemptCount,
}: InteractiveHintProps) => {
  const [expandedLayers, setExpandedLayers] = useState<string[]>([]);
  const [questionAnswered, setQuestionAnswered] = useState(false);
  const [selectedAnswer, setSelectedAnswer] = useState<string | null>(null);
  const [showExperimentResult, setShowExperimentResult] = useState(false);

  const styles = typeStyles[hint.type];

  // Locked state
  if (!isUnlocked) {
    return (
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        className="bg-gray-800/50 border border-gray-700 rounded-lg p-4 flex items-center gap-3"
      >
        <div className="w-8 h-8 rounded-full bg-gray-700 flex items-center justify-center">
          <span className="text-gray-500">🔒</span>
        </div>
        <div className="flex-1">
          <p className="text-gray-500 text-sm">
            {attemptCount === 0
              ? 'Start training to unlock hints'
              : `Keep trying! Hint unlocks with more practice`}
          </p>
        </div>
      </motion.div>
    );
  }

  // Unlocked but not revealed
  if (!isRevealed) {
    return (
      <motion.button
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        whileHover={{ scale: 1.02 }}
        whileTap={{ scale: 0.98 }}
        onClick={onReveal}
        className={`w-full ${styles.bg} border ${styles.border} rounded-lg p-4 flex items-center gap-3 text-left transition-colors hover:bg-opacity-50`}
      >
        <motion.div
          className={`w-8 h-8 rounded-full ${styles.iconBg} flex items-center justify-center`}
          animate={{ rotate: [0, 10, -10, 0] }}
          transition={{ duration: 0.5, repeat: 2 }}
        >
          <span>{hint.icon}</span>
        </motion.div>
        <div className="flex-1">
          <p className="text-white font-medium">{hint.title}</p>
          <p className="text-gray-400 text-sm">Click to reveal</p>
        </div>
        <motion.div
          animate={{ x: [0, 5, 0] }}
          transition={{ duration: 1, repeat: Infinity }}
        >
          <svg className="w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
          </svg>
        </motion.div>
      </motion.button>
    );
  }

  // Fully revealed with interactive content
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      className={`${styles.bg} border ${styles.border} rounded-lg overflow-hidden`}
    >
      {/* Header */}
      <div className="p-4 flex items-center gap-3 border-b border-gray-700/50">
        <div className={`w-8 h-8 rounded-full ${styles.iconBg} flex items-center justify-center`}>
          <span>{hint.icon}</span>
        </div>
        <p className="text-white font-medium">{hint.title}</p>
      </div>

      {/* Content with progressive layers */}
      <motion.div
        className="p-4"
        variants={containerVariants}
        initial="hidden"
        animate="visible"
      >
        <Accordion.Root
          type="multiple"
          value={expandedLayers}
          onValueChange={setExpandedLayers}
          className="space-y-2"
        >
          {hint.layers.map((layer, index) => (
            <Accordion.Item key={index} value={`layer-${index}`}>
              <motion.div variants={itemVariants}>
                {/* First layer always visible */}
                {index === 0 ? (
                  <div className="text-gray-200">{layer.content}</div>
                ) : (
                  <>
                    <Accordion.Trigger className="group flex items-center gap-2 text-blue-400 hover:text-blue-300 text-sm mt-2">
                      <motion.svg
                        className="w-4 h-4"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                        animate={{ rotate: expandedLayers.includes(`layer-${index}`) ? 90 : 0 }}
                      >
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                      </motion.svg>
                      <span>{layer.details ? 'Why?' : 'Learn more'}</span>
                    </Accordion.Trigger>
                    <Accordion.Content className="overflow-hidden data-[state=open]:animate-slideDown data-[state=closed]:animate-slideUp">
                      <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        className="pt-2 pl-6 text-gray-300 text-sm"
                      >
                        {layer.content}
                        {layer.details && (
                          <p className="mt-2 text-gray-400 italic">{layer.details}</p>
                        )}
                      </motion.div>
                    </Accordion.Content>
                  </>
                )}
              </motion.div>
            </Accordion.Item>
          ))}
        </Accordion.Root>

        {/* Experiment section */}
        {hint.experiment && (
          <motion.div
            variants={itemVariants}
            className="mt-4 p-3 bg-purple-900/30 rounded-lg border border-purple-600/30"
          >
            <div className="flex items-center gap-2 mb-2">
              <span className="text-lg">🔬</span>
              <span className="text-purple-300 font-medium text-sm">Try It Yourself</span>
            </div>
            <p className="text-gray-200 text-sm mb-3">{hint.experiment.prompt}</p>

            <div className="flex gap-2">
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={() => setShowExperimentResult(true)}
                className="px-3 py-1.5 bg-purple-600 hover:bg-purple-500 rounded text-sm font-medium transition-colors"
              >
                {hint.experiment.action}
              </motion.button>
            </div>

            <AnimatePresence>
              {showExperimentResult && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                  className="mt-3 pt-3 border-t border-purple-600/30"
                >
                  <p className="text-green-400 text-sm flex items-center gap-2">
                    <span>✓</span>
                    {hint.experiment.successMessage}
                  </p>
                  <p className="text-gray-300 text-sm mt-1">{hint.experiment.expectedResult}</p>
                </motion.div>
              )}
            </AnimatePresence>
          </motion.div>
        )}

        {/* Question section */}
        {hint.question && (
          <motion.div
            variants={itemVariants}
            className="mt-4 p-3 bg-amber-900/30 rounded-lg border border-amber-600/30"
          >
            <div className="flex items-center gap-2 mb-2">
              <span className="text-lg">🤔</span>
              <span className="text-amber-300 font-medium text-sm">Think About It</span>
            </div>
            <p className="text-gray-200 text-sm mb-3">{hint.question.ask}</p>

            {hint.question.options && !questionAnswered && (
              <div className="space-y-2">
                {hint.question.options.map((option, i) => (
                  <motion.button
                    key={i}
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    onClick={() => {
                      setSelectedAnswer(option);
                      setQuestionAnswered(true);
                    }}
                    className="w-full text-left px-3 py-2 bg-gray-800/50 hover:bg-gray-700/50 rounded text-sm text-gray-200 transition-colors"
                  >
                    {option}
                  </motion.button>
                ))}
              </div>
            )}

            <AnimatePresence>
              {questionAnswered && (
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="mt-3 pt-3 border-t border-amber-600/30"
                >
                  {selectedAnswer === hint.question.answer ? (
                    <p className="text-green-400 text-sm flex items-center gap-2">
                      <span>🎉</span> Correct!
                    </p>
                  ) : (
                    <p className="text-amber-400 text-sm flex items-center gap-2">
                      <span>💡</span> The answer is: {hint.question.answer}
                    </p>
                  )}
                  <p className="text-gray-300 text-sm mt-2">{hint.question.explanation}</p>
                </motion.div>
              )}
            </AnimatePresence>

            {!hint.question.options && (
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={() => setQuestionAnswered(true)}
                className="px-3 py-1.5 bg-amber-600 hover:bg-amber-500 rounded text-sm font-medium transition-colors"
              >
                Reveal Answer
              </motion.button>
            )}
          </motion.div>
        )}
      </motion.div>
    </motion.div>
  );
};

export default InteractiveHint;
