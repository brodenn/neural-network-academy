import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import confetti from 'canvas-confetti';

// =============================================================================
// TYPES
// =============================================================================

export interface Experiment {
  id: string;
  title: string;
  icon: string;
  challenge: string;
  steps: ExperimentStep[];
  insight: string;
  celebrateOnComplete?: boolean;
}

export interface ExperimentStep {
  instruction: string;
  action?: {
    type: 'set-architecture' | 'train' | 'observe';
    value?: unknown;
    target?: string;
  };
  observation?: string;
  isQuestion?: boolean;
  answer?: string;
}

interface ExperimentCardProps {
  experiment: Experiment;
  isUnlocked: boolean;
  onAction?: (action: ExperimentStep['action']) => void;
}

// =============================================================================
// COMPONENT
// =============================================================================

export const ExperimentCard = ({
  experiment,
  isUnlocked,
  onAction,
}: ExperimentCardProps) => {
  const [currentStep, setCurrentStep] = useState(0);
  const [isStarted, setIsStarted] = useState(false);
  const [completedSteps, setCompletedSteps] = useState<Set<number>>(new Set());
  const [showAnswer, setShowAnswer] = useState<number | null>(null);
  const [isComplete, setIsComplete] = useState(false);

  const handleStartExperiment = () => {
    setIsStarted(true);
    setCurrentStep(0);
  };

  const handleCompleteStep = (stepIndex: number) => {
    const step = experiment.steps[stepIndex];

    // Trigger action if defined
    if (step.action && onAction) {
      onAction(step.action);
    }

    setCompletedSteps(prev => new Set([...prev, stepIndex]));

    // Auto-advance to next step
    if (stepIndex < experiment.steps.length - 1) {
      setTimeout(() => setCurrentStep(stepIndex + 1), 500);
    } else {
      // Experiment complete!
      setIsComplete(true);
      if (experiment.celebrateOnComplete) {
        confetti({
          particleCount: 80,
          spread: 60,
          origin: { y: 0.7 },
          colors: ['#8B5CF6', '#A855F7', '#C084FC'],
        });
      }
    }
  };

  const handleRevealAnswer = (stepIndex: number) => {
    setShowAnswer(stepIndex);
  };

  // Locked state
  if (!isUnlocked) {
    return (
      <div className="bg-gray-800/30 border border-gray-700/50 rounded-xl p-4 opacity-50">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-full bg-gray-700 flex items-center justify-center">
            <span className="text-xl">🔒</span>
          </div>
          <div>
            <p className="text-gray-400 font-medium">{experiment.title}</p>
            <p className="text-gray-500 text-sm">Complete previous hints to unlock</p>
          </div>
        </div>
      </div>
    );
  }

  // Not started state
  if (!isStarted) {
    return (
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-gradient-to-br from-purple-900/40 to-blue-900/40 border border-purple-500/30 rounded-xl overflow-hidden"
      >
        <div className="p-4">
          <div className="flex items-center gap-3 mb-3">
            <motion.div
              className="w-12 h-12 rounded-full bg-purple-600 flex items-center justify-center"
              animate={{ scale: [1, 1.1, 1] }}
              transition={{ duration: 2, repeat: Infinity }}
            >
              <span className="text-2xl">{experiment.icon}</span>
            </motion.div>
            <div>
              <p className="text-white font-bold">{experiment.title}</p>
              <p className="text-purple-300 text-sm">{experiment.steps.length} steps</p>
            </div>
          </div>

          <p className="text-gray-200 mb-4">{experiment.challenge}</p>

          <motion.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            onClick={handleStartExperiment}
            className="w-full py-3 bg-purple-600 hover:bg-purple-500 rounded-lg font-bold text-white flex items-center justify-center gap-2 transition-colors"
          >
            <span>Start Experiment</span>
            <motion.span
              animate={{ x: [0, 5, 0] }}
              transition={{ duration: 1, repeat: Infinity }}
            >
              →
            </motion.span>
          </motion.button>
        </div>
      </motion.div>
    );
  }

  // In progress or complete
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="bg-gradient-to-br from-purple-900/40 to-blue-900/40 border border-purple-500/30 rounded-xl overflow-hidden"
    >
      {/* Header */}
      <div className="p-4 border-b border-purple-500/20">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-full bg-purple-600 flex items-center justify-center">
              <span className="text-xl">{experiment.icon}</span>
            </div>
            <p className="text-white font-bold">{experiment.title}</p>
          </div>
          <div className="text-sm text-purple-300">
            {completedSteps.size}/{experiment.steps.length}
          </div>
        </div>

        {/* Progress bar */}
        <div className="mt-3 h-1 bg-gray-700 rounded-full overflow-hidden">
          <motion.div
            className="h-full bg-purple-500"
            initial={{ width: 0 }}
            animate={{ width: `${(completedSteps.size / experiment.steps.length) * 100}%` }}
            transition={{ type: 'spring', stiffness: 100 }}
          />
        </div>
      </div>

      {/* Steps */}
      <div className="p-4 space-y-3">
        {experiment.steps.map((step, index) => {
          const isCompleted = completedSteps.has(index);
          const isCurrent = index === currentStep && !isComplete;
          const isLocked = index > currentStep && !isCompleted;

          return (
            <motion.div
              key={index}
              initial={{ opacity: 0, x: -20 }}
              animate={{
                opacity: isLocked ? 0.4 : 1,
                x: 0,
              }}
              transition={{ delay: index * 0.1 }}
              className={`relative pl-8 ${isLocked ? 'pointer-events-none' : ''}`}
            >
              {/* Step indicator */}
              <div className="absolute left-0 top-0">
                {isCompleted ? (
                  <motion.div
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    className="w-6 h-6 rounded-full bg-green-500 flex items-center justify-center"
                  >
                    <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                    </svg>
                  </motion.div>
                ) : isCurrent ? (
                  <motion.div
                    animate={{ scale: [1, 1.2, 1] }}
                    transition={{ duration: 1, repeat: Infinity }}
                    className="w-6 h-6 rounded-full bg-purple-500 flex items-center justify-center text-white text-sm font-bold"
                  >
                    {index + 1}
                  </motion.div>
                ) : (
                  <div className="w-6 h-6 rounded-full bg-gray-600 flex items-center justify-center text-gray-400 text-sm">
                    {index + 1}
                  </div>
                )}
              </div>

              {/* Step content */}
              <div>
                <p className={`text-sm ${isCurrent ? 'text-white' : isCompleted ? 'text-gray-400' : 'text-gray-500'}`}>
                  {step.instruction}
                </p>

                {/* Observation (shown after completing step) */}
                <AnimatePresence>
                  {isCompleted && step.observation && (
                    <motion.div
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: 'auto' }}
                      exit={{ opacity: 0, height: 0 }}
                      className="mt-2 p-2 bg-blue-900/30 rounded border border-blue-500/30"
                    >
                      <p className="text-blue-300 text-xs flex items-center gap-2">
                        <span>👁️</span>
                        {step.observation}
                      </p>
                    </motion.div>
                  )}
                </AnimatePresence>

                {/* Question with answer reveal */}
                {step.isQuestion && isCurrent && (
                  <div className="mt-2">
                    {showAnswer === index ? (
                      <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        className="p-2 bg-amber-900/30 rounded border border-amber-500/30"
                      >
                        <p className="text-amber-300 text-xs">{step.answer}</p>
                      </motion.div>
                    ) : (
                      <button
                        onClick={() => handleRevealAnswer(index)}
                        className="text-amber-400 text-xs hover:text-amber-300 underline"
                      >
                        Show answer
                      </button>
                    )}
                  </div>
                )}

                {/* Action button for current step */}
                {isCurrent && !isCompleted && (
                  <motion.button
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    onClick={() => handleCompleteStep(index)}
                    className="mt-2 px-3 py-1.5 bg-purple-600 hover:bg-purple-500 rounded text-sm font-medium transition-colors"
                  >
                    {step.action ? 'Do It!' : step.isQuestion ? 'Got It!' : 'Done'}
                  </motion.button>
                )}
              </div>
            </motion.div>
          );
        })}
      </div>

      {/* Completion insight */}
      <AnimatePresence>
        {isComplete && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="p-4 bg-green-900/30 border-t border-green-500/30"
          >
            <div className="flex items-center gap-2 mb-2">
              <span className="text-xl">💡</span>
              <span className="text-green-400 font-bold">Key Insight</span>
            </div>
            <p className="text-gray-200 text-sm">{experiment.insight}</p>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
};

export default ExperimentCard;
