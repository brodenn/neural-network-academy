import { motion } from 'framer-motion';
import type { StepProgressData } from '../types';

interface PathProgressBarProps {
  steps: StepProgressData[];
  currentStep: number;
  onStepClick: (stepNumber: number) => void;
}

export const PathProgressBar = ({ steps, currentStep, onStepClick }: PathProgressBarProps) => {
  // Filter out any invalid steps (with undefined/null stepNumber)
  const validSteps = (steps || []).filter(
    (s): s is StepProgressData => s != null && typeof s.stepNumber === 'number'
  );

  // Don't render if no valid steps
  if (validSteps.length === 0) {
    return (
      <div className="bg-gray-800 rounded-lg p-4">
        <div className="text-gray-500 text-sm">Loading steps...</div>
      </div>
    );
  }

  return (
    <div className="bg-gray-800 rounded-lg p-4">
      <div className="flex items-center justify-between mb-1">
        <span className="text-xs text-gray-500">Click circles to navigate steps</span>
        <span className="text-xs text-gray-500">{validSteps.filter(s => s.completed).length}/{validSteps.length} complete</span>
      </div>
      <div className="flex items-center justify-between relative">
        {/* Connection line */}
        <div className="absolute top-1/2 left-0 right-0 h-0.5 bg-gray-700 -translate-y-1/2 z-0" />

        {/* Progress line (filled portion) */}
        <motion.div
          className="absolute top-1/2 left-0 h-0.5 bg-green-500 -translate-y-1/2 z-0"
          initial={{ width: 0 }}
          animate={{
            width: `${(validSteps.filter(s => s.completed).length / Math.max(validSteps.length - 1, 1)) * 100}%`
          }}
          transition={{ duration: 0.5, ease: 'easeOut' }}
        />

        {validSteps.map((step) => {
          const isLocked = !step.unlocked;
          const isCompleted = step.completed;
          const isCurrent = step.stepNumber === currentStep;
          const isClickable = step.unlocked;

          return (
            <motion.button
              key={step.stepNumber}
              onClick={() => isClickable && onStepClick(step.stepNumber)}
              disabled={isLocked}
              className={`relative z-10 w-10 h-10 rounded-full flex items-center justify-center border-2 transition-colors ${
                isLocked
                  ? 'bg-gray-700 border-gray-600 cursor-not-allowed'
                  : isCompleted
                  ? 'bg-green-600 border-green-500 cursor-pointer hover:bg-green-500'
                  : isCurrent
                  ? 'bg-blue-600 border-blue-400 cursor-pointer'
                  : 'bg-gray-700 border-gray-500 cursor-pointer hover:bg-gray-600'
              }`}
              whileHover={isClickable ? { scale: 1.1 } : {}}
              whileTap={isClickable ? { scale: 0.95 } : {}}
              animate={isCurrent ? {
                boxShadow: ['0 0 0 0 rgba(59, 130, 246, 0)', '0 0 0 8px rgba(59, 130, 246, 0.3)', '0 0 0 0 rgba(59, 130, 246, 0)']
              } : {}}
              transition={isCurrent ? {
                duration: 2,
                repeat: Infinity,
                repeatType: 'loop'
              } : {}}
            >
              {isLocked ? (
                <svg className="w-4 h-4 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                    d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
                </svg>
              ) : isCompleted ? (
                <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
              ) : (
                <span className="text-sm font-bold text-white">{step.stepNumber}</span>
              )}
            </motion.button>
          );
        })}
      </div>

      {/* Step labels */}
      <div className="flex items-center justify-between mt-2">
        {validSteps.map((step) => {
          const isLocked = !step.unlocked;
          const isCompleted = step.completed;
          const isCurrent = step.stepNumber === currentStep;

          return (
            <div
              key={`label-${step.stepNumber}`}
              className={`text-xs text-center w-10 ${
                isLocked
                  ? 'text-gray-600'
                  : isCompleted
                  ? 'text-green-400'
                  : isCurrent
                  ? 'text-blue-400 font-medium'
                  : 'text-gray-400'
              }`}
            >
              Step {step.stepNumber}
            </div>
          );
        })}
      </div>
    </div>
  );
};
