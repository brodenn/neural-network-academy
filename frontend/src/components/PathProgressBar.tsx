import { motion } from 'framer-motion';
import * as Tooltip from '@radix-ui/react-tooltip';
import type { StepProgressData } from '../types';

interface StepInfo {
  stepNumber: number;
  title: string;
  problemId: string;
}

interface PathProgressBarProps {
  steps: StepProgressData[];
  stepInfo?: StepInfo[];
  currentStep: number;
  onStepClick: (stepNumber: number) => void;
}

export const PathProgressBar = ({ steps, stepInfo, currentStep, onStepClick }: PathProgressBarProps) => {
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

  // Get step title from stepInfo if available
  const getStepTitle = (stepNumber: number): string => {
    const info = stepInfo?.find(s => s.stepNumber === stepNumber);
    return info?.title || `Step ${stepNumber}`;
  };

  return (
    <Tooltip.Provider delayDuration={200}>
      <nav
        className="bg-gray-800 rounded-lg p-4"
        role="navigation"
        aria-label="Learning path progress"
      >
        <div className="flex items-center justify-between mb-1">
          <span className="text-xs text-gray-500">Hover for step details</span>
          <span className="text-xs text-gray-500" aria-live="polite">
            {validSteps.filter(s => s.completed).length}/{validSteps.length} complete
          </span>
        </div>
        <div className="flex items-center justify-between relative" role="list">
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
            const title = getStepTitle(step.stepNumber);

            // Build tooltip content
            const tooltipContent = (
              <div className="text-center">
                <div className="font-medium text-white">{title}</div>
                <div className="text-xs text-gray-300 mt-1">
                  {isLocked ? (
                    <span className="text-gray-400">🔒 Locked</span>
                  ) : isCompleted ? (
                    <span className="text-green-400">✓ Completed ({(step.bestAccuracy * 100).toFixed(0)}%)</span>
                  ) : isCurrent ? (
                    <span className="text-blue-400">● Current Step</span>
                  ) : (
                    <span className="text-gray-400">○ Not started</span>
                  )}
                </div>
                {step.attempts > 0 && (
                  <div className="text-xs text-gray-400 mt-0.5">
                    {step.attempts} attempt{step.attempts !== 1 ? 's' : ''}
                  </div>
                )}
              </div>
            );

            return (
              <Tooltip.Root key={step.stepNumber}>
                <Tooltip.Trigger asChild>
                  <motion.button
                    role="listitem"
                    onClick={() => isClickable && onStepClick(step.stepNumber)}
                    disabled={isLocked}
                    aria-current={isCurrent ? 'step' : undefined}
                    aria-label={`${title}${isCompleted ? ', completed' : isLocked ? ', locked' : isCurrent ? ', current' : ''}`}
                    data-active={isCurrent || undefined}
                    data-completed={isCompleted || undefined}
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
                </Tooltip.Trigger>
                <Tooltip.Portal>
                  <Tooltip.Content
                    className="bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 shadow-xl z-50"
                    sideOffset={8}
                  >
                    {tooltipContent}
                    <Tooltip.Arrow className="fill-gray-900" />
                  </Tooltip.Content>
                </Tooltip.Portal>
              </Tooltip.Root>
            );
          })}
        </div>

        {/* Step labels - now show truncated title */}
        <div className="flex items-center justify-between mt-2">
          {validSteps.map((step) => {
            const isLocked = !step.unlocked;
            const isCompleted = step.completed;
            const isCurrent = step.stepNumber === currentStep;
            const title = getStepTitle(step.stepNumber);
            // Truncate title for label (show first word or first 8 chars)
            const shortTitle = title.split(' ')[0].slice(0, 8);

            return (
              <div
                key={`label-${step.stepNumber}`}
                className={`text-xs text-center w-10 truncate ${
                  isLocked
                    ? 'text-gray-600'
                    : isCompleted
                    ? 'text-green-400'
                    : isCurrent
                    ? 'text-blue-400 font-medium'
                    : 'text-gray-400'
                }`}
                title={title}
              >
                {shortTitle}
              </div>
            );
          })}
        </div>
      </nav>
    </Tooltip.Provider>
  );
};
