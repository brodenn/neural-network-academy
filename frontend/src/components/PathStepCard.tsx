import { motion } from 'framer-motion';
import type { StepProgressData } from '../types';

interface PathStepInfo {
  stepNumber: number;
  problemId: string;
  title: string;
  learningObjectives: string[];
  requiredAccuracy?: number;
  hints: string[];
}

interface PathStepCardProps {
  step: PathStepInfo;
  progress: StepProgressData | null;
  isCurrentStep: boolean;
}

export const PathStepCard = ({ step, progress, isCurrentStep }: PathStepCardProps) => {
  const isCompleted = progress?.completed ?? false;
  const isLocked = !(progress?.unlocked ?? false);
  const attempts = progress?.attempts ?? 0;
  const bestAccuracy = progress?.bestAccuracy ?? 0;
  const requiredAccuracy = step.requiredAccuracy ?? 0.95;

  // Format accuracy as percentage
  const formatAccuracy = (acc: number) => `${(acc * 100).toFixed(0)}%`;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className={`bg-gray-800 rounded-lg p-4 border-2 ${
        isLocked
          ? 'border-gray-700 opacity-60'
          : isCompleted
          ? 'border-green-600'
          : isCurrentStep
          ? 'border-blue-500'
          : 'border-gray-600'
      }`}
    >
      {/* Header */}
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center gap-2">
          <div className={`w-8 h-8 rounded-full flex items-center justify-center ${
            isLocked
              ? 'bg-gray-700'
              : isCompleted
              ? 'bg-green-600'
              : 'bg-blue-600'
          }`}>
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
          </div>
          <div>
            <h3 className={`font-semibold ${isLocked ? 'text-gray-500' : 'text-white'}`}>
              {step.title}
            </h3>
            <span className="text-xs text-gray-500">{step.problemId}</span>
          </div>
        </div>

        {/* Status Badge */}
        {isCompleted && (
          <motion.span
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            className="bg-green-600 text-white text-xs px-2 py-1 rounded-full"
          >
            Complete
          </motion.span>
        )}
        {isCurrentStep && !isCompleted && (
          <span className="bg-blue-600 text-white text-xs px-2 py-1 rounded-full animate-pulse">
            Current
          </span>
        )}
      </div>

      {/* Instructions */}
      {!isLocked && isCurrentStep && !isCompleted && (
        <div className="mb-3 bg-blue-900/30 border border-blue-700 rounded-lg p-3">
          <div className="flex items-start gap-2">
            <svg className="w-5 h-5 text-blue-400 flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <div>
              <h4 className="text-sm font-semibold text-blue-300 mb-1">What to do:</h4>
              <ol className="text-sm text-gray-300 space-y-1 list-decimal list-inside">
                <li>Review the learning objectives below</li>
                <li>Use the input controls to test different values</li>
                <li>Click "Start Learning" to train the network</li>
                <li>Reach {requiredAccuracy === 0 ? 'any accuracy' : `${formatAccuracy(requiredAccuracy)} accuracy`} to complete this step</li>
                <li>Next step will unlock automatically!</li>
              </ol>
            </div>
          </div>
        </div>
      )}

      {/* Success Message */}
      {!isLocked && isCompleted && (
        <div className="mb-3 bg-green-900/30 border border-green-700 rounded-lg p-3">
          <div className="flex items-center gap-2">
            <svg className="w-5 h-5 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <span className="text-sm font-medium text-green-300">
              Step completed! {step.stepNumber < 7 ? 'Move to the next step below.' : 'You finished the path!'}
            </span>
          </div>
        </div>
      )}

      {/* Learning Objectives */}
      {!isLocked && (
        <div className="mb-3">
          <h4 className="text-xs font-medium text-gray-400 uppercase mb-1">Learning Objectives</h4>
          <ul className="space-y-1">
            {step.learningObjectives.map((obj, i) => (
              <li key={i} className="text-sm text-gray-300 flex items-start gap-2">
                <span className="text-blue-400 mt-0.5">•</span>
                {obj}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Progress Stats */}
      {!isLocked && (
        <div className="grid grid-cols-3 gap-2 text-center bg-gray-900/50 rounded p-2">
          <div>
            <div className="text-xs text-gray-500">Required</div>
            <div className={`text-sm font-medium ${
              requiredAccuracy === 0 ? 'text-yellow-400' : 'text-gray-300'
            }`}>
              {requiredAccuracy === 0 ? 'Any' : formatAccuracy(requiredAccuracy)}
            </div>
          </div>
          <div>
            <div className="text-xs text-gray-500">Best</div>
            <div className={`text-sm font-medium ${
              bestAccuracy >= requiredAccuracy ? 'text-green-400' : 'text-gray-300'
            }`}>
              {bestAccuracy > 0 ? formatAccuracy(bestAccuracy) : '-'}
            </div>
          </div>
          <div>
            <div className="text-xs text-gray-500">Attempts</div>
            <div className="text-sm font-medium text-gray-300">{attempts}</div>
          </div>
        </div>
      )}

      {/* Locked Message */}
      {isLocked && (
        <p className="text-sm text-gray-500 text-center py-2">
          Complete previous steps to unlock
        </p>
      )}
    </motion.div>
  );
};
