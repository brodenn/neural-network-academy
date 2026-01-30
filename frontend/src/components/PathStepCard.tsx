import { motion } from 'framer-motion';
import type { StepProgressData } from '../types';

interface PathStepInfo {
  stepNumber: number;
  problemId: string;
  title: string;
  learningObjectives: string[];
  requiredAccuracy?: number;
  hints: string[];
  isFinalStep?: boolean;
}

interface PathStepCardProps {
  step: PathStepInfo;
  progress: StepProgressData | null;
  isCurrentStep: boolean;
  isFinalStep?: boolean;
}

type Tier = 'gold' | 'silver' | 'bronze' | null;

function getStepTier(progress: StepProgressData | null): Tier {
  if (!progress?.completed) return null;
  const acc = progress.bestAccuracy;
  const attempts = progress.attempts;
  if (acc >= 0.98 && attempts <= 2) return 'gold';
  if (acc >= 0.95 && attempts <= 5) return 'silver';
  return 'bronze';
}

const TIER_CONFIG = {
  gold:   { icon: '🥇', label: 'Gold',   color: 'text-yellow-400', bg: 'bg-yellow-900/30 border-yellow-600/50' },
  silver: { icon: '🥈', label: 'Silver', color: 'text-gray-300',   bg: 'bg-gray-700/50 border-gray-500/50' },
  bronze: { icon: '🥉', label: 'Bronze', color: 'text-orange-400', bg: 'bg-orange-900/30 border-orange-600/50' },
} as const;

export const PathStepCard = ({ step, progress, isCurrentStep, isFinalStep }: PathStepCardProps) => {
  const isCompleted = progress?.completed ?? false;
  const isLocked = !(progress?.unlocked ?? false);
  const attempts = progress?.attempts ?? 0;
  const bestAccuracy = progress?.bestAccuracy ?? 0;
  const requiredAccuracy = step.requiredAccuracy ?? 0.95;
  const tier = getStepTier(progress);
  const isBoss = isFinalStep ?? step.isFinalStep ?? false;

  // Format accuracy as percentage
  const formatAccuracy = (acc: number) => `${(acc * 100).toFixed(0)}%`;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className={`rounded-lg p-4 border-2 relative overflow-hidden ${
        isBoss && !isLocked && !isCompleted
          ? 'bg-gradient-to-br from-gray-800 via-gray-800 to-red-950 border-red-600 shadow-lg shadow-red-900/20'
          : 'bg-gray-800'
      } ${
        isLocked
          ? 'border-gray-700 opacity-60'
          : isCompleted
          ? tier === 'gold' ? 'border-yellow-500' : tier === 'silver' ? 'border-gray-400' : 'border-green-600'
          : isCurrentStep && !isBoss
          ? 'border-blue-500'
          : !isBoss ? 'border-gray-600' : ''
      }`}
    >
      {/* Boss battle banner */}
      {isBoss && !isLocked && !isCompleted && (
        <div className="absolute top-0 right-0 bg-red-600 text-white text-[10px] font-bold px-2 py-0.5 rounded-bl-lg uppercase tracking-wider">
          Final Challenge
        </div>
      )}

      {/* Header */}
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center gap-2">
          <div className={`w-8 h-8 rounded-full flex items-center justify-center ${
            isLocked
              ? 'bg-gray-700'
              : isCompleted
              ? tier === 'gold' ? 'bg-yellow-600' : tier === 'silver' ? 'bg-gray-500' : 'bg-green-600'
              : isBoss ? 'bg-red-600 animate-pulse' : 'bg-blue-600'
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
            ) : isBoss ? (
              <span className="text-sm">⚔️</span>
            ) : (
              <span className="text-sm font-bold text-white">{step.stepNumber}</span>
            )}
          </div>
          <div>
            <h3 className={`font-semibold ${isLocked ? 'text-gray-500' : isBoss && !isCompleted ? 'text-red-300' : 'text-white'}`}>
              {step.title}
            </h3>
            <span className="text-xs text-gray-500">{step.problemId}</span>
          </div>
        </div>

        {/* Status Badge with Tier */}
        {isCompleted && tier && (
          <motion.span
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            className={`text-xs px-2 py-1 rounded-full border flex items-center gap-1 ${TIER_CONFIG[tier].bg}`}
          >
            <span>{TIER_CONFIG[tier].icon}</span>
            <span className={TIER_CONFIG[tier].color}>{TIER_CONFIG[tier].label}</span>
          </motion.span>
        )}
        {isCurrentStep && !isCompleted && (
          <span className={`text-white text-xs px-2 py-1 rounded-full ${isBoss ? 'bg-red-600 animate-pulse' : 'bg-blue-600 animate-pulse'}`}>
            {isBoss ? 'Boss' : 'Current'}
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
