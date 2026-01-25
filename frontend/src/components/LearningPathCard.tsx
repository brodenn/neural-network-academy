import { motion } from 'framer-motion';
import { PathProgressRing } from './PathProgressRing';
import type { LearningPath } from '../types';

interface LearningPathCardProps {
  path: LearningPath;
  completed: number;
  isComplete?: boolean;
  isLocked?: boolean;
  missingPrerequisites?: string[];
  onSelect: () => void;
}

const difficultyColors = {
  beginner: 'bg-blue-100 text-blue-800 border-blue-300',
  intermediate: 'bg-purple-100 text-purple-800 border-purple-300',
  advanced: 'bg-orange-100 text-orange-800 border-orange-300',
  research: 'bg-red-100 text-red-800 border-red-300'
};

export const LearningPathCard = ({
  path,
  completed,
  isComplete: isCompleteOverride,
  isLocked = false,
  missingPrerequisites = [],
  onSelect
}: LearningPathCardProps) => {
  const isComplete = isCompleteOverride ?? completed === path.steps;
  const notStarted = completed === 0;
  const inProgress = !notStarted && !isComplete;

  return (
    <motion.div
      whileHover={isLocked ? {} : { scale: 1.02, y: -4 }}
      whileTap={isLocked ? {} : { scale: 0.98 }}
      className={`bg-white rounded-lg shadow-md p-6 border-2 transition-colors ${
        isLocked
          ? 'border-gray-300 opacity-75 cursor-not-allowed'
          : 'border-gray-200 hover:border-blue-400 cursor-pointer'
      }`}
      onClick={isLocked ? undefined : onSelect}
    >
      <div className="flex items-start justify-between mb-4">
        <div className="flex items-center gap-3">
          <span className="text-4xl">{path.badge.icon}</span>
          <div>
            <h3 className="font-bold text-lg text-gray-900">{path.name}</h3>
            <span className={`text-xs px-2 py-1 rounded-full border ${difficultyColors[path.difficulty]}`}>
              {path.difficulty}
            </span>
          </div>
        </div>
        {isComplete && (
          <motion.div
            initial={{ scale: 0, rotate: -180 }}
            animate={{ scale: 1, rotate: 0 }}
            transition={{ type: 'spring', stiffness: 200 }}
            className="flex flex-col items-center"
          >
            <span className="text-3xl">{path.badge.icon}</span>
            <span className="text-xs text-gray-600 mt-1">{path.badge.title}</span>
          </motion.div>
        )}
      </div>

      <p className="text-gray-600 text-sm mb-4">{path.description}</p>

      <div className="flex items-center justify-between mb-4">
        <div className="text-sm text-gray-500">
          <span className="font-medium">
            {inProgress ? `${completed}/${path.steps} completed` : `${path.steps} steps`}
          </span>
          <span className="mx-2">·</span>
          <span>{path.estimatedTime}</span>
        </div>
        <PathProgressRing completed={completed} total={path.steps} size={60} />
      </div>

      {isLocked ? (
        <div className="space-y-2">
          <div className="flex items-center justify-center gap-2 py-2 bg-gray-200 rounded-lg text-gray-500">
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
            </svg>
            <span className="font-medium">Locked</span>
          </div>
          <p className="text-xs text-gray-500 text-center">
            Complete first: {missingPrerequisites.join(', ')}
          </p>
        </div>
      ) : (
        <motion.button
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          className={`w-full py-2 rounded-lg font-medium transition-colors ${
            notStarted
              ? 'bg-blue-500 text-white hover:bg-blue-600'
              : isComplete
              ? 'bg-green-500 text-white hover:bg-green-600'
              : 'bg-purple-500 text-white hover:bg-purple-600'
          }`}
        >
          {notStarted ? 'Start Path' : isComplete ? 'Review' : 'Continue'}
        </motion.button>
      )}
    </motion.div>
  );
};
