import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { InteractiveHint, type RichHint } from './InteractiveHint';
import { ExperimentCard, type Experiment } from './ExperimentCard';

// =============================================================================
// TYPES
// =============================================================================

interface StepHintPanelProps {
  hints: string[] | RichHint[];
  experiments?: Experiment[];
  attempts: number;
  accuracy?: number;
  requiredAccuracy?: number;
  problemId?: string;
}

interface HintUnlockConfig {
  minAttempts: number;
  maxAccuracy: number;
}

const HINT_UNLOCK_CONFIGS: HintUnlockConfig[] = [
  { minAttempts: 1, maxAccuracy: 0.60 },
  { minAttempts: 2, maxAccuracy: 0.45 },
  { minAttempts: 3, maxAccuracy: 0.30 },
];

// =============================================================================
// HELPER: Check if hints are rich format
// =============================================================================

const isRichHintArray = (hints: string[] | RichHint[]): hints is RichHint[] => {
  return hints.length > 0 && typeof hints[0] === 'object';
};

// =============================================================================
// HELPER: Convert string hints to rich hints for unified handling
// =============================================================================

const convertToRichHints = (hints: string[], problemId?: string): RichHint[] => {
  // Default icons based on hint position
  const defaultIcons = ['💡', '🔍', '🎯'];
  const defaultTypes: RichHint['type'][] = ['concept', 'insight', 'experiment'];

  return hints.map((hint, index) => ({
    id: `${problemId || 'hint'}-${index}`,
    type: defaultTypes[index % defaultTypes.length],
    icon: defaultIcons[index % defaultIcons.length],
    title: `Hint ${index + 1}`,
    layers: [{ content: hint }],
  }));
};

// =============================================================================
// COMPONENT
// =============================================================================

export const StepHintPanel = ({
  hints,
  experiments,
  attempts,
  accuracy = 0,
  requiredAccuracy = 0.95,
  problemId,
}: StepHintPanelProps) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const [revealedHints, setRevealedHints] = useState<Set<number>>(new Set());
  const [justUnlockedHint, setJustUnlockedHint] = useState<number | null>(null);
  const [activeTab, setActiveTab] = useState<'hints' | 'experiments'>('hints');

  // Normalize hints to rich format
  const richHints: RichHint[] = isRichHintArray(hints)
    ? hints
    : convertToRichHints(hints, problemId);

  // Calculate unlocked count
  const getUnlockedCount = () => {
    let unlocked = 0;
    for (const config of HINT_UNLOCK_CONFIGS) {
      if (attempts >= config.minAttempts && accuracy < config.maxAccuracy) {
        unlocked++;
      } else if (attempts >= config.minAttempts && accuracy >= requiredAccuracy) {
        unlocked++;
      } else {
        break;
      }
    }
    return Math.min(unlocked, richHints.length);
  };

  const unlockedCount = getUnlockedCount();

  // Detect new hint unlock
  const [prevUnlockedCount, setPrevUnlockedCount] = useState(0);
  useEffect(() => {
    if (unlockedCount > prevUnlockedCount && prevUnlockedCount > 0) {
      setJustUnlockedHint(unlockedCount - 1);
      setIsExpanded(true);
      const timer = setTimeout(() => setJustUnlockedHint(null), 2000);
      return () => clearTimeout(timer);
    }
    setPrevUnlockedCount(unlockedCount);
  }, [unlockedCount, prevUnlockedCount]);

  const isHintUnlocked = (index: number) => index < unlockedCount;

  const getUnlockStatus = (index: number): { locked: boolean; reason: string } => {
    const config = HINT_UNLOCK_CONFIGS[index];
    if (!config) return { locked: true, reason: 'No more hints' };

    const hasEnoughAttempts = attempts >= config.minAttempts;
    const isStrugglingEnough = accuracy < config.maxAccuracy;

    if (!hasEnoughAttempts) {
      const attemptsNeeded = config.minAttempts - attempts;
      return {
        locked: true,
        reason: attemptsNeeded === 1 ? 'Try 1 more time' : `Try ${attemptsNeeded} more times`,
      };
    }

    if (!isStrugglingEnough) {
      return {
        locked: true,
        reason: `Available below ${(config.maxAccuracy * 100).toFixed(0)}% accuracy`,
      };
    }

    return { locked: false, reason: '' };
  };

  const revealHint = (index: number) => {
    if (isHintUnlocked(index)) {
      setRevealedHints(prev => new Set([...prev, index]));
    }
  };

  if (richHints.length === 0 && (!experiments || experiments.length === 0)) {
    return null;
  }

  const hasExperiments = experiments && experiments.length > 0;

  // Progress indicator for next hint
  const getProgressToNextHint = () => {
    if (unlockedCount >= richHints.length) return null;
    const nextConfig = HINT_UNLOCK_CONFIGS[unlockedCount];
    if (!nextConfig) return null;

    const attemptProgress = Math.min(attempts / nextConfig.minAttempts, 1);
    const accuracyProgress = accuracy < nextConfig.maxAccuracy ? 1 : 0;

    return { attemptProgress, accuracyProgress, config: nextConfig };
  };

  const nextHintProgress = getProgressToNextHint();

  return (
    <div className="bg-gray-800 rounded-lg overflow-hidden">
      {/* Header */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full px-4 py-3 flex items-center justify-between text-left hover:bg-gray-700/50 transition-colors"
      >
        <div className="flex items-center gap-2">
          <motion.svg
            className="w-5 h-5 text-yellow-500"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
            animate={justUnlockedHint !== null ? { scale: [1, 1.3, 1] } : {}}
            transition={{ duration: 0.3 }}
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"
            />
          </motion.svg>
          <span className="font-medium text-gray-200">
            {hasExperiments ? 'Hints & Experiments' : 'Hints'}
          </span>
          <span className="text-xs text-gray-500">
            ({unlockedCount}/{richHints.length} unlocked)
          </span>
          {justUnlockedHint !== null && (
            <motion.span
              initial={{ opacity: 0, scale: 0.5 }}
              animate={{ opacity: 1, scale: 1 }}
              className="text-xs bg-yellow-600 text-white px-2 py-0.5 rounded-full"
            >
              New!
            </motion.span>
          )}
        </div>
        <motion.svg
          className="w-5 h-5 text-gray-400"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
          animate={{ rotate: isExpanded ? 180 : 0 }}
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </motion.svg>
      </button>

      {/* Content */}
      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            {/* Tab navigation (if has experiments) */}
            {hasExperiments && (
              <div className="px-4 pt-2 flex gap-2">
                <button
                  onClick={() => setActiveTab('hints')}
                  className={`px-3 py-1.5 text-sm rounded-lg transition-colors ${
                    activeTab === 'hints'
                      ? 'bg-yellow-600 text-white'
                      : 'bg-gray-700 text-gray-400 hover:text-white'
                  }`}
                >
                  💡 Hints
                </button>
                <button
                  onClick={() => setActiveTab('experiments')}
                  className={`px-3 py-1.5 text-sm rounded-lg transition-colors ${
                    activeTab === 'experiments'
                      ? 'bg-purple-600 text-white'
                      : 'bg-gray-700 text-gray-400 hover:text-white'
                  }`}
                >
                  🔬 Experiments
                </button>
              </div>
            )}

            <div className="px-4 pb-4 pt-3 space-y-3">
              {/* Hints Tab */}
              {activeTab === 'hints' && (
                <>
                  {/* Progress indicator */}
                  {unlockedCount < richHints.length && nextHintProgress && (
                    <div className="text-xs text-gray-500 p-2 bg-gray-700/30 rounded">
                      <div className="flex items-center gap-2 mb-2">
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={2}
                            d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                          />
                        </svg>
                        <span>Hints unlock when you're struggling:</span>
                      </div>
                      <div className="ml-6 space-y-1">
                        <div className="flex items-center gap-2">
                          <div
                            className={`w-3 h-3 rounded-full ${
                              nextHintProgress.attemptProgress >= 1 ? 'bg-green-500' : 'bg-gray-600'
                            }`}
                          />
                          <span>
                            Attempts: {attempts}/{nextHintProgress.config.minAttempts}
                            {nextHintProgress.attemptProgress >= 1 && ' ✓'}
                          </span>
                        </div>
                        <div className="flex items-center gap-2">
                          <div
                            className={`w-3 h-3 rounded-full ${
                              nextHintProgress.accuracyProgress >= 1 ? 'bg-green-500' : 'bg-gray-600'
                            }`}
                          />
                          <span>
                            Accuracy below {(nextHintProgress.config.maxAccuracy * 100).toFixed(0)}%
                            {accuracy > 0 ? `: ${(accuracy * 100).toFixed(0)}%` : ''}
                            {nextHintProgress.accuracyProgress >= 1 && ' ✓'}
                          </span>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Render hints */}
                  {richHints.map((hint, index) => {
                    const isUnlocked = isHintUnlocked(index);
                    const isRevealed = revealedHints.has(index);

                    // Use InteractiveHint for rich hints
                    if (hint.layers.length > 1 || hint.experiment || hint.question) {
                      return (
                        <InteractiveHint
                          key={hint.id}
                          hint={hint}
                          isUnlocked={isUnlocked}
                          isRevealed={isRevealed}
                          onReveal={() => revealHint(index)}
                          attemptCount={attempts}
                        />
                      );
                    }

                    // Simple hint display for basic hints
                    const isJustUnlocked = justUnlockedHint === index;
                    const status = getUnlockStatus(index);

                    return (
                      <motion.div
                        key={hint.id}
                        initial={false}
                        animate={{
                          opacity: isUnlocked ? 1 : 0.6,
                          scale: isJustUnlocked ? [1, 1.02, 1] : 1,
                        }}
                        className={`rounded-lg p-3 ${
                          !isUnlocked
                            ? 'bg-gray-700/30 border border-gray-700'
                            : isJustUnlocked
                            ? 'bg-yellow-900/50 border border-yellow-600'
                            : isRevealed
                            ? 'bg-yellow-900/30 border border-yellow-800/50'
                            : 'bg-gray-700/50 cursor-pointer hover:bg-gray-700 border border-transparent'
                        }`}
                        onClick={() => isUnlocked && !isRevealed && revealHint(index)}
                      >
                        <div className="flex items-start gap-3">
                          <div
                            className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
                              isUnlocked ? 'bg-yellow-600' : 'bg-gray-700'
                            }`}
                          >
                            <span>{isUnlocked ? hint.icon : '🔒'}</span>
                          </div>

                          <div className="flex-1 min-w-0">
                            {!isUnlocked ? (
                              <div className="text-sm text-gray-500">
                                <p className="font-medium">{hint.title}</p>
                                <p className="text-xs">{status.reason}</p>
                              </div>
                            ) : isRevealed ? (
                              <div>
                                <p className="text-xs text-yellow-400 font-medium mb-1">{hint.title}</p>
                                <p className="text-sm text-gray-200">{hint.layers[0].content}</p>
                              </div>
                            ) : (
                              <div>
                                <p className="text-sm font-medium text-white">{hint.title}</p>
                                <motion.p
                                  className="text-xs text-blue-400 flex items-center gap-1"
                                  whileHover={{ x: 2 }}
                                >
                                  Click to reveal
                                  <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                                  </svg>
                                </motion.p>
                              </div>
                            )}
                          </div>
                        </div>
                      </motion.div>
                    );
                  })}

                  {/* Success message */}
                  {unlockedCount === richHints.length && accuracy >= requiredAccuracy && (
                    <motion.div
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      className="text-xs text-green-400 text-center py-2"
                    >
                      ✓ All hints available - you've mastered this step!
                    </motion.div>
                  )}
                </>
              )}

              {/* Experiments Tab */}
              {activeTab === 'experiments' && experiments && (
                <div className="space-y-3">
                  {experiments.map((exp, index) => (
                    <ExperimentCard
                      key={exp.id}
                      experiment={exp}
                      isUnlocked={index === 0 || unlockedCount > index}
                    />
                  ))}
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default StepHintPanel;
