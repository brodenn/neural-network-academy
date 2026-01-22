import { memo, useState, useMemo, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import type { ProblemInfo } from '../types';

interface ContextualTipsProps {
  problem: ProblemInfo | null;
  currentEpoch: number;
  currentAccuracy: number;
  currentLoss: number;
  trainingInProgress: boolean;
}

interface Tip {
  id: string;
  message: string;
  type: 'info' | 'warning' | 'success';
  priority: number;
}

export const ContextualTips = memo(function ContextualTips({
  problem,
  currentEpoch,
  currentAccuracy,
  currentLoss,
  trainingInProgress,
}: ContextualTipsProps) {
  // Track dismissed tips per problem (key is `${problemId}:${tipId}`)
  const [dismissedTips, setDismissedTips] = useState<Set<string>>(new Set());
  // Track which problems have shown success tip
  const [successShownFor, setSuccessShownFor] = useState<Set<string>>(new Set());

  const problemId = problem?.id ?? '';

  // Generate contextual tips based on training state
  const tips = useMemo((): Tip[] => {
    if (!problem) return [];

    const result: Tip[] = [];

    // Success tip - only show once per problem
    const alreadyShownSuccess = successShownFor.has(problem.id);
    if (currentAccuracy >= 0.95 && !alreadyShownSuccess && currentEpoch > 10) {
      result.push({
        id: 'success',
        message: `Great job! The network learned ${problem.name} with ${(currentAccuracy * 100).toFixed(1)}% accuracy.`,
        type: 'success',
        priority: 1,
      });
    }

    // Struggling detection - low accuracy after many epochs
    if (trainingInProgress && currentEpoch > 100 && currentAccuracy < 0.6) {
      // Check for specific failure cases
      if (problem.is_failure_case && problem.failure_reason) {
        result.push({
          id: 'failure-case',
          message: `This is a failure demonstration: ${problem.failure_reason}`,
          type: 'warning',
          priority: 2,
        });
        if (problem.fix_suggestion) {
          result.push({
            id: 'failure-fix',
            message: `Fix: ${problem.fix_suggestion}`,
            type: 'info',
            priority: 3,
          });
        }
      } else {
        // General struggling tips
        result.push({
          id: 'struggling',
          message: 'Network is struggling. Try adjusting the architecture or learning rate.',
          type: 'warning',
          priority: 4,
        });
      }
    }

    // Stuck at 50% (common for XOR without hidden layers)
    if (trainingInProgress && currentEpoch > 50 && currentAccuracy >= 0.45 && currentAccuracy <= 0.55) {
      if (problem.id.includes('xor') || problem.id === 'fail_xor_no_hidden') {
        result.push({
          id: 'xor-linear',
          message: 'XOR is not linearly separable. Make sure you have at least one hidden layer!',
          type: 'warning',
          priority: 2,
        });
      }
    }

    // Loss not decreasing
    if (trainingInProgress && currentEpoch > 200 && currentLoss > 0.5) {
      result.push({
        id: 'high-loss',
        message: 'Loss is still high. Consider: lower learning rate, more neurons, or check for vanishing gradients.',
        type: 'warning',
        priority: 5,
      });
    }

    // Show problem tips if training just started
    if (trainingInProgress && currentEpoch < 20 && currentEpoch > 0 && problem.tips.length > 0) {
      result.push({
        id: 'problem-tip',
        message: problem.tips[0],
        type: 'info',
        priority: 10,
      });
    }

    // Filter out dismissed tips (keyed by problem) and sort by priority
    return result
      .filter((tip) => !dismissedTips.has(`${problem.id}:${tip.id}`))
      .sort((a, b) => a.priority - b.priority)
      .slice(0, 2); // Show max 2 tips
  }, [problem, currentEpoch, currentAccuracy, currentLoss, trainingInProgress, dismissedTips, successShownFor]);

  const dismissTip = useCallback((tipId: string) => {
    setDismissedTips((prev) => new Set([...prev, `${problemId}:${tipId}`]));
    // Mark success as shown when dismissed
    if (tipId === 'success' && problemId) {
      setSuccessShownFor((prev) => new Set([...prev, problemId]));
    }
  }, [problemId]);

  if (tips.length === 0) return null;

  const typeStyles = {
    info: 'bg-blue-900/40 border-blue-700/50 text-blue-200',
    warning: 'bg-yellow-900/40 border-yellow-700/50 text-yellow-200',
    success: 'bg-green-900/40 border-green-700/50 text-green-200',
  };

  const typeIcons = {
    info: '💡',
    warning: '⚠️',
    success: '🎉',
  };

  return (
    <div className="space-y-2">
      <AnimatePresence mode="popLayout">
        {tips.map((tip) => (
          <motion.div
            key={tip.id}
            initial={{ opacity: 0, height: 0, y: -10 }}
            animate={{ opacity: 1, height: 'auto', y: 0 }}
            exit={{ opacity: 0, height: 0, y: -10 }}
            transition={{ duration: 0.2 }}
            className={`rounded-lg border p-2.5 ${typeStyles[tip.type]}`}
          >
            <div className="flex items-start gap-2">
              <span className="text-sm flex-shrink-0">{typeIcons[tip.type]}</span>
              <p className="text-xs flex-1 leading-relaxed">{tip.message}</p>
              <button
                onClick={() => dismissTip(tip.id)}
                className="text-gray-400 hover:text-gray-200 flex-shrink-0 p-0.5"
                aria-label="Dismiss tip"
              >
                <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
          </motion.div>
        ))}
      </AnimatePresence>
    </div>
  );
});
