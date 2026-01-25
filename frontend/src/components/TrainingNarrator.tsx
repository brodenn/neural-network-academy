import { useState, useEffect, useRef } from 'react';
import { animated, useSpring, config } from '@react-spring/web';

// =============================================================================
// TYPES
// =============================================================================

interface TrainingNarratorProps {
  epoch: number;
  loss: number;
  accuracy: number;
  isTraining: boolean;
  isComplete: boolean;
  targetAccuracy?: number;
  problemId?: string;
  isFailureCase?: boolean;
}

interface Insight {
  id: string;
  message: string;
  icon: string;
  type: 'info' | 'success' | 'warning' | 'error' | 'celebration';
  timestamp: number;
}

// =============================================================================
// INSIGHT GENERATION LOGIC
// =============================================================================

const generateInsight = (
  epoch: number,
  loss: number,
  accuracy: number,
  prevLoss: number,
  prevAccuracy: number,
  targetAccuracy: number,
  isFailureCase: boolean
): Insight | null => {
  const now = Date.now();
  const lossChange = prevLoss > 0 ? (loss - prevLoss) / prevLoss : 0;
  const accuracyChange = accuracy - prevAccuracy;

  // Training just started
  if (epoch === 1) {
    return {
      id: `insight-${now}`,
      message: 'Training started! Watch the network learn...',
      icon: '🚀',
      type: 'info',
      timestamp: now,
    };
  }

  // Early epochs
  if (epoch <= 10 && epoch % 5 === 0) {
    return {
      id: `insight-${now}`,
      message: 'Network is initializing weights and finding patterns...',
      icon: '🔍',
      type: 'info',
      timestamp: now,
    };
  }

  // Loss dropping fast (good!)
  if (lossChange < -0.1 && epoch > 5) {
    return {
      id: `insight-${now}`,
      message: 'Loss dropping fast! The network is learning patterns.',
      icon: '📉',
      type: 'success',
      timestamp: now,
    };
  }

  // Loss exploded (failure case or bad LR)
  if (loss > 10 || Number.isNaN(loss)) {
    if (isFailureCase) {
      return {
        id: `insight-${now}`,
        message: 'Loss exploded! This is the expected failure - learning rate is too high.',
        icon: '💥',
        type: 'warning',
        timestamp: now,
      };
    }
    return {
      id: `insight-${now}`,
      message: 'Loss exploded! Try lowering the learning rate.',
      icon: '⚠️',
      type: 'error',
      timestamp: now,
    };
  }

  // Loss increased significantly
  if (lossChange > 0.05 && epoch > 10) {
    return {
      id: `insight-${now}`,
      message: 'Loss increased - this can happen during learning, it may recover.',
      icon: '📈',
      type: 'warning',
      timestamp: now,
    };
  }

  // Plateau detected
  if (Math.abs(lossChange) < 0.001 && epoch > 50 && accuracy < targetAccuracy) {
    if (isFailureCase) {
      return {
        id: `insight-${now}`,
        message: 'Network stuck at ~50% - expected for this architecture!',
        icon: '🎯',
        type: 'info',
        timestamp: now,
      };
    }
    return {
      id: `insight-${now}`,
      message: 'Plateau detected. Try: different architecture, learning rate, or more neurons.',
      icon: '⏸️',
      type: 'warning',
      timestamp: now,
    };
  }

  // Accuracy milestones
  if (accuracy >= 0.95 && prevAccuracy < 0.95) {
    return {
      id: `insight-${now}`,
      message: 'Excellent! 95% accuracy reached!',
      icon: '🎯',
      type: 'celebration',
      timestamp: now,
    };
  }

  if (accuracy >= 0.75 && prevAccuracy < 0.75) {
    return {
      id: `insight-${now}`,
      message: '75% accuracy! Network is learning well.',
      icon: '📊',
      type: 'success',
      timestamp: now,
    };
  }

  if (accuracy >= 0.50 && prevAccuracy < 0.50 && epoch > 20) {
    return {
      id: `insight-${now}`,
      message: 'Passed 50% - better than random guessing!',
      icon: '👍',
      type: 'info',
      timestamp: now,
    };
  }

  // Close to target
  if (accuracy >= targetAccuracy * 0.9 && accuracy < targetAccuracy) {
    return {
      id: `insight-${now}`,
      message: 'Almost there! Fine-tuning for final accuracy...',
      icon: '🔬',
      type: 'info',
      timestamp: now,
    };
  }

  // Accuracy improving steadily
  if (accuracyChange > 0.05) {
    return {
      id: `insight-${now}`,
      message: 'Accuracy improving! Network is finding the solution.',
      icon: '📈',
      type: 'success',
      timestamp: now,
    };
  }

  return null;
};

// =============================================================================
// COMPONENT
// =============================================================================

export const TrainingNarrator = ({
  epoch,
  loss,
  accuracy,
  isTraining,
  isComplete,
  targetAccuracy = 0.95,
  isFailureCase = false,
}: TrainingNarratorProps) => {
  const [, setInsights] = useState<Insight[]>([]);
  const [currentInsight, setCurrentInsight] = useState<Insight | null>(null);
  const prevLossRef = useRef(loss);
  const prevAccuracyRef = useRef(accuracy);
  const lastInsightTimeRef = useRef(0);

  // Spring animation for insight appearance
  const insightSpring = useSpring({
    opacity: currentInsight ? 1 : 0,
    y: currentInsight ? 0 : 20,
    scale: currentInsight ? 1 : 0.9,
    config: config.wobbly,
  });

  // Generate insights during training
  useEffect(() => {
    if (!isTraining || epoch === 0) return;

    const now = Date.now();
    // Throttle insights to every 2 seconds minimum
    if (now - lastInsightTimeRef.current < 2000) {
      prevLossRef.current = loss;
      prevAccuracyRef.current = accuracy;
      return;
    }

    const insight = generateInsight(
      epoch,
      loss,
      accuracy,
      prevLossRef.current,
      prevAccuracyRef.current,
      targetAccuracy,
      isFailureCase
    );

    if (insight) {
      setInsights(prev => [...prev.slice(-4), insight]);
      setCurrentInsight(insight);
      lastInsightTimeRef.current = now;

      // Auto-hide after 4 seconds
      setTimeout(() => {
        setCurrentInsight(prev => (prev?.id === insight.id ? null : prev));
      }, 4000);
    }

    prevLossRef.current = loss;
    prevAccuracyRef.current = accuracy;
  }, [epoch, loss, accuracy, isTraining, targetAccuracy, isFailureCase]);

  // Training complete insight
  useEffect(() => {
    if (isComplete && !isTraining) {
      const now = Date.now();
      const completeInsight: Insight = accuracy >= targetAccuracy
        ? {
            id: `complete-${now}`,
            message: `Training complete! Achieved ${(accuracy * 100).toFixed(1)}% accuracy!`,
            icon: '🎉',
            type: 'celebration',
            timestamp: now,
          }
        : isFailureCase
        ? {
            id: `complete-${now}`,
            message: `Training complete. As expected, this configuration couldn't solve the problem.`,
            icon: '📚',
            type: 'info',
            timestamp: now,
          }
        : {
            id: `complete-${now}`,
            message: `Training complete. Reached ${(accuracy * 100).toFixed(1)}% - try adjusting the network!`,
            icon: '🔧',
            type: 'warning',
            timestamp: now,
          };

      setInsights(prev => [...prev.slice(-4), completeInsight]);
      setCurrentInsight(completeInsight);
    }
  }, [isComplete, isTraining, accuracy, targetAccuracy, isFailureCase]);

  // Type-based styling
  const getTypeStyles = (type: Insight['type']) => {
    switch (type) {
      case 'success':
        return 'bg-green-900/80 border-green-500 text-green-100';
      case 'warning':
        return 'bg-amber-900/80 border-amber-500 text-amber-100';
      case 'error':
        return 'bg-red-900/80 border-red-500 text-red-100';
      case 'celebration':
        return 'bg-purple-900/80 border-purple-500 text-purple-100';
      default:
        return 'bg-blue-900/80 border-blue-500 text-blue-100';
    }
  };

  if (!currentInsight) return null;

  return (
    <animated.div
      style={insightSpring}
      className={`rounded-lg border px-4 py-3 ${getTypeStyles(currentInsight.type)}`}
    >
      <div className="flex items-center gap-3">
        <animated.span
          className="text-2xl"
          style={{
            transform: insightSpring.scale.to(s => `scale(${s * 1.2})`),
          }}
        >
          {currentInsight.icon}
        </animated.span>
        <div className="flex-1">
          <p className="font-medium text-sm">{currentInsight.message}</p>
          {isTraining && (
            <p className="text-xs opacity-70 mt-1">
              Epoch {epoch} · Loss: {loss.toFixed(4)} · Accuracy: {(accuracy * 100).toFixed(1)}%
            </p>
          )}
        </div>
      </div>
    </animated.div>
  );
};

// =============================================================================
// MINI VERSION FOR INLINE USE
// =============================================================================

export const TrainingInsightBadge = ({
  epoch,
  loss,
  accuracy,
  isTraining,
}: Pick<TrainingNarratorProps, 'epoch' | 'loss' | 'accuracy' | 'isTraining'>) => {
  const [message, setMessage] = useState<string | null>(null);

  const spring = useSpring({
    opacity: message ? 1 : 0,
    config: config.gentle,
  });

  useEffect(() => {
    if (!isTraining) {
      setMessage(null);
      return;
    }

    // Simple status messages
    if (epoch <= 5) {
      setMessage('Initializing...');
    } else if (loss > 1) {
      setMessage('Learning patterns...');
    } else if (accuracy < 0.5) {
      setMessage('Finding features...');
    } else if (accuracy < 0.9) {
      setMessage('Improving accuracy...');
    } else {
      setMessage('Fine-tuning...');
    }
  }, [epoch, loss, accuracy, isTraining]);

  if (!message) return null;

  return (
    <animated.span
      style={spring}
      className="inline-flex items-center gap-1 px-2 py-1 bg-blue-600/50 rounded text-xs"
    >
      <span className="animate-pulse">●</span>
      {message}
    </animated.span>
  );
};

export default TrainingNarrator;
