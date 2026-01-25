import { useState, useEffect } from 'react';
import { animated, useSpring, useTrail, config } from '@react-spring/web';
import { motion, AnimatePresence } from 'framer-motion';

// =============================================================================
// TYPES
// =============================================================================

type FailureType =
  | 'zero_init'
  | 'high_lr'
  | 'low_lr'
  | 'no_hidden'
  | 'vanishing_gradient'
  | 'underfitting'
  | 'nan_explosion';

interface FailureDramatizationProps {
  failureType: FailureType;
  isActive: boolean;
  loss: number;
  accuracy: number;
  epoch: number;
}

interface FailureInfo {
  title: string;
  icon: string;
  color: string;
  description: string;
  whatToWatch: string;
  lesson: string;
}

// =============================================================================
// FAILURE CONFIGURATIONS
// =============================================================================

const FAILURE_INFO: Record<FailureType, FailureInfo> = {
  zero_init: {
    title: 'Symmetry Breaking Failure',
    icon: '🔗',
    color: 'purple',
    description: 'All neurons start identical and stay identical!',
    whatToWatch: 'Accuracy stuck at ~50% despite hidden neurons',
    lesson: 'Random initialization breaks symmetry, allowing neurons to learn different features.',
  },
  high_lr: {
    title: 'Learning Rate Explosion',
    icon: '💥',
    color: 'red',
    description: 'Gradients are too large, weights oscillate wildly!',
    whatToWatch: 'Loss spikes, oscillates, or becomes NaN',
    lesson: 'Lower learning rate = smaller steps = stable convergence.',
  },
  low_lr: {
    title: 'Glacial Learning',
    icon: '🐌',
    color: 'blue',
    description: 'Steps are too small, learning takes forever!',
    whatToWatch: 'Loss barely changes across many epochs',
    lesson: 'Too low LR means thousands of epochs for simple tasks.',
  },
  no_hidden: {
    title: 'Linear Limitation',
    icon: '📏',
    color: 'amber',
    description: 'No hidden layer = only linear boundaries possible!',
    whatToWatch: 'Decision boundary is a straight line, can\'t curve',
    lesson: 'Hidden layers enable non-linear transformations.',
  },
  vanishing_gradient: {
    title: 'Vanishing Gradients',
    icon: '👻',
    color: 'gray',
    description: 'Gradients shrink to zero in deep networks!',
    whatToWatch: 'Early layers stop learning, loss plateaus',
    lesson: 'Use ReLU activation and proper initialization for deep networks.',
  },
  underfitting: {
    title: 'Underfitting',
    icon: '📉',
    color: 'cyan',
    description: 'Network too simple for the pattern!',
    whatToWatch: 'Both training and test accuracy are low',
    lesson: 'Add more neurons or layers to increase capacity.',
  },
  nan_explosion: {
    title: 'Numerical Explosion',
    icon: '♾️',
    color: 'red',
    description: 'Values exploded to infinity!',
    whatToWatch: 'Loss becomes NaN (Not a Number)',
    lesson: 'Gradient clipping or lower LR prevents explosions.',
  },
};

// =============================================================================
// VISUAL EFFECTS
// =============================================================================

// Shaking effect for unstable training
const ShakingContainer = ({ children, intensity = 1 }: { children: React.ReactNode; intensity?: number }) => {
  const shake = useSpring({
    from: { x: 0 },
    to: async (next) => {
      while (true) {
        await next({ x: -3 * intensity });
        await next({ x: 3 * intensity });
        await next({ x: -2 * intensity });
        await next({ x: 2 * intensity });
        await next({ x: 0 });
        await new Promise(r => setTimeout(r, 500));
      }
    },
    config: { tension: 300, friction: 5 },
  });

  return <animated.div style={shake}>{children}</animated.div>;
};

// Pulsing warning effect
const PulsingWarning = ({ color }: { color: string }) => {
  const pulse = useSpring({
    from: { opacity: 0.3, scale: 1 },
    to: { opacity: 0.8, scale: 1.05 },
    config: { duration: 800 },
    loop: { reverse: true },
  });

  const colorClasses: Record<string, string> = {
    red: 'bg-red-500',
    amber: 'bg-amber-500',
    purple: 'bg-purple-500',
    blue: 'bg-blue-500',
    cyan: 'bg-cyan-500',
    gray: 'bg-gray-500',
  };

  return (
    <animated.div
      style={pulse}
      className={`absolute inset-0 ${colorClasses[color] || 'bg-red-500'} rounded-lg pointer-events-none`}
    />
  );
};

// Explosion effect for NaN
const ExplosionEffect = () => {
  const particles = useTrail(12, {
    from: { opacity: 1, scale: 0, x: 0, y: 0 },
    to: { opacity: 0, scale: 2, x: Math.random() * 100 - 50, y: Math.random() * 100 - 50 },
    config: { tension: 200, friction: 20 },
  });

  return (
    <div className="absolute inset-0 overflow-hidden pointer-events-none">
      {particles.map((style, i) => (
        <animated.div
          key={i}
          style={{
            ...style,
            position: 'absolute',
            left: '50%',
            top: '50%',
            width: 20,
            height: 20,
            borderRadius: '50%',
            background: `hsl(${i * 30}, 80%, 50%)`,
          }}
        />
      ))}
    </div>
  );
};

// =============================================================================
// MAIN COMPONENT
// =============================================================================

export const FailureDramatization = ({
  failureType,
  isActive,
  loss,
  accuracy,
  epoch,
}: FailureDramatizationProps) => {
  const [showExplosion, setShowExplosion] = useState(false);
  const [phase, setPhase] = useState<'intro' | 'watching' | 'lesson'>('intro');

  const info = FAILURE_INFO[failureType];

  // Detect specific failure conditions
  useEffect(() => {
    if (!isActive) return;

    // NaN detection
    if (Number.isNaN(loss) || loss > 1e6) {
      setShowExplosion(true);
      setTimeout(() => setShowExplosion(false), 2000);
    }

    // Phase progression
    if (epoch === 0) {
      setPhase('intro');
    } else if (epoch > 0 && epoch < 50) {
      setPhase('watching');
    } else if (epoch >= 50) {
      setPhase('lesson');
    }
  }, [isActive, loss, epoch]);

  // Animation spring for content
  const contentSpring = useSpring({
    opacity: isActive ? 1 : 0,
    y: isActive ? 0 : 20,
    config: config.gentle,
  });

  if (!isActive) return null;

  const isUnstable = failureType === 'high_lr' || failureType === 'nan_explosion';
  const colorClasses: Record<string, { bg: string; border: string; text: string }> = {
    red: { bg: 'bg-red-900/40', border: 'border-red-500', text: 'text-red-400' },
    amber: { bg: 'bg-amber-900/40', border: 'border-amber-500', text: 'text-amber-400' },
    purple: { bg: 'bg-purple-900/40', border: 'border-purple-500', text: 'text-purple-400' },
    blue: { bg: 'bg-blue-900/40', border: 'border-blue-500', text: 'text-blue-400' },
    cyan: { bg: 'bg-cyan-900/40', border: 'border-cyan-500', text: 'text-cyan-400' },
    gray: { bg: 'bg-gray-800/40', border: 'border-gray-500', text: 'text-gray-400' },
  };

  const colors = colorClasses[info.color] || colorClasses.red;

  const Container = isUnstable ? ShakingContainer : 'div';

  return (
    <animated.div style={contentSpring} className="relative">
      {/* Pulsing background for warning */}
      <div className="absolute inset-0 overflow-hidden rounded-lg">
        <PulsingWarning color={info.color} />
      </div>

      {/* Explosion effect */}
      <AnimatePresence>
        {showExplosion && <ExplosionEffect />}
      </AnimatePresence>

      <Container intensity={loss > 5 ? 2 : 1}>
        <div className={`relative ${colors.bg} ${colors.border} border-2 rounded-lg p-4 overflow-hidden`}>
          {/* Header */}
          <div className="flex items-center gap-3 mb-3">
            <motion.span
              className="text-3xl"
              animate={isUnstable ? { rotate: [0, 10, -10, 0] } : {}}
              transition={{ duration: 0.3, repeat: Infinity }}
            >
              {info.icon}
            </motion.span>
            <div>
              <h3 className={`font-bold ${colors.text}`}>{info.title}</h3>
              <p className="text-sm text-gray-300">
                {phase === 'intro' && 'Watch what happens...'}
                {phase === 'watching' && info.whatToWatch}
                {phase === 'lesson' && 'Lesson learned!'}
              </p>
            </div>
            <div className="ml-auto">
              <span className="px-2 py-1 bg-gray-800 rounded text-xs text-gray-400">
                ⚠️ EXPECTED FAILURE
              </span>
            </div>
          </div>

          {/* Description */}
          <p className="text-gray-200 mb-3">{info.description}</p>

          {/* Current stats */}
          <div className="flex gap-4 mb-3 text-sm">
            <div className={`px-3 py-1.5 rounded ${loss > 1 ? 'bg-red-900/50' : 'bg-gray-800/50'}`}>
              <span className="text-gray-400">Loss: </span>
              <span className={loss > 1 ? 'text-red-400 font-mono' : 'text-white font-mono'}>
                {Number.isNaN(loss) ? 'NaN 💥' : loss > 1e6 ? '∞' : loss.toFixed(4)}
              </span>
            </div>
            <div className={`px-3 py-1.5 rounded ${accuracy < 0.6 ? 'bg-amber-900/50' : 'bg-gray-800/50'}`}>
              <span className="text-gray-400">Acc: </span>
              <span className={accuracy < 0.6 ? 'text-amber-400 font-mono' : 'text-white font-mono'}>
                {(accuracy * 100).toFixed(1)}%
              </span>
            </div>
            <div className="px-3 py-1.5 bg-gray-800/50 rounded">
              <span className="text-gray-400">Epoch: </span>
              <span className="text-white font-mono">{epoch}</span>
            </div>
          </div>

          {/* Lesson reveal */}
          <AnimatePresence>
            {phase === 'lesson' && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
                className="border-t border-gray-600 pt-3 mt-3"
              >
                <div className="flex items-start gap-2">
                  <span className="text-xl">💡</span>
                  <div>
                    <p className="text-sm font-medium text-green-400">Key Insight:</p>
                    <p className="text-sm text-gray-200">{info.lesson}</p>
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </Container>
    </animated.div>
  );
};

// =============================================================================
// FAILURE BADGE (compact version)
// =============================================================================

export const FailureBadge = ({ failureType }: { failureType: FailureType }) => {
  const info = FAILURE_INFO[failureType];

  return (
    <motion.div
      initial={{ scale: 0.9, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      className="inline-flex items-center gap-2 px-3 py-1.5 bg-amber-900/50 border border-amber-600/50 rounded-lg"
    >
      <span>{info.icon}</span>
      <span className="text-amber-400 text-sm font-medium">{info.title}</span>
      <span className="text-xs text-amber-300/70 bg-amber-900/50 px-1.5 py-0.5 rounded">
        DESIGNED TO FAIL
      </span>
    </motion.div>
  );
};

// =============================================================================
// HELPER: Detect failure type from problem ID
// =============================================================================

export const getFailureTypeFromProblemId = (problemId: string): FailureType | null => {
  const mapping: Record<string, FailureType> = {
    fail_zero_init: 'zero_init',
    fail_high_lr: 'high_lr',
    fail_low_lr: 'low_lr',
    fail_no_hidden: 'no_hidden',
    fail_vanishing_grad: 'vanishing_gradient',
    fail_underfitting: 'underfitting',
  };

  return mapping[problemId] || null;
};

export default FailureDramatization;
