import { useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

export interface TuningConfig {
  parameter: 'learningRate' | 'epochs' | 'hiddenNeurons';
  parameterLabel: string;
  min: number;
  max: number;
  step: number;
  defaultValue: number;
  targetLoss: number;
  maxAttempts: number;
  architecture: number[];
  epochs: number;
  hints: string[];
  successMessage: string;
}

export interface TuningChallengeEntry {
  title: string;
  description: string;
  problem: string;
  config: TuningConfig;
}

export const TUNING_CHALLENGES: Record<string, TuningChallengeEntry> = {
  lr_sweet_spot: {
    title: 'Find the Learning Rate Sweet Spot',
    description: 'Circle classification needs a good learning rate. Too low = glacial, too high = chaos. Find the sweet spot!',
    problem: 'Circle',
    config: {
      parameter: 'learningRate',
      parameterLabel: 'Learning Rate',
      min: 0.001,
      max: 5.0,
      step: 0.01,
      defaultValue: 0.01,
      targetLoss: 0.05,
      maxAttempts: 5,
      architecture: [2, 8, 1],
      epochs: 500,
      hints: [
        'Circle classification converges well with LR between 0.1 and 1.0',
        'Try 0.5 - it is a classic starting point for small networks',
      ],
      successMessage: 'You found a great learning rate! This skill is essential for real ML work.',
    },
  },
  epoch_budget: {
    title: 'Optimize 3D Surface Training',
    description: '3D surface regression needs enough epochs to converge but not so many you waste compute. Find the minimum.',
    problem: '3D Surface',
    config: {
      parameter: 'epochs',
      parameterLabel: 'Training Epochs',
      min: 50,
      max: 5000,
      step: 50,
      defaultValue: 100,
      targetLoss: 0.1,
      maxAttempts: 5,
      architecture: [2, 16, 16, 1],
      epochs: 1000,
      hints: [
        'Regression problems need enough epochs for smooth approximation',
        '3D surfaces typically need 500-1500 epochs with a good architecture',
      ],
      successMessage: 'Efficient! Knowing when to stop training is a valuable skill.',
    },
  },
  capacity_tuning: {
    title: 'Tune Spiral Network Capacity',
    description: 'The spiral problem needs enough neurons but not too many. Find the minimum hidden layer size that works.',
    problem: 'Spiral',
    config: {
      parameter: 'hiddenNeurons',
      parameterLabel: 'Hidden Neurons per Layer',
      min: 1,
      max: 32,
      step: 1,
      defaultValue: 2,
      targetLoss: 0.1,
      maxAttempts: 5,
      architecture: [2, 4, 1],
      epochs: 1000,
      hints: [
        'Spirals need a complex non-linear boundary - at least 8 neurons',
        'Try 12-16 neurons - enough capacity for the spiral pattern',
      ],
      successMessage: 'Right-sizing prevents overfitting and speeds up training!',
    },
  },
  parity_capacity: {
    title: 'Right-Size for 5-Bit Parity',
    description: '5-bit parity (XOR of 5 inputs) is harder than 2-input XOR. How many hidden neurons do you need?',
    problem: '5-Bit XOR Parity',
    config: {
      parameter: 'hiddenNeurons',
      parameterLabel: 'Hidden Neurons per Layer',
      min: 2,
      max: 32,
      step: 1,
      defaultValue: 3,
      targetLoss: 0.1,
      maxAttempts: 5,
      architecture: [5, 4, 1],
      epochs: 2000,
      hints: [
        '5-bit parity needs more capacity than 2-input XOR',
        'Try 8-16 neurons - the problem grows exponentially with inputs',
      ],
      successMessage: 'Great! Scaling capacity with problem complexity is a key ML skill.',
    },
  },
  xor_epochs: {
    title: 'Find the Minimum Epochs for XOR',
    description: 'You know XOR works with [2, 4, 1] and LR=0.5. But how many epochs does it really need? Find the minimum!',
    problem: 'XOR Gate',
    config: {
      parameter: 'epochs',
      parameterLabel: 'Training Epochs',
      min: 10,
      max: 2000,
      step: 10,
      defaultValue: 50,
      targetLoss: 0.05,
      maxAttempts: 5,
      architecture: [2, 4, 1],
      epochs: 500,
      hints: [
        'XOR is simple but still needs enough iterations',
        'Try 200-500 epochs with LR=0.5',
      ],
      successMessage: 'Efficient! You found the sweet spot between too few and wasteful epochs.',
    },
  },
  cnn_epoch_budget: {
    title: 'Find CNN Training Epochs',
    description: 'CNN digit recognition needs enough training. Find the minimum epochs for 90%+ accuracy on 8x8 digits.',
    problem: 'Digits',
    config: {
      parameter: 'epochs',
      parameterLabel: 'Training Epochs',
      min: 5,
      max: 200,
      step: 5,
      defaultValue: 10,
      targetLoss: 0.3,
      maxAttempts: 5,
      architecture: [2, 8, 1],
      epochs: 100,
      hints: [
        'CNNs learn features progressively — early epochs learn edges, later epochs learn shapes',
        'Try 30-50 epochs as a starting point',
      ],
      successMessage: 'CNN training is efficient! Feature reuse means fewer epochs than dense networks.',
    },
  },
};

interface TuningChallengeProps {
  challengeId: string;
  onSolved: (success: boolean) => void;
  onTrain: (config: { architecture: number[]; learningRate: number; epochs: number }) => void;
}

export const TuningChallenge = ({ challengeId, onSolved, onTrain }: TuningChallengeProps) => {
  const challenge = TUNING_CHALLENGES[challengeId];
  if (!challenge) {
    return <div className="text-red-400">Unknown tuning challenge: {challengeId}</div>;
  }

  const { title, description, problem, config } = challenge;

  const [value, setValue] = useState(config.defaultValue);
  const [attempts, setAttempts] = useState(0);
  const [results, setResults] = useState<{ value: number; loss: number }[]>([]);
  const [bestLoss, setBestLoss] = useState<number | null>(null);
  const [solved, setSolved] = useState(false);
  const [training, setTraining] = useState(false);
  const [hintsRevealed, setHintsRevealed] = useState(0);

  const closeness = bestLoss !== null
    ? Math.max(0, 1 - (bestLoss - config.targetLoss) / 2)
    : 0;

  const getTrainConfig = useCallback(() => {
    const arch = [...config.architecture];
    let lr = 0.5;
    let epochs = config.epochs;

    if (config.parameter === 'learningRate') {
      lr = value;
    } else if (config.parameter === 'epochs') {
      epochs = value;
    } else if (config.parameter === 'hiddenNeurons') {
      // Rebuild architecture with the tuned neuron count
      arch.splice(1, arch.length - 2, value);
    }

    return { architecture: arch, learningRate: lr, epochs };
  }, [value, config]);

  const handleTry = async () => {
    if (solved || attempts >= config.maxAttempts) return;

    setTraining(true);
    const trainConfig = getTrainConfig();
    onTrain(trainConfig);

    // Simulate getting loss result from training
    // The actual loss comes from the training callback, but we estimate for the UI
    // A more sophisticated version would hook into real training results
    const simulatedLoss = simulateLoss(config.parameter, value, config);

    setTimeout(() => {
      const newAttempts = attempts + 1;
      const newResults = [...results, { value, loss: simulatedLoss }];

      setAttempts(newAttempts);
      setResults(newResults);
      setTraining(false);

      if (bestLoss === null || simulatedLoss < bestLoss) {
        setBestLoss(simulatedLoss);
      }

      if (simulatedLoss <= config.targetLoss) {
        setSolved(true);
        onSolved(true);
      }
      // When max attempts exhausted without success, don't call onSolved —
      // step stays incomplete and the retry UI appears
    }, 1500);
  };

  const attemptsLeft = config.maxAttempts - attempts;

  return (
    <div className="bg-gray-800 rounded-lg border border-gray-700 overflow-hidden">
      {/* Header */}
      <div className="bg-indigo-900/30 border-b border-indigo-600/30 px-4 py-3">
        <div className="flex items-center gap-2">
          <span className="text-2xl">🎛️</span>
          <div>
            <h3 className="text-lg font-semibold text-white">{title}</h3>
            <p className="text-indigo-300 text-sm">Tuning Challenge - {problem}</p>
          </div>
        </div>
      </div>

      {/* Description */}
      <div className="px-4 py-4 border-b border-gray-700">
        <p className="text-gray-200 mb-3">{description}</p>

        {/* Target */}
        <div className="bg-gray-900 rounded-lg p-3 flex items-center justify-between">
          <div>
            <span className="text-gray-400 text-sm">Target Loss:</span>
            <span className="text-green-400 font-mono font-bold ml-2">&le; {config.targetLoss}</span>
          </div>
          <div>
            <span className="text-gray-400 text-sm">Attempts:</span>
            <span className={`font-bold ml-2 ${attemptsLeft <= 1 ? 'text-red-400' : attemptsLeft <= 2 ? 'text-amber-400' : 'text-blue-400'}`}>
              {attemptsLeft} left
            </span>
          </div>
        </div>
      </div>

      {/* Tuning Slider */}
      <div className="px-4 py-4 border-b border-gray-700">
        <label className="block text-sm font-medium text-gray-300 mb-2">
          {config.parameterLabel}: <span className="text-indigo-400 font-mono">{formatValue(value, config.parameter)}</span>
        </label>

        <div className="relative">
          <input
            type="range"
            min={config.min}
            max={config.max}
            step={config.step}
            value={value}
            onChange={(e) => setValue(Number(e.target.value))}
            disabled={solved || attempts >= config.maxAttempts || training}
            className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-indigo-500 disabled:opacity-50"
          />
          <div className="flex justify-between text-xs text-gray-500 mt-1">
            <span>{formatValue(config.min, config.parameter)}</span>
            <span>{formatValue(config.max, config.parameter)}</span>
          </div>
        </div>

        {/* Previous attempts markers */}
        {results.length > 0 && (
          <div className="mt-3 flex flex-wrap gap-2">
            {results.map((r, i) => (
              <span
                key={i}
                className={`text-xs px-2 py-1 rounded-full font-mono ${
                  r.loss <= config.targetLoss
                    ? 'bg-green-900/50 text-green-400 border border-green-600/50'
                    : 'bg-gray-700 text-gray-400'
                }`}
              >
                {formatValue(r.value, config.parameter)} → {r.loss.toFixed(3)}
              </span>
            ))}
          </div>
        )}
      </div>

      {/* Thermometer */}
      {bestLoss !== null && (
        <div className="px-4 py-3 border-b border-gray-700">
          <div className="flex items-center justify-between text-sm mb-1">
            <span className="text-gray-400">Closeness to target</span>
            <span className={`font-medium ${closeness >= 1 ? 'text-green-400' : closeness > 0.7 ? 'text-amber-400' : 'text-red-400'}`}>
              Best loss: {bestLoss.toFixed(4)}
            </span>
          </div>
          <div className="h-3 bg-gray-700 rounded-full overflow-hidden">
            <motion.div
              className={`h-full rounded-full ${
                closeness >= 1 ? 'bg-green-500' : closeness > 0.7 ? 'bg-amber-500' : closeness > 0.4 ? 'bg-orange-500' : 'bg-red-500'
              }`}
              initial={{ width: 0 }}
              animate={{ width: `${Math.min(closeness * 100, 100)}%` }}
              transition={{ type: 'spring', stiffness: 100 }}
            />
          </div>
          <div className="flex justify-between text-xs text-gray-500 mt-1">
            <span>Far</span>
            <span>Target ({config.targetLoss})</span>
          </div>
        </div>
      )}

      {/* Hints */}
      {attempts >= 2 && !solved && hintsRevealed < config.hints.length && (
        <div className="px-4 py-3 border-b border-gray-700">
          <button
            onClick={() => setHintsRevealed(prev => Math.min(prev + 1, config.hints.length))}
            className="text-sm text-amber-400 hover:text-amber-300 flex items-center gap-1"
          >
            <span>💡</span> Reveal hint ({hintsRevealed}/{config.hints.length})
          </button>
          <AnimatePresence>
            {config.hints.slice(0, hintsRevealed).map((hint, i) => (
              <motion.p
                key={i}
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                className="text-sm text-amber-200 mt-2 pl-6"
              >
                {hint}
              </motion.p>
            ))}
          </AnimatePresence>
        </div>
      )}

      {/* Actions */}
      <div className="px-4 py-4">
        {!solved && attempts < config.maxAttempts && (
          <motion.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            onClick={handleTry}
            disabled={training}
            className={`w-full py-3 rounded-lg font-medium transition-colors ${
              training
                ? 'bg-gray-600 text-gray-400 cursor-wait'
                : 'bg-indigo-600 hover:bg-indigo-500 text-white'
            }`}
          >
            {training ? (
              <span className="flex items-center justify-center gap-2">
                <motion.span
                  animate={{ rotate: 360 }}
                  transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
                >
                  ⚡
                </motion.span>
                Training...
              </span>
            ) : (
              `Try ${config.parameterLabel} = ${formatValue(value, config.parameter)}`
            )}
          </motion.button>
        )}

        {/* Success */}
        <AnimatePresence>
          {solved && (
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              className="bg-green-900/30 border border-green-600/50 rounded-lg p-4 text-center"
            >
              <span className="text-3xl block mb-2">🎉</span>
              <p className="text-green-400 font-medium mb-1">Target Reached!</p>
              <p className="text-gray-300 text-sm">{config.successMessage}</p>
              <p className="text-green-300 font-mono text-sm mt-2">
                {config.parameterLabel} = {formatValue(results[results.length - 1]?.value ?? value, config.parameter)} → Loss: {bestLoss?.toFixed(4)}
              </p>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Failure with retry */}
        <AnimatePresence>
          {!solved && attempts >= config.maxAttempts && (
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              className="bg-amber-900/20 border border-amber-600/30 rounded-lg p-4"
            >
              <div className="text-center mb-3">
                <span className="text-3xl block mb-2">🔄</span>
                <p className="text-amber-400 font-medium mb-1">Out of Attempts</p>
                <p className="text-gray-300 text-sm">
                  Best loss: {bestLoss?.toFixed(4)} (target: {config.targetLoss}).
                  Review the hints and try again!
                </p>
              </div>
              <button
                onClick={() => {
                  setAttempts(0);
                  setResults([]);
                  setBestLoss(null);
                }}
                className="w-full py-2 bg-amber-600 text-white rounded-lg hover:bg-amber-500 font-medium transition-colors"
              >
                Retry
              </button>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
};

// Format display value based on parameter type
function formatValue(value: number, parameter: string): string {
  if (parameter === 'learningRate') return value.toFixed(3);
  if (parameter === 'epochs') return String(Math.round(value));
  return String(Math.round(value));
}

// Simple loss simulation based on parameter value
// This gives immediate feedback while the real training runs
function simulateLoss(parameter: string, value: number, _config: TuningConfig): number {
  const noise = (Math.random() - 0.5) * 0.02;

  if (parameter === 'learningRate') {
    // Sweet spot around 0.3-0.7 for XOR
    const optimal = 0.5;
    const distance = Math.abs(Math.log(value / optimal));
    if (value > 3) return 2.0 + noise; // Exploding
    if (value < 0.01) return 0.5 + noise; // Too slow
    return Math.max(0.001, 0.01 + distance * 0.15 + noise);
  }

  if (parameter === 'epochs') {
    // More epochs = lower loss, diminishing returns
    const effectiveEpochs = Math.min(value, 3000);
    const loss = 0.5 * Math.exp(-effectiveEpochs / 500);
    return Math.max(0.001, loss + noise);
  }

  if (parameter === 'hiddenNeurons') {
    // Need at least ~4 neurons, diminishing returns after 8
    if (value < 3) return 0.4 + noise;
    if (value < 5) return 0.15 + noise;
    return Math.max(0.001, 0.08 * Math.exp(-(value - 4) / 6) + noise);
  }

  return 0.5 + noise;
}

export default TuningChallenge;
