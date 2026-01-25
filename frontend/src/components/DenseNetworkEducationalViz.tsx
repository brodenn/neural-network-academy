import { useState, useEffect, useMemo } from 'react';
import { createPortal } from 'react-dom';
import { motion, AnimatePresence } from 'framer-motion';

interface DenseNetworkEducationalVizProps {
  layerSizes: number[];
  inputLabels: string[];
  outputLabels: string[];
  outputActivation: 'sigmoid' | 'softmax';
  onClose: () => void;
}

type Mode = 'input' | 'weighted_sum' | 'activation' | 'forward' | 'output' | 'backprop';

// Seeded random for consistent values
function seededRandom(seed: number): number {
  const x = Math.sin(seed * 9999) * 10000;
  return x - Math.floor(x);
}

// Stagger animation variants
const containerVariants = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: {
      staggerChildren: 0.08,
      delayChildren: 0.1,
    },
  },
};

const itemVariants = {
  hidden: { opacity: 0, scale: 0.8, y: 10 },
  show: {
    opacity: 1,
    scale: 1,
    y: 0,
    transition: { type: 'spring' as const, stiffness: 300, damping: 25 }
  },
};

// Tab button component
function TabButton({
  mode,
  currentMode,
  onClick,
  color,
  children
}: {
  mode: Mode;
  currentMode: Mode;
  onClick: () => void;
  color: string;
  children: React.ReactNode;
}) {
  const isActive = mode === currentMode;
  const colorClasses: Record<string, string> = {
    purple: isActive ? 'bg-purple-600 text-white' : 'bg-gray-700 text-gray-300 hover:bg-gray-600',
    cyan: isActive ? 'bg-cyan-600 text-white' : 'bg-gray-700 text-gray-300 hover:bg-gray-600',
    orange: isActive ? 'bg-orange-600 text-white' : 'bg-gray-700 text-gray-300 hover:bg-gray-600',
    green: isActive ? 'bg-green-600 text-white' : 'bg-gray-700 text-gray-300 hover:bg-gray-600',
    blue: isActive ? 'bg-blue-600 text-white' : 'bg-gray-700 text-gray-300 hover:bg-gray-600',
    pink: isActive ? 'bg-pink-600 text-white' : 'bg-gray-700 text-gray-300 hover:bg-gray-600',
  };

  return (
    <button
      onClick={onClick}
      className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${colorClasses[color]}`}
    >
      {children}
    </button>
  );
}

// Neuron component
function Neuron({
  value,
  label,
  isHighlighted,
  color = 'cyan',
  size = 'md',
  showValue = true
}: {
  value: number;
  label?: string;
  isHighlighted?: boolean;
  color?: 'cyan' | 'purple' | 'green' | 'orange' | 'pink' | 'blue';
  size?: 'sm' | 'md' | 'lg';
  showValue?: boolean;
}) {
  const sizeClasses = {
    sm: 'w-10 h-10 text-[10px]',
    md: 'w-14 h-14 text-xs',
    lg: 'w-18 h-18 text-sm',
  };

  const colorMap: Record<string, { bg: string; border: string; glow: string }> = {
    cyan: { bg: 'bg-cyan-600', border: 'border-cyan-400', glow: 'shadow-cyan-500/50' },
    purple: { bg: 'bg-purple-600', border: 'border-purple-400', glow: 'shadow-purple-500/50' },
    green: { bg: 'bg-green-600', border: 'border-green-400', glow: 'shadow-green-500/50' },
    orange: { bg: 'bg-orange-600', border: 'border-orange-400', glow: 'shadow-orange-500/50' },
    pink: { bg: 'bg-pink-600', border: 'border-pink-400', glow: 'shadow-pink-500/50' },
    blue: { bg: 'bg-blue-600', border: 'border-blue-400', glow: 'shadow-blue-500/50' },
  };

  const colors = colorMap[color];
  const intensity = Math.abs(value);

  return (
    <motion.div
      className="flex flex-col items-center gap-1"
      variants={itemVariants}
    >
      {label && <span className="text-[10px] text-gray-500">{label}</span>}
      <motion.div
        className={`${sizeClasses[size]} rounded-full flex items-center justify-center font-mono border-2 ${colors.border} ${isHighlighted ? `${colors.bg} shadow-lg ${colors.glow}` : 'bg-gray-800'}`}
        animate={{
          scale: isHighlighted ? 1.1 : 1,
          backgroundColor: isHighlighted ? undefined : `rgba(31, 41, 55, ${0.3 + intensity * 0.7})`,
        }}
        transition={{ type: 'spring', stiffness: 400, damping: 25 }}
      >
        {showValue && (
          <span className={isHighlighted ? 'text-white font-bold' : 'text-gray-300'}>
            {value.toFixed(2)}
          </span>
        )}
      </motion.div>
    </motion.div>
  );
}

export function DenseNetworkEducationalViz({
  layerSizes,
  inputLabels,
  outputLabels,
  outputActivation,
  onClose,
}: DenseNetworkEducationalVizProps) {
  const [mode, setMode] = useState<Mode>('input');
  const [isPlaying, setIsPlaying] = useState(false);
  const [speed, setSpeed] = useState(500);
  const [currentNeuron, setCurrentNeuron] = useState(0);
  const [currentLayer, setCurrentLayer] = useState(0);
  const [forwardStep, setForwardStep] = useState(0);

  // Generate sample input values
  const inputValues = useMemo(() => {
    return inputLabels.map((_, i) => seededRandom(i * 17) * 0.8 + 0.1);
  }, [inputLabels]);

  // Generate simulated weights
  const weights = useMemo(() => {
    const w: number[][][] = [];
    for (let l = 0; l < layerSizes.length - 1; l++) {
      const layerWeights: number[][] = [];
      for (let i = 0; i < layerSizes[l]; i++) {
        const neuronWeights: number[] = [];
        for (let j = 0; j < layerSizes[l + 1]; j++) {
          neuronWeights.push(seededRandom(l * 1000 + i * 100 + j) * 2 - 1);
        }
        layerWeights.push(neuronWeights);
      }
      w.push(layerWeights);
    }
    return w;
  }, [layerSizes]);

  // Generate biases
  const biases = useMemo(() => {
    const b: number[][] = [];
    for (let l = 1; l < layerSizes.length; l++) {
      const layerBiases: number[] = [];
      for (let i = 0; i < layerSizes[l]; i++) {
        layerBiases.push(seededRandom(l * 500 + i * 33) * 0.5 - 0.25);
      }
      b.push(layerBiases);
    }
    return b;
  }, [layerSizes]);

  // Calculate activations for each layer
  const activations = useMemo(() => {
    const acts: number[][] = [inputValues];
    let current = inputValues;

    for (let l = 0; l < weights.length; l++) {
      const next: number[] = [];
      const isOutputLayer = l === weights.length - 1;

      for (let j = 0; j < layerSizes[l + 1]; j++) {
        let sum = biases[l][j];
        for (let i = 0; i < current.length; i++) {
          sum += current[i] * weights[l][i][j];
        }

        // Apply activation
        if (isOutputLayer) {
          // For output, we'll apply sigmoid/softmax later
          next.push(sum);
        } else {
          // ReLU for hidden layers
          next.push(Math.max(0, sum));
        }
      }

      // Apply softmax to output layer if needed
      if (isOutputLayer && outputActivation === 'softmax') {
        const maxVal = Math.max(...next);
        const exps = next.map(v => Math.exp(v - maxVal));
        const sumExps = exps.reduce((a, b) => a + b, 0);
        const softmaxed = exps.map(e => e / sumExps);
        acts.push(softmaxed);
      } else if (isOutputLayer) {
        // Sigmoid for output
        const sigmoided = next.map(v => 1 / (1 + Math.exp(-v)));
        acts.push(sigmoided);
      } else {
        acts.push(next);
      }

      current = acts[acts.length - 1];
    }

    return acts;
  }, [inputValues, weights, biases, layerSizes, outputActivation]);

  // Pre-activation values (before ReLU/Sigmoid)
  const preActivations = useMemo(() => {
    const preActs: number[][] = [inputValues];
    let current = inputValues;

    for (let l = 0; l < weights.length; l++) {
      const next: number[] = [];
      for (let j = 0; j < layerSizes[l + 1]; j++) {
        let sum = biases[l][j];
        for (let i = 0; i < current.length; i++) {
          sum += current[i] * weights[l][i][j];
        }
        next.push(sum);
      }
      preActs.push(next);
      current = activations[l + 1];
    }

    return preActs;
  }, [inputValues, weights, biases, layerSizes, activations]);

  // Auto-play animation
  useEffect(() => {
    if (!isPlaying) return;

    const interval = setInterval(() => {
      if (mode === 'weighted_sum' || mode === 'activation') {
        setCurrentNeuron(prev => {
          const maxNeurons = layerSizes[currentLayer + 1] || layerSizes[1];
          if (prev + 1 >= maxNeurons) {
            setCurrentLayer(l => (l + 1) % (layerSizes.length - 1));
            return 0;
          }
          return prev + 1;
        });
      } else if (mode === 'forward') {
        setForwardStep(prev => (prev + 1) % (layerSizes.length + 2));
      }
    }, speed);

    return () => clearInterval(interval);
  }, [isPlaying, mode, layerSizes, currentLayer, speed]);

  // Reset state when changing modes
  const handleModeChange = (newMode: Mode) => {
    setMode(newMode);
    setCurrentNeuron(0);
    setCurrentLayer(0);
    setForwardStep(0);
    setIsPlaying(false);
  };

  const modalContent = (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 bg-black/90 z-[9999] flex items-center justify-center p-4"
      onClick={onClose}
      style={{ position: 'fixed', top: 0, left: 0, right: 0, bottom: 0 }}
    >
      <motion.div
        initial={{ scale: 0.9, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        exit={{ scale: 0.9, opacity: 0 }}
        className="bg-gray-900 rounded-xl p-6 max-w-5xl w-full max-h-[90vh] overflow-auto"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-xl font-bold text-white">
            Dense Neural Network: Step by Step
          </h2>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-white text-2xl leading-none"
          >
            &times;
          </button>
        </div>

        {/* Pipeline indicator */}
        <div className="mb-4 flex items-center gap-1 text-xs overflow-x-auto pb-2">
          {(['input', 'weighted_sum', 'activation', 'forward', 'output', 'backprop'] as Mode[]).map((m, i) => {
            const labels = ['Input', 'Weighted Sum', 'Activation', 'Forward Pass', 'Output', 'Backprop'];
            return (
              <div key={m} className="flex items-center">
                <div
                  className={`px-2 py-1 rounded cursor-pointer transition-all whitespace-nowrap ${
                    mode === m
                      ? 'bg-white text-gray-900 font-bold scale-110'
                      : 'bg-gray-700 text-gray-400 hover:bg-gray-600'
                  }`}
                  onClick={() => handleModeChange(m)}
                >
                  {i + 1}. {labels[i]}
                </div>
                {i < 5 && <span className="text-gray-600 mx-1">→</span>}
              </div>
            );
          })}
        </div>

        {/* Mode tabs */}
        <div className="flex gap-2 mb-4 flex-wrap">
          <TabButton mode="input" currentMode={mode} onClick={() => handleModeChange('input')} color="purple">
            📥 Input Layer
          </TabButton>
          <TabButton mode="weighted_sum" currentMode={mode} onClick={() => handleModeChange('weighted_sum')} color="cyan">
            ⚖️ Weighted Sum
          </TabButton>
          <TabButton mode="activation" currentMode={mode} onClick={() => handleModeChange('activation')} color="orange">
            ⚡ Activation
          </TabButton>
          <TabButton mode="forward" currentMode={mode} onClick={() => handleModeChange('forward')} color="green">
            ➡️ Forward Pass
          </TabButton>
          <TabButton mode="output" currentMode={mode} onClick={() => handleModeChange('output')} color="blue">
            📊 Output
          </TabButton>
          <TabButton mode="backprop" currentMode={mode} onClick={() => handleModeChange('backprop')} color="pink">
            🔄 Backprop
          </TabButton>
        </div>

        <AnimatePresence mode="wait">
          {/* INPUT LAYER VISUALIZATION */}
          {mode === 'input' && (
            <motion.div
              key="input"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              className="space-y-4"
            >
              <div className="bg-gray-800 rounded-lg p-4">
                <h3 className="text-purple-400 font-semibold mb-2">How Input Layer Works</h3>
                <p className="text-gray-300 text-sm">
                  The <span className="text-purple-400 font-medium">input layer</span> receives raw data from the outside world.
                  Each neuron represents one feature/measurement. Values are typically normalized to 0-1 range.
                </p>
              </div>

              <div className="flex items-center justify-center gap-6 flex-wrap">
                {/* Raw data */}
                <div className="text-center">
                  <div className="text-gray-400 text-sm mb-3">Raw Sensor Data</div>
                  <motion.div
                    className="space-y-2"
                    variants={containerVariants}
                    initial="hidden"
                    animate="show"
                  >
                    {inputLabels.map((label, i) => (
                      <motion.div
                        key={i}
                        variants={itemVariants}
                        className="flex items-center gap-3 bg-gray-800 rounded-lg px-3 py-2"
                      >
                        <span className="text-gray-400 text-sm w-24 truncate">{label}</span>
                        <div className="w-24 h-3 bg-gray-700 rounded overflow-hidden">
                          <motion.div
                            className="h-full bg-purple-500"
                            initial={{ width: 0 }}
                            animate={{ width: `${inputValues[i] * 100}%` }}
                            transition={{ duration: 0.5, delay: i * 0.1 }}
                          />
                        </div>
                        <span className="text-purple-400 font-mono text-sm w-12">
                          {inputValues[i].toFixed(2)}
                        </span>
                      </motion.div>
                    ))}
                  </motion.div>
                </div>

                <motion.div
                  className="text-4xl text-gray-500"
                  animate={{ x: [0, 10, 0] }}
                  transition={{ duration: 1, repeat: Infinity }}
                >
                  →
                </motion.div>

                {/* Input neurons */}
                <div className="text-center">
                  <div className="text-gray-400 text-sm mb-3">Input Neurons</div>
                  <motion.div
                    className="flex flex-col gap-2 items-center"
                    variants={containerVariants}
                    initial="hidden"
                    animate="show"
                  >
                    {inputValues.map((val, i) => (
                      <Neuron
                        key={i}
                        value={val}
                        label={`x${i + 1}`}
                        color="purple"
                        isHighlighted={val > 0.5}
                      />
                    ))}
                  </motion.div>
                </div>
              </div>

              <div className="bg-gray-800 rounded-lg p-4">
                <h4 className="text-gray-300 font-medium mb-2">Why Normalize Inputs?</h4>
                <ul className="text-sm text-gray-400 space-y-1">
                  <li>• <span className="text-purple-400">Faster training</span>: Gradients flow better in 0-1 range</li>
                  <li>• <span className="text-purple-400">Equal importance</span>: All features contribute fairly</li>
                  <li>• <span className="text-purple-400">Stability</span>: Prevents numerical overflow</li>
                </ul>
              </div>
            </motion.div>
          )}

          {/* WEIGHTED SUM VISUALIZATION */}
          {mode === 'weighted_sum' && (
            <motion.div
              key="weighted_sum"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              className="space-y-4"
            >
              <div className="bg-gray-800 rounded-lg p-4">
                <h3 className="text-cyan-400 font-semibold mb-2">How Weighted Sum Works</h3>
                <p className="text-gray-300 text-sm">
                  Each connection has a <span className="text-cyan-400 font-medium">weight</span>.
                  The neuron computes: <code className="bg-gray-700 px-2 py-0.5 rounded">Σ(weight × input) + bias</code>
                </p>
              </div>

              <div className="flex items-center justify-center gap-4 flex-wrap">
                {/* Input neurons */}
                <div className="text-center">
                  <div className="text-gray-400 text-sm mb-2">Inputs (x)</div>
                  <div className="flex flex-col gap-1">
                    {inputValues.slice(0, 4).map((val, i) => (
                      <motion.div
                        key={i}
                        className="flex items-center gap-2"
                        animate={{
                          scale: currentNeuron === i ? 1.1 : 1,
                          opacity: currentNeuron === i ? 1 : 0.6
                        }}
                      >
                        <div className="w-10 h-10 rounded-full bg-purple-600 flex items-center justify-center text-white text-xs font-mono">
                          {val.toFixed(2)}
                        </div>
                      </motion.div>
                    ))}
                  </div>
                </div>

                {/* Weights */}
                <div className="text-center">
                  <div className="text-gray-400 text-sm mb-2">Weights (w)</div>
                  <div className="flex flex-col gap-1">
                    {weights[0]?.slice(0, 4).map((w, i) => (
                      <motion.div
                        key={i}
                        className={`w-14 h-10 rounded flex items-center justify-center text-xs font-mono ${
                          w[currentNeuron] >= 0 ? 'bg-cyan-800 text-cyan-200' : 'bg-pink-900 text-pink-200'
                        }`}
                        animate={{ scale: i === currentNeuron % 4 ? 1.1 : 1 }}
                      >
                        {w[currentNeuron]?.toFixed(2) || '0.00'}
                      </motion.div>
                    ))}
                  </div>
                </div>

                {/* Multiplication */}
                <div className="text-center">
                  <div className="text-gray-400 text-sm mb-2">x × w</div>
                  <div className="flex flex-col gap-1">
                    {inputValues.slice(0, 4).map((val, i) => {
                      const w = weights[0]?.[i]?.[currentNeuron] || 0;
                      const product = val * w;
                      return (
                        <motion.div
                          key={i}
                          className="w-14 h-10 rounded bg-gray-700 flex items-center justify-center text-xs font-mono text-gray-300"
                          initial={{ opacity: 0 }}
                          animate={{ opacity: 1 }}
                          transition={{ delay: i * 0.1 }}
                        >
                          {product.toFixed(2)}
                        </motion.div>
                      );
                    })}
                  </div>
                </div>

                <div className="text-2xl text-gray-500">=</div>

                {/* Sum + Bias */}
                <div className="text-center">
                  <div className="text-gray-400 text-sm mb-2">Σ + bias</div>
                  <motion.div
                    className="w-20 h-20 rounded-xl bg-gradient-to-br from-cyan-600 to-cyan-800 flex flex-col items-center justify-center"
                    key={currentNeuron}
                    initial={{ scale: 0.5 }}
                    animate={{ scale: 1 }}
                    transition={{ type: 'spring', stiffness: 300 }}
                  >
                    <span className="text-xs text-cyan-200">z{currentNeuron + 1}</span>
                    <span className="text-white font-bold">
                      {preActivations[1]?.[currentNeuron]?.toFixed(2) || '0.00'}
                    </span>
                  </motion.div>
                </div>
              </div>

              <div className="bg-gray-800 rounded-lg p-3">
                <div className="text-sm text-gray-400 font-mono text-center">
                  z = {inputValues.slice(0, 4).map((val, i) => {
                    const w = weights[0]?.[i]?.[currentNeuron] || 0;
                    return `(${val.toFixed(1)}×${w.toFixed(1)})`;
                  }).join(' + ')} + <span className="text-yellow-400">{biases[0]?.[currentNeuron]?.toFixed(2) || 0}</span>
                  {' = '}
                  <span className="text-cyan-400">{preActivations[1]?.[currentNeuron]?.toFixed(2) || 0}</span>
                </div>
              </div>

              <div className="bg-gray-800 rounded-lg p-4">
                <h4 className="text-gray-300 font-medium mb-2">Why Weights & Biases?</h4>
                <ul className="text-sm text-gray-400 space-y-1">
                  <li>• <span className="text-cyan-400">Weights</span>: Control importance of each input connection</li>
                  <li>• <span className="text-yellow-400">Bias</span>: Shifts the activation threshold (like an offset)</li>
                  <li>• <span className="text-cyan-400">Learning</span>: Training adjusts these values to minimize error</li>
                </ul>
              </div>
            </motion.div>
          )}

          {/* ACTIVATION FUNCTION VISUALIZATION */}
          {mode === 'activation' && (
            <motion.div
              key="activation"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              className="space-y-4"
            >
              <div className="bg-gray-800 rounded-lg p-4">
                <h3 className="text-orange-400 font-semibold mb-2">How Activation Functions Work</h3>
                <p className="text-gray-300 text-sm">
                  Activation functions add <span className="text-orange-400 font-medium">non-linearity</span>.
                  Without them, stacking layers would just be linear transformations.
                </p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* ReLU */}
                <div className="bg-gray-800 rounded-lg p-4">
                  <h4 className="text-orange-400 font-medium mb-3 flex items-center gap-2">
                    <span>ReLU</span>
                    <span className="text-xs text-gray-500">(Hidden Layers)</span>
                  </h4>
                  <div className="flex items-center justify-center gap-4">
                    <div className="text-center">
                      <div className="text-gray-500 text-xs mb-1">Before (z)</div>
                      <motion.div
                        className="flex flex-col gap-1"
                        variants={containerVariants}
                        initial="hidden"
                        animate="show"
                      >
                        {preActivations[1]?.slice(0, 4).map((z, i) => (
                          <motion.div
                            key={i}
                            variants={itemVariants}
                            className={`w-14 h-8 rounded flex items-center justify-center text-xs font-mono ${
                              z < 0 ? 'bg-red-900/60 text-red-300' : 'bg-gray-700 text-gray-300'
                            }`}
                          >
                            {z.toFixed(2)}
                          </motion.div>
                        ))}
                      </motion.div>
                    </div>

                    <div className="text-center">
                      <motion.div
                        className="text-orange-400 font-mono text-sm"
                        animate={{ scale: [1, 1.1, 1] }}
                        transition={{ duration: 1, repeat: Infinity }}
                      >
                        max(0, z)
                      </motion.div>
                    </div>

                    <div className="text-center">
                      <div className="text-gray-500 text-xs mb-1">After (a)</div>
                      <motion.div
                        className="flex flex-col gap-1"
                        variants={containerVariants}
                        initial="hidden"
                        animate="show"
                      >
                        {preActivations[1]?.slice(0, 4).map((z, i) => {
                          const a = Math.max(0, z);
                          const wasNegative = z < 0;
                          return (
                            <motion.div
                              key={i}
                              variants={itemVariants}
                              className={`w-14 h-8 rounded flex items-center justify-center text-xs font-mono ${
                                wasNegative ? 'bg-orange-600 text-white font-bold' : 'bg-gray-700 text-gray-300'
                              }`}
                            >
                              {a.toFixed(2)}
                            </motion.div>
                          );
                        })}
                      </motion.div>
                    </div>
                  </div>
                </div>

                {/* Sigmoid */}
                <div className="bg-gray-800 rounded-lg p-4">
                  <h4 className="text-blue-400 font-medium mb-3 flex items-center gap-2">
                    <span>Sigmoid</span>
                    <span className="text-xs text-gray-500">(Output Layer - Binary)</span>
                  </h4>
                  <div className="flex items-center justify-center gap-4">
                    <div className="text-center">
                      <div className="text-gray-500 text-xs mb-1">Raw (z)</div>
                      <div className="w-14 h-10 rounded bg-gray-700 flex items-center justify-center text-sm font-mono text-gray-300">
                        {(preActivations[layerSizes.length - 1]?.[0] || 0).toFixed(2)}
                      </div>
                    </div>

                    <div className="text-center">
                      <motion.div
                        className="text-blue-400 font-mono text-xs"
                        animate={{ scale: [1, 1.1, 1] }}
                        transition={{ duration: 1, repeat: Infinity }}
                      >
                        1/(1+e<sup>-z</sup>)
                      </motion.div>
                    </div>

                    <div className="text-center">
                      <div className="text-gray-500 text-xs mb-1">Probability</div>
                      <motion.div
                        className="w-14 h-10 rounded bg-blue-600 flex items-center justify-center text-sm font-mono text-white font-bold"
                        animate={{ scale: [1, 1.05, 1] }}
                        transition={{ duration: 0.5, repeat: Infinity }}
                      >
                        {(activations[layerSizes.length - 1]?.[0] || 0.5).toFixed(2)}
                      </motion.div>
                    </div>
                  </div>
                  <div className="mt-3 text-center text-xs text-gray-500">
                    Squashes any value to (0, 1) range
                  </div>
                </div>
              </div>

              <div className="bg-gray-800 rounded-lg p-4">
                <h4 className="text-gray-300 font-medium mb-2">Why Non-Linearity?</h4>
                <ul className="text-sm text-gray-400 space-y-1">
                  <li>• <span className="text-orange-400">Complex patterns</span>: Can learn XOR, curves, boundaries</li>
                  <li>• <span className="text-orange-400">ReLU benefits</span>: Fast computation, avoids vanishing gradients</li>
                  <li>• <span className="text-blue-400">Sigmoid benefits</span>: Perfect for probabilities (0-1 output)</li>
                </ul>
              </div>
            </motion.div>
          )}

          {/* FORWARD PROPAGATION VISUALIZATION */}
          {mode === 'forward' && (
            <motion.div
              key="forward"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              className="space-y-4"
            >
              <div className="bg-gray-800 rounded-lg p-4">
                <h3 className="text-green-400 font-semibold mb-2">How Forward Propagation Works</h3>
                <p className="text-gray-300 text-sm">
                  Data flows <span className="text-green-400 font-medium">left to right</span> through the network.
                  Each layer transforms the signal: weighted sum → activation → next layer.
                </p>
              </div>

              <div className="overflow-x-auto pb-4">
                <svg width="700" height="300" className="mx-auto">
                  {/* Draw connections first */}
                  {layerSizes.slice(0, -1).map((size, layerIdx) => {
                    const nextSize = layerSizes[layerIdx + 1];
                    const x1 = 80 + layerIdx * 150;
                    const x2 = 80 + (layerIdx + 1) * 150;

                    return Array.from({ length: Math.min(size, 5) }).map((_, i) => {
                      const y1 = 50 + i * (200 / Math.min(size, 5));
                      return Array.from({ length: Math.min(nextSize, 5) }).map((_, j) => {
                        const y2 = 50 + j * (200 / Math.min(nextSize, 5));
                        const isActive = forwardStep > layerIdx;
                        const w = weights[layerIdx]?.[i]?.[j] || 0;

                        return (
                          <motion.line
                            key={`${layerIdx}-${i}-${j}`}
                            x1={x1 + 20}
                            y1={y1}
                            x2={x2 - 20}
                            y2={y2}
                            stroke={w >= 0 ? 'rgba(34, 211, 238, 0.5)' : 'rgba(236, 72, 153, 0.5)'}
                            strokeWidth={1 + Math.abs(w)}
                            initial={{ opacity: 0.1 }}
                            animate={{
                              opacity: isActive ? 0.7 : 0.1,
                              strokeWidth: isActive ? 1 + Math.abs(w) * 2 : 1
                            }}
                            transition={{ duration: 0.3 }}
                          />
                        );
                      });
                    });
                  })}

                  {/* Draw neurons */}
                  {layerSizes.map((size, layerIdx) => {
                    const x = 80 + layerIdx * 150;
                    const isActive = forwardStep >= layerIdx;
                    const displaySize = Math.min(size, 5);

                    return (
                      <g key={layerIdx}>
                        {/* Layer label */}
                        <text
                          x={x}
                          y={280}
                          textAnchor="middle"
                          fill="#9ca3af"
                          fontSize="12"
                        >
                          {layerIdx === 0 ? 'Input' : layerIdx === layerSizes.length - 1 ? 'Output' : `Hidden ${layerIdx}`}
                        </text>

                        {Array.from({ length: displaySize }).map((_, i) => {
                          const y = 50 + i * (200 / displaySize);
                          const activation = activations[layerIdx]?.[i] || 0;

                          return (
                            <g key={i}>
                              {/* Glow effect */}
                              {isActive && activation > 0.3 && (
                                <motion.circle
                                  cx={x}
                                  cy={y}
                                  r={25}
                                  fill="none"
                                  stroke={layerIdx === 0 ? '#a78bfa' : layerIdx === layerSizes.length - 1 ? '#3b82f6' : '#22c55e'}
                                  strokeWidth={2}
                                  initial={{ opacity: 0, scale: 0.5 }}
                                  animate={{ opacity: activation * 0.5, scale: 1 }}
                                  transition={{ duration: 0.3 }}
                                />
                              )}

                              {/* Neuron */}
                              <motion.circle
                                cx={x}
                                cy={y}
                                r={18}
                                fill={
                                  layerIdx === 0 ? '#7c3aed' :
                                  layerIdx === layerSizes.length - 1 ? '#2563eb' :
                                  activation > 0.5 ? '#16a34a' : '#374151'
                                }
                                stroke={isActive ? '#fff' : '#6b7280'}
                                strokeWidth={isActive ? 2 : 1}
                                initial={{ scale: 0.8, opacity: 0.5 }}
                                animate={{
                                  scale: isActive ? 1 : 0.8,
                                  opacity: isActive ? 1 : 0.5
                                }}
                                transition={{ duration: 0.3, delay: layerIdx * 0.1 }}
                              />

                              {/* Activation value */}
                              <motion.text
                                x={x}
                                y={y + 4}
                                textAnchor="middle"
                                fill="white"
                                fontSize="10"
                                fontFamily="monospace"
                                initial={{ opacity: 0 }}
                                animate={{ opacity: isActive ? 1 : 0 }}
                              >
                                {activation.toFixed(2)}
                              </motion.text>
                            </g>
                          );
                        })}

                        {/* Show ellipsis if more neurons */}
                        {size > 5 && (
                          <text x={x} y={260} textAnchor="middle" fill="#6b7280" fontSize="14">
                            ...+{size - 5}
                          </text>
                        )}
                      </g>
                    );
                  })}

                  {/* Signal flow indicator */}
                  <motion.rect
                    x={60 + forwardStep * 150}
                    y={20}
                    width={60}
                    height={250}
                    fill="rgba(34, 197, 94, 0.1)"
                    stroke="rgba(34, 197, 94, 0.5)"
                    strokeWidth={2}
                    strokeDasharray="5,5"
                    rx={10}
                    animate={{ x: 60 + (forwardStep % (layerSizes.length + 1)) * 150 }}
                    transition={{ duration: 0.3 }}
                  />
                </svg>
              </div>

              <div className="bg-gray-800 rounded-lg p-4">
                <h4 className="text-gray-300 font-medium mb-2">Forward Pass Summary</h4>
                <div className="flex items-center justify-center gap-2 text-sm flex-wrap">
                  <span className="px-2 py-1 bg-purple-900/50 text-purple-300 rounded">Input</span>
                  <span className="text-gray-500">→</span>
                  <span className="px-2 py-1 bg-gray-700 text-gray-300 rounded">× weights + bias</span>
                  <span className="text-gray-500">→</span>
                  <span className="px-2 py-1 bg-orange-900/50 text-orange-300 rounded">Activation</span>
                  <span className="text-gray-500">→</span>
                  <span className="px-2 py-1 bg-gray-700 text-gray-300 rounded">... repeat ...</span>
                  <span className="text-gray-500">→</span>
                  <span className="px-2 py-1 bg-blue-900/50 text-blue-300 rounded">Output</span>
                </div>
              </div>
            </motion.div>
          )}

          {/* OUTPUT LAYER VISUALIZATION */}
          {mode === 'output' && (
            <motion.div
              key="output"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              className="space-y-4"
            >
              <div className="bg-gray-800 rounded-lg p-4">
                <h3 className="text-blue-400 font-semibold mb-2">How Output Layer Works</h3>
                <p className="text-gray-300 text-sm">
                  The output layer produces the final prediction.
                  <span className="text-blue-400 font-medium"> Sigmoid</span> for binary (yes/no),
                  <span className="text-pink-400 font-medium"> Softmax</span> for multi-class (which category).
                </p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* Sigmoid output */}
                <div className="bg-gray-800 rounded-lg p-4">
                  <h4 className="text-blue-400 font-medium mb-3">Sigmoid (Binary Classification)</h4>
                  <div className="flex flex-col items-center gap-4">
                    <div className="flex items-center gap-4">
                      <div className="text-center">
                        <div className="text-gray-500 text-xs mb-1">Raw Score</div>
                        <div className="w-16 h-12 rounded bg-gray-700 flex items-center justify-center font-mono text-gray-300">
                          {(preActivations[layerSizes.length - 1]?.[0] || 0).toFixed(2)}
                        </div>
                      </div>
                      <motion.span
                        className="text-2xl text-gray-500"
                        animate={{ x: [0, 5, 0] }}
                        transition={{ duration: 0.5, repeat: Infinity }}
                      >
                        →
                      </motion.span>
                      <div className="text-center">
                        <div className="text-gray-500 text-xs mb-1">Probability</div>
                        <motion.div
                          className="w-16 h-12 rounded bg-blue-600 flex items-center justify-center font-mono text-white font-bold"
                          animate={{ scale: [1, 1.05, 1] }}
                          transition={{ duration: 1, repeat: Infinity }}
                        >
                          {(activations[layerSizes.length - 1]?.[0] || 0.5).toFixed(2)}
                        </motion.div>
                      </div>
                    </div>

                    <div className="w-full">
                      <div className="flex justify-between text-xs text-gray-500 mb-1">
                        <span>0 (False)</span>
                        <span>0.5</span>
                        <span>1 (True)</span>
                      </div>
                      <div className="h-4 bg-gray-700 rounded-full overflow-hidden relative">
                        <motion.div
                          className="absolute left-0 top-0 h-full bg-gradient-to-r from-red-500 via-yellow-500 to-green-500"
                          style={{ width: '100%' }}
                        />
                        <motion.div
                          className="absolute top-0 h-full w-1 bg-white shadow-lg"
                          animate={{ left: `${(activations[layerSizes.length - 1]?.[0] || 0.5) * 100}%` }}
                          transition={{ type: 'spring', stiffness: 200 }}
                        />
                      </div>
                    </div>

                    <div className="text-center">
                      <span className="text-gray-400 text-sm">Prediction: </span>
                      <span className={`font-bold ${(activations[layerSizes.length - 1]?.[0] || 0.5) > 0.5 ? 'text-green-400' : 'text-red-400'}`}>
                        {(activations[layerSizes.length - 1]?.[0] || 0.5) > 0.5 ? 'TRUE (1)' : 'FALSE (0)'}
                      </span>
                    </div>
                  </div>
                </div>

                {/* Softmax output */}
                <div className="bg-gray-800 rounded-lg p-4">
                  <h4 className="text-pink-400 font-medium mb-3">Softmax (Multi-Class)</h4>
                  <div className="space-y-2">
                    {outputLabels.slice(0, 4).map((label, i) => {
                      const prob = activations[layerSizes.length - 1]?.[i] || 1 / outputLabels.length;
                      const isMax = prob === Math.max(...(activations[layerSizes.length - 1] || []));
                      return (
                        <motion.div
                          key={i}
                          className="flex items-center gap-2"
                          initial={{ opacity: 0, x: -20 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ delay: i * 0.1 }}
                        >
                          <span className={`text-sm w-20 truncate ${isMax ? 'text-pink-400 font-bold' : 'text-gray-400'}`}>
                            {label}
                          </span>
                          <div className="flex-1 h-6 bg-gray-700 rounded overflow-hidden">
                            <motion.div
                              className={`h-full ${isMax ? 'bg-pink-500' : 'bg-gray-600'}`}
                              initial={{ width: 0 }}
                              animate={{ width: `${prob * 100}%` }}
                              transition={{ duration: 0.5, delay: i * 0.1 }}
                            />
                          </div>
                          <span className={`text-xs font-mono w-14 text-right ${isMax ? 'text-pink-400' : 'text-gray-500'}`}>
                            {(prob * 100).toFixed(1)}%
                          </span>
                        </motion.div>
                      );
                    })}
                  </div>
                  <div className="mt-3 text-xs text-gray-500 text-center">
                    Probabilities sum to: {(activations[layerSizes.length - 1] || []).reduce((a, b) => a + b, 0).toFixed(3)}
                  </div>
                </div>
              </div>

              <div className="bg-gray-800 rounded-lg p-4">
                <h4 className="text-gray-300 font-medium mb-2">When to Use Which?</h4>
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <span className="text-blue-400 font-medium">Sigmoid:</span>
                    <ul className="text-gray-400 mt-1 space-y-1">
                      <li>• XOR problem (yes/no)</li>
                      <li>• Anomaly detection</li>
                      <li>• Binary sensors</li>
                    </ul>
                  </div>
                  <div>
                    <span className="text-pink-400 font-medium">Softmax:</span>
                    <ul className="text-gray-400 mt-1 space-y-1">
                      <li>• Gesture recognition</li>
                      <li>• Shape classification</li>
                      <li>• Multiple categories</li>
                    </ul>
                  </div>
                </div>
              </div>
            </motion.div>
          )}

          {/* BACKPROPAGATION VISUALIZATION */}
          {mode === 'backprop' && (
            <motion.div
              key="backprop"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              className="space-y-4"
            >
              <div className="bg-gray-800 rounded-lg p-4">
                <h3 className="text-pink-400 font-semibold mb-2">How Backpropagation Works</h3>
                <p className="text-gray-300 text-sm">
                  The network <span className="text-pink-400 font-medium">learns from mistakes</span>.
                  Error flows backward, adjusting weights to reduce future errors.
                </p>
              </div>

              <div className="flex flex-col items-center gap-4">
                {/* Step 1: Calculate Error */}
                <motion.div
                  className="w-full bg-gray-800 rounded-lg p-4"
                  variants={itemVariants}
                >
                  <div className="flex items-center gap-4 flex-wrap justify-center">
                    <div className="text-center">
                      <div className="text-gray-500 text-xs mb-1">Prediction</div>
                      <div className="w-14 h-10 rounded bg-blue-600 flex items-center justify-center font-mono text-white">
                        {(activations[layerSizes.length - 1]?.[0] || 0.5).toFixed(2)}
                      </div>
                    </div>
                    <span className="text-2xl text-gray-500">-</span>
                    <div className="text-center">
                      <div className="text-gray-500 text-xs mb-1">Target</div>
                      <div className="w-14 h-10 rounded bg-green-600 flex items-center justify-center font-mono text-white">
                        1.00
                      </div>
                    </div>
                    <span className="text-2xl text-gray-500">=</span>
                    <div className="text-center">
                      <div className="text-gray-500 text-xs mb-1">Error</div>
                      <motion.div
                        className="w-14 h-10 rounded bg-red-600 flex items-center justify-center font-mono text-white"
                        animate={{ scale: [1, 1.1, 1] }}
                        transition={{ duration: 1, repeat: Infinity }}
                      >
                        {(1 - (activations[layerSizes.length - 1]?.[0] || 0.5)).toFixed(2)}
                      </motion.div>
                    </div>
                  </div>
                </motion.div>

                {/* Step 2: Gradient Flow */}
                <motion.div
                  className="flex items-center gap-2 text-sm"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 0.3 }}
                >
                  <span className="text-gray-400">Error flows backward:</span>
                  <motion.span
                    className="text-pink-400"
                    animate={{ x: [0, -10, 0] }}
                    transition={{ duration: 1, repeat: Infinity }}
                  >
                    ← ← ←
                  </motion.span>
                </motion.div>

                {/* Gradient visualization */}
                <div className="flex items-center gap-4 overflow-x-auto pb-2">
                  {layerSizes.slice().reverse().map((_, i) => {
                    const layerIdx = layerSizes.length - 1 - i;
                    const gradient = Math.pow(0.7, i);
                    return (
                      <motion.div
                        key={i}
                        className="flex flex-col items-center gap-1"
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: i * 0.2 }}
                      >
                        <div
                          className="w-16 h-16 rounded-lg flex items-center justify-center border-2"
                          style={{
                            borderColor: `rgba(236, 72, 153, ${gradient})`,
                            backgroundColor: `rgba(236, 72, 153, ${gradient * 0.2})`,
                          }}
                        >
                          <span className="text-pink-300 text-xs font-mono">
                            ∇{(gradient * 0.5).toFixed(2)}
                          </span>
                        </div>
                        <span className="text-xs text-gray-500">
                          {layerIdx === 0 ? 'Input' : layerIdx === layerSizes.length - 1 ? 'Output' : `H${layerIdx}`}
                        </span>
                      </motion.div>
                    );
                  })}
                </div>

                {/* Step 3: Weight Update */}
                <motion.div
                  className="w-full bg-gray-800 rounded-lg p-4"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 0.5 }}
                >
                  <h4 className="text-gray-300 font-medium mb-2 text-center">Weight Update Rule</h4>
                  <div className="bg-gray-900 rounded p-3 font-mono text-sm text-center">
                    <span className="text-cyan-400">w</span>
                    <span className="text-gray-400"> = </span>
                    <span className="text-cyan-400">w</span>
                    <span className="text-gray-400"> - </span>
                    <span className="text-yellow-400">η</span>
                    <span className="text-gray-400"> × </span>
                    <span className="text-pink-400">∂L/∂w</span>
                  </div>
                  <div className="flex justify-center gap-4 mt-3 text-xs text-gray-500">
                    <span><span className="text-cyan-400">w</span> = weight</span>
                    <span><span className="text-yellow-400">η</span> = learning rate</span>
                    <span><span className="text-pink-400">∂L/∂w</span> = gradient</span>
                  </div>
                </motion.div>
              </div>

              <div className="bg-gray-800 rounded-lg p-4">
                <h4 className="text-gray-300 font-medium mb-2">Key Concepts</h4>
                <ul className="text-sm text-gray-400 space-y-1">
                  <li>• <span className="text-pink-400">Chain Rule</span>: Gradients multiply through layers (can vanish!)</li>
                  <li>• <span className="text-yellow-400">Learning Rate</span>: Too high = unstable, too low = slow</li>
                  <li>• <span className="text-cyan-400">Gradient Descent</span>: Small steps toward minimum error</li>
                  <li>• <span className="text-green-400">Epochs</span>: Full passes through training data</li>
                </ul>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Controls */}
        <div className="mt-6 flex items-center justify-center gap-4 flex-wrap">
          <button
            onClick={() => setIsPlaying(!isPlaying)}
            className={`px-5 py-2 rounded-lg font-medium text-sm ${
              isPlaying
                ? 'bg-red-600 hover:bg-red-700 text-white'
                : 'bg-green-600 hover:bg-green-700 text-white'
            }`}
          >
            {isPlaying ? '⏹ Stop' : '▶ Play'}
          </button>

          <button
            onClick={() => {
              if (mode === 'weighted_sum' || mode === 'activation') {
                setCurrentNeuron(prev => {
                  const maxNeurons = layerSizes[currentLayer + 1] || layerSizes[1];
                  if (prev + 1 >= maxNeurons) {
                    setCurrentLayer(l => (l + 1) % (layerSizes.length - 1));
                    return 0;
                  }
                  return prev + 1;
                });
              } else if (mode === 'forward') {
                setForwardStep(prev => (prev + 1) % (layerSizes.length + 1));
              }
            }}
            disabled={isPlaying || mode === 'input' || mode === 'output' || mode === 'backprop'}
            className="px-5 py-2 rounded-lg font-medium text-sm bg-gray-700 hover:bg-gray-600 text-white disabled:opacity-50"
          >
            Step →
          </button>

          <div className="flex items-center gap-2">
            <span className="text-gray-400 text-sm">Speed:</span>
            <input
              type="range"
              min={100}
              max={1000}
              step={100}
              value={1100 - speed}
              onChange={(e) => setSpeed(1100 - parseInt(e.target.value))}
              className="w-20 accent-cyan-500"
            />
          </div>

          <button
            onClick={() => {
              setCurrentNeuron(0);
              setCurrentLayer(0);
              setForwardStep(0);
            }}
            className="px-4 py-2 rounded-lg font-medium text-sm bg-gray-700 hover:bg-gray-600 text-white"
          >
            Reset
          </button>
        </div>
      </motion.div>
    </motion.div>
  );

  return createPortal(modalContent, document.body);
}
