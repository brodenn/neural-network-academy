import { useState, useEffect, useMemo, useCallback } from 'react';
import { createPortal } from 'react-dom';
import { motion, AnimatePresence } from 'framer-motion';

interface TrainingEducationalVizProps {
  layerSizes: number[];
  onClose: () => void;
}

type TrainingStep = 'overview' | 'forward' | 'loss' | 'backward' | 'update' | 'loop';

// Seeded random for consistent values
function seededRandom(seed: number): number {
  const x = Math.sin(seed * 9999) * 10000;
  return x - Math.floor(x);
}

export function TrainingEducationalViz({
  layerSizes,
  onClose,
}: TrainingEducationalVizProps) {
  const [step, setStep] = useState<TrainingStep>('overview');
  const [isPlaying, setIsPlaying] = useState(true);
  const [speed, setSpeed] = useState(800);
  const [animationPhase, setAnimationPhase] = useState(0);
  const [epoch, setEpoch] = useState(1);
  const [currentLoss, setCurrentLoss] = useState(0.8);

  // Generate sample data
  const sampleInput = useMemo(() => [0.8, 0.3, 0.6], []);
  const sampleTarget = useMemo(() => [1.0], []);

  // Simulated weights that change during training
  const [weights, setWeights] = useState(() => {
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
  });

  // Calculate activations for forward pass visualization
  const activations = useMemo(() => {
    const acts: number[][] = [sampleInput];
    let current = sampleInput;

    for (let l = 0; l < weights.length; l++) {
      const next: number[] = [];
      for (let j = 0; j < layerSizes[l + 1]; j++) {
        let sum = 0;
        for (let i = 0; i < current.length; i++) {
          sum += current[i] * (weights[l]?.[i]?.[j] || 0);
        }
        // ReLU for hidden, sigmoid for output
        if (l === weights.length - 1) {
          next.push(1 / (1 + Math.exp(-sum)));
        } else {
          next.push(Math.max(0, sum));
        }
      }
      acts.push(next);
      current = next;
    }
    return acts;
  }, [sampleInput, weights, layerSizes]);

  const prediction = activations[activations.length - 1][0];
  const error = sampleTarget[0] - prediction;
  const loss = error * error;

  // Animation for training loop
  useEffect(() => {
    if (!isPlaying) return;

    const interval = setInterval(() => {
      if (step === 'forward') {
        setAnimationPhase(prev => (prev + 1) % (layerSizes.length + 1));
      } else if (step === 'backward') {
        setAnimationPhase(prev => (prev + 1) % (layerSizes.length + 1));
      } else if (step === 'update') {
        setAnimationPhase(prev => (prev + 1) % 4);
      } else if (step === 'loop') {
        setAnimationPhase(prev => {
          const next = prev + 1;
          if (next >= 6) {
            setEpoch(e => e + 1);
            setCurrentLoss(l => Math.max(0.01, l * 0.85));
            // Simulate weight update
            setWeights(w => w.map(layer =>
              layer.map(neuron =>
                neuron.map(weight => weight + (seededRandom(weight * 1000) - 0.5) * 0.1)
              )
            ));
            return 0;
          }
          return next;
        });
      }
    }, speed);

    return () => clearInterval(interval);
  }, [isPlaying, step, layerSizes.length, speed]);

  const handleStepChange = (newStep: TrainingStep) => {
    setStep(newStep);
    setAnimationPhase(0);
    setIsPlaying(false);
  };

  const resetTraining = useCallback(() => {
    setEpoch(1);
    setCurrentLoss(0.8);
    setAnimationPhase(0);
    setWeights(() => {
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
    });
  }, [layerSizes]);

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
            How Neural Networks Learn
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
          {(['overview', 'forward', 'loss', 'backward', 'update', 'loop'] as TrainingStep[]).map((s, i) => {
            const labels = ['Overview', 'Forward Pass', 'Loss', 'Backprop', 'Update', 'Training Loop'];
            return (
              <div key={s} className="flex items-center">
                <div
                  className={`px-2 py-1 rounded cursor-pointer transition-all whitespace-nowrap ${
                    step === s
                      ? 'bg-white text-gray-900 font-bold scale-110'
                      : 'bg-gray-700 text-gray-400 hover:bg-gray-600'
                  }`}
                  onClick={() => handleStepChange(s)}
                >
                  {i + 1}. {labels[i]}
                </div>
                {i < 5 && <span className="text-gray-600 mx-1">→</span>}
              </div>
            );
          })}
        </div>

        <AnimatePresence mode="wait">
          {/* OVERVIEW */}
          {step === 'overview' && (
            <motion.div
              key="overview"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              className="space-y-4"
            >
              <div className="bg-gray-800 rounded-lg p-4">
                <h3 className="text-purple-400 font-semibold mb-2">The Training Process</h3>
                <p className="text-gray-300 text-sm">
                  Neural networks learn by <span className="text-purple-400 font-medium">adjusting weights</span> to minimize prediction errors.
                  This happens through a cycle of forward pass → loss calculation → backpropagation → weight update.
                </p>
              </div>

              {/* Visual cycle */}
              <div className="flex justify-center py-6">
                <div className="relative w-80 h-80">
                  {/* Cycle arrows */}
                  <svg className="w-full h-full" viewBox="0 0 200 200">
                    <defs>
                      <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                        <polygon points="0 0, 10 3.5, 0 7" fill="#9ca3af" />
                      </marker>
                    </defs>
                    <motion.path
                      d="M 100 30 A 70 70 0 0 1 170 100"
                      fill="none"
                      stroke="#22d3ee"
                      strokeWidth="3"
                      markerEnd="url(#arrowhead)"
                      initial={{ pathLength: 0 }}
                      animate={{ pathLength: 1 }}
                      transition={{ duration: 0.5, delay: 0 }}
                    />
                    <motion.path
                      d="M 170 100 A 70 70 0 0 1 100 170"
                      fill="none"
                      stroke="#ef4444"
                      strokeWidth="3"
                      markerEnd="url(#arrowhead)"
                      initial={{ pathLength: 0 }}
                      animate={{ pathLength: 1 }}
                      transition={{ duration: 0.5, delay: 0.3 }}
                    />
                    <motion.path
                      d="M 100 170 A 70 70 0 0 1 30 100"
                      fill="none"
                      stroke="#ec4899"
                      strokeWidth="3"
                      markerEnd="url(#arrowhead)"
                      initial={{ pathLength: 0 }}
                      animate={{ pathLength: 1 }}
                      transition={{ duration: 0.5, delay: 0.6 }}
                    />
                    <motion.path
                      d="M 30 100 A 70 70 0 0 1 100 30"
                      fill="none"
                      stroke="#22c55e"
                      strokeWidth="3"
                      markerEnd="url(#arrowhead)"
                      initial={{ pathLength: 0 }}
                      animate={{ pathLength: 1 }}
                      transition={{ duration: 0.5, delay: 0.9 }}
                    />
                  </svg>

                  {/* Labels */}
                  <motion.div
                    className="absolute top-2 left-1/2 -translate-x-1/2 px-3 py-2 bg-cyan-600 rounded-lg text-white text-sm font-medium"
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.2 }}
                  >
                    1. Forward Pass
                  </motion.div>
                  <motion.div
                    className="absolute right-0 top-1/2 -translate-y-1/2 px-3 py-2 bg-red-600 rounded-lg text-white text-sm font-medium"
                    initial={{ opacity: 0, x: 10 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.5 }}
                  >
                    2. Loss
                  </motion.div>
                  <motion.div
                    className="absolute bottom-2 left-1/2 -translate-x-1/2 px-3 py-2 bg-pink-600 rounded-lg text-white text-sm font-medium"
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.8 }}
                  >
                    3. Backprop
                  </motion.div>
                  <motion.div
                    className="absolute left-0 top-1/2 -translate-y-1/2 px-3 py-2 bg-green-600 rounded-lg text-white text-sm font-medium"
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 1.1 }}
                  >
                    4. Update
                  </motion.div>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div className="bg-gray-800 rounded-lg p-3">
                  <div className="text-cyan-400 font-medium text-sm mb-1">➡️ Forward Pass</div>
                  <p className="text-xs text-gray-400">Data flows through network, producing a prediction</p>
                </div>
                <div className="bg-gray-800 rounded-lg p-3">
                  <div className="text-red-400 font-medium text-sm mb-1">📉 Loss Calculation</div>
                  <p className="text-xs text-gray-400">Measure how wrong the prediction was</p>
                </div>
                <div className="bg-gray-800 rounded-lg p-3">
                  <div className="text-pink-400 font-medium text-sm mb-1">⬅️ Backpropagation</div>
                  <p className="text-xs text-gray-400">Calculate how each weight contributed to error</p>
                </div>
                <div className="bg-gray-800 rounded-lg p-3">
                  <div className="text-green-400 font-medium text-sm mb-1">🔧 Weight Update</div>
                  <p className="text-xs text-gray-400">Adjust weights to reduce future errors</p>
                </div>
              </div>
            </motion.div>
          )}

          {/* FORWARD PASS */}
          {step === 'forward' && (
            <motion.div
              key="forward"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              className="space-y-4"
            >
              <div className="bg-gray-800 rounded-lg p-4">
                <h3 className="text-cyan-400 font-semibold mb-2">Forward Pass</h3>
                <p className="text-gray-300 text-sm">
                  Input data flows <span className="text-cyan-400 font-medium">left to right</span> through the network.
                  Each neuron computes: <code className="bg-gray-700 px-1 rounded">activation(Σ(weights × inputs) + bias)</code>
                </p>
              </div>

              {/* Network visualization */}
              <div className="overflow-x-auto py-4">
                <svg width="600" height="250" className="mx-auto">
                  {/* Connections */}
                  {layerSizes.slice(0, -1).map((size, layerIdx) => {
                    const nextSize = layerSizes[layerIdx + 1];
                    const x1 = 80 + layerIdx * 140;
                    const x2 = 80 + (layerIdx + 1) * 140;

                    return Array.from({ length: Math.min(size, 4) }).map((_, i) => {
                      const y1 = 50 + i * 50;
                      return Array.from({ length: Math.min(nextSize, 4) }).map((_, j) => {
                        const y2 = 50 + j * 50;
                        const isActive = animationPhase > layerIdx;

                        return (
                          <motion.line
                            key={`${layerIdx}-${i}-${j}`}
                            x1={x1 + 15}
                            y1={y1}
                            x2={x2 - 15}
                            y2={y2}
                            stroke={isActive ? '#22d3ee' : '#374151'}
                            strokeWidth={isActive ? 2 : 1}
                            initial={{ opacity: 0.2 }}
                            animate={{
                              opacity: isActive ? 0.8 : 0.2,
                              stroke: isActive ? '#22d3ee' : '#374151'
                            }}
                            transition={{ duration: 0.3 }}
                          />
                        );
                      });
                    });
                  })}

                  {/* Signal flow animation */}
                  {animationPhase > 0 && animationPhase <= layerSizes.length && (
                    <motion.circle
                      cx={80 + (animationPhase - 1) * 140 + 70}
                      cy={100}
                      r={8}
                      fill="#22d3ee"
                      initial={{ opacity: 0, x: -50 }}
                      animate={{ opacity: [0, 1, 0], x: [0, 70, 140] }}
                      transition={{ duration: 0.8 }}
                    />
                  )}

                  {/* Neurons */}
                  {layerSizes.map((size, layerIdx) => {
                    const x = 80 + layerIdx * 140;
                    const isActive = animationPhase >= layerIdx;
                    const displaySize = Math.min(size, 4);

                    return (
                      <g key={layerIdx}>
                        <text x={x} y={230} textAnchor="middle" fill="#9ca3af" fontSize="11">
                          {layerIdx === 0 ? 'Input' : layerIdx === layerSizes.length - 1 ? 'Output' : `Hidden ${layerIdx}`}
                        </text>

                        {Array.from({ length: displaySize }).map((_, i) => {
                          const y = 50 + i * 50;
                          const activation = activations[layerIdx]?.[i] || 0;

                          return (
                            <g key={i}>
                              <motion.circle
                                cx={x}
                                cy={y}
                                r={15}
                                fill={isActive ? (activation > 0.5 ? '#22d3ee' : '#0891b2') : '#374151'}
                                stroke={isActive ? '#22d3ee' : '#6b7280'}
                                strokeWidth={2}
                                initial={{ scale: 0.8 }}
                                animate={{ scale: isActive ? 1 : 0.8 }}
                                transition={{ duration: 0.3, delay: layerIdx * 0.1 }}
                              />
                              <text
                                x={x}
                                y={y + 4}
                                textAnchor="middle"
                                fill="white"
                                fontSize="9"
                                fontFamily="monospace"
                              >
                                {isActive ? activation.toFixed(2) : '?'}
                              </text>
                            </g>
                          );
                        })}
                      </g>
                    );
                  })}
                </svg>
              </div>

              <div className="bg-gray-800 rounded-lg p-4">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-gray-400">Input:</span>
                  <span className="text-cyan-400 font-mono">[{sampleInput.map(v => v.toFixed(2)).join(', ')}]</span>
                </div>
                <div className="flex items-center justify-between text-sm mt-2">
                  <span className="text-gray-400">Prediction:</span>
                  <motion.span
                    className="text-cyan-400 font-mono font-bold"
                    key={prediction}
                    initial={{ scale: 1.2 }}
                    animate={{ scale: 1 }}
                  >
                    {prediction.toFixed(4)}
                  </motion.span>
                </div>
              </div>
            </motion.div>
          )}

          {/* LOSS CALCULATION */}
          {step === 'loss' && (
            <motion.div
              key="loss"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              className="space-y-4"
            >
              <div className="bg-gray-800 rounded-lg p-4">
                <h3 className="text-red-400 font-semibold mb-2">Loss Calculation</h3>
                <p className="text-gray-300 text-sm">
                  The <span className="text-red-400 font-medium">loss function</span> measures how wrong the prediction was.
                  Common loss: <code className="bg-gray-700 px-1 rounded">MSE = (target - prediction)²</code>
                </p>
              </div>

              <div className="flex items-center justify-center gap-8 flex-wrap py-6">
                {/* Target */}
                <motion.div
                  className="text-center"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0 }}
                >
                  <div className="text-gray-500 text-sm mb-2">Target</div>
                  <div className="w-20 h-20 rounded-xl bg-green-600 flex items-center justify-center text-2xl font-bold text-white">
                    {sampleTarget[0].toFixed(2)}
                  </div>
                </motion.div>

                <motion.div
                  className="text-4xl text-gray-500"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 0.2 }}
                >
                  -
                </motion.div>

                {/* Prediction */}
                <motion.div
                  className="text-center"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.3 }}
                >
                  <div className="text-gray-500 text-sm mb-2">Prediction</div>
                  <div className="w-20 h-20 rounded-xl bg-cyan-600 flex items-center justify-center text-2xl font-bold text-white">
                    {prediction.toFixed(2)}
                  </div>
                </motion.div>

                <motion.div
                  className="text-4xl text-gray-500"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 0.5 }}
                >
                  =
                </motion.div>

                {/* Error */}
                <motion.div
                  className="text-center"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.6 }}
                >
                  <div className="text-gray-500 text-sm mb-2">Error</div>
                  <motion.div
                    className="w-20 h-20 rounded-xl bg-yellow-600 flex items-center justify-center text-2xl font-bold text-white"
                    animate={{ scale: [1, 1.1, 1] }}
                    transition={{ duration: 1, repeat: Infinity }}
                  >
                    {error.toFixed(2)}
                  </motion.div>
                </motion.div>
              </div>

              {/* Loss calculation */}
              <motion.div
                className="bg-gray-800 rounded-lg p-4 text-center"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.8 }}
              >
                <div className="text-sm text-gray-400 mb-2">Mean Squared Error (MSE)</div>
                <div className="text-lg font-mono">
                  <span className="text-yellow-400">{error.toFixed(3)}</span>
                  <span className="text-gray-400">² = </span>
                  <motion.span
                    className="text-red-400 font-bold text-2xl"
                    animate={{ scale: [1, 1.1, 1] }}
                    transition={{ duration: 0.5, repeat: Infinity }}
                  >
                    {loss.toFixed(4)}
                  </motion.span>
                </div>
              </motion.div>

              {/* Loss visualization bar */}
              <div className="bg-gray-800 rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-gray-400 text-sm">Loss</span>
                  <span className="text-red-400 font-mono">{loss.toFixed(4)}</span>
                </div>
                <div className="h-6 bg-gray-700 rounded-full overflow-hidden">
                  <motion.div
                    className="h-full bg-gradient-to-r from-green-500 via-yellow-500 to-red-500"
                    initial={{ width: 0 }}
                    animate={{ width: `${Math.min(loss * 100, 100)}%` }}
                    transition={{ duration: 0.5 }}
                  />
                </div>
                <div className="flex justify-between text-xs text-gray-500 mt-1">
                  <span>0 (Perfect)</span>
                  <span>0.5</span>
                  <span>1.0 (Bad)</span>
                </div>
              </div>
            </motion.div>
          )}

          {/* BACKPROPAGATION */}
          {step === 'backward' && (
            <motion.div
              key="backward"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              className="space-y-4"
            >
              <div className="bg-gray-800 rounded-lg p-4">
                <h3 className="text-pink-400 font-semibold mb-2">Backpropagation</h3>
                <p className="text-gray-300 text-sm">
                  Error flows <span className="text-pink-400 font-medium">right to left</span>, calculating how much each weight
                  contributed to the error. Uses the <span className="text-pink-400">chain rule</span> of calculus.
                </p>
              </div>

              {/* Gradient flow visualization */}
              <div className="overflow-x-auto py-4">
                <svg width="600" height="250" className="mx-auto">
                  {/* Gradient flow arrows */}
                  {layerSizes.slice(0, -1).map((_, layerIdx) => {
                    const reverseIdx = layerSizes.length - 2 - layerIdx;
                    const x1 = 80 + (reverseIdx + 1) * 140;
                    const x2 = 80 + reverseIdx * 140;
                    const isActive = animationPhase > layerIdx;

                    return (
                      <motion.line
                        key={layerIdx}
                        x1={x1}
                        y1={100}
                        x2={x2 + 30}
                        y2={100}
                        stroke={isActive ? '#ec4899' : '#374151'}
                        strokeWidth={isActive ? 4 : 2}
                        strokeDasharray="10,5"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: isActive ? 1 : 0.3 }}
                        transition={{ duration: 0.3 }}
                      />
                    );
                  })}

                  {/* Gradient magnitude indicators */}
                  {layerSizes.map((_, layerIdx) => {
                    const reverseIdx = layerSizes.length - 1 - layerIdx;
                    const x = 80 + reverseIdx * 140;
                    const isActive = animationPhase >= layerIdx;
                    const gradient = Math.pow(0.7, layerIdx);

                    return (
                      <g key={layerIdx}>
                        <motion.circle
                          cx={x}
                          cy={100}
                          r={25}
                          fill={`rgba(236, 72, 153, ${isActive ? gradient : 0.1})`}
                          stroke={isActive ? '#ec4899' : '#6b7280'}
                          strokeWidth={2}
                          initial={{ scale: 0 }}
                          animate={{ scale: isActive ? 1 : 0.5 }}
                          transition={{ duration: 0.3, delay: layerIdx * 0.2 }}
                        />
                        <text
                          x={x}
                          y={105}
                          textAnchor="middle"
                          fill="white"
                          fontSize="10"
                          fontFamily="monospace"
                        >
                          {isActive ? `∇${gradient.toFixed(2)}` : '?'}
                        </text>
                        <text x={x} y={150} textAnchor="middle" fill="#9ca3af" fontSize="10">
                          {reverseIdx === layerSizes.length - 1 ? 'Output' : reverseIdx === 0 ? 'Input' : `H${reverseIdx}`}
                        </text>
                      </g>
                    );
                  })}
                </svg>
              </div>

              <div className="bg-gray-800 rounded-lg p-4">
                <h4 className="text-gray-300 font-medium mb-2">Chain Rule in Action</h4>
                <div className="bg-gray-900 rounded p-3 font-mono text-sm overflow-x-auto">
                  <div className="text-pink-400">∂Loss/∂w = ∂Loss/∂output × ∂output/∂w</div>
                  <div className="text-gray-500 mt-2 text-xs">
                    "How does changing this weight affect the loss?"
                  </div>
                </div>
              </div>

              <div className="bg-gray-800 rounded-lg p-4">
                <h4 className="text-gray-300 font-medium mb-2">Why Gradients Get Smaller</h4>
                <p className="text-sm text-gray-400">
                  Each layer multiplies gradients together. Small values (0.7 × 0.7 × ...) shrink quickly.
                  This is the <span className="text-pink-400">"vanishing gradient"</span> problem - why deep networks are hard to train.
                </p>
              </div>
            </motion.div>
          )}

          {/* WEIGHT UPDATE */}
          {step === 'update' && (
            <motion.div
              key="update"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              className="space-y-4"
            >
              <div className="bg-gray-800 rounded-lg p-4">
                <h3 className="text-green-400 font-semibold mb-2">Weight Update</h3>
                <p className="text-gray-300 text-sm">
                  Adjust each weight in the <span className="text-green-400 font-medium">opposite direction</span> of its gradient.
                  Learning rate controls step size.
                </p>
              </div>

              {/* Update formula */}
              <div className="bg-gray-800 rounded-lg p-4 text-center">
                <div className="text-lg font-mono">
                  <span className="text-cyan-400">w</span>
                  <span className="text-gray-400"><sub>new</sub> = </span>
                  <span className="text-cyan-400">w</span>
                  <span className="text-gray-400"><sub>old</sub> - </span>
                  <span className="text-yellow-400">η</span>
                  <span className="text-gray-400"> × </span>
                  <span className="text-pink-400">∇w</span>
                </div>
                <div className="flex justify-center gap-6 mt-3 text-xs text-gray-500">
                  <span><span className="text-yellow-400">η</span> = learning rate (0.01-0.5)</span>
                  <span><span className="text-pink-400">∇w</span> = gradient</span>
                </div>
              </div>

              {/* Visual weight update */}
              <div className="flex items-center justify-center gap-4 flex-wrap py-4">
                <motion.div
                  className="text-center"
                  animate={{ scale: animationPhase === 0 ? 1.1 : 1 }}
                >
                  <div className="text-gray-500 text-xs mb-1">Old Weight</div>
                  <div className="w-16 h-16 rounded-lg bg-gray-700 flex items-center justify-center font-mono text-cyan-400">
                    0.50
                  </div>
                </motion.div>

                <span className="text-gray-500 text-2xl">-</span>

                <motion.div
                  className="text-center"
                  animate={{ scale: animationPhase === 1 ? 1.1 : 1 }}
                >
                  <div className="text-gray-500 text-xs mb-1">LR</div>
                  <div className="w-16 h-16 rounded-lg bg-yellow-900/50 flex items-center justify-center font-mono text-yellow-400">
                    0.1
                  </div>
                </motion.div>

                <span className="text-gray-500 text-2xl">×</span>

                <motion.div
                  className="text-center"
                  animate={{ scale: animationPhase === 2 ? 1.1 : 1 }}
                >
                  <div className="text-gray-500 text-xs mb-1">Gradient</div>
                  <div className="w-16 h-16 rounded-lg bg-pink-900/50 flex items-center justify-center font-mono text-pink-400">
                    0.25
                  </div>
                </motion.div>

                <span className="text-gray-500 text-2xl">=</span>

                <motion.div
                  className="text-center"
                  animate={{ scale: animationPhase === 3 ? 1.1 : 1 }}
                >
                  <div className="text-gray-500 text-xs mb-1">New Weight</div>
                  <motion.div
                    className="w-16 h-16 rounded-lg bg-green-600 flex items-center justify-center font-mono text-white font-bold"
                    animate={{ scale: animationPhase === 3 ? [1, 1.2, 1] : 1 }}
                    transition={{ duration: 0.5 }}
                  >
                    0.475
                  </motion.div>
                </motion.div>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div className="bg-gray-800 rounded-lg p-3">
                  <div className="text-yellow-400 font-medium text-sm mb-1">Learning Rate Too High</div>
                  <p className="text-xs text-gray-400">Overshoots minimum, unstable training, loss oscillates</p>
                </div>
                <div className="bg-gray-800 rounded-lg p-3">
                  <div className="text-yellow-400 font-medium text-sm mb-1">Learning Rate Too Low</div>
                  <p className="text-xs text-gray-400">Very slow progress, may get stuck in local minima</p>
                </div>
              </div>
            </motion.div>
          )}

          {/* FULL TRAINING LOOP */}
          {step === 'loop' && (
            <motion.div
              key="loop"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              className="space-y-4"
            >
              <div className="bg-gray-800 rounded-lg p-4">
                <h3 className="text-yellow-400 font-semibold mb-2">Full Training Loop</h3>
                <p className="text-gray-300 text-sm">
                  Repeat the cycle many times (<span className="text-yellow-400 font-medium">epochs</span>).
                  Watch the loss decrease as the network learns!
                </p>
              </div>

              {/* Training stats */}
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-gray-800 rounded-lg p-4 text-center">
                  <div className="text-gray-500 text-sm">Epoch</div>
                  <motion.div
                    className="text-4xl font-bold text-yellow-400"
                    key={epoch}
                    initial={{ scale: 1.5 }}
                    animate={{ scale: 1 }}
                  >
                    {epoch}
                  </motion.div>
                </div>
                <div className="bg-gray-800 rounded-lg p-4 text-center">
                  <div className="text-gray-500 text-sm">Loss</div>
                  <motion.div
                    className="text-4xl font-bold text-red-400"
                    key={currentLoss.toFixed(4)}
                    initial={{ scale: 1.5 }}
                    animate={{ scale: 1 }}
                  >
                    {currentLoss.toFixed(4)}
                  </motion.div>
                </div>
              </div>

              {/* Current phase indicator */}
              <div className="flex justify-center gap-2">
                {['Forward', 'Loss', 'Backprop', 'Update', 'Next'].map((phase, i) => (
                  <motion.div
                    key={phase}
                    className={`px-3 py-2 rounded-lg text-sm font-medium ${
                      animationPhase === i
                        ? 'bg-yellow-600 text-white'
                        : animationPhase > i
                        ? 'bg-green-800 text-green-300'
                        : 'bg-gray-700 text-gray-500'
                    }`}
                    animate={{ scale: animationPhase === i ? 1.1 : 1 }}
                  >
                    {animationPhase === i ? '▶ ' : animationPhase > i ? '✓ ' : ''}{phase}
                  </motion.div>
                ))}
              </div>

              {/* Loss curve visualization */}
              <div className="bg-gray-800 rounded-lg p-4">
                <div className="text-gray-400 text-sm mb-2">Loss Over Time</div>
                <div className="h-32 flex items-end gap-1">
                  {Array.from({ length: Math.min(epoch, 20) }).map((_, i) => {
                    const height = Math.pow(0.85, i) * 100;
                    return (
                      <motion.div
                        key={i}
                        className="flex-1 bg-gradient-to-t from-red-600 to-yellow-500 rounded-t"
                        initial={{ height: 0 }}
                        animate={{ height: `${height}%` }}
                        transition={{ duration: 0.3 }}
                      />
                    );
                  })}
                </div>
                <div className="flex justify-between text-xs text-gray-500 mt-1">
                  <span>Epoch 1</span>
                  <span>Epoch {epoch}</span>
                </div>
              </div>

              <div className="bg-gray-800 rounded-lg p-4">
                <h4 className="text-gray-300 font-medium mb-2">Training Tips</h4>
                <ul className="text-sm text-gray-400 space-y-1">
                  <li>• <span className="text-green-400">Stop early</span> if validation loss starts increasing (overfitting)</li>
                  <li>• <span className="text-yellow-400">Adjust learning rate</span> if loss plateaus or oscillates</li>
                  <li>• <span className="text-cyan-400">More data</span> usually helps more than more epochs</li>
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
              if (step === 'forward' || step === 'backward') {
                setAnimationPhase(prev => (prev + 1) % (layerSizes.length + 1));
              } else if (step === 'update') {
                setAnimationPhase(prev => (prev + 1) % 4);
              } else if (step === 'loop') {
                setAnimationPhase(prev => {
                  const next = prev + 1;
                  if (next >= 6) {
                    setEpoch(e => e + 1);
                    setCurrentLoss(l => Math.max(0.01, l * 0.85));
                    return 0;
                  }
                  return next;
                });
              }
            }}
            disabled={isPlaying || step === 'overview' || step === 'loss'}
            className="px-5 py-2 rounded-lg font-medium text-sm bg-gray-700 hover:bg-gray-600 text-white disabled:opacity-50"
          >
            Step →
          </button>

          <div className="flex items-center gap-2">
            <span className="text-gray-400 text-sm">Speed:</span>
            <input
              type="range"
              min={200}
              max={1500}
              step={100}
              value={1700 - speed}
              onChange={(e) => setSpeed(1700 - parseInt(e.target.value))}
              className="w-20 accent-yellow-500"
            />
          </div>

          <button
            onClick={resetTraining}
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
