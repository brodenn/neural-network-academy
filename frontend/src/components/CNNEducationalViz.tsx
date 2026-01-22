import { useState, useEffect, useMemo } from 'react';
import { createPortal } from 'react-dom';
import { motion, AnimatePresence } from 'framer-motion';

interface CNNEducationalVizProps {
  inputGrid: number[][];
  onClose: () => void;
}

type Mode = 'padding' | 'conv' | 'relu' | 'maxpool' | 'flatten' | 'dense' | 'softmax';

// Example 3x3 kernel for edge detection
const EXAMPLE_KERNELS = {
  edge: [
    [-1, -1, -1],
    [-1, 8, -1],
    [-1, -1, -1],
  ],
  horizontal: [
    [-1, -1, -1],
    [2, 2, 2],
    [-1, -1, -1],
  ],
  vertical: [
    [-1, 2, -1],
    [-1, 2, -1],
    [-1, 2, -1],
  ],
};

type KernelType = keyof typeof EXAMPLE_KERNELS;

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
      staggerChildren: 0.05,
      delayChildren: 0.1,
    },
  },
};

const itemVariants = {
  hidden: { opacity: 0, scale: 0.8, y: 10 },
  show: { opacity: 1, scale: 1, y: 0 },
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
    yellow: isActive ? 'bg-yellow-600 text-white' : 'bg-gray-700 text-gray-300 hover:bg-gray-600',
    cyan: isActive ? 'bg-cyan-600 text-white' : 'bg-gray-700 text-gray-300 hover:bg-gray-600',
    orange: isActive ? 'bg-orange-600 text-white' : 'bg-gray-700 text-gray-300 hover:bg-gray-600',
    green: isActive ? 'bg-green-600 text-white' : 'bg-gray-700 text-gray-300 hover:bg-gray-600',
    purple: isActive ? 'bg-purple-600 text-white' : 'bg-gray-700 text-gray-300 hover:bg-gray-600',
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

export function CNNEducationalViz({ inputGrid, onClose }: CNNEducationalVizProps) {
  const [mode, setMode] = useState<Mode>('conv');
  const [kernelPos, setKernelPos] = useState({ row: 0, col: 0 });
  const [isPlaying, setIsPlaying] = useState(false);
  const [speed, setSpeed] = useState(500);
  const [selectedKernel, setSelectedKernel] = useState<KernelType>('edge');
  const [flattenStep, setFlattenStep] = useState(0);
  const [denseNeuronIdx, setDenseNeuronIdx] = useState(0);

  const gridSize = inputGrid.length || 8;
  const kernelSize = 3;
  const maxPos = gridSize - kernelSize;
  const poolSize = 2;
  const stride = 2;
  const featureMapSize = 6;
  const paddingSize = 1;

  // Create padded input grid
  const paddedGrid = useMemo(() => {
    const padded: number[][] = [];
    const paddedSize = gridSize + paddingSize * 2;
    for (let i = 0; i < paddedSize; i++) {
      const row: number[] = [];
      for (let j = 0; j < paddedSize; j++) {
        const origI = i - paddingSize;
        const origJ = j - paddingSize;
        if (origI >= 0 && origI < gridSize && origJ >= 0 && origJ < gridSize) {
          row.push(inputGrid[origI]?.[origJ] ?? 0);
        } else {
          row.push(0); // Zero padding
        }
      }
      padded.push(row);
    }
    return padded;
  }, [inputGrid, gridSize, paddingSize]);

  // Create simulated feature map (conv output before ReLU)
  const convOutputRaw = useMemo(() => {
    const map: number[][] = [];
    for (let i = 0; i < featureMapSize; i++) {
      const row: number[] = [];
      for (let j = 0; j < featureMapSize; j++) {
        const inputVal = inputGrid[i]?.[j];
        // Simulate conv output with some negative values
        const val = inputVal !== undefined && inputVal > 0
          ? inputVal * 1.5 - 0.5 + seededRandom(i * 10 + j) * 0.4
          : seededRandom(i * 100 + j * 7) * 0.8 - 0.4;
        row.push(val);
      }
      map.push(row);
    }
    return map;
  }, [inputGrid, featureMapSize]);

  // ReLU applied
  const convOutputRelu = useMemo(() => {
    return convOutputRaw.map(row => row.map(val => Math.max(0, val)));
  }, [convOutputRaw]);

  // Pooled output (3x3)
  const pooledOutput = useMemo(() => {
    const pooled: number[][] = [];
    for (let i = 0; i < featureMapSize / stride; i++) {
      const row: number[] = [];
      for (let j = 0; j < featureMapSize / stride; j++) {
        const vals = [
          convOutputRelu[i * stride]?.[j * stride] ?? 0,
          convOutputRelu[i * stride]?.[j * stride + 1] ?? 0,
          convOutputRelu[i * stride + 1]?.[j * stride] ?? 0,
          convOutputRelu[i * stride + 1]?.[j * stride + 1] ?? 0,
        ];
        row.push(Math.max(...vals));
      }
      pooled.push(row);
    }
    return pooled;
  }, [convOutputRelu, featureMapSize, stride]);

  // Flattened vector
  const flattenedVector = useMemo(() => {
    return pooledOutput.flat();
  }, [pooledOutput]);

  // Dense layer weights (simulated)
  const denseWeights = useMemo(() => {
    const numOutputs = 3; // 3 classes: circle, square, triangle
    const weights: number[][] = [];
    for (let i = 0; i < numOutputs; i++) {
      const neuronWeights: number[] = [];
      for (let j = 0; j < flattenedVector.length; j++) {
        neuronWeights.push(seededRandom(i * 100 + j * 17) * 2 - 1);
      }
      weights.push(neuronWeights);
    }
    return weights;
  }, [flattenedVector.length]);

  // Dense layer output (before softmax)
  const denseOutput = useMemo(() => {
    return denseWeights.map((neuronWeights, i) => {
      const sum = neuronWeights.reduce((acc, w, j) => acc + w * flattenedVector[j], 0);
      return sum + seededRandom(i * 999) * 0.5; // Add bias-like term
    });
  }, [denseWeights, flattenedVector]);

  // Softmax output
  const softmaxOutput = useMemo(() => {
    const maxVal = Math.max(...denseOutput);
    const exps = denseOutput.map(v => Math.exp(v - maxVal));
    const sumExps = exps.reduce((a, b) => a + b, 0);
    return exps.map(e => e / sumExps);
  }, [denseOutput]);

  // Get pool window values
  const getPoolWindowValues = () => {
    const values: number[] = [];
    for (let i = 0; i < poolSize; i++) {
      for (let j = 0; j < poolSize; j++) {
        values.push(convOutputRelu[kernelPos.row + i]?.[kernelPos.col + j] ?? 0);
      }
    }
    return values;
  };

  const poolValues = getPoolWindowValues();
  const maxPoolValue = Math.max(...poolValues);
  const maxPoolIndex = poolValues.indexOf(maxPoolValue);

  // Calculate convolution output for current position
  const calculateConvOutput = () => {
    const kernel = EXAMPLE_KERNELS[selectedKernel];
    let sum = 0;
    for (let ki = 0; ki < kernelSize; ki++) {
      for (let kj = 0; kj < kernelSize; kj++) {
        const inputVal = inputGrid[kernelPos.row + ki]?.[kernelPos.col + kj] || 0;
        sum += inputVal * kernel[ki][kj];
      }
    }
    return sum;
  };

  const convResult = calculateConvOutput();
  const kernel = EXAMPLE_KERNELS[selectedKernel];

  // Auto-play animation
  useEffect(() => {
    if (!isPlaying) return;

    const interval = setInterval(() => {
      if (mode === 'conv' || mode === 'padding') {
        setKernelPos((prev) => {
          let newCol = prev.col + 1;
          let newRow = prev.row;
          if (newCol > maxPos) {
            newCol = 0;
            newRow = prev.row + 1;
          }
          if (newRow > maxPos) {
            newRow = 0;
            newCol = 0;
          }
          return { row: newRow, col: newCol };
        });
      } else if (mode === 'maxpool' || mode === 'relu') {
        setKernelPos((prev) => {
          let newCol = prev.col + stride;
          let newRow = prev.row;
          if (newCol >= featureMapSize) {
            newCol = 0;
            newRow = prev.row + stride;
          }
          if (newRow >= featureMapSize) {
            newRow = 0;
            newCol = 0;
          }
          return { row: newRow, col: newCol };
        });
      } else if (mode === 'flatten') {
        setFlattenStep((prev) => (prev + 1) % (flattenedVector.length + 1));
      } else if (mode === 'dense') {
        setDenseNeuronIdx((prev) => (prev + 1) % 3);
      }
    }, speed);

    return () => clearInterval(interval);
  }, [isPlaying, mode, maxPos, stride, featureMapSize, speed, flattenedVector.length]);

  // Reset position when switching modes
  const handleModeChange = (newMode: Mode) => {
    setMode(newMode);
    setKernelPos({ row: 0, col: 0 });
    setFlattenStep(0);
    setDenseNeuronIdx(0);
    setIsPlaying(false);
  };

  const classLabels = ['Circle', 'Square', 'Triangle'];

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
            CNN Pipeline: Step by Step
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
          {(['padding', 'conv', 'relu', 'maxpool', 'flatten', 'dense', 'softmax'] as Mode[]).map((m, i) => (
            <div key={m} className="flex items-center">
              <div
                className={`px-2 py-1 rounded cursor-pointer transition-all ${
                  mode === m
                    ? 'bg-white text-gray-900 font-bold scale-110'
                    : 'bg-gray-700 text-gray-400 hover:bg-gray-600'
                }`}
                onClick={() => handleModeChange(m)}
              >
                {i + 1}. {m.charAt(0).toUpperCase() + m.slice(1)}
              </div>
              {i < 6 && <span className="text-gray-600 mx-1">→</span>}
            </div>
          ))}
        </div>

        {/* Mode tabs */}
        <div className="flex gap-2 mb-4 flex-wrap">
          <TabButton mode="padding" currentMode={mode} onClick={() => handleModeChange('padding')} color="yellow">
            🔲 Padding
          </TabButton>
          <TabButton mode="conv" currentMode={mode} onClick={() => handleModeChange('conv')} color="cyan">
            🔍 Convolution
          </TabButton>
          <TabButton mode="relu" currentMode={mode} onClick={() => handleModeChange('relu')} color="orange">
            ⚡ ReLU
          </TabButton>
          <TabButton mode="maxpool" currentMode={mode} onClick={() => handleModeChange('maxpool')} color="green">
            📉 MaxPool
          </TabButton>
          <TabButton mode="flatten" currentMode={mode} onClick={() => handleModeChange('flatten')} color="purple">
            📏 Flatten
          </TabButton>
          <TabButton mode="dense" currentMode={mode} onClick={() => handleModeChange('dense')} color="blue">
            🔗 Dense
          </TabButton>
          <TabButton mode="softmax" currentMode={mode} onClick={() => handleModeChange('softmax')} color="pink">
            📊 Softmax
          </TabButton>
        </div>

        <AnimatePresence mode="wait">
          {/* PADDING VISUALIZATION */}
          {mode === 'padding' && (
            <motion.div
              key="padding"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              className="space-y-4"
            >
              <div className="bg-gray-800 rounded-lg p-4">
                <h3 className="text-yellow-400 font-semibold mb-2">How Padding Works</h3>
                <p className="text-gray-300 text-sm">
                  <span className="text-yellow-400 font-medium">Zero padding</span> adds zeros around the input borders.
                  This preserves spatial dimensions after convolution and ensures edge pixels are processed fairly.
                </p>
              </div>

              <div className="flex items-center justify-center gap-8 flex-wrap">
                {/* Original input */}
                <div className="text-center">
                  <div className="text-gray-400 text-sm mb-2">Original ({gridSize}×{gridSize})</div>
                  <div
                    className="grid gap-0.5 bg-gray-950 p-1 rounded"
                    style={{ gridTemplateColumns: `repeat(${gridSize}, 20px)` }}
                  >
                    {inputGrid.map((row, ri) =>
                      row.map((val, ci) => (
                        <div
                          key={`${ri}-${ci}`}
                          className="w-5 h-5 rounded-sm"
                          style={{ backgroundColor: `rgba(34, 211, 238, ${val})` }}
                        />
                      ))
                    )}
                  </div>
                </div>

                <motion.div
                  className="text-4xl text-gray-500"
                  animate={{ scale: [1, 1.2, 1] }}
                  transition={{ duration: 1, repeat: Infinity }}
                >
                  →
                </motion.div>

                {/* Padded input */}
                <div className="text-center">
                  <div className="text-gray-400 text-sm mb-2">Padded ({gridSize + 2}×{gridSize + 2})</div>
                  <motion.div
                    className="grid gap-0.5 bg-gray-950 p-1 rounded"
                    style={{ gridTemplateColumns: `repeat(${gridSize + 2}, 20px)` }}
                    variants={containerVariants}
                    initial="hidden"
                    animate="show"
                  >
                    {paddedGrid.map((row, ri) =>
                      row.map((val, ci) => {
                        const isPadding = ri === 0 || ri === gridSize + 1 || ci === 0 || ci === gridSize + 1;
                        return (
                          <motion.div
                            key={`${ri}-${ci}`}
                            variants={itemVariants}
                            className={`w-5 h-5 rounded-sm flex items-center justify-center text-[8px] ${
                              isPadding ? 'border border-yellow-500/50' : ''
                            }`}
                            style={{
                              backgroundColor: isPadding
                                ? 'rgba(234, 179, 8, 0.3)'
                                : `rgba(34, 211, 238, ${val})`,
                            }}
                          >
                            {isPadding && <span className="text-yellow-400">0</span>}
                          </motion.div>
                        );
                      })
                    )}
                  </motion.div>
                </div>
              </div>

              <div className="bg-gray-800 rounded-lg p-4">
                <h4 className="text-gray-300 font-medium mb-2">Why Padding?</h4>
                <ul className="text-sm text-gray-400 space-y-1">
                  <li>• <span className="text-yellow-400">Preserves size</span>: With padding=1 and 3×3 kernel, output size = input size</li>
                  <li>• <span className="text-yellow-400">Edge features</span>: Corner and edge pixels get equal attention</li>
                  <li>• <span className="text-yellow-400">Types</span>: Zero (most common), Reflect, Replicate</li>
                </ul>
              </div>
            </motion.div>
          )}

          {/* CONVOLUTION VISUALIZATION */}
          {mode === 'conv' && (
            <motion.div
              key="conv"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              className="space-y-4"
            >
              <div className="bg-gray-800 rounded-lg p-4">
                <h3 className="text-cyan-400 font-semibold mb-2">How Convolution Works</h3>
                <p className="text-gray-300 text-sm">
                  A <span className="text-purple-400 font-medium">kernel (filter)</span> slides across the input.
                  At each position, it multiplies each pixel by the corresponding weight and sums them up.
                </p>
              </div>

              <div className="flex gap-2 items-center">
                <span className="text-gray-400 text-sm">Kernel:</span>
                {(Object.keys(EXAMPLE_KERNELS) as KernelType[]).map((k) => (
                  <button
                    key={k}
                    onClick={() => setSelectedKernel(k)}
                    className={`px-3 py-1 rounded text-sm capitalize ${
                      selectedKernel === k
                        ? 'bg-purple-600 text-white'
                        : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                    }`}
                  >
                    {k}
                  </button>
                ))}
              </div>

              <div className="flex items-center justify-center gap-6 flex-wrap">
                <div className="text-center">
                  <div className="text-gray-400 text-sm mb-2">Input</div>
                  <div
                    className="grid gap-0.5 bg-gray-950 p-1 rounded"
                    style={{ gridTemplateColumns: `repeat(${gridSize}, 24px)` }}
                  >
                    {inputGrid.map((row, ri) =>
                      row.map((val, ci) => {
                        const inKernel =
                          ri >= kernelPos.row &&
                          ri < kernelPos.row + kernelSize &&
                          ci >= kernelPos.col &&
                          ci < kernelPos.col + kernelSize;
                        return (
                          <motion.div
                            key={`${ri}-${ci}`}
                            className="w-6 h-6 rounded-sm flex items-center justify-center text-[8px]"
                            animate={{
                              backgroundColor: inKernel
                                ? 'rgba(168, 85, 247, 0.8)'
                                : `rgba(34, 211, 238, ${val})`,
                              scale: inKernel ? 1.1 : 1,
                            }}
                            transition={{ duration: 0.15 }}
                          >
                            {inKernel && <span className="text-white font-bold">{val.toFixed(1)}</span>}
                          </motion.div>
                        );
                      })
                    )}
                  </div>
                </div>

                <div className="text-3xl text-gray-500">×</div>

                <div className="text-center">
                  <div className="text-gray-400 text-sm mb-2">Kernel</div>
                  <div
                    className="grid gap-0.5 bg-purple-900/30 p-2 rounded border-2 border-purple-500"
                    style={{ gridTemplateColumns: `repeat(${kernelSize}, 36px)` }}
                  >
                    {kernel.map((row, ri) =>
                      row.map((val, ci) => (
                        <div
                          key={`k-${ri}-${ci}`}
                          className="w-9 h-9 rounded flex items-center justify-center text-sm font-mono bg-purple-800/50 text-purple-200"
                        >
                          {val}
                        </div>
                      ))
                    )}
                  </div>
                </div>

                <div className="text-3xl text-gray-500">=</div>

                <div className="text-center">
                  <div className="text-gray-400 text-sm mb-2">Output</div>
                  <motion.div
                    key={`${kernelPos.row}-${kernelPos.col}`}
                    initial={{ scale: 0.5 }}
                    animate={{ scale: 1 }}
                    className="w-14 h-14 rounded-lg bg-cyan-600 flex items-center justify-center text-lg font-bold text-white"
                  >
                    {convResult.toFixed(1)}
                  </motion.div>
                  <div className="text-gray-500 text-xs mt-1">Raw sum</div>
                </div>
              </div>

              <div className="bg-gray-800 rounded-lg p-3">
                <div className="text-sm text-gray-400 font-mono overflow-x-auto">
                  {kernel.flatMap((row, ri) =>
                    row.map((kval, ci) => {
                      const inputVal = inputGrid[kernelPos.row + ri]?.[kernelPos.col + ci] || 0;
                      return `(${inputVal.toFixed(1)}×${kval})`;
                    })
                  ).join(' + ')} = <span className="text-cyan-400">{convResult.toFixed(2)}</span>
                </div>
              </div>
            </motion.div>
          )}

          {/* RELU VISUALIZATION */}
          {mode === 'relu' && (
            <motion.div
              key="relu"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              className="space-y-4"
            >
              <div className="bg-gray-800 rounded-lg p-4">
                <h3 className="text-orange-400 font-semibold mb-2">How ReLU Works</h3>
                <p className="text-gray-300 text-sm">
                  <span className="text-orange-400 font-medium">ReLU (Rectified Linear Unit)</span> is simple:
                  keep positive values, replace negatives with zero. Formula: <code className="bg-gray-700 px-1 rounded">max(0, x)</code>
                </p>
              </div>

              <div className="flex items-center justify-center gap-8 flex-wrap">
                {/* Before ReLU */}
                <div className="text-center">
                  <div className="text-gray-400 text-sm mb-2">Before ReLU (Conv Output)</div>
                  <motion.div
                    className="grid gap-0.5 bg-gray-950 p-1 rounded"
                    style={{ gridTemplateColumns: `repeat(${featureMapSize}, 36px)` }}
                    variants={containerVariants}
                    initial="hidden"
                    animate="show"
                  >
                    {convOutputRaw.map((row, ri) =>
                      row.map((val, ci) => {
                        const isNegative = val < 0;
                        return (
                          <motion.div
                            key={`${ri}-${ci}`}
                            variants={itemVariants}
                            className={`w-9 h-9 rounded-sm flex items-center justify-center text-[10px] font-mono ${
                              isNegative ? 'bg-red-900/60 text-red-300' : 'bg-cyan-900/60 text-cyan-300'
                            }`}
                          >
                            {val.toFixed(2)}
                          </motion.div>
                        );
                      })
                    )}
                  </motion.div>
                </div>

                <motion.div
                  className="text-center"
                  animate={{ scale: [1, 1.1, 1] }}
                  transition={{ duration: 0.5, repeat: Infinity }}
                >
                  <div className="text-4xl text-gray-500">→</div>
                  <div className="text-orange-400 text-sm font-medium mt-1">max(0, x)</div>
                </motion.div>

                {/* After ReLU */}
                <div className="text-center">
                  <div className="text-gray-400 text-sm mb-2">After ReLU</div>
                  <motion.div
                    className="grid gap-0.5 bg-gray-950 p-1 rounded"
                    style={{ gridTemplateColumns: `repeat(${featureMapSize}, 36px)` }}
                    variants={containerVariants}
                    initial="hidden"
                    animate="show"
                  >
                    {convOutputRelu.map((row, ri) =>
                      row.map((val, ci) => {
                        const wasNegative = convOutputRaw[ri][ci] < 0;
                        return (
                          <motion.div
                            key={`${ri}-${ci}`}
                            variants={itemVariants}
                            className={`w-9 h-9 rounded-sm flex items-center justify-center text-[10px] font-mono ${
                              wasNegative
                                ? 'bg-orange-600 text-white font-bold'
                                : 'bg-cyan-900/60 text-cyan-300'
                            }`}
                          >
                            {val.toFixed(2)}
                          </motion.div>
                        );
                      })
                    )}
                  </motion.div>
                </div>
              </div>

              <div className="bg-gray-800 rounded-lg p-4">
                <h4 className="text-gray-300 font-medium mb-2">Why ReLU?</h4>
                <ul className="text-sm text-gray-400 space-y-1">
                  <li>• <span className="text-orange-400">Non-linearity</span>: Allows network to learn complex patterns</li>
                  <li>• <span className="text-orange-400">Sparsity</span>: Zeros create sparse representations (efficient)</li>
                  <li>• <span className="text-orange-400">Fast</span>: Simple comparison, no expensive math</li>
                  <li>• <span className="text-orange-400">Gradient flow</span>: Avoids vanishing gradients for positive values</li>
                </ul>
              </div>
            </motion.div>
          )}

          {/* MAX POOLING VISUALIZATION */}
          {mode === 'maxpool' && (
            <motion.div
              key="maxpool"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              className="space-y-4"
            >
              <div className="bg-gray-800 rounded-lg p-4">
                <h3 className="text-green-400 font-semibold mb-2">How Max Pooling Works</h3>
                <p className="text-gray-300 text-sm">
                  Takes the <span className="text-green-400 font-medium">maximum value</span> from each 2×2 window.
                  Reduces spatial size by 4× while keeping the strongest activations.
                </p>
              </div>

              <div className="flex items-center justify-center gap-6 flex-wrap">
                <div className="text-center">
                  <div className="text-gray-400 text-sm mb-2">Feature Map ({featureMapSize}×{featureMapSize})</div>
                  <div
                    className="grid gap-0.5 bg-gray-950 p-1 rounded"
                    style={{ gridTemplateColumns: `repeat(${featureMapSize}, 32px)` }}
                  >
                    {convOutputRelu.map((row, ri) =>
                      row.map((val, ci) => {
                        const inPool =
                          ri >= kernelPos.row &&
                          ri < kernelPos.row + poolSize &&
                          ci >= kernelPos.col &&
                          ci < kernelPos.col + poolSize;
                        const localIdx = inPool
                          ? (ri - kernelPos.row) * poolSize + (ci - kernelPos.col)
                          : -1;
                        const isMax = inPool && localIdx === maxPoolIndex;

                        return (
                          <motion.div
                            key={`${ri}-${ci}`}
                            className="w-8 h-8 rounded-sm flex items-center justify-center text-[10px] font-mono"
                            animate={{
                              backgroundColor: isMax
                                ? 'rgba(34, 197, 94, 1)'
                                : inPool
                                ? 'rgba(34, 197, 94, 0.4)'
                                : `rgba(34, 211, 238, ${Math.min(1, val)})`,
                              scale: isMax ? 1.15 : inPool ? 1.05 : 1,
                            }}
                            style={{
                              border: inPool ? '2px solid rgba(34, 197, 94, 1)' : 'none',
                            }}
                            transition={{ duration: 0.15 }}
                          >
                            <span className={isMax ? 'text-white font-bold' : 'text-gray-300'}>
                              {val.toFixed(2)}
                            </span>
                          </motion.div>
                        );
                      })
                    )}
                  </div>
                </div>

                <div className="text-3xl text-gray-500">→</div>

                <div className="text-center">
                  <div className="text-gray-400 text-sm mb-2">2×2 Window</div>
                  <div className="grid grid-cols-2 gap-1 bg-green-900/30 p-2 rounded border-2 border-green-500">
                    {poolValues.map((val, i) => (
                      <motion.div
                        key={i}
                        className={`w-10 h-10 rounded flex items-center justify-center text-xs font-mono ${
                          i === maxPoolIndex
                            ? 'bg-green-500 text-white font-bold'
                            : 'bg-gray-700 text-gray-300'
                        }`}
                        animate={{ scale: i === maxPoolIndex ? 1.1 : 1 }}
                      >
                        {val.toFixed(2)}
                      </motion.div>
                    ))}
                  </div>
                  <div className="text-xs text-gray-500 mt-1">
                    Max: <span className="text-green-400 font-bold">{maxPoolValue.toFixed(2)}</span>
                  </div>
                </div>

                <div className="text-3xl text-gray-500">=</div>

                <div className="text-center">
                  <div className="text-gray-400 text-sm mb-2">Output (3×3)</div>
                  <div className="grid grid-cols-3 gap-0.5 bg-gray-950 p-1 rounded">
                    {pooledOutput.map((row, ri) =>
                      row.map((val, ci) => {
                        const isCurrentOutput =
                          ri === Math.floor(kernelPos.row / stride) &&
                          ci === Math.floor(kernelPos.col / stride);
                        return (
                          <motion.div
                            key={`${ri}-${ci}`}
                            className={`w-8 h-8 rounded-sm flex items-center justify-center text-[10px] font-mono ${
                              isCurrentOutput
                                ? 'bg-green-500 text-white font-bold'
                                : 'bg-green-900/40 text-green-300'
                            }`}
                            animate={{ scale: isCurrentOutput ? 1.15 : 1 }}
                          >
                            {val.toFixed(2)}
                          </motion.div>
                        );
                      })
                    )}
                  </div>
                </div>
              </div>
            </motion.div>
          )}

          {/* FLATTEN VISUALIZATION */}
          {mode === 'flatten' && (
            <motion.div
              key="flatten"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              className="space-y-4"
            >
              <div className="bg-gray-800 rounded-lg p-4">
                <h3 className="text-purple-400 font-semibold mb-2">How Flatten Works</h3>
                <p className="text-gray-300 text-sm">
                  <span className="text-purple-400 font-medium">Flatten</span> converts the 2D feature map into a 1D vector.
                  This prepares spatial features for the fully-connected (dense) layers.
                </p>
              </div>

              <div className="flex items-center justify-center gap-8 flex-wrap">
                {/* 2D pooled output */}
                <div className="text-center">
                  <div className="text-gray-400 text-sm mb-2">Pooled (3×3)</div>
                  <motion.div
                    className="grid grid-cols-3 gap-1 bg-gray-950 p-2 rounded"
                    variants={containerVariants}
                    initial="hidden"
                    animate="show"
                  >
                    {pooledOutput.map((row, ri) =>
                      row.map((val, ci) => {
                        const idx = ri * 3 + ci;
                        const isHighlighted = idx < flattenStep;
                        return (
                          <motion.div
                            key={`${ri}-${ci}`}
                            variants={itemVariants}
                            className={`w-12 h-12 rounded flex flex-col items-center justify-center text-xs font-mono transition-colors ${
                              isHighlighted
                                ? 'bg-purple-500 text-white'
                                : 'bg-gray-800 text-gray-300'
                            }`}
                          >
                            <span className="text-[8px] text-gray-500">[{ri},{ci}]</span>
                            <span>{val.toFixed(2)}</span>
                          </motion.div>
                        );
                      })
                    )}
                  </motion.div>
                </div>

                <motion.div
                  animate={{ x: [0, 10, 0] }}
                  transition={{ duration: 0.8, repeat: Infinity }}
                  className="text-4xl text-gray-500"
                >
                  →
                </motion.div>

                {/* 1D flattened vector */}
                <div className="text-center">
                  <div className="text-gray-400 text-sm mb-2">Flattened (9×1)</div>
                  <motion.div
                    className="flex flex-col gap-0.5 bg-gray-950 p-2 rounded"
                    variants={containerVariants}
                    initial="hidden"
                    animate="show"
                  >
                    {flattenedVector.map((val, i) => {
                      const isHighlighted = i < flattenStep;
                      const row = Math.floor(i / 3);
                      const col = i % 3;
                      return (
                        <motion.div
                          key={i}
                          variants={itemVariants}
                          className={`w-20 h-6 rounded flex items-center justify-between px-2 text-xs font-mono transition-colors ${
                            isHighlighted
                              ? 'bg-purple-500 text-white'
                              : 'bg-gray-800 text-gray-300'
                          }`}
                        >
                          <span className="text-[8px] opacity-60">[{row},{col}]→[{i}]</span>
                          <span>{val.toFixed(2)}</span>
                        </motion.div>
                      );
                    })}
                  </motion.div>
                </div>
              </div>

              <div className="bg-gray-800 rounded-lg p-4">
                <h4 className="text-gray-300 font-medium mb-2">Flatten Operation</h4>
                <p className="text-sm text-gray-400">
                  Shape: <span className="text-purple-400">(3, 3)</span> → <span className="text-purple-400">(9,)</span>
                  <br />
                  Row-major order: [0,0] → [0,1] → [0,2] → [1,0] → ... → [2,2]
                </p>
              </div>
            </motion.div>
          )}

          {/* DENSE VISUALIZATION */}
          {mode === 'dense' && (
            <motion.div
              key="dense"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              className="space-y-4"
            >
              <div className="bg-gray-800 rounded-lg p-4">
                <h3 className="text-blue-400 font-semibold mb-2">How Dense (Fully Connected) Works</h3>
                <p className="text-gray-300 text-sm">
                  Each output neuron connects to <span className="text-blue-400 font-medium">every input</span>.
                  It computes a weighted sum: <code className="bg-gray-700 px-1 rounded">Σ(weight × input) + bias</code>
                </p>
              </div>

              <div className="flex items-center justify-center gap-6 flex-wrap">
                {/* Input vector */}
                <div className="text-center">
                  <div className="text-gray-400 text-sm mb-2">Input (9)</div>
                  <div className="flex flex-col gap-0.5">
                    {flattenedVector.map((val, i) => (
                      <motion.div
                        key={i}
                        className="w-14 h-5 rounded text-[10px] font-mono flex items-center justify-center"
                        animate={{
                          backgroundColor:
                            denseWeights[denseNeuronIdx][i] > 0
                              ? 'rgba(59, 130, 246, 0.6)'
                              : 'rgba(239, 68, 68, 0.4)',
                          scale: 1,
                        }}
                      >
                        {val.toFixed(2)}
                      </motion.div>
                    ))}
                  </div>
                </div>

                {/* Weights visualization */}
                <div className="text-center">
                  <div className="text-gray-400 text-sm mb-2">Weights</div>
                  <div className="flex flex-col gap-0.5">
                    {denseWeights[denseNeuronIdx].map((w, i) => (
                      <motion.div
                        key={i}
                        className={`w-14 h-5 rounded text-[10px] font-mono flex items-center justify-center ${
                          w > 0 ? 'bg-blue-800 text-blue-200' : 'bg-red-900 text-red-200'
                        }`}
                        initial={{ opacity: 0, x: -10 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: i * 0.03 }}
                      >
                        {w.toFixed(2)}
                      </motion.div>
                    ))}
                  </div>
                </div>

                <div className="text-3xl text-gray-500">=</div>

                {/* Output neurons */}
                <div className="text-center">
                  <div className="text-gray-400 text-sm mb-2">Output (3 classes)</div>
                  <div className="flex flex-col gap-2">
                    {classLabels.map((label, i) => (
                      <motion.div
                        key={i}
                        className={`px-4 py-2 rounded-lg text-sm font-medium flex items-center justify-between gap-4 ${
                          i === denseNeuronIdx
                            ? 'bg-blue-600 text-white ring-2 ring-blue-400'
                            : 'bg-gray-700 text-gray-300'
                        }`}
                        animate={{ scale: i === denseNeuronIdx ? 1.05 : 1 }}
                        onClick={() => setDenseNeuronIdx(i)}
                        style={{ cursor: 'pointer' }}
                      >
                        <span>{label}</span>
                        <span className="font-mono">{denseOutput[i].toFixed(2)}</span>
                      </motion.div>
                    ))}
                  </div>
                </div>
              </div>

              <div className="bg-gray-800 rounded-lg p-4">
                <h4 className="text-gray-300 font-medium mb-2">
                  Calculation for "{classLabels[denseNeuronIdx]}" neuron:
                </h4>
                <p className="text-sm text-gray-400 font-mono">
                  Σ(input × weight) = {denseOutput[denseNeuronIdx].toFixed(2)}
                </p>
              </div>
            </motion.div>
          )}

          {/* SOFTMAX VISUALIZATION */}
          {mode === 'softmax' && (
            <motion.div
              key="softmax"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              className="space-y-4"
            >
              <div className="bg-gray-800 rounded-lg p-4">
                <h3 className="text-pink-400 font-semibold mb-2">How Softmax Works</h3>
                <p className="text-gray-300 text-sm">
                  <span className="text-pink-400 font-medium">Softmax</span> converts raw scores into probabilities.
                  Formula: <code className="bg-gray-700 px-1 rounded">e^x / Σe^x</code> — all outputs sum to 1.
                </p>
              </div>

              <div className="flex items-center justify-center gap-8 flex-wrap">
                {/* Raw scores */}
                <div className="text-center">
                  <div className="text-gray-400 text-sm mb-2">Raw Scores (Logits)</div>
                  <div className="flex flex-col gap-2">
                    {classLabels.map((label, i) => (
                      <motion.div
                        key={i}
                        className="flex items-center gap-3"
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: i * 0.1 }}
                      >
                        <span className="text-gray-400 text-sm w-16">{label}</span>
                        <div className="w-20 h-8 rounded bg-gray-700 flex items-center justify-center font-mono text-sm">
                          {denseOutput[i].toFixed(2)}
                        </div>
                      </motion.div>
                    ))}
                  </div>
                </div>

                <motion.div
                  className="text-center"
                  animate={{ scale: [1, 1.1, 1] }}
                  transition={{ duration: 1, repeat: Infinity }}
                >
                  <div className="text-4xl text-gray-500">→</div>
                  <div className="text-pink-400 text-xs mt-1">e^x / Σe^x</div>
                </motion.div>

                {/* Probabilities */}
                <div className="text-center">
                  <div className="text-gray-400 text-sm mb-2">Probabilities</div>
                  <div className="flex flex-col gap-2">
                    {classLabels.map((label, i) => {
                      const prob = softmaxOutput[i];
                      const isMax = prob === Math.max(...softmaxOutput);
                      return (
                        <motion.div
                          key={i}
                          className="flex items-center gap-3"
                          initial={{ opacity: 0, x: 20 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ delay: i * 0.1 + 0.3 }}
                        >
                          <span className={`text-sm w-16 ${isMax ? 'text-pink-400 font-bold' : 'text-gray-400'}`}>
                            {label}
                          </span>
                          <div className="w-32 h-8 bg-gray-800 rounded overflow-hidden flex items-center">
                            <motion.div
                              className={`h-full ${isMax ? 'bg-pink-500' : 'bg-gray-600'}`}
                              initial={{ width: 0 }}
                              animate={{ width: `${prob * 100}%` }}
                              transition={{ duration: 0.5, delay: i * 0.1 + 0.5 }}
                            />
                            <span className={`ml-2 text-xs font-mono ${isMax ? 'text-white' : 'text-gray-400'}`}>
                              {(prob * 100).toFixed(1)}%
                            </span>
                          </div>
                        </motion.div>
                      );
                    })}
                  </div>
                </div>
              </div>

              <div className="bg-gray-800 rounded-lg p-4">
                <h4 className="text-gray-300 font-medium mb-2">Result</h4>
                <p className="text-sm text-gray-400">
                  Prediction: <span className="text-pink-400 font-bold text-lg">
                    {classLabels[softmaxOutput.indexOf(Math.max(...softmaxOutput))]}
                  </span>
                  <span className="ml-2">
                    ({(Math.max(...softmaxOutput) * 100).toFixed(1)}% confidence)
                  </span>
                </p>
                <p className="text-xs text-gray-500 mt-2">
                  Sum of probabilities: {softmaxOutput.reduce((a, b) => a + b, 0).toFixed(4)} ≈ 1.0
                </p>
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
              if (mode === 'flatten') {
                setFlattenStep((prev) => (prev + 1) % (flattenedVector.length + 1));
              } else if (mode === 'dense') {
                setDenseNeuronIdx((prev) => (prev + 1) % 3);
              } else {
                setKernelPos((prev) => {
                  if (mode === 'conv' || mode === 'padding') {
                    let newCol = prev.col + 1;
                    let newRow = prev.row;
                    if (newCol > maxPos) {
                      newCol = 0;
                      newRow = (prev.row + 1) % (maxPos + 1);
                    }
                    return { row: newRow, col: newCol };
                  } else {
                    let newCol = prev.col + stride;
                    let newRow = prev.row;
                    if (newCol >= featureMapSize) {
                      newCol = 0;
                      newRow = prev.row + stride;
                    }
                    if (newRow >= featureMapSize) {
                      newRow = 0;
                      newCol = 0;
                    }
                    return { row: newRow, col: newCol };
                  }
                });
              }
            }}
            disabled={isPlaying || mode === 'softmax'}
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
              setKernelPos({ row: 0, col: 0 });
              setFlattenStep(0);
              setDenseNeuronIdx(0);
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
