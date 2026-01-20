import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import type { CNNFeatureMaps, NetworkArchitecture, LayerWeights } from '../types';
import { CNNEducationalViz } from './CNNEducationalViz';

interface FeatureMapVisualizationProps {
  inputGrid: number[][];
  featureMaps: CNNFeatureMaps | null;
  architecture: NetworkArchitecture | null;
  weights: LayerWeights[];
  prediction: number[] | null;
  outputLabels: string[];
}

// Heatmap component for displaying 2D data
function Heatmap({
  data,
  title,
  size = 'md',
  colorScheme = 'cyan',
}: {
  data: number[][];
  title?: string;
  size?: 'sm' | 'md' | 'lg';
  colorScheme?: 'cyan' | 'green' | 'purple' | 'orange';
}) {
  const cellSizes = { sm: 12, md: 16, lg: 20 };
  const cellSize = cellSizes[size];

  // Normalize data to 0-1 range
  const flat = data.flat();
  const min = Math.min(...flat);
  const max = Math.max(...flat);
  const range = max - min || 1;

  const colorMap: Record<string, (intensity: number) => string> = {
    cyan: (i) => `rgba(34, 211, 238, ${i})`,
    green: (i) => `rgba(34, 197, 94, ${i})`,
    purple: (i) => `rgba(168, 85, 247, ${i})`,
    orange: (i) => `rgba(251, 146, 60, ${i})`,
  };

  const getColor = colorMap[colorScheme];

  return (
    <div className="flex flex-col items-center">
      {title && <span className="text-[10px] text-gray-500 mb-1">{title}</span>}
      <div
        className="grid gap-px bg-gray-900 rounded"
        style={{
          gridTemplateColumns: `repeat(${data[0]?.length || 1}, ${cellSize}px)`,
        }}
      >
        {data.map((row, ri) =>
          row.map((val, ci) => {
            const normalized = (val - min) / range;
            return (
              <div
                key={`${ri}-${ci}`}
                style={{
                  width: cellSize,
                  height: cellSize,
                  backgroundColor: getColor(normalized),
                }}
                title={`${val.toFixed(3)}`}
              />
            );
          })
        )}
      </div>
    </div>
  );
}

// Display CNN filter weights as heatmaps
function FilterVisualization({
  weights,
}: {
  weights: LayerWeights[];
}) {
  const convLayer = weights.find((w) => w.layer === 0);
  if (!convLayer || !convLayer.weights) return null;

  // Conv2D weights are [kernel_h, kernel_w, in_channels, out_filters]
  // We need to reshape them for visualization
  const filterData = convLayer.weights as unknown as number[][][][];
  if (!filterData || !filterData[0]) return null;

  const numFilters = filterData[0]?.[0]?.[0]?.length || 0;
  const kernelSize = filterData.length;

  // Extract each filter (assuming single input channel)
  const filters: number[][][] = [];
  for (let f = 0; f < numFilters; f++) {
    const filter: number[][] = [];
    for (let i = 0; i < kernelSize; i++) {
      const row: number[] = [];
      for (let j = 0; j < kernelSize; j++) {
        // Get weight for position (i,j) channel 0, filter f
        row.push(filterData[i]?.[j]?.[0]?.[f] || 0);
      }
      filter.push(row);
    }
    filters.push(filter);
  }

  return (
    <div className="bg-gray-800 rounded-lg p-3">
      <h3 className="text-sm font-medium text-gray-300 mb-2">Conv Filters</h3>
      <div className="flex gap-2 flex-wrap">
        {filters.map((filter, i) => (
          <Heatmap
            key={i}
            data={filter}
            title={`F${i + 1}`}
            size="md"
            colorScheme="purple"
          />
        ))}
      </div>
    </div>
  );
}

// Layer explanations for tooltips
const LAYER_EXPLANATIONS: Record<string, { title: string; description: string; icon: string }> = {
  conv: {
    title: 'Convolution Layer',
    description: 'Detects patterns (edges, shapes) by sliding filters across the input. Each filter learns to detect different features.',
    icon: '🔍',
  },
  pool: {
    title: 'Max Pooling Layer',
    description: 'Reduces size by keeping only the maximum value in each region. Makes the network faster and more robust to small shifts.',
    icon: '📉',
  },
};

// Info tooltip component
function InfoTooltip({ type }: { type: 'conv' | 'pool' }) {
  const [showTooltip, setShowTooltip] = useState(false);
  const info = LAYER_EXPLANATIONS[type];

  return (
    <div className="relative inline-block">
      <button
        onMouseEnter={() => setShowTooltip(true)}
        onMouseLeave={() => setShowTooltip(false)}
        onClick={() => setShowTooltip(!showTooltip)}
        className="ml-2 w-5 h-5 rounded-full bg-gray-600 hover:bg-gray-500 text-xs text-gray-300 inline-flex items-center justify-center"
      >
        ?
      </button>
      {showTooltip && (
        <div className="absolute z-10 left-6 top-0 w-64 p-3 bg-gray-700 rounded-lg shadow-xl border border-gray-600">
          <div className="flex items-center gap-2 mb-1">
            <span>{info.icon}</span>
            <span className="font-medium text-white text-sm">{info.title}</span>
          </div>
          <p className="text-xs text-gray-300">{info.description}</p>
        </div>
      )}
    </div>
  );
}

// Display feature maps from CNN layers
function FeatureMapsDisplay({
  featureMaps,
}: {
  featureMaps: CNNFeatureMaps;
}) {
  const layers = Object.entries(featureMaps);

  if (layers.length === 0) {
    return (
      <div className="text-gray-500 text-sm">No feature maps available</div>
    );
  }

  return (
    <div className="space-y-3">
      {layers.map(([layerName, data]) => {
        // data is (height, width, channels) - need to extract each channel
        if (!data || !data[0] || !data[0][0]) return null;

        const height = data.length;
        const width = data[0].length;
        const channels = data[0][0].length;

        const channelMaps: number[][][] = [];
        for (let c = 0; c < channels; c++) {
          const channelData: number[][] = [];
          for (let h = 0; h < height; h++) {
            const row: number[] = [];
            for (let w = 0; w < width; w++) {
              row.push(data[h][w][c]);
            }
            channelData.push(row);
          }
          channelMaps.push(channelData);
        }

        const isPool = layerName.includes('pool');
        const colorScheme = isPool ? 'green' : 'cyan';
        const layerType = isPool ? 'pool' : 'conv';

        return (
          <div key={layerName} className="bg-gray-800 rounded-lg p-3">
            <h3 className="text-sm font-medium text-gray-300 mb-2 flex items-center">
              <span className={isPool ? 'text-green-400' : 'text-cyan-400'}>
                {layerName.replace('_', ' ').replace('conv2d', 'Conv').replace('maxpool', 'MaxPool')}
              </span>
              <span className="text-xs text-gray-500 ml-2">
                {height}×{width}×{channels}
              </span>
              <InfoTooltip type={layerType} />
            </h3>
            <div className="flex gap-3 flex-wrap">
              {channelMaps.map((channelData, i) => (
                <Heatmap
                  key={i}
                  data={channelData}
                  title={`Ch${i + 1}`}
                  size="md"
                  colorScheme={colorScheme}
                />
              ))}
            </div>
          </div>
        );
      })}
    </div>
  );
}

// Prediction output bars
function PredictionBars({
  prediction,
  labels,
}: {
  prediction: number[];
  labels: string[];
}) {
  const maxIdx = prediction.indexOf(Math.max(...prediction));

  return (
    <div className="bg-gray-800 rounded-lg p-3">
      <h3 className="text-sm font-medium text-gray-300 mb-2">Prediction</h3>
      <div className="space-y-2">
        {labels.map((label, i) => {
          const prob = prediction[i] || 0;
          const isMax = i === maxIdx;
          return (
            <div key={label} className="flex items-center gap-2">
              <span
                className={`text-xs w-16 ${
                  isMax ? 'text-cyan-400 font-medium' : 'text-gray-400'
                }`}
              >
                {label}
              </span>
              <div className="flex-1 h-4 bg-gray-700 rounded overflow-hidden">
                <div
                  className={`h-full transition-all duration-300 ${
                    isMax
                      ? 'bg-gradient-to-r from-cyan-600 to-cyan-400'
                      : 'bg-gray-500'
                  }`}
                  style={{ width: `${prob * 100}%` }}
                />
              </div>
              <span
                className={`text-xs font-mono w-12 text-right ${
                  isMax ? 'text-cyan-400' : 'text-gray-500'
                }`}
              >
                {(prob * 100).toFixed(1)}%
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export function FeatureMapVisualization({
  inputGrid,
  featureMaps,
  architecture,
  weights,
  prediction,
  outputLabels,
}: FeatureMapVisualizationProps) {
  const [showEducational, setShowEducational] = useState(false);
  const [showLearnHint, setShowLearnHint] = useState(true);

  return (
    <div className="space-y-4">
      {/* Educational visualization modal */}
      <AnimatePresence>
        {showEducational && inputGrid.length > 0 && (
          <CNNEducationalViz
            inputGrid={inputGrid}
            onClose={() => setShowEducational(false)}
          />
        )}
      </AnimatePresence>

      {/* Prominent Learn Banner */}
      <AnimatePresence>
        {showLearnHint && inputGrid.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="bg-gradient-to-r from-purple-900/80 via-cyan-900/80 to-purple-900/80 rounded-lg p-3 border border-purple-500/30 relative overflow-hidden"
          >
            {/* Animated background shimmer */}
            <motion.div
              className="absolute inset-0 bg-gradient-to-r from-transparent via-white/5 to-transparent"
              animate={{ x: ['-100%', '100%'] }}
              transition={{ duration: 2, repeat: Infinity, ease: 'linear' }}
            />

            <div className="relative flex items-center justify-between gap-3">
              <div className="flex items-center gap-3">
                <motion.div
                  className="w-10 h-10 rounded-full bg-purple-500/20 flex items-center justify-center text-xl"
                  animate={{ scale: [1, 1.1, 1] }}
                  transition={{ duration: 1.5, repeat: Infinity }}
                >
                  🔍
                </motion.div>
                <div>
                  <div className="text-white font-medium text-sm">Learn CNN Step-by-Step</div>
                  <div className="text-purple-300/80 text-xs">See how convolution, pooling & more work</div>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <motion.button
                  onClick={() => {
                    setShowEducational(true);
                    setShowLearnHint(false);
                  }}
                  className="px-4 py-2 bg-purple-500 hover:bg-purple-400 text-white font-medium rounded-lg text-sm flex items-center gap-2 shadow-lg shadow-purple-500/25"
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  <span>📚</span> Start Learning
                </motion.button>
                <button
                  onClick={() => setShowLearnHint(false)}
                  className="text-gray-400 hover:text-white p-1"
                  title="Dismiss"
                >
                  ✕
                </button>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Architecture info with Learn button */}
      {architecture && (
        <div className="bg-gray-800 rounded-lg p-3">
          <div className="flex justify-between items-start mb-1">
            <h3 className="text-sm font-medium text-gray-300">CNN Architecture</h3>
            {!showLearnHint && (
              <motion.button
                onClick={() => setShowEducational(true)}
                className="px-3 py-1.5 text-xs font-medium bg-gradient-to-r from-purple-600 to-cyan-600 hover:from-purple-500 hover:to-cyan-500 text-white rounded-lg transition-all flex items-center gap-1.5 shadow-md"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <span>📚</span> Learn How It Works
              </motion.button>
            )}
          </div>
          <div className="text-xs text-gray-500">
            {architecture.layers?.join(' → ')}
          </div>
        </div>
      )}

      {/* Input grid visualization */}
      {inputGrid && inputGrid.length > 0 && (
        <div className="bg-gray-800 rounded-lg p-3">
          <h3 className="text-sm font-medium text-gray-300 mb-2">Input</h3>
          <Heatmap data={inputGrid} size="lg" colorScheme="cyan" />
        </div>
      )}

      {/* Filter weights */}
      {weights && weights.length > 0 && (
        <FilterVisualization weights={weights} />
      )}

      {/* Feature maps */}
      {featureMaps && Object.keys(featureMaps).length > 0 && (
        <FeatureMapsDisplay featureMaps={featureMaps} />
      )}

      {/* Prediction output */}
      {prediction && prediction.length > 0 && (
        <PredictionBars prediction={prediction} labels={outputLabels} />
      )}
    </div>
  );
}
