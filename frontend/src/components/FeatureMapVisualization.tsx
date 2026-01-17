import type { CNNFeatureMaps, NetworkArchitecture, LayerWeights } from '../types';

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
  const cellSizes = { sm: 6, md: 10, lg: 16 };
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

        return (
          <div key={layerName} className="bg-gray-800 rounded-lg p-3">
            <h3 className="text-sm font-medium text-gray-300 mb-2">
              {layerName.replace('_', ' ').replace('conv2d', 'Conv').replace('maxpool', 'MaxPool')}
              <span className="text-xs text-gray-500 ml-2">
                {height}×{width}×{channels}
              </span>
            </h3>
            <div className="flex gap-2 flex-wrap">
              {channelMaps.map((channelData, i) => (
                <Heatmap
                  key={i}
                  data={channelData}
                  title={`Ch${i + 1}`}
                  size="sm"
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
  return (
    <div className="space-y-4">
      {/* Architecture info */}
      {architecture && (
        <div className="bg-gray-800 rounded-lg p-3">
          <h3 className="text-sm font-medium text-gray-300 mb-1">CNN Architecture</h3>
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
