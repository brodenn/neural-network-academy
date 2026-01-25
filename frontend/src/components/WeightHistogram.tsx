import { useMemo } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';

interface LayerWeights {
  layer: number;
  weights: number[][];
  biases: number[][];
}

interface WeightHistogramProps {
  weights: LayerWeights[];
  trainingInProgress?: boolean;
}

export function WeightHistogram({ weights, trainingInProgress }: WeightHistogramProps) {
  const histogramData = useMemo(() => {
    // Flatten all weights into a single array
    const allWeights: number[] = [];

    if (!Array.isArray(weights)) {
      return { bins: [], stats: { min: 0, max: 0, mean: 0, std: 0, count: 0 } };
    }

    for (const layerData of weights) {
      // Handle LayerWeights structure
      const weightMatrix = layerData?.weights;
      if (!Array.isArray(weightMatrix)) continue;

      for (const row of weightMatrix) {
        if (!Array.isArray(row)) continue;
        for (const weight of row) {
          if (typeof weight === 'number' && isFinite(weight)) {
            allWeights.push(weight);
          }
        }
      }
    }

    if (allWeights.length === 0) {
      return { bins: [], stats: { min: 0, max: 0, mean: 0, std: 0, count: 0 } };
    }

    // Calculate statistics
    const min = Math.min(...allWeights);
    const max = Math.max(...allWeights);
    const mean = allWeights.reduce((a, b) => a + b, 0) / allWeights.length;
    const variance = allWeights.reduce((sum, w) => sum + (w - mean) ** 2, 0) / allWeights.length;
    const std = Math.sqrt(variance);

    // Create histogram bins
    const numBins = 20;
    const binWidth = (max - min) / numBins || 0.1;
    const bins: { range: string; count: number; center: number }[] = [];

    for (let i = 0; i < numBins; i++) {
      const binStart = min + i * binWidth;
      const binEnd = binStart + binWidth;
      const count = allWeights.filter(w => w >= binStart && w < binEnd).length;
      bins.push({
        range: `${binStart.toFixed(2)}`,
        count,
        center: (binStart + binEnd) / 2,
      });
    }

    return { bins, stats: { min, max, mean, std, count: allWeights.length } };
  }, [weights]);

  if (histogramData.bins.length === 0) {
    return (
      <div className="bg-gray-800 rounded-lg p-3 h-full">
        <h3 className="text-sm font-semibold mb-2 text-gray-300">Weight Distribution</h3>
        <div className="h-32 flex items-center justify-center text-gray-500 text-sm">
          No weights to display
        </div>
      </div>
    );
  }

  const { bins, stats } = histogramData;

  return (
    <div className="bg-gray-800 rounded-lg p-3 h-full">
      <div className="flex items-center justify-between mb-1">
        <h3 className="text-sm font-semibold text-gray-300">Weight Distribution</h3>
        {trainingInProgress && (
          <span className="text-xs text-cyan-400 animate-pulse">updating...</span>
        )}
      </div>

      <div className="h-32">
        <ResponsiveContainer width="100%" height="100%" minWidth={100} minHeight={100}>
          <BarChart data={bins} margin={{ top: 5, right: 5, bottom: 5, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis
              dataKey="center"
              stroke="#9CA3AF"
              tick={{ fill: '#9CA3AF', fontSize: 9 }}
              tickFormatter={(v) => v.toFixed(1)}
              interval="preserveStartEnd"
            />
            <YAxis
              stroke="#9CA3AF"
              tick={{ fill: '#9CA3AF', fontSize: 9 }}
              width={30}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: '#1F2937',
                border: '1px solid #374151',
                borderRadius: '8px',
                fontSize: '12px',
              }}
              formatter={(value) => [value ?? 0, 'Count']}
              labelFormatter={(label) => `Weight: ${Number(label).toFixed(3)}`}
            />
            <ReferenceLine x={0} stroke="#EF4444" strokeDasharray="3 3" />
            <ReferenceLine x={stats.mean} stroke="#22C55E" strokeDasharray="3 3" />
            <Bar dataKey="count" fill="#0891b2" radius={[2, 2, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="mt-1 grid grid-cols-4 gap-1 text-xs text-gray-400">
        <div className="text-center">
          <div className="text-gray-500">Min</div>
          <div className="font-mono">{stats.min.toFixed(2)}</div>
        </div>
        <div className="text-center">
          <div className="text-gray-500">Max</div>
          <div className="font-mono">{stats.max.toFixed(2)}</div>
        </div>
        <div className="text-center">
          <div className="text-gray-500">Mean</div>
          <div className="font-mono text-green-400">{stats.mean.toFixed(2)}</div>
        </div>
        <div className="text-center">
          <div className="text-gray-500">Std</div>
          <div className="font-mono">{stats.std.toFixed(2)}</div>
        </div>
      </div>

      <div className="mt-1 text-xs text-gray-500 text-center">
        {stats.count} weights | <span className="text-red-400">red</span>=zero <span className="text-green-400">green</span>=mean
      </div>
    </div>
  );
}
