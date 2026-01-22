import { memo, useMemo, useState, useCallback } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  ReferenceLine,
} from 'recharts';
import { TrainingScrubber } from './TrainingScrubber';

interface LossCurveProps {
  lossHistory: number[];
  accuracyHistory: number[];
  totalEpochs?: number;
  trainingComplete?: boolean;
  onScrubToEpoch?: (epoch: number) => void;
}

export const LossCurve = memo(function LossCurve({
  lossHistory,
  accuracyHistory,
  totalEpochs,
  trainingComplete = false,
  onScrubToEpoch,
}: LossCurveProps) {
  // Scrubber state
  const [scrubEpoch, setScrubEpoch] = useState<number | null>(null);
  const [showScrubber, setShowScrubber] = useState(false);

  const data = useMemo(() => {
    if (lossHistory.length === 0) return [];

    // Calculate the step size based on total epochs vs data points
    const actualTotal = totalEpochs || lossHistory.length;
    const step = actualTotal / lossHistory.length;

    return lossHistory.map((loss, i) => ({
      epoch: Math.round(i * step),
      loss,
      accuracy: accuracyHistory[i] ?? 0,
    }));
  }, [lossHistory, accuracyHistory, totalEpochs]);

  // Handle scrubbing
  const handleScrubToEpoch = useCallback((epoch: number) => {
    setScrubEpoch(epoch);
    onScrubToEpoch?.(epoch);
  }, [onScrubToEpoch]);

  if (data.length === 0) {
    return (
      <div className="bg-gray-800 rounded-lg p-3">
        <h2 className="text-lg font-semibold mb-2">Training Progress</h2>
        <div className="h-36 flex items-center justify-center text-gray-500 text-sm">
          No training data yet. Start training to see the loss curve.
        </div>
      </div>
    );
  }

  const actualTotalEpochs = totalEpochs || lossHistory.length;

  return (
    <div className="bg-gray-800 rounded-lg p-3">
      <div className="flex items-center justify-between mb-2">
        <h2 className="text-lg font-semibold">Training Progress</h2>
        {trainingComplete && lossHistory.length > 10 && (
          <button
            onClick={() => setShowScrubber(!showScrubber)}
            className={`px-2 py-1 text-xs rounded transition-colors ${
              showScrubber
                ? 'bg-cyan-600 text-white'
                : 'bg-gray-700 text-gray-400 hover:text-gray-300'
            }`}
          >
            {showScrubber ? '⏱ Timeline On' : '⏱ Timeline'}
          </button>
        )}
      </div>

      <div className="h-36">
        <ResponsiveContainer width="100%" height="100%" minWidth={100} minHeight={100}>
          <LineChart data={data} margin={{ top: 5, right: 5, bottom: 5, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis
              dataKey="epoch"
              stroke="#9CA3AF"
              tick={{ fill: '#9CA3AF', fontSize: 11 }}
            />
            <YAxis
              yAxisId="loss"
              stroke="#EF4444"
              tick={{ fill: '#9CA3AF', fontSize: 11 }}
              domain={[0, 'auto']}
              width={40}
            />
            <YAxis
              yAxisId="accuracy"
              orientation="right"
              stroke="#22C55E"
              tick={{ fill: '#9CA3AF', fontSize: 11 }}
              domain={[0, 1]}
              tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
              width={40}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: '#1F2937',
                border: '1px solid #374151',
                borderRadius: '8px',
              }}
              formatter={(value, name) => {
                const numValue = Number(value);
                if (name === 'Accuracy') return [`${(numValue * 100).toFixed(1)}%`, 'Accuracy'];
                return [numValue.toFixed(4), 'Loss'];
              }}
            />
            <Legend verticalAlign="top" height={20} />
            <Line
              yAxisId="loss"
              type="monotone"
              dataKey="loss"
              stroke="#EF4444"
              dot={false}
              strokeWidth={2}
              name="Loss"
            />
            <Line
              yAxisId="accuracy"
              type="monotone"
              dataKey="accuracy"
              stroke="#22C55E"
              dot={false}
              strokeWidth={2}
              name="Accuracy"
            />
            {/* Scrubber reference line */}
            {showScrubber && scrubEpoch !== null && (
              <ReferenceLine
                x={scrubEpoch}
                stroke="#06b6d4"
                strokeWidth={2}
                strokeDasharray="4 4"
                yAxisId="loss"
              />
            )}
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="mt-2 text-xs text-gray-500 text-center">
        Epochs: {actualTotalEpochs} | Loss: {lossHistory[lossHistory.length - 1]?.toFixed(4) ?? '-'} | Acc: {((accuracyHistory[accuracyHistory.length - 1] ?? 0) * 100).toFixed(1)}%
      </div>

      {/* Training Timeline Scrubber */}
      {showScrubber && trainingComplete && (
        <TrainingScrubber
          lossHistory={lossHistory}
          accuracyHistory={accuracyHistory}
          totalEpochs={actualTotalEpochs}
          currentEpoch={scrubEpoch ?? actualTotalEpochs}
          onScrubToEpoch={handleScrubToEpoch}
        />
      )}
    </div>
  );
});
