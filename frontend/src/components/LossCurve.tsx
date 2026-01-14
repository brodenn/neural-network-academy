import { memo, useMemo } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';

interface LossCurveProps {
  lossHistory: number[];
  accuracyHistory: number[];
  totalEpochs?: number;
}

export const LossCurve = memo(function LossCurve({
  lossHistory,
  accuracyHistory,
  totalEpochs,
}: LossCurveProps) {
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

  if (data.length === 0) {
    return (
      <div className="bg-gray-800 rounded-lg p-6">
        <h2 className="text-xl font-semibold mb-4">Training Progress</h2>
        <div className="h-64 flex items-center justify-center text-gray-500">
          No training data yet. Start training to see the loss curve.
        </div>
      </div>
    );
  }

  return (
    <div className="bg-gray-800 rounded-lg p-6">
      <h2 className="text-xl font-semibold mb-4">Training Progress</h2>

      <div className="h-64 min-h-[256px]">
        <ResponsiveContainer width="100%" height="100%" minWidth={100} minHeight={100}>
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis
              dataKey="epoch"
              stroke="#9CA3AF"
              tick={{ fill: '#9CA3AF' }}
              label={{ value: 'Epoch', position: 'bottom', fill: '#9CA3AF' }}
            />
            <YAxis
              yAxisId="loss"
              stroke="#EF4444"
              tick={{ fill: '#9CA3AF' }}
              domain={[0, 'auto']}
            />
            <YAxis
              yAxisId="accuracy"
              orientation="right"
              stroke="#22C55E"
              tick={{ fill: '#9CA3AF' }}
              domain={[0, 1]}
              tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
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
            <Legend />
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
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="mt-4 text-sm text-gray-400 text-center">
        Epochs: {totalEpochs || lossHistory.length} |
        Final Loss: {lossHistory[lossHistory.length - 1]?.toFixed(4) ?? '-'} |
        Final Accuracy: {((accuracyHistory[accuracyHistory.length - 1] ?? 0) * 100).toFixed(1)}%
      </div>
    </div>
  );
});
