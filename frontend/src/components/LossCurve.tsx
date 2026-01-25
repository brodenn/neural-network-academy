import { memo, useMemo, useState, useCallback } from 'react';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  ReferenceLine,
  ReferenceArea,
} from 'recharts';
import { TrainingScrubber } from './TrainingScrubber';

// =============================================================================
// TYPES
// =============================================================================

interface LossCurveProps {
  lossHistory: number[];
  accuracyHistory: number[];
  totalEpochs?: number;
  trainingComplete?: boolean;
  targetAccuracy?: number;
  isFailureCase?: boolean;
  onScrubToEpoch?: (epoch: number) => void;
}

interface DataPoint {
  epoch: number;
  loss: number;
  accuracy: number;
  insight?: string;
}

// =============================================================================
// INSIGHT DETECTION
// =============================================================================

const detectPlateaus = (data: DataPoint[]): { start: number; end: number }[] => {
  const plateaus: { start: number; end: number }[] = [];
  if (data.length < 20) return plateaus;

  let plateauStart: number | null = null;
  const threshold = 0.001;

  for (let i = 5; i < data.length; i++) {
    const recentLosses = data.slice(Math.max(0, i - 5), i).map(d => d.loss);
    const avgLoss = recentLosses.reduce((a, b) => a + b, 0) / recentLosses.length;
    const variance = recentLosses.reduce((a, b) => a + Math.abs(b - avgLoss), 0) / recentLosses.length;

    if (variance < threshold) {
      if (plateauStart === null) plateauStart = data[i - 5].epoch;
    } else {
      if (plateauStart !== null) {
        plateaus.push({ start: plateauStart, end: data[i - 1].epoch });
        plateauStart = null;
      }
    }
  }

  if (plateauStart !== null) {
    plateaus.push({ start: plateauStart, end: data[data.length - 1].epoch });
  }

  return plateaus.filter(p => p.end - p.start >= 10); // Only significant plateaus
};

const detectSpikes = (data: DataPoint[]): number[] => {
  const spikes: number[] = [];
  if (data.length < 5) return spikes;

  for (let i = 1; i < data.length; i++) {
    const prevLoss = data[i - 1].loss;
    const currLoss = data[i].loss;
    if (prevLoss > 0 && currLoss > prevLoss * 2) {
      spikes.push(data[i].epoch);
    }
  }

  return spikes;
};

// =============================================================================
// CUSTOM TOOLTIP
// =============================================================================

interface CustomTooltipProps {
  active?: boolean;
  payload?: Array<{ value: number; name: string; payload: DataPoint }>;
  label?: string | number;
  targetAccuracy: number;
}

const CustomTooltip = ({ active, payload, label, targetAccuracy }: CustomTooltipProps) => {
  if (!active || !payload || payload.length === 0) return null;

  const data = payload[0].payload;
  const accuracy = data.accuracy;
  const loss = data.loss;

  // Generate contextual insight
  let insight = '';
  let insightColor = 'text-gray-400';

  if (loss > 1) {
    insight = 'High loss - network is still learning basic patterns';
    insightColor = 'text-amber-400';
  } else if (loss > 0.5) {
    insight = 'Loss decreasing - features emerging';
    insightColor = 'text-blue-400';
  } else if (accuracy >= targetAccuracy) {
    insight = 'Target reached! Network has learned the pattern';
    insightColor = 'text-green-400';
  } else if (accuracy >= 0.9) {
    insight = 'Almost there - fine-tuning in progress';
    insightColor = 'text-cyan-400';
  } else if (accuracy >= 0.75) {
    insight = 'Good progress - network finding the solution';
    insightColor = 'text-blue-400';
  } else if (accuracy <= 0.55 && data.epoch > 50) {
    insight = 'Stuck at random chance - architecture issue?';
    insightColor = 'text-red-400';
  }

  return (
    <div className="bg-gray-900 border border-gray-600 rounded-lg p-3 shadow-xl">
      <p className="text-white font-medium mb-2">Epoch {label}</p>
      <div className="space-y-1 text-sm">
        <p className="flex items-center gap-2">
          <span className="w-3 h-3 rounded-full bg-red-500" />
          <span className="text-gray-400">Loss:</span>
          <span className="text-white font-mono">{loss.toFixed(4)}</span>
        </p>
        <p className="flex items-center gap-2">
          <span className="w-3 h-3 rounded-full bg-green-500" />
          <span className="text-gray-400">Accuracy:</span>
          <span className="text-white font-mono">{(accuracy * 100).toFixed(1)}%</span>
        </p>
      </div>
      {insight && (
        <p className={`mt-2 text-xs ${insightColor} border-t border-gray-700 pt-2`}>
          💡 {insight}
        </p>
      )}
    </div>
  );
};

// =============================================================================
// COMPONENT
// =============================================================================

export const LossCurve = memo(function LossCurve({
  lossHistory,
  accuracyHistory,
  totalEpochs,
  trainingComplete = false,
  targetAccuracy = 0.95,
  isFailureCase = false,
  onScrubToEpoch,
}: LossCurveProps) {
  const [scrubEpoch, setScrubEpoch] = useState<number | null>(null);
  const [showScrubber, setShowScrubber] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);

  // Transform data
  const data = useMemo(() => {
    if (lossHistory.length === 0) return [];

    const actualTotal = totalEpochs || lossHistory.length;
    const step = actualTotal / lossHistory.length;

    return lossHistory.map((loss, i) => ({
      epoch: Math.round(i * step),
      loss,
      accuracy: accuracyHistory[i] ?? 0,
    }));
  }, [lossHistory, accuracyHistory, totalEpochs]);

  // Detect patterns
  const plateaus = useMemo(() => detectPlateaus(data), [data]);
  const spikes = useMemo(() => detectSpikes(data), [data]);

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
  const finalAccuracy = accuracyHistory[accuracyHistory.length - 1] ?? 0;
  const finalLoss = lossHistory[lossHistory.length - 1] ?? 0;

  // Determine status
  const getStatus = () => {
    if (isFailureCase && finalAccuracy < 0.6) {
      return { label: 'Expected Failure', color: 'text-amber-400', icon: '📚' };
    }
    if (finalAccuracy >= targetAccuracy) {
      return { label: 'Target Reached!', color: 'text-green-400', icon: '🎯' };
    }
    if (plateaus.length > 0 && !trainingComplete) {
      return { label: 'Plateau Detected', color: 'text-amber-400', icon: '⏸️' };
    }
    if (spikes.length > 0) {
      return { label: 'Unstable', color: 'text-red-400', icon: '⚠️' };
    }
    if (trainingComplete) {
      return { label: 'Complete', color: 'text-blue-400', icon: '✓' };
    }
    return { label: 'Training...', color: 'text-cyan-400', icon: '●' };
  };

  const status = getStatus();

  return (
    <div className="bg-gray-800 rounded-lg p-3">
      {/* Header */}
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <h2 className="text-lg font-semibold">Training Progress</h2>
          <span className={`text-xs px-2 py-0.5 rounded ${status.color} bg-gray-700`}>
            {status.icon} {status.label}
          </span>
        </div>
        <div className="flex gap-1">
          {trainingComplete && lossHistory.length > 10 && (
            <button
              onClick={() => setShowScrubber(!showScrubber)}
              className={`px-2 py-1 text-xs rounded transition-colors ${
                showScrubber ? 'bg-cyan-600 text-white' : 'bg-gray-700 text-gray-400 hover:text-gray-300'
              }`}
            >
              ⏱
            </button>
          )}
          <button
            onClick={() => setShowAdvanced(!showAdvanced)}
            className={`px-2 py-1 text-xs rounded transition-colors ${
              showAdvanced ? 'bg-purple-600 text-white' : 'bg-gray-700 text-gray-400 hover:text-gray-300'
            }`}
            title="Show annotations"
          >
            📊
          </button>
        </div>
      </div>

      {/* Chart */}
      <div className="h-40">
        <ResponsiveContainer width="100%" height="100%" minWidth={100} minHeight={100}>
          <AreaChart data={data} margin={{ top: 5, right: 5, bottom: 5, left: 0 }}>
            <defs>
              <linearGradient id="lossGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#EF4444" stopOpacity={0.3} />
                <stop offset="95%" stopColor="#EF4444" stopOpacity={0} />
              </linearGradient>
              <linearGradient id="accuracyGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#22C55E" stopOpacity={0.3} />
                <stop offset="95%" stopColor="#22C55E" stopOpacity={0} />
              </linearGradient>
            </defs>

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
              width={35}
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

            <Tooltip content={<CustomTooltip targetAccuracy={targetAccuracy} />} />

            <Legend verticalAlign="top" height={20} />

            {/* Target accuracy line */}
            {showAdvanced && (
              <ReferenceLine
                yAxisId="accuracy"
                y={targetAccuracy}
                stroke="#22C55E"
                strokeDasharray="5 5"
                strokeOpacity={0.7}
                label={{
                  value: `Target: ${(targetAccuracy * 100).toFixed(0)}%`,
                  fill: '#22C55E',
                  fontSize: 10,
                  position: 'right',
                }}
              />
            )}

            {/* Plateau regions */}
            {showAdvanced && plateaus.map((plateau, i) => (
              <ReferenceArea
                key={`plateau-${i}`}
                yAxisId="loss"
                x1={plateau.start}
                x2={plateau.end}
                fill="#F59E0B"
                fillOpacity={0.1}
                stroke="#F59E0B"
                strokeOpacity={0.3}
              />
            ))}

            {/* Spike markers */}
            {showAdvanced && spikes.map((spike, i) => (
              <ReferenceLine
                key={`spike-${i}`}
                yAxisId="loss"
                x={spike}
                stroke="#EF4444"
                strokeWidth={2}
                strokeDasharray="2 2"
              />
            ))}

            {/* Scrubber line */}
            {showScrubber && scrubEpoch !== null && (
              <ReferenceLine
                x={scrubEpoch}
                stroke="#06b6d4"
                strokeWidth={2}
                strokeDasharray="4 4"
                yAxisId="loss"
              />
            )}

            {/* Loss area */}
            <Area
              yAxisId="loss"
              type="monotone"
              dataKey="loss"
              stroke="#EF4444"
              fill="url(#lossGradient)"
              strokeWidth={2}
              name="Loss"
              isAnimationActive={!trainingComplete}
              animationDuration={300}
            />

            {/* Accuracy area */}
            <Area
              yAxisId="accuracy"
              type="monotone"
              dataKey="accuracy"
              stroke="#22C55E"
              fill="url(#accuracyGradient)"
              strokeWidth={2}
              name="Accuracy"
              isAnimationActive={!trainingComplete}
              animationDuration={300}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {/* Stats bar */}
      <div className="mt-2 flex items-center justify-between text-xs text-gray-400">
        <span>Epochs: {actualTotalEpochs}</span>
        <span className="text-red-400">Loss: {finalLoss.toFixed(4)}</span>
        <span className="text-green-400">Acc: {(finalAccuracy * 100).toFixed(1)}%</span>
      </div>

      {/* Pattern annotations */}
      {showAdvanced && (plateaus.length > 0 || spikes.length > 0) && (
        <div className="mt-2 flex flex-wrap gap-2 text-xs">
          {plateaus.length > 0 && (
            <span className="px-2 py-1 bg-amber-900/30 border border-amber-600/30 rounded text-amber-400">
              ⏸️ {plateaus.length} plateau{plateaus.length > 1 ? 's' : ''} detected
            </span>
          )}
          {spikes.length > 0 && (
            <span className="px-2 py-1 bg-red-900/30 border border-red-600/30 rounded text-red-400">
              ⚠️ {spikes.length} spike{spikes.length > 1 ? 's' : ''} (high LR?)
            </span>
          )}
        </div>
      )}

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

export default LossCurve;
