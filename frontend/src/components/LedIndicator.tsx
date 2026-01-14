import { memo } from 'react';

interface LedIndicatorProps {
  isOn: boolean;
  prediction?: number;
  expected?: number;
  correct?: boolean;
}

export const LedIndicator = memo(function LedIndicator({
  isOn,
  prediction,
  expected,
  correct,
}: LedIndicatorProps) {
  return (
    <div className="bg-gray-800 rounded-lg p-6">
      <h2 className="text-xl font-semibold mb-4">LED Output</h2>

      <div className="flex flex-col items-center gap-4">
        {/* LED visualization */}
        <div
          className={`
            w-24 h-24 rounded-full transition-all duration-300
            ${isOn
              ? 'bg-green-400 shadow-lg shadow-green-400/70'
              : 'bg-gray-700 border-2 border-gray-600'
            }
          `}
        />

        {/* Status text */}
        <div className="text-center">
          <div className={`text-2xl font-bold ${isOn ? 'text-green-400' : 'text-gray-500'}`}>
            {isOn ? 'ON' : 'OFF'}
          </div>

          {prediction !== undefined && (
            <div className="mt-2 text-sm">
              <span className="text-gray-400">Prediction: </span>
              <span className="font-mono">{prediction.toFixed(3)}</span>
            </div>
          )}

          {expected !== undefined && (
            <div className="text-sm">
              <span className="text-gray-400">Expected: </span>
              <span className="font-mono">{expected}</span>
              {correct !== undefined && (
                <span className={`ml-2 ${correct ? 'text-green-400' : 'text-red-400'}`}>
                  {correct ? '✓' : '✗'}
                </span>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
});
