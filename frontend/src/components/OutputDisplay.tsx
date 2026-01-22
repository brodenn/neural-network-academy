import type { PredictionResult, ProblemInfo } from '../types';

interface OutputDisplayProps {
  problem: ProblemInfo | null;
  prediction: PredictionResult | null;
}

export function OutputDisplay({ problem, prediction }: OutputDisplayProps) {
  if (!problem) {
    return (
      <div className="bg-gray-800 rounded-lg p-3">
        <h2 className="text-sm font-semibold mb-2">Output</h2>
        <p className="text-gray-500 text-sm">Select a problem</p>
      </div>
    );
  }

  const isMultiClass = problem.category === 'multi-class';
  const isBinary = problem.category === 'binary';

  // Handle multi-class predictions (array of probabilities)
  const renderMultiClass = () => {
    if (!prediction || !Array.isArray(prediction.prediction)) {
      return (
        <div className="space-y-2">
          {problem.output_labels.map((label, i) => (
            <div key={i} className="flex items-center gap-2">
              <span className="w-16 text-sm text-gray-400">{label}</span>
              <div className="flex-1 h-4 bg-gray-700 rounded overflow-hidden">
                <div className="h-full bg-gray-600 w-0" />
              </div>
              <span className="w-12 text-right text-sm font-mono text-gray-500">--</span>
            </div>
          ))}
        </div>
      );
    }

    const probs = prediction.prediction as number[];
    const maxIndex = probs.indexOf(Math.max(...probs));
    const expectedIndex = Array.isArray(prediction.expected)
      ? (prediction.expected as number[]).indexOf(Math.max(...(prediction.expected as number[])))
      : 0;

    return (
      <div className="space-y-2">
        {problem.output_labels.map((label, i) => {
          const prob = probs[i] || 0;
          const isPredicted = i === maxIndex;
          const isExpected = i === expectedIndex;

          return (
            <div key={i} className="flex items-center gap-2">
              <span
                className={`w-16 text-sm ${isPredicted ? 'text-cyan-400 font-semibold' : 'text-gray-400'}`}
              >
                {label}
              </span>
              <div className="flex-1 h-4 bg-gray-700 rounded overflow-hidden">
                <div
                  className={`h-full transition-all ${
                    isPredicted
                      ? 'bg-gradient-to-r from-cyan-600 to-cyan-400'
                      : 'bg-gray-600'
                  }`}
                  style={{ width: `${prob * 100}%` }}
                />
              </div>
              <span
                className={`w-12 text-right text-sm font-mono ${
                  isPredicted ? 'text-cyan-400' : 'text-gray-500'
                }`}
              >
                {(prob * 100).toFixed(0)}%
              </span>
              {isExpected && (
                <span className="text-green-400 text-xs">expected</span>
              )}
            </div>
          );
        })}

        <div className="mt-3 pt-3 border-t border-gray-700">
          <div className="flex justify-between text-sm">
            <span className="text-gray-400">Prediction:</span>
            <span className={prediction.correct ? 'text-green-400' : 'text-red-400'}>
              {problem.output_labels[maxIndex]} {prediction.correct ? '✓' : '✗'}
            </span>
          </div>
        </div>
      </div>
    );
  };

  // Handle binary/regression predictions (single value)
  const renderSingleValue = () => {
    const value = typeof prediction?.prediction === 'number' ? prediction.prediction : 0;
    const expected = typeof prediction?.expected === 'number' ? prediction.expected : 0;
    const rounded = prediction?.prediction_rounded ?? 0;
    const ledOn = prediction?.led_state ?? false;

    return (
      <div className="space-y-3">
        {/* Compact LED + Result row for binary */}
        {isBinary && (
          <div className="flex items-center justify-between bg-gray-900/50 rounded-lg p-3">
            <div className="flex items-center gap-3">
              <div
                className={`
                  w-12 h-12 rounded-full flex items-center justify-center text-xl font-bold
                  transition-all duration-300
                  ${
                    ledOn
                      ? 'bg-green-500 text-white shadow-md shadow-green-500/40'
                      : 'bg-gray-700 text-gray-500'
                  }
                `}
              >
                {rounded}
              </div>
              <div className="flex flex-col">
                <span className="text-xs text-gray-500">Output</span>
                <span className={`font-bold ${ledOn ? 'text-green-400' : 'text-gray-400'}`}>
                  {ledOn ? 'ON (1)' : 'OFF (0)'}
                </span>
              </div>
            </div>
            {prediction && (
              <div className={`px-3 py-1.5 rounded-lg text-sm font-semibold ${
                prediction.correct
                  ? 'bg-green-600/20 text-green-400 ring-1 ring-green-500/30'
                  : 'bg-red-600/20 text-red-400 ring-1 ring-red-500/30'
              }`}>
                {prediction.correct ? '✓ Correct' : '✗ Wrong'}
              </div>
            )}
          </div>
        )}

        {/* Compact value comparison */}
        <div className="bg-gray-900/50 rounded-lg p-3 space-y-2">
          <div className="grid grid-cols-2 gap-3">
            {/* Prediction */}
            <div>
              <div className="text-xs text-gray-500 mb-1">Prediction</div>
              <div className="font-mono text-lg text-cyan-400">{value.toFixed(4)}</div>
              <div className="h-2 bg-gray-700 rounded overflow-hidden mt-1">
                <div
                  className="h-full bg-gradient-to-r from-cyan-600 to-cyan-400 transition-all"
                  style={{ width: `${value * 100}%` }}
                />
              </div>
            </div>
            {/* Expected */}
            <div>
              <div className="text-xs text-gray-500 mb-1">Expected</div>
              <div className="font-mono text-lg text-gray-300">{expected.toFixed(4)}</div>
              <div className="h-2 bg-gray-700 rounded overflow-hidden mt-1">
                <div
                  className="h-full bg-gray-500 transition-all"
                  style={{ width: `${expected * 100}%` }}
                />
              </div>
            </div>
          </div>
          {/* Difference indicator */}
          <div className="flex justify-between items-center text-xs pt-2 border-t border-gray-700">
            <span className="text-gray-500">Difference</span>
            <span className={`font-mono ${Math.abs(value - expected) < 0.1 ? 'text-green-400' : 'text-yellow-400'}`}>
              {Math.abs(value - expected).toFixed(4)}
            </span>
          </div>
        </div>

        {/* Non-binary result indicator */}
        {!isBinary && prediction && (
          <div className={`flex justify-between items-center p-2 rounded-lg ${
            prediction.correct
              ? 'bg-green-600/10 border border-green-500/20'
              : 'bg-red-600/10 border border-red-500/20'
          }`}>
            <span className="text-sm text-gray-400">Result</span>
            <span
              className={`text-sm font-semibold ${
                prediction.correct ? 'text-green-400' : 'text-red-400'
              }`}
            >
              {prediction.correct ? 'Correct ✓' : 'Incorrect ✗'}
            </span>
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="bg-gray-800 rounded-lg p-3">
      <h2 className="text-sm font-semibold mb-2">
        Output
        <span className="text-sm font-normal text-gray-500 ml-2">
          ({problem.output_labels.join(', ')})
        </span>
      </h2>

      {isMultiClass ? renderMultiClass() : renderSingleValue()}
    </div>
  );
}
