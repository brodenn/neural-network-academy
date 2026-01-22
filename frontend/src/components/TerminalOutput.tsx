import { memo, useRef, useEffect } from 'react';
import type { PredictionResult } from '../types';

interface TerminalOutputProps {
  predictions: PredictionResult[];
  maxLines?: number;
}

// Helper to format prediction value (can be number or array)
function formatPrediction(value: number | number[]): string {
  if (Array.isArray(value)) {
    // For multi-class, show the max probability class
    const maxIndex = value.indexOf(Math.max(...value));
    const maxProb = value[maxIndex];
    return `Class ${maxIndex} (${(maxProb * 100).toFixed(0)}%)`;
  }
  return value.toFixed(2);
}

// Helper to format expected value
function formatExpected(value: number | number[], labels?: string[]): string {
  if (Array.isArray(value)) {
    // For one-hot encoded, find the class index
    const classIndex = value.indexOf(1);
    if (labels && labels[classIndex]) {
      return labels[classIndex];
    }
    return `Class ${classIndex}`;
  }
  return String(value);
}

export const TerminalOutput = memo(function TerminalOutput({
  predictions,
  maxLines = 10,
}: TerminalOutputProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when new predictions arrive
  useEffect(() => {
    if (containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [predictions]);

  const displayPredictions = predictions.slice(-maxLines);

  return (
    <div className="bg-gray-800 rounded-lg p-3">
      <h2 className="text-lg font-semibold mb-2">Terminal Output (G6)</h2>

      <div
        ref={containerRef}
        className="bg-black rounded p-3 font-mono text-xs h-32 overflow-y-auto"
      >
        {displayPredictions.length === 0 ? (
          <div className="text-gray-500">
            Waiting for predictions...<br />
            Click inputs after training.
          </div>
        ) : (
          displayPredictions
            .filter((pred) => pred && pred.inputs && pred.prediction !== undefined)
            .map((pred, i) => (
              <div
                key={i}
                className={`mb-1 ${pred.correct ? 'text-green-400' : 'text-red-400'}`}
              >
                <span className="text-gray-500">Input: </span>
                <span className="text-white">[{pred.inputs.join(', ')}]</span>
                <span className="text-gray-500">, Prediction: </span>
                <span className="text-yellow-400">{formatPrediction(pred.prediction)}</span>
                <span className="text-gray-500">, LED: </span>
                <span className={pred.led_state ? 'text-green-400' : 'text-gray-400'}>
                  {pred.led_state ? 'ON' : 'OFF'}
                </span>
                <span className="text-gray-500">, Expected: </span>
                <span className="text-white">{formatExpected(pred.expected ?? 0, pred.output_labels)}</span>
                <span className="ml-2">
                  {pred.correct ? '✓' : '✗'}
                </span>
              </div>
            ))
        )}
      </div>

      <div className="mt-2 text-xs text-gray-500 text-center">
        Format matches assignment Bilaga A
      </div>
    </div>
  );
});
