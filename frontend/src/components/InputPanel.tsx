import { useCallback } from 'react';
import type { ProblemInfo, InputConfig } from '../types';
import { getInputConfigForProblem } from '../types';

interface InputPanelProps {
  problem: ProblemInfo | null;
  values: number[];
  onChange: (values: number[]) => void;
  disabled?: boolean;
}

export function InputPanel({ problem, values, onChange, disabled = false }: InputPanelProps) {
  if (!problem) {
    return (
      <div className="bg-gray-800 rounded-lg p-4">
        <h2 className="text-lg font-semibold mb-3">Input</h2>
        <p className="text-gray-500 text-sm">Select a problem to configure inputs</p>
      </div>
    );
  }

  const config = getInputConfigForProblem(problem);

  const handleValueChange = useCallback(
    (index: number, newValue: number) => {
      const newValues = [...values];
      newValues[index] = newValue;
      onChange(newValues);
    },
    [values, onChange]
  );

  const handleToggle = useCallback(
    (index: number) => {
      const newValues = [...values];
      newValues[index] = newValues[index] === 1 ? 0 : 1;
      onChange(newValues);
    },
    [values, onChange]
  );

  const renderBinaryInputs = () => (
    <div className="grid grid-cols-5 gap-2">
      {config.labels.map((label, i) => (
        <button
          key={i}
          onClick={() => handleToggle(i)}
          disabled={disabled}
          className={`
            flex flex-col items-center justify-center p-3 rounded-lg transition-all
            ${
              values[i] === 1
                ? 'bg-green-600 text-white shadow-lg shadow-green-500/30'
                : 'bg-gray-700 text-gray-400 hover:bg-gray-600'
            }
            ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
          `}
        >
          <span className="text-2xl font-bold">{values[i]}</span>
          <span className="text-xs mt-1">{label}</span>
        </button>
      ))}
    </div>
  );

  const renderSliderInputs = () => (
    <div className="space-y-4">
      {config.labels.map((label, i) => (
        <div key={i} className="space-y-1">
          <div className="flex justify-between text-sm">
            <span className="text-gray-400">{label}</span>
            <span className="font-mono text-cyan-400">{values[i]?.toFixed(2) || '0.00'}</span>
          </div>
          <input
            type="range"
            min={config.min || 0}
            max={config.max || 1}
            step={config.step || 0.01}
            value={values[i] || 0}
            onChange={(e) => handleValueChange(i, parseFloat(e.target.value))}
            disabled={disabled}
            className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-cyan-500 disabled:opacity-50"
          />
        </div>
      ))}
    </div>
  );

  const renderPatternInputs = () => (
    <div className="space-y-3">
      <div className="flex items-end gap-1 h-24 bg-gray-900 rounded p-2">
        {config.labels.map((label, i) => {
          const height = (values[i] || 0) * 100;
          return (
            <div key={i} className="flex-1 flex flex-col items-center">
              <div
                className="w-full bg-gradient-to-t from-cyan-600 to-cyan-400 rounded-t transition-all"
                style={{ height: `${height}%` }}
              />
              <span className="text-[8px] text-gray-500 mt-1">{label}</span>
            </div>
          );
        })}
      </div>

      <div className="grid grid-cols-4 gap-2">
        <button
          onClick={() => {
            // Tap pattern
            const pattern = new Array(8).fill(0);
            pattern[3] = 0.9;
            onChange(pattern);
          }}
          disabled={disabled}
          className="px-2 py-1 bg-gray-700 hover:bg-gray-600 rounded text-xs disabled:opacity-50"
        >
          Tap
        </button>
        <button
          onClick={() => {
            // Swipe pattern
            onChange(Array.from({ length: 8 }, (_, i) => 0.2 + (i / 7) * 0.6));
          }}
          disabled={disabled}
          className="px-2 py-1 bg-gray-700 hover:bg-gray-600 rounded text-xs disabled:opacity-50"
        >
          Swipe
        </button>
        <button
          onClick={() => {
            // Shake pattern
            onChange(
              Array.from({ length: 8 }, (_, i) => 0.5 + 0.3 * Math.sin((i / 8) * Math.PI * 4))
            );
          }}
          disabled={disabled}
          className="px-2 py-1 bg-gray-700 hover:bg-gray-600 rounded text-xs disabled:opacity-50"
        >
          Shake
        </button>
        <button
          onClick={() => onChange(new Array(8).fill(0))}
          disabled={disabled}
          className="px-2 py-1 bg-gray-700 hover:bg-gray-600 rounded text-xs disabled:opacity-50"
        >
          Clear
        </button>
      </div>

      {/* Individual sliders for fine control */}
      <div className="grid grid-cols-8 gap-1">
        {config.labels.map((_, i) => (
          <input
            key={i}
            type="range"
            min={0}
            max={1}
            step={0.1}
            value={values[i] || 0}
            onChange={(e) => handleValueChange(i, parseFloat(e.target.value))}
            disabled={disabled}
            className="w-full h-1 bg-gray-700 rounded appearance-none cursor-pointer accent-cyan-500 disabled:opacity-50"
            style={{ writingMode: 'vertical-lr', direction: 'rtl', height: '40px' }}
          />
        ))}
      </div>
    </div>
  );

  return (
    <div className="bg-gray-800 rounded-lg p-4">
      <h2 className="text-lg font-semibold mb-3">
        Input
        <span className="text-sm font-normal text-gray-500 ml-2">
          ({problem.input_labels.length} values)
        </span>
      </h2>

      {config.type === 'binary' && renderBinaryInputs()}
      {config.type === 'slider' && renderSliderInputs()}
      {config.type === 'pattern' && renderPatternInputs()}

      {disabled && (
        <p className="text-xs text-yellow-500 mt-3">Train the network first</p>
      )}
    </div>
  );
}
