import { memo } from 'react';

interface ButtonSimulatorProps {
  buttons: number[];
  onToggle: (index: number) => void;
  disabled?: boolean;
}

export const ButtonSimulator = memo(function ButtonSimulator({
  buttons,
  onToggle,
  disabled = false,
}: ButtonSimulatorProps) {
  return (
    <div className="bg-gray-800 rounded-lg p-6">
      <h2 className="text-xl font-semibold mb-4">GPIO Buttons (Input)</h2>
      <p className="text-gray-400 text-sm mb-4">
        Click buttons to toggle. XOR rule: LED ON if odd number of buttons pressed.
      </p>

      <div className="flex gap-4 justify-center mb-4">
        {buttons.map((state, index) => (
          <button
            key={index}
            onClick={() => onToggle(index)}
            disabled={disabled}
            className={`
              w-16 h-16 rounded-lg font-bold text-lg transition-all duration-200
              ${state === 1
                ? 'bg-blue-500 text-white shadow-lg shadow-blue-500/50'
                : 'bg-gray-700 text-gray-400 hover:bg-gray-600'
              }
              ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
            `}
          >
            {index + 1}
          </button>
        ))}
      </div>

      <div className="text-center">
        <span className="text-gray-400">Input: </span>
        <span className="font-mono text-lg">
          [{buttons.join(', ')}]
        </span>
        <span className="text-gray-500 ml-2">
          ({buttons.filter(b => b === 1).length} pressed)
        </span>
      </div>
    </div>
  );
});
