import { useState } from 'react';

interface ShortcutInfo {
  key: string;
  description: string;
  available: boolean;
}

interface KeyboardShortcutsProps {
  trainingInProgress: boolean;
  trainingComplete: boolean;
  hasStep: boolean;
}

export function KeyboardShortcuts({
  trainingInProgress,
  trainingComplete,
  hasStep,
}: KeyboardShortcutsProps) {
  const [isOpen, setIsOpen] = useState(false);

  const shortcuts: ShortcutInfo[] = [
    {
      key: 'Space',
      description: trainingInProgress ? 'Stop training' : 'Start training',
      available: true,
    },
    {
      key: 'R',
      description: 'Reset network',
      available: !trainingInProgress,
    },
    {
      key: 'S',
      description: 'Step (single epoch)',
      available: hasStep && !trainingInProgress && trainingComplete,
    },
  ];

  return (
    <div className="relative">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="text-xs text-gray-400 hover:text-gray-200 px-2 py-1 rounded hover:bg-gray-700 transition-colors"
        title="Keyboard shortcuts"
      >
        <span className="font-mono">?</span>
      </button>

      {isOpen && (
        <>
          {/* Backdrop */}
          <div
            className="fixed inset-0 z-40"
            onClick={() => setIsOpen(false)}
          />

          {/* Popup */}
          <div className="absolute right-0 top-full mt-1 w-56 bg-gray-800 border border-gray-600 rounded-lg shadow-xl z-50 p-3">
            <div className="text-xs font-medium text-gray-300 mb-2">
              Keyboard Shortcuts
            </div>
            <div className="space-y-1">
              {shortcuts.map(({ key, description, available }) => (
                <div
                  key={key}
                  className={`flex items-center justify-between text-xs ${
                    available ? 'text-gray-200' : 'text-gray-500'
                  }`}
                >
                  <span
                    className={`font-mono px-1.5 py-0.5 rounded ${
                      available
                        ? 'bg-gray-700 text-cyan-400'
                        : 'bg-gray-800 text-gray-600'
                    }`}
                  >
                    {key}
                  </span>
                  <span className="ml-2">{description}</span>
                </div>
              ))}
            </div>
          </div>
        </>
      )}
    </div>
  );
}
