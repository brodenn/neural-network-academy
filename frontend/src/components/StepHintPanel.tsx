import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

interface StepHintPanelProps {
  hints: string[];
  attempts: number;
}

export const StepHintPanel = ({ hints, attempts }: StepHintPanelProps) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const [revealedHints, setRevealedHints] = useState<Set<number>>(new Set());

  // Calculate how many hints are available (1 hint per 2 attempts, starting at 0)
  const availableHints = Math.min(Math.floor(attempts / 2) + 1, hints.length);

  const revealHint = (index: number) => {
    if (index < availableHints) {
      setRevealedHints(prev => new Set([...prev, index]));
    }
  };

  if (hints.length === 0) return null;

  return (
    <div className="bg-gray-800 rounded-lg">
      {/* Header */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full px-4 py-3 flex items-center justify-between text-left hover:bg-gray-700/50 rounded-lg transition-colors"
      >
        <div className="flex items-center gap-2">
          <svg className="w-5 h-5 text-yellow-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
              d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
          </svg>
          <span className="font-medium text-gray-200">Hints</span>
          <span className="text-xs text-gray-500">
            ({availableHints}/{hints.length} available)
          </span>
        </div>
        <motion.svg
          className="w-5 h-5 text-gray-400"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
          animate={{ rotate: isExpanded ? 180 : 0 }}
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </motion.svg>
      </button>

      {/* Hints List */}
      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="px-4 pb-3 space-y-2">
              {hints.map((hint, index) => {
                const isAvailable = index < availableHints;
                const isRevealed = revealedHints.has(index);

                return (
                  <div
                    key={index}
                    className={`rounded p-3 ${
                      isAvailable
                        ? isRevealed
                          ? 'bg-yellow-900/30 border border-yellow-800/50'
                          : 'bg-gray-700/50 cursor-pointer hover:bg-gray-700'
                        : 'bg-gray-900/50 opacity-50'
                    }`}
                    onClick={() => isAvailable && !isRevealed && revealHint(index)}
                  >
                    <div className="flex items-start gap-2">
                      <span className={`text-sm font-medium ${
                        isAvailable ? 'text-yellow-400' : 'text-gray-600'
                      }`}>
                        #{index + 1}
                      </span>
                      {isAvailable ? (
                        isRevealed ? (
                          <p className="text-sm text-gray-300">{hint}</p>
                        ) : (
                          <button className="text-sm text-blue-400 hover:text-blue-300">
                            Click to reveal hint
                          </button>
                        )
                      ) : (
                        <span className="text-sm text-gray-600">
                          Train {(index + 1) * 2 - attempts} more time(s) to unlock
                        </span>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};
