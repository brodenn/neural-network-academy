import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import type { DebugOption, BrokenConfig } from '../data/challenges';

// Re-export for convenience
export type { DebugOption, BrokenConfig } from '../data/challenges';
export { DEBUG_CHALLENGES } from '../data/challenges';

export interface DebugChallengeProps {
  title: string;
  description: string;
  problem: string;
  config: BrokenConfig;
  symptoms: string[];
  options: DebugOption[];
  onSolved: (correct: boolean) => void;
  onTryFix: (fixId: string) => void;
}

export const DebugChallenge = ({
  title,
  description,
  problem,
  config,
  symptoms,
  options,
  onSolved,
  onTryFix,
}: DebugChallengeProps) => {
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [revealed, setRevealed] = useState(false);
  const [attempts, setAttempts] = useState(0);

  const selectedOption = options.find(o => o.id === selectedId);
  const correctOption = options.find(o => o.isCorrect);

  const handleSelect = (id: string) => {
    if (revealed) return;
    setSelectedId(id);
  };

  const handleCheck = () => {
    if (!selectedId) return;
    setAttempts(prev => prev + 1);

    const isCorrect = selectedOption?.isCorrect ?? false;
    if (isCorrect) {
      setRevealed(true);
      onSolved(true);
    } else {
      setSelectedId(null);
    }
  };

  const handleRevealAnswer = () => {
    setRevealed(true);
    onSolved(false);
  };

  return (
    <div className="bg-gray-800 rounded-lg border border-gray-700 overflow-hidden">
      {/* Header */}
      <div className="bg-red-900/30 border-b border-red-600/30 px-4 py-3">
        <div className="flex items-center gap-2">
          <span className="text-2xl">🔍</span>
          <div>
            <h3 className="text-lg font-semibold text-white">{title}</h3>
            <p className="text-red-300 text-sm">Debug Challenge</p>
          </div>
        </div>
      </div>

      {/* Scenario */}
      <div className="px-4 py-4 border-b border-gray-700">
        <p className="text-gray-200 mb-3">{description}</p>

        {/* Broken Config Display */}
        <div className="bg-gray-900 rounded-lg p-3 font-mono text-sm">
          <p className="text-gray-500 mb-2"># Configuration that isn't working:</p>
          <div className="space-y-1">
            <p><span className="text-purple-400">problem</span> = <span className="text-green-400">"{problem}"</span></p>
            <p><span className="text-purple-400">architecture</span> = <span className="text-yellow-400">[{config.architecture.join(', ')}]</span></p>
            <p><span className="text-purple-400">learning_rate</span> = <span className="text-cyan-400">{config.learningRate}</span></p>
            <p><span className="text-purple-400">epochs</span> = <span className="text-cyan-400">{config.epochs}</span></p>
            {config.weightInit && (
              <p><span className="text-purple-400">weight_init</span> = <span className="text-green-400">"{config.weightInit}"</span></p>
            )}
            {config.hiddenActivation && (
              <p><span className="text-purple-400">activation</span> = <span className="text-green-400">"{config.hiddenActivation}"</span></p>
            )}
          </div>
        </div>

        {/* Symptoms */}
        <div className="mt-3 p-3 bg-amber-900/20 border border-amber-600/30 rounded-lg">
          <p className="text-amber-400 text-sm font-medium mb-2">Observed symptoms:</p>
          <ul className="space-y-1">
            {symptoms.map((symptom, i) => (
              <li key={i} className="text-amber-200 text-sm flex items-center gap-2">
                <span className="text-amber-500">•</span>
                {symptom}
              </li>
            ))}
          </ul>
        </div>
      </div>

      {/* Question */}
      <div className="px-4 py-4">
        <p className="text-white font-medium mb-3">What's causing the problem?</p>

        {/* Options */}
        <div className="space-y-2">
          {options.map((option) => {
            const isSelected = selectedId === option.id;
            const isCorrect = option.isCorrect;

            let bgClass = 'bg-gray-700/50 hover:bg-gray-700 border-gray-600';
            let textClass = 'text-gray-200';

            if (revealed) {
              if (isCorrect) {
                bgClass = 'bg-green-900/50 border-green-500';
                textClass = 'text-green-300';
              } else {
                bgClass = 'bg-gray-800/50 border-gray-700';
                textClass = 'text-gray-500';
              }
            } else if (isSelected) {
              bgClass = 'bg-blue-900/50 border-blue-500';
              textClass = 'text-blue-200';
            }

            return (
              <motion.button
                key={option.id}
                onClick={() => handleSelect(option.id)}
                disabled={revealed}
                whileHover={!revealed ? { scale: 1.01 } : {}}
                whileTap={!revealed ? { scale: 0.99 } : {}}
                className={`w-full text-left px-4 py-3 rounded-lg border-2 transition-colors ${bgClass}`}
              >
                <div className="flex items-center gap-3">
                  <div className={`w-5 h-5 rounded border-2 flex items-center justify-center ${
                    isSelected ? 'border-current bg-current/20' : 'border-gray-500'
                  }`}>
                    {revealed && isCorrect && <span className="text-green-400 text-xs">✓</span>}
                    {isSelected && !revealed && <span className="text-blue-400 text-xs">●</span>}
                  </div>
                  <span className={textClass}>{option.text}</span>
                </div>
              </motion.button>
            );
          })}
        </div>

        {/* Attempts indicator */}
        {attempts > 0 && !revealed && (
          <motion.p
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="mt-2 text-amber-400 text-sm"
          >
            Attempts: {attempts} {attempts >= 2 && '- Hint: Think about what the symptoms tell you!'}
          </motion.p>
        )}

        {/* Explanation */}
        <AnimatePresence>
          {revealed && correctOption && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="mt-4"
            >
              <div className="p-4 rounded-lg border bg-green-900/30 border-green-600/50">
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-xl">💡</span>
                  <span className="text-green-400 font-medium">The Bug:</span>
                </div>
                <p className="text-gray-200 text-sm">{correctOption.explanation}</p>
              </div>

              <motion.button
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                onClick={() => onTryFix(correctOption.id)}
                className="mt-3 w-full py-2 bg-green-600 hover:bg-green-500 rounded-lg text-white font-medium"
              >
                Apply the Fix & Train
              </motion.button>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Actions */}
        {!revealed && (
          <div className="mt-4 flex gap-2">
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={handleCheck}
              disabled={!selectedId}
              className={`flex-1 py-2 rounded-lg font-medium transition-colors ${
                selectedId
                  ? 'bg-blue-600 hover:bg-blue-500 text-white'
                  : 'bg-gray-700 text-gray-500 cursor-not-allowed'
              }`}
            >
              Check Answer
            </motion.button>
            {attempts >= 3 && (
              <motion.button
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                onClick={handleRevealAnswer}
                className="px-4 py-2 bg-gray-600 hover:bg-gray-500 rounded-lg text-white font-medium"
              >
                Show Answer
              </motion.button>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default DebugChallenge;
