import { memo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import type { ProblemInfo } from '../types';

interface ProblemInfoModalProps {
  problem: ProblemInfo | null;
  isOpen: boolean;
  onClose: () => void;
}

export const ProblemInfoModal = memo(function ProblemInfoModal({
  problem,
  isOpen,
  onClose,
}: ProblemInfoModalProps) {
  if (!problem) return null;

  const difficultyStars = '★'.repeat(problem.difficulty) + '☆'.repeat(5 - problem.difficulty);
  const difficultyLabel = ['Beginner', 'Easy', 'Intermediate', 'Advanced', 'Expert'][problem.difficulty - 1];

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Backdrop */}
          <motion.div
            className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
          />

          {/* Modal */}
          <motion.div
            className="fixed inset-0 z-50 flex items-center justify-center p-4 pointer-events-none"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <motion.div
              className="bg-gray-800 rounded-xl shadow-2xl max-w-lg w-full max-h-[85vh] overflow-hidden pointer-events-auto border border-gray-700"
              initial={{ scale: 0.9, y: 20 }}
              animate={{ scale: 1, y: 0 }}
              exit={{ scale: 0.9, y: 20 }}
              transition={{ type: 'spring', damping: 25, stiffness: 300 }}
              onClick={(e) => e.stopPropagation()}
            >
              {/* Header */}
              <div className={`px-5 py-4 border-b border-gray-700 ${
                problem.is_failure_case
                  ? 'bg-gradient-to-r from-red-900/30 to-orange-900/30'
                  : 'bg-gradient-to-r from-cyan-900/30 to-blue-900/30'
              }`}>
                <div className="flex items-start justify-between">
                  <div>
                    <div className="flex items-center gap-2 mb-1">
                      <span className={`text-xs px-2 py-0.5 rounded-full ${
                        problem.is_failure_case
                          ? 'bg-red-500/20 text-red-400'
                          : 'bg-cyan-500/20 text-cyan-400'
                      }`}>
                        Level {problem.level}
                      </span>
                      <span className={`text-xs px-2 py-0.5 rounded-full ${
                        problem.category === 'binary' ? 'bg-green-500/20 text-green-400' :
                        problem.category === 'regression' ? 'bg-blue-500/20 text-blue-400' :
                        'bg-purple-500/20 text-purple-400'
                      }`}>
                        {problem.category}
                      </span>
                      {problem.is_failure_case && (
                        <span className="text-xs px-2 py-0.5 rounded-full bg-red-500/20 text-red-400">
                          ⚠ Failure Case
                        </span>
                      )}
                    </div>
                    <h2 className="text-xl font-bold text-white">{problem.name}</h2>
                  </div>
                  <button
                    onClick={onClose}
                    className="text-gray-400 hover:text-white transition-colors p-1"
                    aria-label="Close"
                  >
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </button>
                </div>
              </div>

              {/* Content */}
              <div className="px-5 py-4 overflow-y-auto max-h-[60vh] space-y-4">
                {/* Difficulty */}
                <div className="flex items-center gap-3">
                  <span className="text-yellow-400 tracking-wider">{difficultyStars}</span>
                  <span className="text-sm text-gray-400">{difficultyLabel}</span>
                </div>

                {/* Description */}
                <div>
                  <h3 className="text-sm font-semibold text-gray-300 mb-1">Description</h3>
                  <p className="text-gray-400 text-sm leading-relaxed">{problem.description}</p>
                </div>

                {/* What You'll Learn */}
                <div className="bg-cyan-900/20 rounded-lg p-3 border border-cyan-800/30">
                  <h3 className="text-sm font-semibold text-cyan-400 mb-1 flex items-center gap-2">
                    <span>🎯</span> What You'll Learn
                  </h3>
                  <p className="text-gray-300 text-sm">{problem.learning_goal}</p>
                </div>

                {/* Key Concept */}
                <div className="bg-blue-900/20 rounded-lg p-3 border border-blue-800/30">
                  <h3 className="text-sm font-semibold text-blue-400 mb-1 flex items-center gap-2">
                    <span>💡</span> Key Concept
                  </h3>
                  <p className="text-gray-300 text-sm">{problem.concept}</p>
                </div>

                {/* Failure Case Warning */}
                {problem.is_failure_case && problem.failure_reason && (
                  <div className="bg-red-900/20 rounded-lg p-3 border border-red-800/30">
                    <h3 className="text-sm font-semibold text-red-400 mb-1 flex items-center gap-2">
                      <span>⚠️</span> Why This Fails
                    </h3>
                    <p className="text-gray-300 text-sm">{problem.failure_reason}</p>
                    {problem.fix_suggestion && (
                      <div className="mt-2 pt-2 border-t border-red-800/30">
                        <h4 className="text-xs font-semibold text-green-400 mb-1">How to Fix It:</h4>
                        <p className="text-gray-300 text-sm">{problem.fix_suggestion}</p>
                      </div>
                    )}
                  </div>
                )}

                {/* Tips */}
                {problem.tips && problem.tips.length > 0 && (
                  <div>
                    <h3 className="text-sm font-semibold text-gray-300 mb-2 flex items-center gap-2">
                      <span>📝</span> Tips
                    </h3>
                    <ul className="space-y-2">
                      {problem.tips.map((tip, index) => (
                        <li key={index} className="flex items-start gap-2 text-sm text-gray-400">
                          <span className="text-cyan-500 mt-0.5">•</span>
                          <span>{tip}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {/* Architecture Info */}
                <div className="bg-gray-900/50 rounded-lg p-3 border border-gray-700/50">
                  <h3 className="text-sm font-semibold text-gray-300 mb-2">Network Architecture</h3>
                  <div className="flex flex-wrap gap-2 text-xs">
                    <span className="px-2 py-1 bg-gray-700 rounded text-gray-300">
                      {problem.network_type === 'cnn' ? 'CNN' : 'Dense'}
                    </span>
                    <span className="px-2 py-1 bg-gray-700 rounded text-gray-300">
                      Layers: [{problem.default_architecture.join(' → ')}]
                    </span>
                    <span className="px-2 py-1 bg-gray-700 rounded text-gray-300">
                      Output: {problem.output_activation}
                    </span>
                    {problem.locked_architecture && (
                      <span className="px-2 py-1 bg-yellow-900/50 rounded text-yellow-400">
                        🔒 Architecture Locked
                      </span>
                    )}
                  </div>
                </div>

                {/* Input/Output Labels */}
                <div className="grid grid-cols-2 gap-3 text-xs">
                  <div className="bg-gray-900/50 rounded-lg p-2 border border-gray-700/50">
                    <h4 className="text-gray-500 mb-1">Inputs</h4>
                    <div className="flex flex-wrap gap-1">
                      {problem.input_labels.map((label, i) => (
                        <span key={i} className="px-1.5 py-0.5 bg-gray-700 rounded text-gray-300">
                          {label}
                        </span>
                      ))}
                    </div>
                  </div>
                  <div className="bg-gray-900/50 rounded-lg p-2 border border-gray-700/50">
                    <h4 className="text-gray-500 mb-1">Outputs</h4>
                    <div className="flex flex-wrap gap-1">
                      {problem.output_labels.map((label, i) => (
                        <span key={i} className="px-1.5 py-0.5 bg-gray-700 rounded text-gray-300">
                          {label}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              </div>

              {/* Footer */}
              <div className="px-5 py-3 border-t border-gray-700 bg-gray-900/50">
                <button
                  onClick={onClose}
                  className="w-full py-2 bg-cyan-600 hover:bg-cyan-500 text-white rounded-lg font-medium transition-colors"
                >
                  Got it!
                </button>
              </div>
            </motion.div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
});
