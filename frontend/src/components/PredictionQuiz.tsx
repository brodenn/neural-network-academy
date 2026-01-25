import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import type { PredictionOption } from '../data/challenges';

// Re-export for convenience
export type { PredictionOption } from '../data/challenges';
export { PREDICTION_QUIZZES } from '../data/challenges';
export type { QuizId } from '../data/challenges';

export interface PredictionQuizProps {
  question: string;
  context: {
    architecture: number[];
    learningRate: number;
    epochs: number;
    problem: string;
  };
  options: PredictionOption[];
  onAnswer: (correct: boolean, selectedId: string) => void;
  onRevealAndTrain: () => void;
}

// =============================================================================
// COMPONENT
// =============================================================================

export const PredictionQuiz = ({
  question,
  context,
  options,
  onAnswer,
  onRevealAndTrain,
}: PredictionQuizProps) => {
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [revealed, setRevealed] = useState(false);
  const [showExplanation, setShowExplanation] = useState(false);

  const selectedOption = options.find(o => o.id === selectedId);
  const correctOption = options.find(o => o.isCorrect);

  const handleSelect = (id: string) => {
    if (revealed) return;
    setSelectedId(id);
  };

  const handleReveal = () => {
    if (!selectedId) return;
    setRevealed(true);
    setShowExplanation(true);
    onAnswer(selectedOption?.isCorrect ?? false, selectedId);
  };

  const handleTrainAndSee = () => {
    onRevealAndTrain();
  };

  return (
    <div className="bg-gray-800 rounded-lg border border-gray-700 overflow-hidden">
      {/* Header */}
      <div className="bg-purple-900/30 border-b border-purple-600/30 px-4 py-3">
        <div className="flex items-center gap-2">
          <span className="text-2xl">🤔</span>
          <h3 className="text-lg font-semibold text-white">Predict the Outcome</h3>
        </div>
      </div>

      {/* Context */}
      <div className="px-4 py-3 bg-gray-900/50 border-b border-gray-700">
        <p className="text-sm text-gray-400 mb-2">Configuration:</p>
        <div className="flex flex-wrap gap-2 text-sm">
          <span className="px-2 py-1 bg-blue-900/50 rounded text-blue-300">
            Architecture: [{context.architecture.join(', ')}]
          </span>
          <span className="px-2 py-1 bg-amber-900/50 rounded text-amber-300">
            LR: {context.learningRate}
          </span>
          <span className="px-2 py-1 bg-green-900/50 rounded text-green-300">
            Epochs: {context.epochs}
          </span>
          <span className="px-2 py-1 bg-purple-900/50 rounded text-purple-300">
            Problem: {context.problem}
          </span>
        </div>
      </div>

      {/* Question */}
      <div className="px-4 py-4">
        <p className="text-white text-lg mb-4">{question}</p>

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
              } else if (isSelected && !isCorrect) {
                bgClass = 'bg-red-900/50 border-red-500';
                textClass = 'text-red-300';
              } else {
                bgClass = 'bg-gray-800/50 border-gray-700';
                textClass = 'text-gray-500';
              }
            } else if (isSelected) {
              bgClass = 'bg-purple-900/50 border-purple-500';
              textClass = 'text-purple-200';
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
                  <div className={`w-6 h-6 rounded-full border-2 flex items-center justify-center ${
                    isSelected ? 'border-current bg-current/20' : 'border-gray-500'
                  }`}>
                    {revealed && isCorrect && <span className="text-green-400">✓</span>}
                    {revealed && isSelected && !isCorrect && <span className="text-red-400">✗</span>}
                    {!revealed && isSelected && <span className="text-purple-400">●</span>}
                  </div>
                  <span className={textClass}>{option.text}</span>
                </div>
              </motion.button>
            );
          })}
        </div>

        {/* Explanation */}
        <AnimatePresence>
          {showExplanation && selectedOption && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="mt-4"
            >
              <div className={`p-4 rounded-lg border ${
                selectedOption.isCorrect
                  ? 'bg-green-900/30 border-green-600/50'
                  : 'bg-amber-900/30 border-amber-600/50'
              }`}>
                <div className="flex items-center gap-2 mb-2">
                  {selectedOption.isCorrect ? (
                    <>
                      <span className="text-xl">🎉</span>
                      <span className="text-green-400 font-medium">Correct!</span>
                    </>
                  ) : (
                    <>
                      <span className="text-xl">💡</span>
                      <span className="text-amber-400 font-medium">Not quite!</span>
                    </>
                  )}
                </div>
                <p className="text-gray-200 text-sm">{selectedOption.explanation}</p>
                {!selectedOption.isCorrect && correctOption && (
                  <p className="text-green-300 text-sm mt-2">
                    <strong>The correct answer:</strong> {correctOption.text}
                  </p>
                )}
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Actions */}
        <div className="mt-4 flex gap-2">
          {!revealed ? (
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={handleReveal}
              disabled={!selectedId}
              className={`flex-1 py-2 rounded-lg font-medium transition-colors ${
                selectedId
                  ? 'bg-purple-600 hover:bg-purple-500 text-white'
                  : 'bg-gray-700 text-gray-500 cursor-not-allowed'
              }`}
            >
              Check My Prediction
            </motion.button>
          ) : (
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={handleTrainAndSee}
              className="flex-1 py-2 bg-green-600 hover:bg-green-500 rounded-lg text-white font-medium"
            >
              Train & See It Happen
            </motion.button>
          )}
        </div>
      </div>
    </div>
  );
};

export default PredictionQuiz;
