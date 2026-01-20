import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import type { ProblemInfo } from '../types';

interface ProblemSelectorProps {
  problems: ProblemInfo[];
  currentProblem: ProblemInfo | null;
  onSelect: (problemId: string) => void;
  disabled?: boolean;
}

// Group problems by difficulty level
function groupProblemsByLevel(problems: ProblemInfo[]): Map<string, ProblemInfo[]> {
  const levels = new Map<string, ProblemInfo[]>();
  const levelNames: Record<number, string> = {
    1: 'Level 1: Single Neuron',
    2: 'Level 2: Hidden Layers',
    3: 'Level 3: Decision Boundaries',
    4: 'Level 4: Advanced',
    5: 'Level 5: Expert',
  };

  problems.forEach((p) => {
    // CNN problems get their own category
    const levelKey = p.network_type === 'cnn'
      ? 'Level 6: CNN (Images)'
      : levelNames[p.difficulty] || `Level ${p.difficulty}`;

    if (!levels.has(levelKey)) {
      levels.set(levelKey, []);
    }
    levels.get(levelKey)!.push(p);
  });

  return levels;
}

export function ProblemSelector({
  problems,
  currentProblem,
  onSelect,
  disabled = false,
}: ProblemSelectorProps) {
  const [showTips, setShowTips] = useState(false);
  const [expandedLevel, setExpandedLevel] = useState<string | null>(null);

  const getCategoryColor = (category: string) => {
    switch (category) {
      case 'binary':
        return 'bg-blue-500/20 text-blue-400 border-blue-500/30';
      case 'regression':
        return 'bg-green-500/20 text-green-400 border-green-500/30';
      case 'multi-class':
        return 'bg-purple-500/20 text-purple-400 border-purple-500/30';
      default:
        return 'bg-gray-500/20 text-gray-400 border-gray-500/30';
    }
  };

  const getDifficultyColor = (difficulty: number) => {
    switch (difficulty) {
      case 1: return 'text-green-400';
      case 2: return 'text-lime-400';
      case 3: return 'text-yellow-400';
      case 4: return 'text-orange-400';
      case 5: return 'text-red-400';
      default: return 'text-gray-400';
    }
  };

  const groupedProblems = groupProblemsByLevel(problems);

  return (
    <div className="bg-gray-800 rounded-lg p-4">
      <h2 className="text-lg font-semibold mb-3 flex items-center gap-2">
        <span>📚</span> Learning Problems
      </h2>

      {/* Grouped problem selection */}
      <div className="space-y-2 mb-4">
        {Array.from(groupedProblems.entries()).map(([level, levelProblems]) => (
          <div key={level} className="bg-gray-700/50 rounded-lg overflow-hidden">
            <button
              onClick={() => setExpandedLevel(expandedLevel === level ? null : level)}
              className="w-full px-3 py-2 flex items-center justify-between text-left hover:bg-gray-700/80 transition-colors"
            >
              <span className="text-sm font-medium text-gray-300">{level}</span>
              <div className="flex items-center gap-2">
                <span className="text-xs text-gray-500">{levelProblems.length} problems</span>
                <span className="text-gray-500 text-xs">
                  {expandedLevel === level ? '▼' : '▶'}
                </span>
              </div>
            </button>

            <AnimatePresence>
              {expandedLevel === level && (
                <motion.div
                  initial={{ height: 0, opacity: 0 }}
                  animate={{ height: 'auto', opacity: 1 }}
                  exit={{ height: 0, opacity: 0 }}
                  transition={{ duration: 0.2 }}
                  className="border-t border-gray-600"
                >
                  {levelProblems.map((p) => (
                    <button
                      key={p.id}
                      onClick={() => !disabled && onSelect(p.id)}
                      disabled={disabled}
                      className={`w-full px-3 py-2 text-left transition-colors flex items-center justify-between ${
                        currentProblem?.id === p.id
                          ? 'bg-blue-600/30 border-l-2 border-blue-500'
                          : 'hover:bg-gray-600/50 border-l-2 border-transparent'
                      } ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
                    >
                      <div className="flex items-center gap-2">
                        <span className="text-sm text-white">{p.name}</span>
                        <span className={`text-xs px-1.5 py-0.5 rounded border ${getCategoryColor(p.category)}`}>
                          {p.category}
                        </span>
                      </div>
                      <span className={`text-xs ${getDifficultyColor(p.difficulty)}`}>
                        {'★'.repeat(p.difficulty)}
                      </span>
                    </button>
                  ))}
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        ))}
      </div>

      {/* Quick dropdown for mobile/compact view */}
      <div className="mb-4 sm:hidden">
        <select
          value={currentProblem?.id || ''}
          onChange={(e) => onSelect(e.target.value)}
          disabled={disabled}
          className="w-full bg-gray-700 text-white rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
        >
          {problems.map((p) => (
            <option key={p.id} value={p.id}>
              {p.name} {'★'.repeat(p.difficulty)}
            </option>
          ))}
        </select>
      </div>

      {/* Current problem details */}
      {currentProblem && (
        <div className="space-y-3 border-t border-gray-700 pt-3">
          {/* Header with difficulty */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span
                className={`px-2 py-0.5 rounded text-xs font-medium border ${getCategoryColor(
                  currentProblem.category
                )}`}
              >
                {currentProblem.category}
              </span>
              <span className="text-xs text-gray-500">
                {currentProblem.output_activation}
              </span>
            </div>
            <div className={`text-sm ${getDifficultyColor(currentProblem.difficulty)}`}>
              {'★'.repeat(currentProblem.difficulty)}
              <span className="text-gray-500 ml-1">
                ({['Beginner', 'Easy', 'Medium', 'Hard', 'Expert'][currentProblem.difficulty - 1]})
              </span>
            </div>
          </div>

          {/* Concept badge */}
          {currentProblem.concept && (
            <div className="flex items-center gap-2">
              <span className="text-xs text-gray-500">Teaches:</span>
              <span className="px-2 py-0.5 bg-cyan-500/20 text-cyan-400 rounded text-xs font-medium">
                {currentProblem.concept}
              </span>
            </div>
          )}

          {/* Description */}
          <p className="text-sm text-gray-400">{currentProblem.description}</p>

          {/* Learning goal */}
          {currentProblem.learning_goal && (
            <div className="bg-gray-700/50 rounded-lg p-3 border-l-2 border-yellow-500/50">
              <div className="flex items-center gap-2 mb-1">
                <span>🎯</span>
                <span className="text-xs font-medium text-yellow-400">Learning Goal</span>
              </div>
              <p className="text-xs text-gray-300">{currentProblem.learning_goal}</p>
            </div>
          )}

          {/* Tips toggle */}
          {currentProblem.tips && currentProblem.tips.length > 0 && (
            <div>
              <button
                onClick={() => setShowTips(!showTips)}
                className="flex items-center gap-2 text-xs text-gray-400 hover:text-white transition-colors"
              >
                <span>{showTips ? '▼' : '▶'}</span>
                <span>💡 Tips ({currentProblem.tips.length})</span>
              </button>

              <AnimatePresence>
                {showTips && (
                  <motion.ul
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: 'auto', opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    className="mt-2 space-y-1 pl-4 overflow-hidden"
                  >
                    {currentProblem.tips.map((tip, i) => (
                      <motion.li
                        key={i}
                        initial={{ x: -10, opacity: 0 }}
                        animate={{ x: 0, opacity: 1 }}
                        transition={{ delay: i * 0.1 }}
                        className="text-xs text-gray-400 flex items-start gap-2"
                      >
                        <span className="text-green-400">•</span>
                        <span>{tip}</span>
                      </motion.li>
                    ))}
                  </motion.ul>
                )}
              </AnimatePresence>
            </div>
          )}

          {/* Architecture details (collapsible) */}
          <details className="text-xs text-gray-500">
            <summary className="cursor-pointer hover:text-gray-400">
              Technical Details
            </summary>
            <div className="mt-2 space-y-1 pl-4">
              <div>
                <span className="text-gray-400">Architecture:</span>{' '}
                <code className="bg-gray-700 px-1 rounded">
                  {currentProblem.default_architecture.join(' → ')}
                </code>
              </div>
              <div>
                <span className="text-gray-400">Inputs:</span>{' '}
                {currentProblem.input_labels.length <= 3
                  ? currentProblem.input_labels.join(', ')
                  : `${currentProblem.input_labels.length} inputs`}
              </div>
              <div>
                <span className="text-gray-400">Outputs:</span>{' '}
                {currentProblem.output_labels.join(', ')}
              </div>
            </div>
          </details>
        </div>
      )}

      {disabled && (
        <p className="text-xs text-yellow-500 mt-3">
          Cannot change problem during training
        </p>
      )}
    </div>
  );
}
