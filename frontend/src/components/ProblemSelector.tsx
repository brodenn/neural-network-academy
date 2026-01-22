import { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import type { ProblemInfo } from '../types';

interface ProblemSelectorProps {
  problems: ProblemInfo[];
  currentProblem: ProblemInfo | null;
  onSelect: (problemId: string) => void;
  disabled?: boolean;
}

// Level names for the curriculum
const LEVEL_NAMES: Record<number, string> = {
  1: 'Level 1: Single Neuron',
  2: 'Level 2: Hidden Layers',
  3: 'Level 3: Decision Boundaries',
  4: 'Level 4: Regression',
  5: 'Level 5: Failure Cases',
  6: 'Level 6: Multi-Class',
  7: 'Level 7: CNN (Images)',
};

// Group problems by level number
function groupProblemsByLevel(problems: ProblemInfo[]): Map<string, ProblemInfo[]> {
  const levels = new Map<string, ProblemInfo[]>();

  // Sort problems by level
  const sortedProblems = [...problems].sort((a, b) => a.level - b.level);

  sortedProblems.forEach((p) => {
    const levelKey = LEVEL_NAMES[p.level] || `Level ${p.level}`;

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
  const [isOpen, setIsOpen] = useState(false);
  const [expandedLevel, setExpandedLevel] = useState<string | null>(null);
  const menuRef = useRef<HTMLDivElement>(null);

  // Close menu when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // Close menu on escape key
  useEffect(() => {
    const handleEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setIsOpen(false);
      }
    };

    document.addEventListener('keydown', handleEscape);
    return () => document.removeEventListener('keydown', handleEscape);
  }, []);

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

  const getDifficultyStars = (difficulty: number) => {
    return '★'.repeat(difficulty);
  };

  const groupedProblems = groupProblemsByLevel(problems);

  const handleSelect = (problemId: string) => {
    if (!disabled) {
      onSelect(problemId);
      setIsOpen(false);
    }
  };

  return (
    <div className="relative" ref={menuRef}>
      {/* Dropdown Button */}
      <button
        onClick={() => !disabled && setIsOpen(!isOpen)}
        disabled={disabled}
        className={`flex items-center gap-2 px-3 py-1.5 rounded-lg border transition-colors ${
          disabled
            ? 'bg-gray-700 border-gray-600 text-gray-400 cursor-not-allowed'
            : isOpen
            ? 'bg-gray-700 border-cyan-500 text-white'
            : 'bg-gray-800 border-gray-600 text-gray-200 hover:border-gray-500'
        }`}
      >
        <span className="text-sm font-medium">
          {currentProblem?.name || 'Select Problem'}
        </span>
        {currentProblem?.is_failure_case && (
          <span className="text-red-400 text-xs">!</span>
        )}
        <svg
          className={`w-4 h-4 transition-transform ${isOpen ? 'rotate-180' : ''}`}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {/* Dropdown Menu */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.15 }}
            className="absolute top-full left-0 mt-1 w-80 max-h-[70vh] overflow-y-auto bg-gray-800 border border-gray-600 rounded-lg shadow-xl z-50"
          >
            {/* Current problem info */}
            {currentProblem && (
              <div className="p-3 border-b border-gray-700">
                <div className="flex items-center justify-between mb-1">
                  <span className={`text-xs px-1.5 py-0.5 rounded border ${
                    currentProblem.is_failure_case
                      ? 'bg-red-500/20 text-red-400 border-red-500/30'
                      : getCategoryColor(currentProblem.category)
                  }`}>
                    {currentProblem.is_failure_case ? 'failure-case' : currentProblem.category}
                  </span>
                  <span className="text-yellow-400 text-xs">
                    {getDifficultyStars(currentProblem.difficulty)}
                  </span>
                </div>
                <p className="text-xs text-gray-400 line-clamp-2">{currentProblem.description}</p>
                {currentProblem.concept && (
                  <div className="mt-1">
                    <span className="text-xs text-cyan-400">Teaches: {currentProblem.concept}</span>
                  </div>
                )}
              </div>
            )}

            {/* Level groups */}
            <div className="py-1">
              {Array.from(groupedProblems.entries()).map(([level, levelProblems]) => {
                const isFailureLevel = level.includes('Failure Cases');
                const isExpanded = expandedLevel === level;

                return (
                  <div key={level}>
                    {/* Level header */}
                    <button
                      onClick={() => setExpandedLevel(isExpanded ? null : level)}
                      className={`w-full px-3 py-2 flex items-center justify-between text-left transition-colors ${
                        isFailureLevel
                          ? 'hover:bg-red-900/30'
                          : 'hover:bg-gray-700/50'
                      }`}
                    >
                      <div className="flex items-center gap-2">
                        {isFailureLevel && <span className="text-red-400">!</span>}
                        <span className={`text-sm font-medium ${
                          isFailureLevel ? 'text-red-300' : 'text-gray-300'
                        }`}>
                          {level}
                        </span>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="text-xs text-gray-500">{levelProblems.length}</span>
                        <span className="text-xs text-gray-500">
                          {isExpanded ? '▼' : '▶'}
                        </span>
                      </div>
                    </button>

                    {/* Problems in this level */}
                    <AnimatePresence>
                      {isExpanded && (
                        <motion.div
                          initial={{ height: 0, opacity: 0 }}
                          animate={{ height: 'auto', opacity: 1 }}
                          exit={{ height: 0, opacity: 0 }}
                          transition={{ duration: 0.15 }}
                          className="overflow-hidden"
                        >
                          {levelProblems.map((p) => (
                            <button
                              key={p.id}
                              onClick={() => handleSelect(p.id)}
                              disabled={disabled}
                              className={`w-full px-4 py-2 text-left transition-colors flex items-center justify-between ${
                                currentProblem?.id === p.id
                                  ? p.is_failure_case
                                    ? 'bg-red-600/30 border-l-2 border-red-500'
                                    : 'bg-cyan-600/30 border-l-2 border-cyan-500'
                                  : p.is_failure_case
                                  ? 'hover:bg-red-900/20 border-l-2 border-transparent'
                                  : 'hover:bg-gray-700/50 border-l-2 border-transparent'
                              } ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
                            >
                              <div className="flex items-center gap-2 flex-wrap">
                                <span className={`text-sm ${p.is_failure_case ? 'text-red-200' : 'text-white'}`}>
                                  {p.name}
                                </span>
                                <span className={`text-xs px-1 py-0.5 rounded border ${
                                  p.is_failure_case
                                    ? 'bg-red-500/20 text-red-400 border-red-500/30'
                                    : getCategoryColor(p.category)
                                }`}>
                                  {p.is_failure_case ? 'fail' : p.category.slice(0, 3)}
                                </span>
                              </div>
                              <span className="text-yellow-400 text-xs">
                                {getDifficultyStars(p.difficulty)}
                              </span>
                            </button>
                          ))}
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </div>
                );
              })}
            </div>

            {disabled && (
              <div className="px-3 py-2 text-xs text-yellow-500 border-t border-gray-700">
                Cannot change during training
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
