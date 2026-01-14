import type { ProblemInfo } from '../types';

interface ProblemSelectorProps {
  problems: ProblemInfo[];
  currentProblem: ProblemInfo | null;
  onSelect: (problemId: string) => void;
  disabled?: boolean;
}

export function ProblemSelector({
  problems,
  currentProblem,
  onSelect,
  disabled = false,
}: ProblemSelectorProps) {
  const getCategoryColor = (category: string) => {
    switch (category) {
      case 'binary':
        return 'bg-blue-500/20 text-blue-400';
      case 'regression':
        return 'bg-green-500/20 text-green-400';
      case 'multi-class':
        return 'bg-purple-500/20 text-purple-400';
      default:
        return 'bg-gray-500/20 text-gray-400';
    }
  };

  return (
    <div className="bg-gray-800 rounded-lg p-4">
      <h2 className="text-lg font-semibold mb-3">Problem Selection</h2>

      {/* Dropdown for quick selection */}
      <div className="mb-4">
        <select
          value={currentProblem?.id || ''}
          onChange={(e) => onSelect(e.target.value)}
          disabled={disabled}
          className="w-full bg-gray-700 text-white rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
        >
          {problems.map((p) => (
            <option key={p.id} value={p.id}>
              {p.name}
            </option>
          ))}
        </select>
      </div>

      {/* Current problem details */}
      {currentProblem && (
        <div className="space-y-3">
          <div className="flex items-center gap-2">
            <span
              className={`px-2 py-0.5 rounded text-xs font-medium ${getCategoryColor(
                currentProblem.category
              )}`}
            >
              {currentProblem.category}
            </span>
            <span className="text-xs text-gray-500">
              {currentProblem.output_activation}
            </span>
          </div>

          <p className="text-sm text-gray-400">{currentProblem.description}</p>

          <div className="text-xs text-gray-500 space-y-1">
            <div>
              <span className="text-gray-400">Architecture:</span>{' '}
              {currentProblem.default_architecture.join(' → ')}
            </div>
            <div>
              <span className="text-gray-400">Inputs:</span>{' '}
              {currentProblem.input_labels.join(', ')}
            </div>
            <div>
              <span className="text-gray-400">Outputs:</span>{' '}
              {currentProblem.output_labels.join(', ')}
            </div>
          </div>

          {currentProblem.embedded_context && (
            <div className="text-xs text-cyan-400/70 italic border-l-2 border-cyan-500/30 pl-2">
              {currentProblem.embedded_context}
            </div>
          )}
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
