import { useCallback, useState } from 'react';
import type { ProblemInfo } from '../types';
import { getInputConfigForProblem } from '../types';

interface InputPanelProps {
  problem: ProblemInfo | null;
  values: number[] | number[][];  // 1D for dense, 2D grid for CNN
  onChange: (values: number[] | number[][]) => void;
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
      const vals = values as number[];
      const newValues = [...vals];
      newValues[index] = newValue;
      onChange(newValues);
    },
    [values, onChange]
  );

  const handleToggle = useCallback(
    (index: number) => {
      const vals = values as number[];
      const newValues = [...vals];
      newValues[index] = newValues[index] === 1 ? 0 : 1;
      onChange(newValues);
    },
    [values, onChange]
  );

  const renderBinaryInputs = () => {
    const vals = values as number[];
    return (
      <div className="grid grid-cols-5 gap-2">
        {config.labels.map((label, i) => (
          <button
            key={i}
            onClick={() => handleToggle(i)}
            disabled={disabled}
            className={`
              flex flex-col items-center justify-center p-3 rounded-lg transition-all
              ${
                vals[i] === 1
                  ? 'bg-green-600 text-white shadow-lg shadow-green-500/30'
                  : 'bg-gray-700 text-gray-400 hover:bg-gray-600'
              }
              ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
            `}
          >
            <span className="text-2xl font-bold">{vals[i]}</span>
            <span className="text-xs mt-1">{label}</span>
          </button>
        ))}
      </div>
    );
  };

  const renderSliderInputs = () => {
    const vals = values as number[];
    return (
      <div className="space-y-4">
        {config.labels.map((label, i) => (
          <div key={i} className="space-y-1">
            <div className="flex justify-between text-sm">
              <span className="text-gray-400">{label}</span>
              <span className="font-mono text-cyan-400">{vals[i]?.toFixed(2) || '0.00'}</span>
            </div>
            <input
              type="range"
              min={config.min || 0}
              max={config.max || 1}
              step={config.step || 0.01}
              value={vals[i] || 0}
              onChange={(e) => handleValueChange(i, parseFloat(e.target.value))}
              disabled={disabled}
              className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-cyan-500 disabled:opacity-50"
            />
          </div>
        ))}
      </div>
    );
  };

  const renderPatternInputs = () => (
    <div className="space-y-3">
      <div className="flex items-end gap-1 h-24 bg-gray-900 rounded p-2">
        {config.labels.map((label, i) => {
          const height = ((values as number[])[i] || 0) * 100;
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
            value={(values as number[])[i] || 0}
            onChange={(e) => handleValueChange(i, parseFloat(e.target.value))}
            disabled={disabled}
            className="w-full h-1 bg-gray-700 rounded appearance-none cursor-pointer accent-cyan-500 disabled:opacity-50"
            style={{ writingMode: 'vertical-lr', direction: 'rtl', height: '40px' }}
          />
        ))}
      </div>
    </div>
  );

  // Grid input for CNN shape detection
  const gridSize = config.gridSize || 8;
  const gridValues = Array.isArray(values[0])
    ? (values as number[][])
    : Array.from({ length: gridSize }, () => Array(gridSize).fill(0));

  const [isDrawing, setIsDrawing] = useState(false);
  const [brushValue, setBrushValue] = useState(1.0);

  const handleGridCellChange = useCallback((row: number, col: number, value: number) => {
    const newGrid = gridValues.map((r, ri) =>
      ri === row ? r.map((c, ci) => (ci === col ? value : c)) : [...r]
    );
    onChange(newGrid);
  }, [gridValues, onChange]);

  const handleMouseDown = (row: number, col: number) => {
    if (disabled) return;
    setIsDrawing(true);
    handleGridCellChange(row, col, brushValue);
  };

  const handleMouseEnter = (row: number, col: number) => {
    if (!isDrawing || disabled) return;
    handleGridCellChange(row, col, brushValue);
  };

  const handleMouseUp = () => {
    setIsDrawing(false);
  };

  // Shape preset generators
  const generateCircle = () => {
    const grid = Array.from({ length: gridSize }, () => Array(gridSize).fill(0));
    const center = 3.5;  // Center of 8x8 grid
    const outerRadius = 3.5;
    const innerRadius = 1.8;  // Creates a ring/donut shape
    for (let i = 0; i < gridSize; i++) {
      for (let j = 0; j < gridSize; j++) {
        const dist = Math.sqrt((i - center) ** 2 + (j - center) ** 2);
        // Ring: fill pixels between inner and outer radius
        if (dist <= outerRadius && dist >= innerRadius) {
          grid[i][j] = 1.0;
        }
      }
    }
    onChange(grid);
  };

  const generateSquare = () => {
    const grid = Array.from({ length: gridSize }, () => Array(gridSize).fill(0));
    // Sharp corners distinguish it from circle
    for (let i = 1; i < 7; i++) {
      for (let j = 1; j < 7; j++) {
        grid[i][j] = 1.0;
      }
    }
    onChange(grid);
  };

  const generateTriangle = () => {
    const grid = Array.from({ length: gridSize }, () => Array(gridSize).fill(0));
    // Upward pointing triangle
    for (let row = 0; row < gridSize; row++) {
      // Width increases as we go down (apex at top)
      const halfWidth = Math.floor(row / 2) + 1;
      const center = Math.floor(gridSize / 2);
      for (let col = center - halfWidth; col <= center + halfWidth; col++) {
        if (col >= 0 && col < gridSize && row >= 1) {
          grid[row][col] = 1.0;
        }
      }
    }
    onChange(grid);
  };

  const clearGrid = () => {
    const grid = Array.from({ length: gridSize }, () => Array(gridSize).fill(0));
    onChange(grid);
  };

  // Digit patterns for digit recognition problem
  const digitPatterns: Record<number, number[][]> = {
    0: [[0,0,1,1,1,1,0,0],[0,1,1,0,0,1,1,0],[0,1,1,0,0,1,1,0],[0,1,1,0,0,1,1,0],[0,1,1,0,0,1,1,0],[0,1,1,0,0,1,1,0],[0,1,1,0,0,1,1,0],[0,0,1,1,1,1,0,0]],
    1: [[0,0,0,1,1,0,0,0],[0,0,1,1,1,0,0,0],[0,0,0,1,1,0,0,0],[0,0,0,1,1,0,0,0],[0,0,0,1,1,0,0,0],[0,0,0,1,1,0,0,0],[0,0,0,1,1,0,0,0],[0,0,1,1,1,1,0,0]],
    2: [[0,0,1,1,1,1,0,0],[0,1,1,0,0,1,1,0],[0,0,0,0,0,1,1,0],[0,0,0,0,1,1,0,0],[0,0,0,1,1,0,0,0],[0,0,1,1,0,0,0,0],[0,1,1,0,0,0,0,0],[0,1,1,1,1,1,1,0]],
    3: [[0,0,1,1,1,1,0,0],[0,1,1,0,0,1,1,0],[0,0,0,0,0,1,1,0],[0,0,0,1,1,1,0,0],[0,0,0,0,0,1,1,0],[0,0,0,0,0,1,1,0],[0,1,1,0,0,1,1,0],[0,0,1,1,1,1,0,0]],
    4: [[0,0,0,0,1,1,0,0],[0,0,0,1,1,1,0,0],[0,0,1,1,1,1,0,0],[0,1,1,0,1,1,0,0],[0,1,1,1,1,1,1,0],[0,0,0,0,1,1,0,0],[0,0,0,0,1,1,0,0],[0,0,0,0,1,1,0,0]],
    5: [[0,1,1,1,1,1,1,0],[0,1,1,0,0,0,0,0],[0,1,1,0,0,0,0,0],[0,1,1,1,1,1,0,0],[0,0,0,0,0,1,1,0],[0,0,0,0,0,1,1,0],[0,1,1,0,0,1,1,0],[0,0,1,1,1,1,0,0]],
    6: [[0,0,1,1,1,1,0,0],[0,1,1,0,0,0,0,0],[0,1,1,0,0,0,0,0],[0,1,1,1,1,1,0,0],[0,1,1,0,0,1,1,0],[0,1,1,0,0,1,1,0],[0,1,1,0,0,1,1,0],[0,0,1,1,1,1,0,0]],
    7: [[0,1,1,1,1,1,1,0],[0,0,0,0,0,1,1,0],[0,0,0,0,1,1,0,0],[0,0,0,0,1,1,0,0],[0,0,0,1,1,0,0,0],[0,0,0,1,1,0,0,0],[0,0,0,1,1,0,0,0],[0,0,0,1,1,0,0,0]],
    8: [[0,0,1,1,1,1,0,0],[0,1,1,0,0,1,1,0],[0,1,1,0,0,1,1,0],[0,0,1,1,1,1,0,0],[0,1,1,0,0,1,1,0],[0,1,1,0,0,1,1,0],[0,1,1,0,0,1,1,0],[0,0,1,1,1,1,0,0]],
    9: [[0,0,1,1,1,1,0,0],[0,1,1,0,0,1,1,0],[0,1,1,0,0,1,1,0],[0,0,1,1,1,1,1,0],[0,0,0,0,0,1,1,0],[0,0,0,0,0,1,1,0],[0,0,0,0,1,1,0,0],[0,0,1,1,1,0,0,0]],
  };

  const generateDigit = (digit: number) => {
    onChange(digitPatterns[digit]);
  };

  // Arrow patterns for arrow direction problem
  const arrowPatterns: Record<string, number[][]> = {
    up: [[0,0,0,1,1,0,0,0],[0,0,1,1,1,1,0,0],[0,1,1,1,1,1,1,0],[1,1,0,1,1,0,1,1],[0,0,0,1,1,0,0,0],[0,0,0,1,1,0,0,0],[0,0,0,1,1,0,0,0],[0,0,0,1,1,0,0,0]],
    down: [[0,0,0,1,1,0,0,0],[0,0,0,1,1,0,0,0],[0,0,0,1,1,0,0,0],[0,0,0,1,1,0,0,0],[1,1,0,1,1,0,1,1],[0,1,1,1,1,1,1,0],[0,0,1,1,1,1,0,0],[0,0,0,1,1,0,0,0]],
    left: [[0,0,0,1,0,0,0,0],[0,0,1,1,0,0,0,0],[0,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1],[0,1,1,1,1,1,1,1],[0,0,1,1,0,0,0,0],[0,0,0,1,0,0,0,0]],
    right: [[0,0,0,0,1,0,0,0],[0,0,0,0,1,1,0,0],[1,1,1,1,1,1,1,0],[1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,0],[0,0,0,0,1,1,0,0],[0,0,0,0,1,0,0,0]],
  };

  const generateArrow = (direction: string) => {
    onChange(arrowPatterns[direction]);
  };

  const isDigitProblem = problem?.id === 'digit_recognition';
  const isArrowProblem = problem?.id === 'arrow_direction';

  const renderGridInputs = () => (
    <div className="space-y-3" onMouseUp={handleMouseUp} onMouseLeave={handleMouseUp}>
      {/* Drawing grid */}
      <div
        className="grid gap-0.5 bg-gray-900 p-2 rounded-lg select-none"
        style={{ gridTemplateColumns: `repeat(${gridSize}, 1fr)` }}
      >
        {gridValues.map((row, rowIdx) =>
          row.map((val, colIdx) => {
            const intensity = Math.min(1, Math.max(0, val));
            const bgColor = intensity > 0
              ? `rgba(34, 211, 238, ${intensity})`  // cyan with opacity
              : 'rgb(31, 41, 55)';  // gray-800
            return (
              <div
                key={`${rowIdx}-${colIdx}`}
                onMouseDown={() => handleMouseDown(rowIdx, colIdx)}
                onMouseEnter={() => handleMouseEnter(rowIdx, colIdx)}
                className={`aspect-square rounded-sm transition-colors cursor-pointer ${
                  disabled ? 'opacity-50 cursor-not-allowed' : ''
                }`}
                style={{ backgroundColor: bgColor }}
              />
            );
          })
        )}
      </div>

      {/* Brush intensity slider */}
      <div className="flex items-center gap-2">
        <span className="text-xs text-gray-400">Brush:</span>
        <input
          type="range"
          min={0}
          max={1}
          step={0.1}
          value={brushValue}
          onChange={(e) => setBrushValue(parseFloat(e.target.value))}
          disabled={disabled}
          className="flex-1 h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-cyan-500"
        />
        <span className="text-xs text-cyan-400 font-mono w-8">{brushValue.toFixed(1)}</span>
      </div>

      {/* Preset buttons - shapes, digits, or arrows depending on problem */}
      {isDigitProblem ? (
        <div className="space-y-2">
          <div className="grid grid-cols-5 gap-1">
            {[0, 1, 2, 3, 4].map((digit) => (
              <button
                key={digit}
                onClick={() => generateDigit(digit)}
                disabled={disabled}
                className="px-2 py-2 bg-gray-700 hover:bg-gray-600 rounded text-sm font-bold disabled:opacity-50 transition-colors"
              >
                {digit}
              </button>
            ))}
          </div>
          <div className="grid grid-cols-5 gap-1">
            {[5, 6, 7, 8, 9].map((digit) => (
              <button
                key={digit}
                onClick={() => generateDigit(digit)}
                disabled={disabled}
                className="px-2 py-2 bg-gray-700 hover:bg-gray-600 rounded text-sm font-bold disabled:opacity-50 transition-colors"
              >
                {digit}
              </button>
            ))}
          </div>
          <button
            onClick={clearGrid}
            disabled={disabled}
            className="w-full px-2 py-1 bg-gray-700 hover:bg-gray-600 rounded text-xs disabled:opacity-50 transition-colors"
          >
            🗑️ Clear
          </button>
        </div>
      ) : isArrowProblem ? (
        <div className="space-y-2">
          <div className="grid grid-cols-3 gap-2">
            <div />
            <button
              onClick={() => generateArrow('up')}
              disabled={disabled}
              className="px-2 py-3 bg-gray-700 hover:bg-gray-600 rounded text-xl disabled:opacity-50 transition-colors"
            >
              ↑
            </button>
            <div />
          </div>
          <div className="grid grid-cols-3 gap-2">
            <button
              onClick={() => generateArrow('left')}
              disabled={disabled}
              className="px-2 py-3 bg-gray-700 hover:bg-gray-600 rounded text-xl disabled:opacity-50 transition-colors"
            >
              ←
            </button>
            <button
              onClick={clearGrid}
              disabled={disabled}
              className="px-2 py-3 bg-gray-700 hover:bg-gray-600 rounded text-xs disabled:opacity-50 transition-colors"
            >
              🗑️
            </button>
            <button
              onClick={() => generateArrow('right')}
              disabled={disabled}
              className="px-2 py-3 bg-gray-700 hover:bg-gray-600 rounded text-xl disabled:opacity-50 transition-colors"
            >
              →
            </button>
          </div>
          <div className="grid grid-cols-3 gap-2">
            <div />
            <button
              onClick={() => generateArrow('down')}
              disabled={disabled}
              className="px-2 py-3 bg-gray-700 hover:bg-gray-600 rounded text-xl disabled:opacity-50 transition-colors"
            >
              ↓
            </button>
            <div />
          </div>
        </div>
      ) : (
        <div className="grid grid-cols-4 gap-2">
          <button
            onClick={generateCircle}
            disabled={disabled}
            className="px-2 py-2 bg-gray-700 hover:bg-gray-600 rounded text-xs disabled:opacity-50 transition-colors"
          >
            ⭕ Circle
          </button>
          <button
            onClick={generateSquare}
            disabled={disabled}
            className="px-2 py-2 bg-gray-700 hover:bg-gray-600 rounded text-xs disabled:opacity-50 transition-colors"
          >
            ⬜ Square
          </button>
          <button
            onClick={generateTriangle}
            disabled={disabled}
            className="px-2 py-2 bg-gray-700 hover:bg-gray-600 rounded text-xs disabled:opacity-50 transition-colors"
          >
            🔺 Triangle
          </button>
          <button
            onClick={clearGrid}
            disabled={disabled}
            className="px-2 py-2 bg-gray-700 hover:bg-gray-600 rounded text-xs disabled:opacity-50 transition-colors"
          >
            🗑️ Clear
          </button>
        </div>
      )}
    </div>
  );

  const inputDescription = config.type === 'grid'
    ? `${gridSize}×${gridSize} grid`
    : `${problem.input_labels.length} values`;

  return (
    <div className="bg-gray-800 rounded-lg p-4">
      <h2 className="text-lg font-semibold mb-3">
        Input
        <span className="text-sm font-normal text-gray-500 ml-2">
          ({inputDescription})
        </span>
      </h2>

      {config.type === 'binary' && renderBinaryInputs()}
      {config.type === 'slider' && renderSliderInputs()}
      {config.type === 'pattern' && renderPatternInputs()}
      {config.type === 'grid' && renderGridInputs()}

      {disabled && (
        <p className="text-xs text-yellow-500 mt-3">Train the network first</p>
      )}
    </div>
  );
}
