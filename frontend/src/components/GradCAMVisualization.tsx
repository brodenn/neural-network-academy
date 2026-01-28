import React, { useState, useCallback, useEffect, useMemo } from 'react';
import { motion } from 'framer-motion';

interface GradCAMData {
  heatmap: number[][];
  input_shape: [number, number];
  target_class: number;
  predicted_class: number;
  probabilities: number[];
  output_labels: string[];
}

interface GradCAMVisualizationProps {
  inputGrid: number[][];
  trainingComplete: boolean;
  outputLabels: string[];
}

type ViewMode = 'input' | 'heatmap' | 'overlay';

export function GradCAMVisualization({
  inputGrid,
  trainingComplete,
  outputLabels,
}: GradCAMVisualizationProps) {
  const [gradcamData, setGradcamData] = useState<GradCAMData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<ViewMode>('overlay');
  const [targetClass, setTargetClass] = useState<number | null>(null);

  const fetchGradCAM = useCallback(async () => {
    if (!trainingComplete || inputGrid.length === 0) return;

    setLoading(true);
    setError(null);

    try {
      const res = await fetch('http://localhost:5000/api/gradcam', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          inputs: inputGrid,
          target_class: targetClass,
        }),
      });

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.error || 'Failed to fetch');
      }

      const data = await res.json();
      setGradcamData(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  }, [inputGrid, trainingComplete, targetClass]);

  // Fetch when input changes or training completes
  useEffect(() => {
    if (trainingComplete && inputGrid.length > 0) {
      // Small debounce to avoid too many requests
      const timer = setTimeout(fetchGradCAM, 300);
      return () => clearTimeout(timer);
    }
  }, [inputGrid, trainingComplete, fetchGradCAM]);

  // Render the visualization canvas
  const canvasContent = useMemo(() => {
    if (!gradcamData || inputGrid.length === 0) return null;

    const { heatmap } = gradcamData;
    const height = inputGrid.length;
    const width = inputGrid[0].length;
    const cellSize = 20;

    // Create cells for the grid
    const cells: React.ReactNode[] = [];

    for (let i = 0; i < height; i++) {
      for (let j = 0; j < width; j++) {
        const inputValue = inputGrid[i][j];
        const heatmapValue = heatmap[i]?.[j] ?? 0;

        let backgroundColor: string;
        let borderColor = 'rgba(255, 255, 255, 0.1)';

        if (viewMode === 'input') {
          // Show only input (grayscale)
          const gray = Math.round(inputValue * 255);
          backgroundColor = `rgb(${gray}, ${gray}, ${gray})`;
        } else if (viewMode === 'heatmap') {
          // Show only heatmap (yellow to red gradient)
          const r = 255;
          const g = Math.round(255 * (1 - heatmapValue));
          const b = 0;
          backgroundColor = `rgb(${r}, ${g}, ${b})`;
        } else {
          // Overlay mode: blend input with heatmap
          // Input as base brightness, heatmap as red overlay
          const inputGray = inputValue * 255;
          const heatAlpha = heatmapValue * 0.7;

          // Blend: base grayscale + red heat overlay
          const r = Math.min(255, inputGray + heatmapValue * 255 * 0.6);
          const g = Math.max(0, inputGray * (1 - heatAlpha * 0.5));
          const b = Math.max(0, inputGray * (1 - heatAlpha * 0.7));

          backgroundColor = `rgb(${Math.round(r)}, ${Math.round(g)}, ${Math.round(b)})`;

          // Highlight high importance regions
          if (heatmapValue > 0.7) {
            borderColor = 'rgba(255, 200, 0, 0.5)';
          }
        }

        cells.push(
          <div
            key={`${i}-${j}`}
            style={{
              width: cellSize,
              height: cellSize,
              backgroundColor,
              border: `1px solid ${borderColor}`,
            }}
            title={`Input: ${inputValue.toFixed(2)}, Importance: ${(heatmapValue * 100).toFixed(1)}%`}
          />
        );
      }
    }

    return (
      <div
        className="grid gap-0 rounded overflow-hidden"
        style={{
          gridTemplateColumns: `repeat(${width}, ${cellSize}px)`,
        }}
      >
        {cells}
      </div>
    );
  }, [inputGrid, gradcamData, viewMode]);

  if (!trainingComplete) {
    return null;
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-gray-800 rounded-lg p-3"
    >
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-lg font-semibold flex items-center gap-2">
          <span className="text-xl">🔍</span>
          Grad-CAM
        </h3>
        <div className="flex items-center gap-2">
          {/* View mode toggle */}
          <div className="flex gap-0.5 bg-gray-900 rounded p-0.5">
            <button
              onClick={() => setViewMode('input')}
              className={`px-2 py-0.5 text-xs rounded ${
                viewMode === 'input' ? 'bg-gray-600 text-white' : 'text-gray-400 hover:text-white'
              }`}
            >
              Input
            </button>
            <button
              onClick={() => setViewMode('heatmap')}
              className={`px-2 py-0.5 text-xs rounded ${
                viewMode === 'heatmap' ? 'bg-orange-600 text-white' : 'text-gray-400 hover:text-white'
              }`}
            >
              Heat
            </button>
            <button
              onClick={() => setViewMode('overlay')}
              className={`px-2 py-0.5 text-xs rounded ${
                viewMode === 'overlay' ? 'bg-cyan-600 text-white' : 'text-gray-400 hover:text-white'
              }`}
            >
              Overlay
            </button>
          </div>
          <button
            onClick={fetchGradCAM}
            disabled={loading}
            className="px-2 py-1 text-xs rounded bg-gray-700 text-gray-300 hover:bg-gray-600 disabled:opacity-50"
          >
            {loading ? '...' : 'Refresh'}
          </button>
        </div>
      </div>

      {/* Educational explanation */}
      <div className="mb-3 p-2 bg-gray-700/50 rounded text-xs text-gray-400">
        <p>
          <span className="text-cyan-400 font-medium">Grad-CAM</span> highlights which parts of the input
          were most important for the prediction.{' '}
          <span className="text-orange-400">Hot regions</span> contributed most to the output class.
        </p>
      </div>

      {/* Target class selector */}
      {outputLabels.length > 0 && (
        <div className="mb-3 flex items-center gap-2 text-xs">
          <span className="text-gray-400">Explain class:</span>
          <select
            value={targetClass ?? ''}
            onChange={(e) => setTargetClass(e.target.value === '' ? null : parseInt(e.target.value))}
            className="bg-gray-700 text-white rounded px-2 py-1 text-xs border border-gray-600"
          >
            <option value="">Predicted</option>
            {outputLabels.map((label, i) => (
              <option key={i} value={i}>
                {label}
              </option>
            ))}
          </select>
          {gradcamData && (
            <span className="text-gray-500">
              (Predicted: {outputLabels[gradcamData.predicted_class]})
            </span>
          )}
        </div>
      )}

      {/* Visualization */}
      {loading && !gradcamData ? (
        <div className="h-40 flex items-center justify-center text-gray-500">
          Loading Grad-CAM...
        </div>
      ) : error ? (
        <div className="h-40 flex items-center justify-center text-red-400 text-sm">
          Error: {error}
        </div>
      ) : gradcamData ? (
        <div className="flex flex-col items-center">
          {canvasContent}

          {/* Probability bars */}
          <div className="mt-3 w-full max-w-xs">
            <div className="text-xs text-gray-400 mb-1">Class Probabilities:</div>
            <div className="space-y-1">
              {gradcamData.probabilities.map((prob, i) => {
                const isTarget = i === gradcamData.target_class;
                const isPredicted = i === gradcamData.predicted_class;
                return (
                  <div key={i} className="flex items-center gap-2">
                    <span
                      className={`text-xs w-16 truncate ${
                        isPredicted ? 'text-cyan-400 font-medium' : 'text-gray-400'
                      }`}
                    >
                      {outputLabels[i] || `Class ${i}`}
                    </span>
                    <div className="flex-1 h-3 bg-gray-700 rounded overflow-hidden">
                      <div
                        className={`h-full transition-all duration-300 ${
                          isTarget
                            ? 'bg-orange-500'
                            : isPredicted
                            ? 'bg-cyan-500'
                            : 'bg-gray-500'
                        }`}
                        style={{ width: `${prob * 100}%` }}
                      />
                    </div>
                    <span className="text-xs text-gray-500 font-mono w-12 text-right">
                      {(prob * 100).toFixed(1)}%
                    </span>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Legend */}
          <div className="mt-3 flex justify-center gap-4 text-xs">
            <div className="flex items-center gap-1">
              <div className="w-3 h-3 rounded bg-gray-400" />
              <span className="text-gray-400">Low importance</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-3 h-3 rounded bg-gradient-to-r from-yellow-500 to-red-500" />
              <span className="text-gray-400">High importance</span>
            </div>
          </div>
        </div>
      ) : (
        <div className="h-40 flex items-center justify-center text-gray-500 text-sm">
          Draw on the input grid to see Grad-CAM visualization
        </div>
      )}
    </motion.div>
  );
}
