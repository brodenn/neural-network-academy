import { useState, useEffect, useCallback, useRef } from 'react';
import { motion } from 'framer-motion';

interface DecisionBoundaryData {
  predictions: number[][];
  confidence: number[][];
  x_range: number[];
  y_range: number[];
  resolution: number;
  problem_id: string;
  category: string;
  output_labels: string[];
  training_data: {
    inputs: number[][];
    labels: number[][];
  };
}

interface DecisionBoundaryVizProps {
  problemId: string;
  trainingComplete: boolean;
  currentEpoch: number;
  onPointClick?: (x: number, y: number) => void;
}

export function DecisionBoundaryViz({
  problemId,
  trainingComplete,
  currentEpoch,
  onPointClick,
}: DecisionBoundaryVizProps) {
  const [data, setData] = useState<DecisionBoundaryData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showTrainingData, setShowTrainingData] = useState(true);
  const [showConfidence, setShowConfidence] = useState(false);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [hoveredPoint, setHoveredPoint] = useState<{ x: number; y: number; pred: number } | null>(null);

  // Only fetch for 2D problems
  const is2DProblem = ['circle', 'donut', 'spiral', 'quadrants', 'colors'].includes(problemId);

  const fetchBoundary = useCallback(async () => {
    if (!is2DProblem || !trainingComplete) return;

    setLoading(true);
    setError(null);

    try {
      const res = await fetch('http://localhost:5000/api/decision-boundary?resolution=60');
      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.error || 'Failed to fetch');
      }
      const boundaryData = await res.json();
      setData(boundaryData);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  }, [is2DProblem, trainingComplete]);

  // Fetch when training completes or epoch changes significantly
  useEffect(() => {
    if (trainingComplete && is2DProblem) {
      fetchBoundary();
    }
  }, [trainingComplete, is2DProblem, fetchBoundary]);

  // Auto-refresh during training (every 100 epochs)
  useEffect(() => {
    if (currentEpoch > 0 && currentEpoch % 100 === 0 && is2DProblem) {
      fetchBoundary();
    }
  }, [currentEpoch, is2DProblem, fetchBoundary]);

  // Render to canvas
  useEffect(() => {
    if (!data || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const { predictions, confidence, resolution, category, training_data } = data;
    const width = canvas.width;
    const height = canvas.height;
    const cellWidth = width / resolution;
    const cellHeight = height / resolution;

    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    // Draw decision boundary heatmap
    for (let row = 0; row < resolution; row++) {
      for (let col = 0; col < resolution; col++) {
        const pred = predictions[row][col];
        const conf = confidence[row][col];

        // Map prediction to color
        let color: string;
        if (category === 'multi-class') {
          // Multi-class: use different hues for different classes
          const hue = (pred * 360) / data.output_labels.length;
          const saturation = showConfidence ? conf * 100 : 70;
          color = `hsl(${hue}, ${saturation}%, 50%)`;
        } else {
          // Binary: blue (0) to red (1) gradient
          if (showConfidence) {
            const intensity = Math.abs(pred - 0.5) * 2; // 0 at boundary, 1 at extremes
            if (pred >= 0.5) {
              color = `rgba(239, 68, 68, ${0.3 + intensity * 0.7})`; // Red
            } else {
              color = `rgba(59, 130, 246, ${0.3 + intensity * 0.7})`; // Blue
            }
          } else {
            if (pred >= 0.5) {
              color = `rgba(239, 68, 68, ${0.5 + pred * 0.3})`; // Red
            } else {
              color = `rgba(59, 130, 246, ${0.5 + (1 - pred) * 0.3})`; // Blue
            }
          }
        }

        ctx.fillStyle = color;
        // Note: y is flipped because canvas y increases downward
        ctx.fillRect(col * cellWidth, (resolution - 1 - row) * cellHeight, cellWidth + 1, cellHeight + 1);
      }
    }

    // Draw decision boundary contour (where prediction ~= 0.5 for binary)
    if (category === 'binary') {
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.8)';
      ctx.lineWidth = 2;
      ctx.beginPath();

      for (let row = 0; row < resolution - 1; row++) {
        for (let col = 0; col < resolution - 1; col++) {
          const p00 = predictions[row][col];
          const p01 = predictions[row][col + 1];
          const p10 = predictions[row + 1][col];
          const p11 = predictions[row + 1][col + 1];

          // Check if 0.5 contour passes through this cell
          const crosses = (
            (p00 < 0.5 && (p01 >= 0.5 || p10 >= 0.5 || p11 >= 0.5)) ||
            (p00 >= 0.5 && (p01 < 0.5 || p10 < 0.5 || p11 < 0.5))
          );

          if (crosses) {
            const cx = (col + 0.5) * cellWidth;
            const cy = (resolution - 1 - row - 0.5) * cellHeight;
            ctx.moveTo(cx - cellWidth / 2, cy);
            ctx.lineTo(cx + cellWidth / 2, cy);
          }
        }
      }
      ctx.stroke();
    }

    // Draw training data points
    if (showTrainingData && training_data) {
      const { inputs, labels } = training_data;
      const xMin = data.x_range[0];
      const xMax = data.x_range[data.x_range.length - 1];
      const yMin = data.y_range[0];
      const yMax = data.y_range[data.y_range.length - 1];

      for (let i = 0; i < inputs.length; i++) {
        const [x, y] = inputs[i];
        const label = labels[i];

        // Convert to canvas coordinates
        const canvasX = ((x - xMin) / (xMax - xMin)) * width;
        const canvasY = ((yMax - y) / (yMax - yMin)) * height;

        // Determine point color based on label
        let pointColor: string;
        if (category === 'multi-class') {
          const classIdx = label.indexOf(Math.max(...label));
          const hue = (classIdx * 360) / data.output_labels.length;
          pointColor = `hsl(${hue}, 90%, 40%)`;
        } else {
          pointColor = label[0] >= 0.5 ? '#dc2626' : '#2563eb';
        }

        // Draw point with white border
        ctx.beginPath();
        ctx.arc(canvasX, canvasY, 5, 0, Math.PI * 2);
        ctx.fillStyle = pointColor;
        ctx.fill();
        ctx.strokeStyle = 'white';
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }
    }
  }, [data, showTrainingData, showConfidence]);

  // Handle canvas click
  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!data || !canvasRef.current || !onPointClick) return;

    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    const canvasX = (e.clientX - rect.left) * scaleX;
    const canvasY = (e.clientY - rect.top) * scaleY;

    // Convert to data coordinates
    const xMin = data.x_range[0];
    const xMax = data.x_range[data.x_range.length - 1];
    const yMin = data.y_range[0];
    const yMax = data.y_range[data.y_range.length - 1];

    const x = xMin + (canvasX / canvas.width) * (xMax - xMin);
    const y = yMax - (canvasY / canvas.height) * (yMax - yMin);

    onPointClick(parseFloat(x.toFixed(2)), parseFloat(y.toFixed(2)));
  };

  // Handle canvas hover
  const handleCanvasMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!data || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    const canvasX = (e.clientX - rect.left) * scaleX;
    const canvasY = (e.clientY - rect.top) * scaleY;

    // Convert to grid indices
    const col = Math.floor((canvasX / canvas.width) * data.resolution);
    const row = data.resolution - 1 - Math.floor((canvasY / canvas.height) * data.resolution);

    if (row >= 0 && row < data.resolution && col >= 0 && col < data.resolution) {
      const xMin = data.x_range[0];
      const xMax = data.x_range[data.x_range.length - 1];
      const yMin = data.y_range[0];
      const yMax = data.y_range[data.y_range.length - 1];

      const x = xMin + (col / data.resolution) * (xMax - xMin);
      const y = yMin + (row / data.resolution) * (yMax - yMin);

      setHoveredPoint({
        x: parseFloat(x.toFixed(2)),
        y: parseFloat(y.toFixed(2)),
        pred: data.predictions[row][col],
      });
    }
  };

  if (!is2DProblem) {
    return null;
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-gray-800 rounded-lg p-4"
    >
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-lg font-semibold flex items-center gap-2">
          <span className="text-xl">🎯</span>
          Decision Boundary
        </h3>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowTrainingData(!showTrainingData)}
            className={`px-2 py-1 text-xs rounded ${
              showTrainingData
                ? 'bg-blue-600 text-white'
                : 'bg-gray-700 text-gray-400'
            }`}
          >
            Data Points
          </button>
          <button
            onClick={() => setShowConfidence(!showConfidence)}
            className={`px-2 py-1 text-xs rounded ${
              showConfidence
                ? 'bg-purple-600 text-white'
                : 'bg-gray-700 text-gray-400'
            }`}
          >
            Confidence
          </button>
          <button
            onClick={fetchBoundary}
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
          This visualization shows how the neural network divides the 2D input space.{' '}
          <span className="text-blue-400">Blue regions</span> are classified as class 0,{' '}
          <span className="text-red-400">red regions</span> as class 1.
          The <span className="text-white">white line</span> marks the decision boundary where the network is uncertain (50% probability).
        </p>
      </div>

      {!trainingComplete ? (
        <div className="aspect-square bg-gray-700/50 rounded flex items-center justify-center text-gray-500">
          <p>Train the network to see decision boundaries</p>
        </div>
      ) : loading && !data ? (
        <div className="aspect-square bg-gray-700/50 rounded flex items-center justify-center text-gray-500">
          <p>Loading decision boundary...</p>
        </div>
      ) : error ? (
        <div className="aspect-square bg-gray-700/50 rounded flex items-center justify-center text-red-400">
          <p>Error: {error}</p>
        </div>
      ) : data ? (
        <div className="relative">
          <canvas
            ref={canvasRef}
            width={300}
            height={300}
            className="w-full aspect-square rounded cursor-crosshair"
            onClick={handleCanvasClick}
            onMouseMove={handleCanvasMove}
            onMouseLeave={() => setHoveredPoint(null)}
          />

          {/* Axis labels */}
          <div className="absolute bottom-0 left-1/2 -translate-x-1/2 text-xs text-gray-400 -mb-5">
            {data.x_range[0].toFixed(1)} ← x → {data.x_range[data.x_range.length - 1].toFixed(1)}
          </div>
          <div className="absolute left-0 top-1/2 -translate-y-1/2 -translate-x-8 text-xs text-gray-400 rotate-[-90deg]">
            {data.y_range[0].toFixed(1)} ← y → {data.y_range[data.y_range.length - 1].toFixed(1)}
          </div>

          {/* Hover tooltip */}
          {hoveredPoint && (
            <div className="absolute top-2 right-2 bg-gray-900/90 px-2 py-1 rounded text-xs">
              <div className="text-gray-400">
                ({hoveredPoint.x}, {hoveredPoint.y})
              </div>
              <div className={hoveredPoint.pred >= 0.5 ? 'text-red-400' : 'text-blue-400'}>
                P(class 1) = {(hoveredPoint.pred * 100).toFixed(1)}%
              </div>
            </div>
          )}

          {/* Legend */}
          <div className="mt-2 flex justify-center gap-4 text-xs">
            <div className="flex items-center gap-1">
              <div className="w-3 h-3 rounded-full bg-blue-500" />
              <span className="text-gray-400">Class 0 (Outside)</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-3 h-3 rounded-full bg-red-500" />
              <span className="text-gray-400">Class 1 (Inside)</span>
            </div>
          </div>
        </div>
      ) : null}

      {/* Click instruction */}
      {trainingComplete && data && onPointClick && (
        <p className="mt-2 text-xs text-gray-500 text-center">
          Click anywhere to test a point
        </p>
      )}
    </motion.div>
  );
}
