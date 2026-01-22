import { memo, useState, useCallback, useEffect, useRef } from 'react';
import { motion } from 'framer-motion';

interface TrainingScrubberProps {
  lossHistory: number[];
  accuracyHistory: number[];
  totalEpochs: number;
  currentEpoch: number;
  onScrubToEpoch?: (epoch: number) => void;
  isPlaying?: boolean;
  onPlayPause?: () => void;
}

export const TrainingScrubber = memo(function TrainingScrubber({
  lossHistory,
  accuracyHistory,
  totalEpochs,
  currentEpoch: _currentEpoch, // Used for display sync
  onScrubToEpoch,
  isPlaying = false,
  onPlayPause,
}: TrainingScrubberProps) {
  const [scrubPosition, setScrubPosition] = useState(totalEpochs);
  const [isDragging, setIsDragging] = useState(false);
  const sliderRef = useRef<HTMLDivElement>(null);

  // Update scrub position when training progresses (unless actively scrubbing)
  useEffect(() => {
    if (!isDragging) {
      setScrubPosition(totalEpochs);
    }
  }, [totalEpochs, isDragging]);

  // Handle scrubbing
  const handleScrub = useCallback((clientX: number) => {
    if (!sliderRef.current) return;

    const rect = sliderRef.current.getBoundingClientRect();
    const x = clientX - rect.left;
    const width = rect.width;
    const ratio = Math.max(0, Math.min(1, x / width));
    const epoch = Math.round(ratio * totalEpochs);

    setScrubPosition(epoch);
    onScrubToEpoch?.(epoch);
  }, [totalEpochs, onScrubToEpoch]);

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    setIsDragging(true);
    handleScrub(e.clientX);
  }, [handleScrub]);

  const handleMouseMove = useCallback((e: MouseEvent) => {
    if (isDragging) {
      handleScrub(e.clientX);
    }
  }, [isDragging, handleScrub]);

  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
  }, []);

  useEffect(() => {
    if (isDragging) {
      window.addEventListener('mousemove', handleMouseMove);
      window.addEventListener('mouseup', handleMouseUp);
      return () => {
        window.removeEventListener('mousemove', handleMouseMove);
        window.removeEventListener('mouseup', handleMouseUp);
      };
    }
  }, [isDragging, handleMouseMove, handleMouseUp]);

  // Calculate displayed values at scrub position
  const getValueAtPosition = useCallback((history: number[], position: number) => {
    if (history.length === 0) return 0;
    const ratio = position / totalEpochs;
    const index = Math.min(Math.floor(ratio * history.length), history.length - 1);
    return history[index] ?? 0;
  }, [totalEpochs]);

  const displayedLoss = getValueAtPosition(lossHistory, scrubPosition);
  const displayedAccuracy = getValueAtPosition(accuracyHistory, scrubPosition);
  const progressPercent = totalEpochs > 0 ? (scrubPosition / totalEpochs) * 100 : 0;

  // Don't render if no training data
  if (lossHistory.length === 0) {
    return null;
  }

  return (
    <div className="mt-3 pt-3 border-t border-gray-700/50">
      {/* Scrubber label */}
      <div className="flex items-center justify-between mb-2 text-xs text-gray-400">
        <span className="flex items-center gap-1">
          <span className="text-cyan-400">⏱</span>
          Training Timeline
        </span>
        <span className="text-gray-500">
          {isDragging ? 'Scrubbing...' : 'Drag to replay'}
        </span>
      </div>

      {/* Slider track */}
      <div
        ref={sliderRef}
        className="relative h-8 bg-gray-700/50 rounded-lg cursor-pointer group"
        onMouseDown={handleMouseDown}
      >
        {/* Mini loss sparkline background */}
        <div className="absolute inset-0 overflow-hidden rounded-lg opacity-30">
          <svg className="w-full h-full" preserveAspectRatio="none">
            <defs>
              <linearGradient id="lossGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#ef4444" stopOpacity="0.5" />
                <stop offset="100%" stopColor="#ef4444" stopOpacity="0" />
              </linearGradient>
            </defs>
            {lossHistory.length > 1 && (
              <path
                d={`M 0,32 ${lossHistory.map((loss, i) => {
                  const x = (i / (lossHistory.length - 1)) * 100;
                  const maxLoss = Math.max(...lossHistory);
                  const y = 32 - (loss / maxLoss) * 28;
                  return `L ${x},${y}`;
                }).join(' ')} L 100,32 Z`}
                fill="url(#lossGradient)"
              />
            )}
          </svg>
        </div>

        {/* Progress fill */}
        <motion.div
          className="absolute left-0 top-0 bottom-0 bg-gradient-to-r from-cyan-600/40 to-cyan-500/40 rounded-lg"
          style={{ width: `${progressPercent}%` }}
          animate={{ width: `${progressPercent}%` }}
          transition={{ type: 'tween', duration: isDragging ? 0 : 0.1 }}
        />

        {/* Thumb */}
        <motion.div
          className="absolute top-1/2 -translate-y-1/2 w-4 h-6 bg-cyan-500 rounded shadow-lg shadow-cyan-500/30 cursor-grab active:cursor-grabbing"
          style={{ left: `calc(${progressPercent}% - 8px)` }}
          animate={{
            left: `calc(${progressPercent}% - 8px)`,
            scale: isDragging ? 1.2 : 1,
          }}
          transition={{ type: 'spring', stiffness: 300, damping: 30 }}
          whileHover={{ scale: 1.1 }}
        >
          {/* Grip lines */}
          <div className="absolute inset-x-1 top-1/2 -translate-y-1/2 space-y-0.5">
            <div className="h-px bg-cyan-300/50" />
            <div className="h-px bg-cyan-300/50" />
            <div className="h-px bg-cyan-300/50" />
          </div>
        </motion.div>

        {/* Epoch markers */}
        <div className="absolute bottom-0.5 left-2 right-2 flex justify-between text-[9px] text-gray-500">
          <span>0</span>
          <span>{Math.round(totalEpochs / 2)}</span>
          <span>{totalEpochs}</span>
        </div>
      </div>

      {/* Scrubbed values display */}
      <div className="mt-2 flex items-center justify-between text-xs">
        <div className="flex items-center gap-3">
          <div className="px-2 py-1 bg-gray-900/60 rounded border border-gray-700/50">
            <span className="text-gray-500">Epoch:</span>{' '}
            <span className="text-cyan-400 font-mono">{scrubPosition}</span>
          </div>
          <div className="px-2 py-1 bg-gray-900/60 rounded border border-gray-700/50">
            <span className="text-gray-500">Loss:</span>{' '}
            <span className="text-red-400 font-mono">{displayedLoss.toFixed(4)}</span>
          </div>
          <div className="px-2 py-1 bg-gray-900/60 rounded border border-gray-700/50">
            <span className="text-gray-500">Acc:</span>{' '}
            <span className="text-green-400 font-mono">{(displayedAccuracy * 100).toFixed(1)}%</span>
          </div>
        </div>

        {/* Playback control (optional) */}
        {onPlayPause && (
          <button
            onClick={onPlayPause}
            className="px-2 py-1 bg-gray-700 hover:bg-gray-600 rounded text-gray-300"
          >
            {isPlaying ? '⏸' : '▶'}
          </button>
        )}
      </div>

      {/* Keyboard hint */}
      <div className="mt-1 text-[10px] text-gray-600 text-center">
        Tip: Scrub to see how the network learned over time
      </div>
    </div>
  );
});
