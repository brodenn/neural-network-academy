import { useState, useEffect, memo } from 'react';
import { NetworkSettings } from './NetworkSettings';
import type { WeightInit, HiddenActivation } from '../types';

interface TrainingPanelProps {
  currentEpoch: number;
  currentLoss: number;
  currentAccuracy: number;
  trainingInProgress: boolean;
  trainingComplete: boolean;
  onStartStatic: (epochs: number, learningRate: number) => void;
  onStartAdaptive: (targetAccuracy: number) => void;
  onStep?: (learningRate: number) => void;  // Optional for CNN
  onReset: () => void;
  onSettingsChange?: (settings: {  // Optional for CNN
    layers: number[];
    weightInit: WeightInit;
    hiddenActivation: HiddenActivation;
    useBiases: boolean;
  }) => void;
  currentArchitecture: number[];
  currentWeightInit: WeightInit;
  currentHiddenActivation: HiddenActivation;
  currentUseBiases: boolean;
  isCNN?: boolean;
}

export const TrainingPanel = memo(function TrainingPanel({
  currentEpoch,
  currentLoss,
  currentAccuracy,
  trainingInProgress,
  trainingComplete,
  onStartStatic,
  onStartAdaptive,
  onStep,
  onReset,
  onSettingsChange,
  currentArchitecture,
  currentWeightInit,
  currentHiddenActivation,
  currentUseBiases,
  isCNN = false,
}: TrainingPanelProps) {
  const [epochs, setEpochs] = useState(1000);
  const [learningRate, setLearningRate] = useState(0.5);
  const [layerInput, setLayerInput] = useState('');
  const [showSettings, setShowSettings] = useState(false);
  // Target accuracy: default 95% for CNN, 99% for dense
  const [targetAccuracy, setTargetAccuracy] = useState(isCNN ? 0.95 : 0.99);

  // Local state for settings (to batch changes)
  const [weightInit, setWeightInit] = useState<WeightInit>(currentWeightInit);
  const [hiddenActivation, setHiddenActivation] = useState<HiddenActivation>(currentHiddenActivation);
  const [useBiases, setUseBiases] = useState(currentUseBiases);

  // Get input/output sizes from current architecture
  const inputSize = currentArchitecture[0] ?? 5;
  const outputSize = currentArchitecture[currentArchitecture.length - 1] ?? 1;

  // Update local state when props change (e.g., problem switch)
  useEffect(() => {
    const hiddenLayers = currentArchitecture.slice(1, -1);
    setLayerInput(hiddenLayers.join(', '));
  }, [currentArchitecture]);

  useEffect(() => {
    setWeightInit(currentWeightInit);
    setHiddenActivation(currentHiddenActivation);
    setUseBiases(currentUseBiases);
  }, [currentWeightInit, currentHiddenActivation, currentUseBiases]);

  // Update target accuracy default when network type changes
  useEffect(() => {
    setTargetAccuracy(isCNN ? 0.95 : 0.99);
  }, [isCNN]);

  // Listen for learning rate preset clicks
  useEffect(() => {
    const handler = (e: CustomEvent<number>) => {
      setLearningRate(e.detail);
    };
    window.addEventListener('setLearningRate', handler as EventListener);
    return () => window.removeEventListener('setLearningRate', handler as EventListener);
  }, []);

  const handleApplySettings = () => {
    if (!onSettingsChange) return;  // Not available for CNN

    const hiddenLayers = layerInput
      .split(',')
      .map((s) => parseInt(s.trim(), 10))
      .filter((n) => !isNaN(n) && n > 0);

    if (hiddenLayers.length > 0) {
      const newArchitecture = [inputSize, ...hiddenLayers, outputSize];
      onSettingsChange({
        layers: newArchitecture,
        weightInit,
        hiddenActivation,
        useBiases,
      });
    }
  };

  // Check if customization is available (not for CNN)
  const canCustomize = !!onSettingsChange;

  return (
    <div className="bg-gray-800 rounded-lg p-6">
      <h2 className="text-xl font-semibold mb-4">Training Controls</h2>

      {/* Status display */}
      <div className="grid grid-cols-3 gap-4 mb-6">
        <div className="bg-gray-700 rounded p-3 text-center">
          <div className="text-gray-400 text-sm">Epoch</div>
          <div className="text-xl font-mono">{currentEpoch}</div>
        </div>
        <div className="bg-gray-700 rounded p-3 text-center">
          <div className="text-gray-400 text-sm">Loss</div>
          <div className="text-xl font-mono">{currentLoss.toFixed(4)}</div>
        </div>
        <div className="bg-gray-700 rounded p-3 text-center">
          <div className="text-gray-400 text-sm">Accuracy</div>
          <div className={`text-xl font-mono ${currentAccuracy >= 0.99 ? 'text-green-400' : ''}`}>
            {(currentAccuracy * 100).toFixed(1)}%
          </div>
        </div>
      </div>

      {/* Architecture config - only for dense networks */}
      {canCustomize && (
        <>
          <div className="mb-4">
            <label className="block text-gray-400 text-sm mb-2">
              Hidden Layers (comma-separated)
            </label>
            <div className="flex gap-2">
              <div className="bg-gray-700 px-3 py-2 rounded text-gray-400">{inputSize} →</div>
              <input
                type="text"
                value={layerInput}
                onChange={(e) => setLayerInput(e.target.value)}
                placeholder="12, 8, 4"
                className="flex-1 bg-gray-700 rounded px-3 py-2 text-white"
                disabled={trainingInProgress}
              />
              <div className="bg-gray-700 px-3 py-2 rounded text-gray-400">→ {outputSize}</div>
            </div>
            <div className="text-gray-500 text-sm mt-1">
              Current: [{currentArchitecture.join(' → ')}]
            </div>
          </div>

          {/* Toggle for advanced settings */}
          <button
            onClick={() => setShowSettings(!showSettings)}
            className="w-full mb-4 py-2 text-sm text-gray-400 hover:text-white border border-gray-600 rounded flex items-center justify-center gap-2"
          >
            <span>{showSettings ? '▼' : '▶'}</span>
            <span>Network Settings (learn what each does!)</span>
          </button>

          {/* Collapsible settings */}
          {showSettings && (
            <div className="mb-4 p-4 bg-gray-700/50 rounded-lg">
              <NetworkSettings
                weightInit={weightInit}
                hiddenActivation={hiddenActivation}
                useBiases={useBiases}
                onWeightInitChange={setWeightInit}
                onHiddenActivationChange={setHiddenActivation}
                onUseBiasesChange={setUseBiases}
                disabled={trainingInProgress}
              />
            </div>
          )}

          {/* Apply button for architecture + settings */}
          <button
            onClick={handleApplySettings}
            disabled={trainingInProgress}
            className="w-full mb-4 bg-purple-600 hover:bg-purple-700 disabled:opacity-50 py-2 rounded font-semibold"
          >
            Apply Architecture & Settings
          </button>
        </>
      )}

      {/* CNN mode indicator */}
      {!canCustomize && (
        <div className="mb-4 p-3 bg-cyan-900/30 border border-cyan-600/30 rounded text-sm text-cyan-300">
          CNN Mode - Architecture is fixed for shape detection
        </div>
      )}

      {/* Training params */}
      <div className="grid grid-cols-2 gap-4 mb-4">
        <div>
          <label className="block text-gray-400 text-sm mb-1">Epochs (static)</label>
          <input
            type="number"
            value={epochs}
            onChange={(e) => setEpochs(parseInt(e.target.value, 10) || 1000)}
            className="w-full bg-gray-700 rounded px-3 py-2 text-white"
            disabled={trainingInProgress}
          />
        </div>
        <div>
          <label className="block text-gray-400 text-sm mb-1">Learning Rate</label>
          <input
            type="number"
            step="0.01"
            value={learningRate}
            onChange={(e) => setLearningRate(parseFloat(e.target.value) || 0.1)}
            className="w-full bg-gray-700 rounded px-3 py-2 text-white"
            disabled={trainingInProgress}
          />
        </div>
      </div>

      {/* Target Accuracy for adaptive training */}
      <div className="mb-4">
        <label className="block text-gray-400 text-sm mb-1">
          Target Accuracy (adaptive): <span className="text-green-400 font-mono">{(targetAccuracy * 100).toFixed(0)}%</span>
        </label>
        <input
          type="range"
          min={50}
          max={100}
          step={1}
          value={targetAccuracy * 100}
          onChange={(e) => setTargetAccuracy(parseInt(e.target.value, 10) / 100)}
          disabled={trainingInProgress}
          className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-green-500"
        />
        <div className="flex justify-between text-xs text-gray-500 mt-1">
          <span>50%</span>
          <span className="text-gray-400">Default: {isCNN ? '95%' : '99%'}</span>
          <span>100%</span>
        </div>
      </div>

      {/* Action buttons */}
      <div className="flex gap-3 mb-3">
        <button
          onClick={() => onStartStatic(epochs, learningRate)}
          disabled={trainingInProgress}
          className="flex-1 bg-blue-600 hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed py-3 rounded font-semibold"
        >
          Train Static
        </button>
        <button
          onClick={() => onStartAdaptive(targetAccuracy)}
          disabled={trainingInProgress}
          className="flex-1 bg-green-600 hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed py-3 rounded font-semibold"
        >
          Train Adaptive
        </button>
        <button
          onClick={onReset}
          disabled={trainingInProgress}
          className="bg-gray-600 hover:bg-gray-500 disabled:opacity-50 px-4 py-3 rounded"
        >
          Reset
        </button>
      </div>

      {/* Step-by-step training - only for dense networks */}
      {onStep && (
        <div className="flex gap-3">
          <button
            onClick={() => onStep(learningRate)}
            disabled={trainingInProgress}
            className="flex-1 bg-orange-600 hover:bg-orange-700 disabled:opacity-50 disabled:cursor-not-allowed py-2 rounded font-semibold text-sm"
            title="Train for 1 epoch - great for watching weights change!"
          >
            Step (1 Epoch)
          </button>
        </div>
      )}

      {/* Status message */}
      {trainingInProgress && (
        <div className="mt-4 text-center text-yellow-400 animate-pulse">
          Training in progress...
        </div>
      )}
      {trainingComplete && !trainingInProgress && (
        <div className="mt-4 text-center text-green-400">
          Training complete! Try the inputs above.
        </div>
      )}
    </div>
  );
});
