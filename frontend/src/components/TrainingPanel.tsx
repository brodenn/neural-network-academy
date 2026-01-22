import { useState, useEffect, memo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { NetworkSettings } from './NetworkSettings';
import { TrainingEducationalViz } from './TrainingEducationalViz';
import type { WeightInit, HiddenActivation, ProblemInfo } from '../types';

interface TrainingPanelProps {
  currentEpoch: number;
  currentLoss: number;
  currentAccuracy: number;
  trainingInProgress: boolean;
  trainingComplete: boolean;
  onStartStatic: (epochs: number, learningRate: number) => void;
  onStartAdaptive: (targetAccuracy: number) => void;
  onStep?: (learningRate: number) => void;  // Optional for CNN
  onStop: () => void;
  onUpdateTarget: (targetAccuracy: number) => void;
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
  // Failure case support
  currentProblem?: ProblemInfo | null;
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
  onStop,
  onUpdateTarget,
  onReset,
  onSettingsChange,
  currentArchitecture,
  currentWeightInit,
  currentHiddenActivation,
  currentUseBiases,
  isCNN = false,
  currentProblem,
}: TrainingPanelProps) {
  // Failure case detection
  const isFailureCase = currentProblem?.is_failure_case ?? false;
  const isArchitectureLocked = currentProblem?.locked_architecture ?? false;
  const forcedLR = currentProblem?.forced_learning_rate;
  const forcedInit = currentProblem?.forced_weight_init;
  const [epochs, setEpochs] = useState(1000);
  const [learningRate, setLearningRate] = useState(0.5);
  const [layerInput, setLayerInput] = useState('');
  const [showSettings, setShowSettings] = useState(false);
  // Target accuracy: default 95% for CNN, 99% for dense
  const [targetAccuracy, setTargetAccuracy] = useState(isCNN ? 0.95 : 0.99);
  // Educational visualization state
  const [showEducational, setShowEducational] = useState(false);
  const [showLearnHint, setShowLearnHint] = useState(true);

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
    <div className="bg-gray-800 rounded-lg p-3" data-testid="training-panel" role="region" aria-label="Training Controls">
      {/* Educational visualization modal */}
      <AnimatePresence>
        {showEducational && (
          <TrainingEducationalViz
            layerSizes={currentArchitecture}
            onClose={() => setShowEducational(false)}
          />
        )}
      </AnimatePresence>

      <div className="flex justify-between items-center mb-3">
        <h2 className="text-lg font-semibold">Training Controls</h2>
        {!showLearnHint && (
          <motion.button
            onClick={() => setShowEducational(true)}
            className="px-3 py-1.5 text-xs font-medium bg-gradient-to-r from-yellow-600 to-orange-600 hover:from-yellow-500 hover:to-orange-500 text-white rounded-lg transition-all flex items-center gap-1.5 shadow-md"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            <span>🎓</span> Learn How Training Works
          </motion.button>
        )}
      </div>

      {/* Failure Case Warning Banner */}
      {isFailureCase && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-red-900/40 border border-red-500/50 rounded-lg p-3 mb-4"
        >
          <div className="flex items-center gap-2 mb-2">
            <span className="text-xl">⚠️</span>
            <span className="text-red-300 font-semibold text-sm">Intentional Failure Case</span>
          </div>
          <p className="text-red-200 text-xs mb-2">
            This problem is designed to fail. Watch what happens when you train!
          </p>
          {currentProblem?.failure_reason && (
            <div className="text-xs mb-2">
              <span className="text-red-400 font-medium">Why it fails: </span>
              <span className="text-red-200">{currentProblem.failure_reason}</span>
            </div>
          )}
          {isArchitectureLocked && (
            <div className="text-xs text-yellow-400 flex items-center gap-1">
              <span>🔒</span>
              <span>Architecture is locked to demonstrate the failure</span>
            </div>
          )}
          {forcedLR != null && (
            <div className="text-xs text-yellow-400 flex items-center gap-1 mt-1">
              <span>⚡</span>
              <span>Learning rate is forced to {forcedLR}</span>
            </div>
          )}
          {forcedInit && (
            <div className="text-xs text-yellow-400 flex items-center gap-1 mt-1">
              <span>🎲</span>
              <span>Weight initialization is forced to {forcedInit}</span>
            </div>
          )}
        </motion.div>
      )}

      {/* Prominent Learn Banner */}
      <AnimatePresence>
        {showLearnHint && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="bg-gradient-to-r from-yellow-900/80 via-orange-900/80 to-yellow-900/80 rounded-lg p-3 border border-yellow-500/30 relative overflow-hidden mb-4"
          >
            {/* Animated background shimmer */}
            <motion.div
              className="absolute inset-0 bg-gradient-to-r from-transparent via-white/5 to-transparent"
              animate={{ x: ['-100%', '100%'] }}
              transition={{ duration: 2, repeat: Infinity, ease: 'linear' }}
            />

            <div className="relative flex items-center justify-between gap-3">
              <div className="flex items-center gap-3">
                <motion.div
                  className="w-10 h-10 rounded-full bg-yellow-500/20 flex items-center justify-center text-xl"
                  animate={{ scale: [1, 1.1, 1] }}
                  transition={{ duration: 1.5, repeat: Infinity }}
                >
                  🎓
                </motion.div>
                <div>
                  <div className="text-white font-medium text-sm">Learn How Training Works</div>
                  <div className="text-yellow-300/80 text-xs">Forward pass, backprop, weight updates & more</div>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <motion.button
                  onClick={() => {
                    setShowEducational(true);
                    setShowLearnHint(false);
                  }}
                  className="px-4 py-2 bg-yellow-500 hover:bg-yellow-400 text-gray-900 font-medium rounded-lg text-sm flex items-center gap-2 shadow-lg shadow-yellow-500/25"
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  <span>📚</span> Start Learning
                </motion.button>
                <button
                  onClick={() => setShowLearnHint(false)}
                  className="text-gray-400 hover:text-white p-1"
                  title="Dismiss"
                >
                  ✕
                </button>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Status display with training state indicator */}
      {(() => {
        // Determine training state for visual indicator
        const isUntrained = currentEpoch === 0 && currentAccuracy === 0;
        const isTrained = trainingComplete && !trainingInProgress && currentAccuracy >= 0.9;

        // Detect if training is "stuck" (failure case behavior)
        const isStuck = currentEpoch > 500 && currentAccuracy < 0.6;
        const isVeryStuck = currentEpoch > 2000 && currentAccuracy < 0.55;

        // Border styles based on state
        let borderClass = 'border-2 border-gray-600'; // Untrained (gray)
        let statusText = '';
        let statusColor = '';

        if (trainingInProgress) {
          if (isFailureCase && isVeryStuck) {
            borderClass = 'border-2 border-red-500 animate-pulse';
            statusText = '✗ Failing as expected...';
            statusColor = 'text-red-400';
          } else if (isFailureCase && isStuck) {
            borderClass = 'border-2 border-orange-500 animate-pulse';
            statusText = '⚠ Struggling...';
            statusColor = 'text-orange-400';
          } else {
            borderClass = 'border-2 border-yellow-500 animate-pulse';
            statusText = '● Training...';
            statusColor = 'text-yellow-400';
          }
        } else if (isFailureCase && currentAccuracy < 0.6 && currentEpoch > 100) {
          borderClass = 'border-2 border-red-500';
          statusText = '✗ Failed (as expected!)';
          statusColor = 'text-red-400';
        } else if (isTrained) {
          borderClass = 'border-2 border-green-500';
          statusText = '✓ Trained';
          statusColor = 'text-green-400';
        } else if (isUntrained) {
          borderClass = 'border-2 border-gray-600';
          statusText = '○ Untrained';
          statusColor = 'text-gray-500';
        } else {
          // Partially trained (some epochs but not complete)
          borderClass = 'border-2 border-orange-500/50';
          statusText = '◐ Partial';
          statusColor = 'text-orange-400';
        }

        return (
          <>
            {/* Training state badge */}
            <div className={`text-sm font-medium mb-2 ${statusColor}`}>
              {statusText}
            </div>

            <div className="grid grid-cols-3 gap-3 mb-4" role="group" aria-label="Training metrics">
              <div className={`bg-gray-700 rounded p-3 text-center ${borderClass} transition-all duration-300`} data-testid="metric-epoch">
                <div className="text-gray-400 text-sm">Epoch</div>
                <div className="text-xl font-mono" aria-label={`Current epoch: ${currentEpoch}`}>{currentEpoch}</div>
              </div>
              <div className={`bg-gray-700 rounded p-3 text-center ${borderClass} transition-all duration-300`} data-testid="metric-loss">
                <div className="text-gray-400 text-sm">Loss</div>
                <div className="text-xl font-mono" aria-label={`Loss: ${currentLoss.toFixed(4)}`}>{currentLoss.toFixed(4)}</div>
              </div>
              <div className={`bg-gray-700 rounded p-3 text-center ${borderClass} transition-all duration-300`} data-testid="metric-accuracy">
                <div className="text-gray-400 text-sm">Accuracy</div>
                <div className={`text-xl font-mono ${currentAccuracy >= 0.95 ? 'text-green-400' : ''}`} aria-label={`Accuracy: ${(currentAccuracy * 100).toFixed(1)} percent`}>
                  {(currentAccuracy * 100).toFixed(1)}%
                </div>
              </div>
            </div>
          </>
        );
      })()}

      {/* Architecture config - only for dense networks */}
      {canCustomize && (
        <>
          <div className="mb-4">
            <label htmlFor="hidden-layers-input" className="block text-gray-400 text-sm mb-2">
              Hidden Layers (comma-separated)
              {isArchitectureLocked && (
                <span className="ml-2 text-yellow-400 text-xs">🔒 Locked</span>
              )}
            </label>
            <div className="flex gap-2">
              <div className="bg-gray-700 px-3 py-2 rounded text-gray-400" aria-hidden="true">{inputSize} →</div>
              <input
                id="hidden-layers-input"
                type="text"
                value={layerInput}
                onChange={(e) => setLayerInput(e.target.value)}
                placeholder="12, 8, 4"
                className={`flex-1 bg-gray-700 rounded px-3 py-2 text-white ${
                  isArchitectureLocked ? 'opacity-50 cursor-not-allowed' : ''
                }`}
                disabled={trainingInProgress || isArchitectureLocked}
                data-testid="hidden-layers-input"
              />
              <div className="bg-gray-700 px-3 py-2 rounded text-gray-400" aria-hidden="true">→ {outputSize}</div>
            </div>
            <div className="text-gray-500 text-sm mt-1">
              Current: [{currentArchitecture.join(' → ')}]
              {isArchitectureLocked && (
                <span className="text-yellow-500 ml-2">(cannot change - demonstrating failure)</span>
              )}
            </div>
          </div>

          {/* Toggle for advanced settings */}
          <button
            onClick={() => setShowSettings(!showSettings)}
            className="w-full mb-4 py-2 text-sm text-gray-400 hover:text-white border border-gray-600 rounded flex items-center justify-center gap-2"
            disabled={isArchitectureLocked}
          >
            <span>{showSettings ? '▼' : '▶'}</span>
            <span>Network Settings (learn what each does!)</span>
            {isArchitectureLocked && <span className="text-yellow-400">🔒</span>}
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
                disabled={trainingInProgress || isArchitectureLocked}
              />
              {forcedInit && (
                <div className="mt-2 text-xs text-yellow-400">
                  ⚠ Weight init is forced to "{forcedInit}" for this problem
                </div>
              )}
            </div>
          )}

          {/* Apply button for architecture + settings */}
          <button
            onClick={handleApplySettings}
            disabled={trainingInProgress || isArchitectureLocked}
            className={`w-full mb-4 py-2 rounded font-semibold ${
              isArchitectureLocked
                ? 'bg-gray-600 text-gray-400 cursor-not-allowed'
                : 'bg-purple-600 hover:bg-purple-700 disabled:opacity-50'
            }`}
          >
            {isArchitectureLocked ? '🔒 Architecture Locked' : 'Apply Architecture & Settings'}
          </button>
        </>
      )}

      {/* CNN mode indicator */}
      {!canCustomize && (
        <div className="mb-4 p-3 bg-cyan-900/30 border border-cyan-600/30 rounded text-sm text-cyan-300">
          CNN Mode - Architecture is fixed for shape detection
        </div>
      )}

      {/* Static Training Section */}
      <div className={`mb-4 p-3 rounded ${
        isFailureCase && forcedLR != null
          ? 'bg-red-900/20 border border-red-600/30'
          : 'bg-blue-900/20 border border-blue-600/30'
      }`}>
        <div className={`text-xs font-semibold mb-2 uppercase tracking-wide ${
          isFailureCase && forcedLR != null ? 'text-red-400' : 'text-blue-400'
        }`}>Static Training</div>
        <div className="grid grid-cols-2 gap-3 mb-3">
          <div>
            <label htmlFor="epochs-input" className="block text-gray-400 text-xs mb-1">Epochs</label>
            <input
              id="epochs-input"
              type="number"
              value={epochs}
              onChange={(e) => setEpochs(parseInt(e.target.value, 10) || 1000)}
              className="w-full bg-gray-700 rounded px-3 py-2 text-white text-sm"
              disabled={trainingInProgress}
              data-testid="epochs-input"
            />
          </div>
          <div>
            <label htmlFor="learning-rate-input" className="block text-gray-400 text-xs mb-1">
              Learning Rate
              {forcedLR != null && (
                <span className="ml-1 text-yellow-400">🔒 {forcedLR}</span>
              )}
            </label>
            <input
              id="learning-rate-input"
              type="number"
              step="0.01"
              value={forcedLR != null ? forcedLR : learningRate}
              onChange={(e) => setLearningRate(parseFloat(e.target.value) || 0.1)}
              className={`w-full bg-gray-700 rounded px-3 py-2 text-white text-sm ${
                forcedLR != null ? 'opacity-50 cursor-not-allowed text-yellow-400' : ''
              }`}
              disabled={trainingInProgress || forcedLR != null}
              data-testid="learning-rate-input"
            />
          </div>
        </div>
        <button
          onClick={() => onStartStatic(epochs, forcedLR != null ? forcedLR : learningRate)}
          disabled={trainingInProgress}
          data-testid="train-static-btn"
          aria-label={isFailureCase ? 'Train Static - Watch it Fail' : 'Train Static'}
          className={`w-full py-2 rounded font-semibold text-sm disabled:opacity-50 disabled:cursor-not-allowed ${
            isFailureCase
              ? 'bg-red-600 hover:bg-red-700'
              : 'bg-blue-600 hover:bg-blue-700'
          }`}
        >
          {isFailureCase ? 'Train Static (Watch it Fail!)' : 'Train Static'}
        </button>
      </div>

      {/* Adaptive Training Section */}
      <div className="mb-4 p-3 bg-green-900/20 border border-green-600/30 rounded">
        <div className="text-green-400 text-xs font-semibold mb-2 uppercase tracking-wide">Adaptive Training</div>
        <div className="text-gray-500 text-xs mb-2">Auto-adjusts learning rate (starts at 1.0, decays to 0.01)</div>
        <div className="mb-3">
          <label htmlFor="target-accuracy-slider" className="block text-gray-400 text-xs mb-1">
            Target Accuracy: <span className="text-green-400 font-mono">{(targetAccuracy * 100).toFixed(0)}%</span>
            {trainingInProgress && (
              <span className="text-yellow-400 ml-2">(adjustable during training)</span>
            )}
          </label>
          <input
            id="target-accuracy-slider"
            type="range"
            min={50}
            max={100}
            step={1}
            value={targetAccuracy * 100}
            onChange={(e) => {
              const newTarget = parseInt(e.target.value, 10) / 100;
              setTargetAccuracy(newTarget);
              // If training is in progress, update the backend target
              if (trainingInProgress) {
                onUpdateTarget(newTarget);
              }
            }}
            className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-green-500"
            aria-valuemin={50}
            aria-valuemax={100}
            aria-valuenow={targetAccuracy * 100}
            data-testid="target-accuracy-slider"
          />
          <div className="flex justify-between text-xs text-gray-500 mt-1">
            <span>50%</span>
            <span>Default: {isCNN ? '95%' : '99%'}</span>
            <span>100%</span>
          </div>
          {trainingInProgress && currentAccuracy > 0 && (
            <div className="mt-2 text-xs">
              {currentAccuracy >= targetAccuracy ? (
                <span className="text-green-400">✓ Current accuracy meets target - will stop soon</span>
              ) : (
                <span className="text-gray-400">
                  Need {((targetAccuracy - currentAccuracy) * 100).toFixed(1)}% more to reach target
                </span>
              )}
            </div>
          )}
        </div>
        <button
          onClick={() => onStartAdaptive(targetAccuracy)}
          disabled={trainingInProgress}
          data-testid="train-adaptive-btn"
          aria-label={`Train Adaptive to ${(targetAccuracy * 100).toFixed(0)} percent accuracy`}
          className="w-full bg-green-600 hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed py-2 rounded font-semibold text-sm"
        >
          Train Adaptive
        </button>
      </div>

      {/* Stop button - shown during training */}
      {trainingInProgress && (
        <button
          onClick={onStop}
          data-testid="stop-training-btn"
          aria-label="Stop Training and keep current progress"
          className="w-full mb-4 bg-red-600 hover:bg-red-700 py-3 rounded font-semibold text-sm flex items-center justify-center gap-2 animate-pulse"
        >
          <span aria-hidden="true">■</span> Stop Training (Keep Current Progress)
        </button>
      )}

      {/* Reset and Step buttons */}
      <div className="flex gap-3 mb-3" role="group" aria-label="Training actions">
        {onStep && (
          <button
            onClick={() => onStep(learningRate)}
            disabled={trainingInProgress}
            data-testid="step-btn"
            aria-label="Train for 1 epoch"
            className="flex-1 bg-orange-600 hover:bg-orange-700 disabled:opacity-50 disabled:cursor-not-allowed py-2 rounded font-semibold text-sm"
            title="Train for 1 epoch using static learning rate"
          >
            Step (1 Epoch)
          </button>
        )}
        <button
          onClick={onReset}
          disabled={trainingInProgress}
          data-testid="reset-btn"
          aria-label="Reset network to initial state"
          className="flex-1 bg-gray-600 hover:bg-gray-500 disabled:opacity-50 py-2 rounded font-semibold text-sm"
        >
          Reset Network
        </button>
      </div>

      {/* Status message - show fix suggestion for failure cases */}
      {trainingComplete && !trainingInProgress && (
        <>
          {isFailureCase && currentAccuracy < 0.6 && currentProblem?.fix_suggestion ? (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="mt-4 bg-green-900/30 border border-green-500/30 rounded-lg p-3"
            >
              <div className="flex items-center gap-2 mb-2">
                <span className="text-lg">💡</span>
                <span className="text-green-400 font-medium text-sm">How to Fix This</span>
              </div>
              <p className="text-green-200 text-xs">{currentProblem.fix_suggestion}</p>
              <p className="text-gray-400 text-xs mt-2">
                Try a different problem to see this concept work correctly!
              </p>
            </motion.div>
          ) : (
            <div className="mt-4 text-center text-green-400 text-sm">
              Try the inputs above to test the network!
            </div>
          )}
        </>
      )}

      {/* Show fix suggestion during training if stuck */}
      {trainingInProgress && isFailureCase && currentEpoch > 1000 && currentAccuracy < 0.55 && currentProblem?.fix_suggestion && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-4 bg-yellow-900/30 border border-yellow-500/30 rounded-lg p-3"
        >
          <div className="text-yellow-400 text-xs font-medium mb-1">💡 This is expected!</div>
          <p className="text-yellow-200 text-xs">{currentProblem.failure_reason}</p>
        </motion.div>
      )}
    </div>
  );
});
