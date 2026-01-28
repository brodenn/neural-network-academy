import { useState, useEffect, useCallback } from 'react';
import { useSocket } from './hooks/useSocket';
import { useKeyboardShortcuts } from './hooks/useKeyboardShortcuts';
import { ProblemSelector } from './components/ProblemSelector';
import { InputPanel } from './components/InputPanel';
import { OutputDisplay } from './components/OutputDisplay';
import { TrainingPanel } from './components/TrainingPanel';
import { LossCurve } from './components/LossCurve';
import { NetworkVisualization } from './components/NetworkVisualization';
import { FeatureMapVisualization } from './components/FeatureMapVisualization';
import { WeightHistogram } from './components/WeightHistogram';
import { DecisionBoundaryViz } from './components/DecisionBoundaryViz';
import { KeyboardShortcuts } from './components/KeyboardShortcuts';
import { ProblemInfoModal } from './components/ProblemInfoModal';
import { ContextualTips } from './components/ContextualTips';
import { LossLandscape3D } from './components/LossLandscape3D';
import { LearningPathSelector } from './components/LearningPathSelector';
import { PathDetailView } from './components/PathDetailView';
import { ToastProvider } from './components/Toast';
import { AchievementProvider } from './components/AchievementProvider';
import type { NetworkState, ProblemInfo, NetworkType, CNNFeatureMaps } from './types';

const API_URL = 'http://localhost:5000';

function App() {
  const {
    connected,
    trainingProgress,
    lastPrediction,
    trainingComplete,
    trainingError,
    socket,
  } = useSocket();

  const [networkState, setNetworkState] = useState<NetworkState | null>(null);
  const [trainingInProgress, setTrainingInProgress] = useState(false);
  const [apiError, setApiError] = useState<string | null>(null);

  // View state - toggle between learning paths, problems, and path detail
  type ViewMode = 'paths' | 'problems' | 'path-detail';
  const [currentView, setCurrentView] = useState<ViewMode>('problems');
  const [currentPathId, setCurrentPathId] = useState<string | null>(null);

  // Clear API error after 5 seconds
  useEffect(() => {
    if (apiError) {
      const timer = setTimeout(() => setApiError(null), 5000);
      return () => clearTimeout(timer);
    }
  }, [apiError]);

  // Problem state
  const [problems, setProblems] = useState<ProblemInfo[]>([]);
  const [currentProblem, setCurrentProblem] = useState<ProblemInfo | null>(null);
  const [inputValues, setInputValues] = useState<number[] | number[][]>([]);
  const [networkType, setNetworkType] = useState<NetworkType>('dense');
  const [featureMaps, setFeatureMaps] = useState<CNNFeatureMaps | null>(null);
  const [isInfoModalOpen, setIsInfoModalOpen] = useState(false);
  const [show3DLandscape, setShow3DLandscape] = useState(false);

  // Fetch available problems
  const fetchProblems = useCallback(async () => {
    try {
      const res = await fetch(`${API_URL}/api/problems`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setProblems(data);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Unknown error';
      console.error('Failed to fetch problems:', message);
      setApiError(`Failed to fetch problems: ${message}`);
    }
  }, []);

  // Fetch network state
  const fetchNetworkState = useCallback(async () => {
    try {
      const res = await fetch(`${API_URL}/api/network`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setNetworkState(data);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Unknown error';
      console.error('Failed to fetch network state:', message);
      // Don't show error for network state polling - too noisy
    }
  }, []);

  // Initial fetch
  useEffect(() => {
    fetchProblems();
    fetchNetworkState();
  }, [fetchProblems, fetchNetworkState]);

  // Helper to initialize input values based on network type
  const initializeInputValues = (info: ProblemInfo) => {
    if (info.network_type === 'cnn' && info.input_shape) {
      // Initialize 2D grid for CNN
      const [h, w] = info.input_shape;
      return Array.from({ length: h }, () => Array(w).fill(0));
    }
    // Initialize 1D array for dense networks
    return new Array(info.input_labels.length).fill(0);
  };

  // Listen for problem info and training events from websocket
  useEffect(() => {
    if (socket) {
      // Sync training state from backend on status events (initial connection + reconnection)
      const handleStatus = (status: { training_in_progress?: boolean }) => {
        if (status.training_in_progress !== undefined) {
          setTrainingInProgress(status.training_in_progress);
        }
      };

      const handleProblemInfo = (info: ProblemInfo) => {
        setCurrentProblem(info);
        setNetworkType(info.network_type || 'dense');
        setInputValues(initializeInputValues(info));
        setFeatureMaps(null);
      };

      const handleProblemChanged = (data: { problem_id: string; info: ProblemInfo }) => {
        setCurrentProblem(data.info);
        setNetworkType(data.info.network_type || 'dense');
        setInputValues(initializeInputValues(data.info));
        setFeatureMaps(null);
        fetchNetworkState();
      };

      const handleTrainingStarted = () => {
        setTrainingInProgress(true);
      };

      const handleTrainingComplete = () => {
        setTrainingInProgress(false);
        fetchNetworkState();
      };

      const handleTrainingStopped = () => {
        setTrainingInProgress(false);
        fetchNetworkState();
      };

      const handleTrainingError = (data: { error: string }) => {
        // Training crashed - reset state
        console.error('Training error:', data.error);
        setTrainingInProgress(false);
      };

      socket.on('status', handleStatus);
      socket.on('problem_info', handleProblemInfo);
      socket.on('problem_changed', handleProblemChanged);
      socket.on('training_started', handleTrainingStarted);
      socket.on('training_complete', handleTrainingComplete);
      socket.on('training_stopped', handleTrainingStopped);
      socket.on('training_error', handleTrainingError);

      return () => {
        socket.off('status', handleStatus);
        socket.off('problem_info', handleProblemInfo);
        socket.off('problem_changed', handleProblemChanged);
        socket.off('training_started', handleTrainingStarted);
        socket.off('training_complete', handleTrainingComplete);
        socket.off('training_stopped', handleTrainingStopped);
        socket.off('training_error', handleTrainingError);
      };
    }
  }, [socket, fetchNetworkState]);

  // Poll network state during training
  useEffect(() => {
    if (trainingInProgress) {
      const interval = setInterval(fetchNetworkState, 500);
      return () => clearInterval(interval);
    }
  }, [trainingInProgress, fetchNetworkState]);

  // Track feature maps from predictions
  useEffect(() => {
    if (lastPrediction?.feature_maps) {
      setFeatureMaps(lastPrediction.feature_maps);
    }
  }, [lastPrediction]);

  // Handle problem selection
  const handleProblemSelect = async (problemId: string) => {
    try {
      const res = await fetch(`${API_URL}/api/problems/${problemId}/select`, { method: 'POST' });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.error || `HTTP ${res.status}`);
      }
      // Problem change will be handled via websocket
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Unknown error';
      console.error('Failed to select problem:', message);
      setApiError(`Failed to select problem: ${message}`);
    }
  };

  // Handle learning path selection - navigate to path detail view
  const handlePathSelect = (pathId: string) => {
    setCurrentPathId(pathId);
    setCurrentView('path-detail');
  };

  // Handle exiting path detail view
  const handleExitPath = () => {
    setCurrentPathId(null);
    setCurrentView('paths');
  };

  // Handle input changes
  const handleInputChange = (values: number[] | number[][]) => {
    setInputValues(values);
    if (socket && trainingComplete) {
      socket.emit('set_inputs', { inputs: values });
    }
  };

  // Training handlers
  const handleStartStatic = async (epochs: number, learningRate: number) => {
    setTrainingInProgress(true);
    try {
      const res = await fetch(`${API_URL}/api/train`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ epochs, learning_rate: learningRate }),
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.error || `HTTP ${res.status}`);
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Unknown error';
      console.error('Failed to start training:', message);
      setApiError(`Failed to start training: ${message}`);
      setTrainingInProgress(false);
    }
  };

  const handleStartAdaptive = async (targetAccuracy: number) => {
    setTrainingInProgress(true);
    try {
      const res = await fetch(`${API_URL}/api/train/adaptive`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ target_accuracy: targetAccuracy }),
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.error || `HTTP ${res.status}`);
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Unknown error';
      console.error('Failed to start adaptive training:', message);
      setApiError(`Failed to start training: ${message}`);
      setTrainingInProgress(false);
    }
  };

  const handleStop = async () => {
    try {
      const res = await fetch(`${API_URL}/api/train/stop`, { method: 'POST' });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.error || `HTTP ${res.status}`);
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Unknown error';
      console.error('Failed to stop training:', message);
      setApiError(`Failed to stop training: ${message}`);
    }
  };

  const handleUpdateTarget = async (targetAccuracy: number) => {
    try {
      const res = await fetch(`${API_URL}/api/train/target`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ target_accuracy: targetAccuracy }),
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.error || `HTTP ${res.status}`);
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Unknown error';
      console.error('Failed to update target accuracy:', message);
      // Don't show error for target updates - minor operation
    }
  };

  const handleReset = async () => {
    try {
      const res = await fetch(`${API_URL}/api/network/reset`, { method: 'POST' });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.error || `HTTP ${res.status}`);
      }
      await fetchNetworkState();
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Unknown error';
      console.error('Failed to reset network:', message);
      setApiError(`Failed to reset network: ${message}`);
    }
  };

  const handleStep = async (learningRate: number) => {
    try {
      const res = await fetch(`${API_URL}/api/train/step`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ learning_rate: learningRate }),
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.error || `HTTP ${res.status}`);
      }
      await fetchNetworkState();
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Unknown error';
      console.error('Failed to step train:', message);
      setApiError(`Failed to step train: ${message}`);
    }
  };

  const handleSettingsChange = async (settings: {
    layers: number[];
    weightInit: string;
    hiddenActivation: string;
    useBiases: boolean;
  }) => {
    try {
      const res = await fetch(`${API_URL}/api/network/architecture`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          layer_sizes: settings.layers,
          weight_init: settings.weightInit,
          hidden_activation: settings.hiddenActivation,
          use_biases: settings.useBiases,
        }),
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.error || `HTTP ${res.status}`);
      }
      await fetchNetworkState();
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Unknown error';
      console.error('Failed to change settings:', message);
      setApiError(`Failed to change settings: ${message}`);
    }
  };

  // Update training state when complete - only when we were training
  useEffect(() => {
    if (trainingComplete && trainingInProgress) {
      setTrainingInProgress(false);
      // Fetch final state after training completes
      fetchNetworkState();
    }
  }, [trainingComplete, trainingInProgress, fetchNetworkState]);

  // Keyboard shortcuts
  useKeyboardShortcuts({
    onStartTraining: () => handleStartAdaptive(0.99),
    onStopTraining: handleStop,
    onReset: handleReset,
    onStep: networkType === 'dense' ? () => handleStep(0.1) : undefined,
    trainingInProgress,
    trainingComplete,
    disabled: !connected,
  });

  // Combine errors for display
  const displayError = apiError || trainingError;

  return (
    <ToastProvider>
    <AchievementProvider>
    <div className="h-screen bg-gray-900 text-gray-100 p-2 lg:p-3 flex flex-col overflow-hidden">
      {/* Error Banner */}
      {displayError && (
        <div className="max-w-[1600px] w-full mx-auto mb-2 flex-shrink-0">
          <div className="bg-red-900/80 border border-red-600 text-red-100 px-4 py-2 rounded flex justify-between items-center">
            <span className="text-sm">{displayError}</span>
            <button
              onClick={() => setApiError(null)}
              className="text-red-300 hover:text-red-100 ml-4"
            >
              ✕
            </button>
          </div>
        </div>
      )}

      {/* Header with Problem Menu */}
      <header className="max-w-[1600px] w-full mx-auto mb-2 flex-shrink-0">
        <div className="flex items-center justify-between flex-wrap gap-2">
          <div className="flex items-center gap-3">
            <h1 className="text-xl font-bold">Neural Network Learning Lab</h1>

            {/* View Navigation Tabs - hidden when in path-detail view */}
            {currentView !== 'path-detail' && (
              <div className="flex items-center bg-gray-800 rounded-lg p-1">
                <div
                  role="tablist"
                  className="flex items-center"
                >
                  <div
                    role="tab"
                    aria-selected={currentView === 'paths'}
                    onClick={() => setCurrentView('paths')}
                    className={`px-3 py-1 rounded text-sm transition-colors cursor-pointer ${
                      currentView === 'paths'
                        ? 'bg-blue-600 text-white'
                        : 'text-gray-400 hover:text-gray-200'
                    }`}
                  >
                    Learning Paths
                  </div>
                  <div
                    role="tab"
                    aria-selected={currentView === 'problems'}
                    onClick={() => setCurrentView('problems')}
                    className={`px-3 py-1 rounded text-sm transition-colors cursor-pointer ${
                      currentView === 'problems'
                        ? 'bg-blue-600 text-white'
                        : 'text-gray-400 hover:text-gray-200'
                    }`}
                  >
                    All Problems
                  </div>
                </div>
              </div>
            )}

            {/* Only show problem selector when in problems view */}
            {currentView === 'problems' && (
              <>
                <ProblemSelector
                  problems={problems}
                  currentProblem={currentProblem}
                  onSelect={handleProblemSelect}
                  disabled={trainingInProgress}
                />
                {currentProblem && (
                  <button
                    onClick={() => setIsInfoModalOpen(true)}
                    className="p-1.5 rounded-lg bg-gray-700 hover:bg-gray-600 text-cyan-400 hover:text-cyan-300 transition-colors"
                    title="Problem Info"
                    aria-label="Show problem information"
                  >
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                  </button>
                )}
              </>
            )}
          </div>
          <div className="flex items-center gap-2 text-xs">
            <KeyboardShortcuts
              trainingInProgress={trainingInProgress}
              trainingComplete={trainingComplete}
              hasStep={networkType === 'dense'}
            />
            <span className={`px-2 py-1 rounded ${connected ? 'bg-green-600' : 'bg-red-600'}`}>
              {connected ? 'Connected' : 'Disconnected'}
            </span>
            <span className={`px-2 py-1 rounded ${trainingComplete ? 'bg-green-600' : 'bg-yellow-600'}`}>
              {trainingComplete ? 'Ready' : 'Training...'}
            </span>
          </div>
        </div>
      </header>

      {/* Main content */}
      <main className="max-w-[1600px] w-full mx-auto flex-1 min-h-0">
        {currentView === 'path-detail' && currentPathId ? (
          <PathDetailView
            pathId={currentPathId}
            socket={socket}
            trainingProgress={trainingProgress}
            trainingComplete={trainingComplete}
            lastPrediction={lastPrediction}
            onExitPath={handleExitPath}
          />
        ) : currentView === 'paths' ? (
          <LearningPathSelector onSelectPath={handlePathSelect} />
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-3 h-full">
          {/* Left column: Training Controls + Input/Output + Loss Curve */}
          <div className="space-y-3 overflow-y-auto">
            {/* Input + Output row */}
            <div className="grid grid-cols-2 gap-3">
              <InputPanel
                problem={currentProblem}
                values={inputValues}
                onChange={handleInputChange}
                disabled={!trainingComplete}
              />
              <OutputDisplay
                problem={currentProblem}
                prediction={lastPrediction}
              />
            </div>
            <TrainingPanel
              currentEpoch={trainingProgress?.epoch ?? 0}
              currentLoss={trainingProgress?.loss ?? 0}
              currentAccuracy={trainingProgress?.accuracy ?? 0}
              trainingInProgress={trainingInProgress}
              trainingComplete={trainingComplete}
              onStartStatic={handleStartStatic}
              onStartAdaptive={handleStartAdaptive}
              onStep={networkType === 'dense' ? handleStep : undefined}
              onStop={handleStop}
              onUpdateTarget={handleUpdateTarget}
              onReset={handleReset}
              onSettingsChange={networkType === 'dense' ? handleSettingsChange : undefined}
              currentArchitecture={networkState?.architecture.layer_sizes ?? currentProblem?.default_architecture ?? [5, 12, 8, 4, 1]}
              currentWeightInit={networkState?.architecture.weight_init ?? 'xavier'}
              currentHiddenActivation={networkState?.architecture.hidden_activation ?? 'relu'}
              currentUseBiases={networkState?.architecture.use_biases ?? true}
              isCNN={networkType === 'cnn'}
              currentProblem={currentProblem}
            />
            <ContextualTips
              problem={currentProblem}
              currentEpoch={trainingProgress?.epoch ?? 0}
              currentAccuracy={trainingProgress?.accuracy ?? 0}
              currentLoss={trainingProgress?.loss ?? 0}
              trainingInProgress={trainingInProgress}
            />
            {/* Loss Visualization with 2D/3D toggle */}
            <div className="relative">
              {/* 3D Toggle Button - positioned absolutely to overlay on both views */}
              <button
                onClick={() => setShow3DLandscape(!show3DLandscape)}
                className="absolute top-2 right-2 z-10 px-2 py-1 text-xs bg-gray-700 hover:bg-gray-600 rounded transition-colors"
              >
                {show3DLandscape ? '2D Chart' : '3D View'}
              </button>
              {show3DLandscape ? (
                <div className="bg-gray-800 rounded-lg p-3">
                  <h2 className="text-lg font-semibold mb-2">Loss Landscape</h2>
                  <LossLandscape3D
                    lossHistory={networkState?.loss_history ?? []}
                    currentEpoch={trainingProgress?.epoch}
                    trainingInProgress={trainingInProgress}
                    totalEpochs={networkState?.total_epochs}
                    networkType={networkType}
                  />
                  <div className="mt-2 text-xs text-gray-500 text-center">
                    Drag to rotate | Scroll to zoom | Ball follows gradient descent
                  </div>
                </div>
              ) : (
                <LossCurve
                  lossHistory={networkState?.loss_history ?? []}
                  accuracyHistory={networkState?.accuracy_history ?? []}
                  totalEpochs={networkState?.total_epochs}
                  trainingComplete={trainingComplete}
                />
              )}
            </div>
          </div>

          {/* Right column: Network Viz + Decision Boundary + Terminal */}
          <div className="space-y-3 overflow-y-auto">
            {networkType === 'cnn' ? (
              <FeatureMapVisualization
                inputGrid={Array.isArray(inputValues[0]) ? (inputValues as number[][]) : []}
                featureMaps={featureMaps}
                architecture={networkState?.architecture ?? null}
                weights={networkState?.weights ?? []}
                prediction={
                  lastPrediction?.prediction && Array.isArray(lastPrediction.prediction)
                    ? lastPrediction.prediction
                    : null
                }
                outputLabels={currentProblem?.output_labels ?? []}
                trainingComplete={trainingComplete}
              />
            ) : (
              <NetworkVisualization
                layerSizes={networkState?.architecture.layer_sizes ?? currentProblem?.default_architecture ?? [5, 12, 8, 4, 1]}
                weights={networkState?.weights ?? []}
                activations={lastPrediction?.activations}
                inputLabels={currentProblem?.input_labels ?? []}
                outputLabels={currentProblem?.output_labels ?? []}
                outputActivation={currentProblem?.output_activation ?? 'sigmoid'}
                trainingInProgress={trainingInProgress}
                currentEpoch={trainingProgress?.epoch ?? 0}
                gradients={trainingProgress?.gradients}
              />
            )}
            <div className="grid grid-cols-2 gap-3">
              <DecisionBoundaryViz
                problemId={currentProblem?.id ?? ''}
                trainingComplete={trainingComplete}
                currentEpoch={trainingProgress?.epoch ?? 0}
                onPointClick={(x, y) => {
                  setInputValues([x, y]);
                  if (socket && trainingComplete) {
                    socket.emit('set_inputs', { inputs: [x, y] });
                  }
                }}
              />
              <WeightHistogram
                weights={networkState?.weights ?? []}
                trainingInProgress={trainingInProgress}
              />
            </div>
          </div>
        </div>
        )}
      </main>

      {/* Problem Info Modal */}
      <ProblemInfoModal
        problem={currentProblem}
        isOpen={isInfoModalOpen}
        onClose={() => setIsInfoModalOpen(false)}
      />
    </div>
    </AchievementProvider>
    </ToastProvider>
  );
}

export default App;
