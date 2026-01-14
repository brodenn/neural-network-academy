import { useState, useEffect, useCallback } from 'react';
import { useSocket } from './hooks/useSocket';
import { ProblemSelector } from './components/ProblemSelector';
import { InputPanel } from './components/InputPanel';
import { OutputDisplay } from './components/OutputDisplay';
import { TrainingPanel } from './components/TrainingPanel';
import { LossCurve } from './components/LossCurve';
import { NetworkVisualization } from './components/NetworkVisualization';
import { TerminalOutput } from './components/TerminalOutput';
import type { NetworkState, PredictionResult, ProblemInfo } from './types';

const API_URL = 'http://localhost:5000';

function App() {
  const {
    connected,
    trainingProgress,
    lastPrediction,
    trainingComplete,
    socket,
  } = useSocket();

  const [networkState, setNetworkState] = useState<NetworkState | null>(null);
  const [predictions, setPredictions] = useState<PredictionResult[]>([]);
  const [trainingInProgress, setTrainingInProgress] = useState(false);

  // Problem state
  const [problems, setProblems] = useState<ProblemInfo[]>([]);
  const [currentProblem, setCurrentProblem] = useState<ProblemInfo | null>(null);
  const [inputValues, setInputValues] = useState<number[]>([]);

  // Fetch available problems
  const fetchProblems = useCallback(async () => {
    try {
      const res = await fetch(`${API_URL}/api/problems`);
      const data = await res.json();
      setProblems(data);
    } catch (err) {
      console.error('Failed to fetch problems:', err);
    }
  }, []);

  // Fetch network state
  const fetchNetworkState = useCallback(async () => {
    try {
      const res = await fetch(`${API_URL}/api/network`);
      const data = await res.json();
      setNetworkState(data);
    } catch (err) {
      console.error('Failed to fetch network state:', err);
    }
  }, []);

  // Initial fetch
  useEffect(() => {
    fetchProblems();
    fetchNetworkState();
  }, [fetchProblems, fetchNetworkState]);

  // Listen for problem info and training events from websocket
  useEffect(() => {
    if (socket) {
      const handleProblemInfo = (info: ProblemInfo) => {
        setCurrentProblem(info);
        setInputValues(new Array(info.input_labels.length).fill(0));
      };

      const handleProblemChanged = (data: { problem_id: string; info: ProblemInfo }) => {
        setCurrentProblem(data.info);
        setInputValues(new Array(data.info.input_labels.length).fill(0));
        setPredictions([]);
        fetchNetworkState();
      };

      const handleTrainingComplete = () => {
        // Always fetch network state when training completes to get final data
        setTrainingInProgress(false);
        fetchNetworkState();
      };

      socket.on('problem_info', handleProblemInfo);
      socket.on('problem_changed', handleProblemChanged);
      socket.on('training_complete', handleTrainingComplete);

      return () => {
        socket.off('problem_info', handleProblemInfo);
        socket.off('problem_changed', handleProblemChanged);
        socket.off('training_complete', handleTrainingComplete);
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

  // Track predictions
  useEffect(() => {
    if (lastPrediction) {
      setPredictions((prev) => [...prev.slice(-50), lastPrediction]);
    }
  }, [lastPrediction]);

  // Handle problem selection
  const handleProblemSelect = async (problemId: string) => {
    try {
      await fetch(`${API_URL}/api/problems/${problemId}/select`, { method: 'POST' });
      // Problem change will be handled via websocket
    } catch (err) {
      console.error('Failed to select problem:', err);
    }
  };

  // Handle input changes
  const handleInputChange = (values: number[]) => {
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
        const data = await res.json();
        console.error('Training failed:', data.error);
        setTrainingInProgress(false);
      }
    } catch (err) {
      console.error('Failed to start training:', err);
      setTrainingInProgress(false);
    }
  };

  const handleStartAdaptive = async () => {
    setTrainingInProgress(true);
    try {
      const res = await fetch(`${API_URL}/api/train/adaptive`, { method: 'POST' });
      if (!res.ok) {
        const data = await res.json();
        console.error('Adaptive training failed:', data.error);
        setTrainingInProgress(false);
      }
    } catch (err) {
      console.error('Failed to start adaptive training:', err);
      setTrainingInProgress(false);
    }
  };

  const handleReset = async () => {
    try {
      await fetch(`${API_URL}/api/network/reset`, { method: 'POST' });
      await fetchNetworkState();
      setPredictions([]);
    } catch (err) {
      console.error('Failed to reset network:', err);
    }
  };

  const handleStep = async (learningRate: number) => {
    try {
      const res = await fetch(`${API_URL}/api/train/step`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ learning_rate: learningRate }),
      });
      if (res.ok) {
        await fetchNetworkState();
      } else {
        const data = await res.json();
        console.error('Step training failed:', data.error);
      }
    } catch (err) {
      console.error('Failed to step train:', err);
    }
  };

  const handleSettingsChange = async (settings: {
    layers: number[];
    weightInit: string;
    hiddenActivation: string;
    useBiases: boolean;
  }) => {
    try {
      await fetch(`${API_URL}/api/network/architecture`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          layer_sizes: settings.layers,
          weight_init: settings.weightInit,
          hidden_activation: settings.hiddenActivation,
          use_biases: settings.useBiases,
        }),
      });
      await fetchNetworkState();
    } catch (err) {
      console.error('Failed to change settings:', err);
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

  return (
    <div className="min-h-screen bg-gray-900 text-gray-100 p-6">
      {/* Header */}
      <header className="max-w-7xl mx-auto mb-6">
        <h1 className="text-3xl font-bold text-center">
          Embedded NN Learning Lab
        </h1>
        <p className="text-gray-400 text-center mt-2">
          Interactive Neural Network for Embedded Systems Problems
        </p>
        <div className="flex justify-center gap-4 mt-4 text-sm">
          <span className={`px-3 py-1 rounded ${connected ? 'bg-green-600' : 'bg-red-600'}`}>
            {connected ? 'Connected' : 'Disconnected'}
          </span>
          <span className={`px-3 py-1 rounded ${trainingComplete ? 'bg-green-600' : 'bg-yellow-600'}`}>
            {trainingComplete ? 'Ready' : 'Training...'}
          </span>
          {currentProblem && (
            <span className="px-3 py-1 rounded bg-cyan-600/50">
              {currentProblem.name}
            </span>
          )}
        </div>
      </header>

      {/* Main content */}
      <main className="max-w-7xl mx-auto">
        {/* Top row: Problem selector + Input + Output */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
          <ProblemSelector
            problems={problems}
            currentProblem={currentProblem}
            onSelect={handleProblemSelect}
            disabled={trainingInProgress}
          />
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

        {/* Middle row: Training controls + Network viz */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
          <TrainingPanel
            currentEpoch={trainingProgress?.epoch ?? 0}
            currentLoss={trainingProgress?.loss ?? 0}
            currentAccuracy={trainingProgress?.accuracy ?? 0}
            trainingInProgress={trainingInProgress}
            trainingComplete={trainingComplete}
            onStartStatic={handleStartStatic}
            onStartAdaptive={handleStartAdaptive}
            onStep={handleStep}
            onReset={handleReset}
            onSettingsChange={handleSettingsChange}
            currentArchitecture={networkState?.architecture.layer_sizes ?? currentProblem?.default_architecture ?? [5, 12, 8, 4, 1]}
            currentWeightInit={networkState?.architecture.weight_init ?? 'xavier'}
            currentHiddenActivation={networkState?.architecture.hidden_activation ?? 'relu'}
            currentUseBiases={networkState?.architecture.use_biases ?? true}
          />
          <NetworkVisualization
            layerSizes={networkState?.architecture.layer_sizes ?? currentProblem?.default_architecture ?? [5, 12, 8, 4, 1]}
            weights={networkState?.weights ?? []}
            activations={lastPrediction?.activations}
          />
        </div>

        {/* Bottom row: Loss curve + Terminal */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <LossCurve
            lossHistory={networkState?.loss_history ?? []}
            accuracyHistory={networkState?.accuracy_history ?? []}
            totalEpochs={networkState?.total_epochs}
          />
          <TerminalOutput predictions={predictions} />
        </div>
      </main>

      {/* Footer */}
      <footer className="max-w-7xl mx-auto mt-8 text-center text-gray-500 text-sm">
        <p>
          Neural Network from Scratch | NumPy Only | {problems.length} Problems | Interactive Learning Tool
        </p>
      </footer>
    </div>
  );
}

export default App;
