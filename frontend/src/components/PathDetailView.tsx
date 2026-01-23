import { useState, useEffect, useCallback } from 'react';
import { motion } from 'framer-motion';
import { usePathProgress } from '../hooks/usePathProgress';
import { PathProgressBar } from './PathProgressBar';
import { PathStepCard } from './PathStepCard';
import { StepHintPanel } from './StepHintPanel';
import { PathCompletionModal } from './PathCompletionModal';
import { InputPanel } from './InputPanel';
import { OutputDisplay } from './OutputDisplay';
import { TrainingPanel } from './TrainingPanel';
import { NetworkVisualization } from './NetworkVisualization';
import { FeatureMapVisualization } from './FeatureMapVisualization';
import { LossCurve } from './LossCurve';
import type {
  NetworkState,
  PredictionResult,
  ProblemInfo,
  NetworkType,
  CNNFeatureMaps,
  TrainingProgress
} from '../types';
import type { Socket } from 'socket.io-client';

const API_URL = 'http://localhost:5000';

interface PathStep {
  step_number: number;
  problem_id: string;
  title: string;
  learning_objectives: string[];
  required_accuracy: number;
  hints: string[];
}

interface PathDetailData {
  id: string;
  name: string;
  description: string;
  difficulty: string;
  estimated_time: string;
  badge: {
    icon: string;
    color: string;
    title: string;
  };
  steps: PathStep[];
}

interface PathDetailViewProps {
  pathId: string;
  socket: Socket | null;
  trainingProgress: TrainingProgress | null;
  trainingComplete: boolean;
  lastPrediction: PredictionResult | null;
  onExitPath: () => void;
}

export const PathDetailView = ({
  pathId,
  socket,
  trainingProgress,
  trainingComplete,
  lastPrediction,
  onExitPath
}: PathDetailViewProps) => {
  const {
    getPathProgress,
    initializePath,
    completeStep,
    recordAttempt,
    setCurrentStep,
    getStepProgress
  } = usePathProgress();

  // Path data state
  const [pathData, setPathData] = useState<PathDetailData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Current step state
  const [currentStepNum, setCurrentStepNum] = useState(1);
  const [currentProblem, setCurrentProblem] = useState<ProblemInfo | null>(null);

  // Network state
  const [networkState, setNetworkState] = useState<NetworkState | null>(null);
  const [trainingInProgress, setTrainingInProgress] = useState(false);
  const [inputValues, setInputValues] = useState<number[] | number[][]>([]);
  const [featureMaps, setFeatureMaps] = useState<CNNFeatureMaps | null>(null);

  // Completion modal state
  const [showCompletionModal, setShowCompletionModal] = useState(false);
  const [justCompletedStep, setJustCompletedStep] = useState(false);

  // Fetch path data
  useEffect(() => {
    const fetchPathData = async () => {
      try {
        const res = await fetch(`${API_URL}/api/paths/${pathId}`);
        if (!res.ok) throw new Error('Failed to load path');
        const data = await res.json();
        setPathData(data);

        // Initialize progress in localStorage
        const stepConfigs = data.steps.map((s: PathStep) => ({
          stepNumber: s.step_number,
          problemId: s.problem_id
        }));
        const progress = initializePath(pathId, stepConfigs);
        setCurrentStepNum(progress.currentStep);

        // Load the current step's problem
        await selectProblem(data.steps[progress.currentStep - 1].problem_id);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unknown error');
      } finally {
        setLoading(false);
      }
    };

    fetchPathData();
  }, [pathId, initializePath]);

  // Fetch network state
  const fetchNetworkState = useCallback(async () => {
    try {
      const res = await fetch(`${API_URL}/api/network`);
      if (!res.ok) return;
      const data = await res.json();
      setNetworkState(data);
    } catch {
      // Silently ignore network fetch errors
    }
  }, []);

  // Select a problem
  const selectProblem = async (problemId: string) => {
    try {
      const res = await fetch(`${API_URL}/api/problems/${problemId}/select`, { method: 'POST' });
      if (!res.ok) throw new Error('Failed to select problem');
      await fetchNetworkState();
    } catch {
      // Problem selection failed, ignore
    }
  };

  // Listen for problem changes and training events
  useEffect(() => {
    if (socket) {
      const handleProblemChanged = (data: { info: ProblemInfo }) => {
        setCurrentProblem(data.info);
        // Initialize input values
        if (data.info.network_type === 'cnn' && data.info.input_shape) {
          const [h, w] = data.info.input_shape;
          setInputValues(Array.from({ length: h }, () => Array(w).fill(0)));
        } else {
          setInputValues(new Array(data.info.input_labels.length).fill(0));
        }
        setFeatureMaps(null);
        fetchNetworkState();
      };

      const handleTrainingStarted = () => setTrainingInProgress(true);
      const handleTrainingComplete = () => {
        setTrainingInProgress(false);
        fetchNetworkState();
      };
      const handleTrainingStopped = () => {
        setTrainingInProgress(false);
        fetchNetworkState();
      };

      socket.on('problem_changed', handleProblemChanged);
      socket.on('training_started', handleTrainingStarted);
      socket.on('training_complete', handleTrainingComplete);
      socket.on('training_stopped', handleTrainingStopped);

      return () => {
        socket.off('problem_changed', handleProblemChanged);
        socket.off('training_started', handleTrainingStarted);
        socket.off('training_complete', handleTrainingComplete);
        socket.off('training_stopped', handleTrainingStopped);
      };
    }
  }, [socket, fetchNetworkState]);

  // Track feature maps from predictions
  useEffect(() => {
    if (lastPrediction?.feature_maps) {
      setFeatureMaps(lastPrediction.feature_maps);
    }
  }, [lastPrediction]);

  // Step completion detection
  useEffect(() => {
    if (!pathData || !trainingComplete || trainingInProgress || justCompletedStep) return;

    const currentStep = pathData.steps[currentStepNum - 1];
    if (!currentStep) return;

    const stepProgress = getStepProgress(pathId, currentStepNum);
    if (stepProgress?.completed) return; // Already completed

    const achieved = trainingProgress?.accuracy ?? 0;
    const required = currentStep.required_accuracy ?? 0.95;

    // Record the attempt
    recordAttempt(pathId, currentStepNum, achieved);

    // Check if step is complete
    // Failure cases (required_accuracy: 0) auto-complete after any training
    if (achieved >= required || required === 0.0) {
      completeStep(pathId, currentStepNum, achieved);
      setJustCompletedStep(true);

      // Check if path is complete
      if (currentStepNum === pathData.steps.length) {
        setShowCompletionModal(true);
      } else {
        // Auto-advance to next step after a short delay
        setTimeout(() => {
          handleStepChange(currentStepNum + 1);
          setJustCompletedStep(false);
        }, 1500);
      }
    }
  }, [trainingComplete, trainingProgress, pathData, currentStepNum, justCompletedStep, pathId, getStepProgress, recordAttempt, completeStep]);

  // Handle step navigation
  const handleStepChange = async (stepNum: number) => {
    if (!pathData) return;

    const stepProgress = getStepProgress(pathId, stepNum);
    if (!stepProgress?.unlocked) return;

    setCurrentStep(pathId, stepNum);
    setCurrentStepNum(stepNum);
    setJustCompletedStep(false);

    const step = pathData.steps[stepNum - 1];
    if (step) {
      await selectProblem(step.problem_id);
    }
  };

  // Training handlers
  const handleStartAdaptive = async (targetAccuracy: number) => {
    setTrainingInProgress(true);
    setJustCompletedStep(false);
    try {
      await fetch(`${API_URL}/api/train/adaptive`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ target_accuracy: targetAccuracy })
      });
    } catch {
      setTrainingInProgress(false);
    }
  };

  const handleStartStatic = async (epochs: number, learningRate: number) => {
    setTrainingInProgress(true);
    setJustCompletedStep(false);
    try {
      await fetch(`${API_URL}/api/train`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ epochs, learning_rate: learningRate })
      });
    } catch {
      setTrainingInProgress(false);
    }
  };

  const handleStop = async () => {
    try {
      await fetch(`${API_URL}/api/train/stop`, { method: 'POST' });
    } catch {
      // Ignore
    }
  };

  const handleReset = async () => {
    try {
      await fetch(`${API_URL}/api/network/reset`, { method: 'POST' });
      await fetchNetworkState();
    } catch {
      // Ignore
    }
  };

  const handleStep = async (learningRate: number) => {
    try {
      await fetch(`${API_URL}/api/train/step`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ learning_rate: learningRate })
      });
      await fetchNetworkState();
    } catch {
      // Ignore
    }
  };

  const handleInputChange = (values: number[] | number[][]) => {
    setInputValues(values);
    if (socket && trainingComplete) {
      socket.emit('set_inputs', { inputs: values });
    }
  };

  // Loading state
  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-lg text-gray-400">Loading path...</div>
      </div>
    );
  }

  // Error state
  if (error || !pathData) {
    return (
      <div className="flex flex-col items-center justify-center h-full gap-4">
        <div className="text-lg text-red-400">{error || 'Failed to load path'}</div>
        <button
          onClick={onExitPath}
          className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg"
        >
          Back to Paths
        </button>
      </div>
    );
  }

  const currentStep = pathData.steps[currentStepNum - 1];
  const stepProgress = getStepProgress(pathId, currentStepNum);
  const progress = getPathProgress(pathId);
  const networkType: NetworkType = currentProblem?.network_type ?? 'dense';

  // Build step progress for progress bar
  const stepsForProgressBar = pathData.steps.map(s => {
    const sp = getStepProgress(pathId, s.step_number);
    return {
      stepNumber: s.step_number,
      problemId: s.problem_id,
      unlocked: sp?.unlocked ?? s.step_number === 1,
      completed: sp?.completed ?? false,
      attempts: sp?.attempts ?? 0,
      bestAccuracy: sp?.bestAccuracy ?? 0
    };
  });

  // Calculate completion stats for modal
  const completionStats = {
    totalSteps: pathData.steps.length,
    totalAttempts: stepsForProgressBar.reduce((sum, s) => sum + s.attempts, 0),
    avgAccuracy: stepsForProgressBar.reduce((sum, s) => sum + s.bestAccuracy, 0) / pathData.steps.length
  };

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between mb-3 flex-shrink-0">
        <div className="flex items-center gap-3">
          <button
            onClick={onExitPath}
            className="p-2 rounded-lg bg-gray-800 hover:bg-gray-700 transition-colors"
            title="Exit Path"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
            </svg>
          </button>
          <div>
            <h2 className="text-lg font-bold">{pathData.name}</h2>
            <p className="text-sm text-gray-400">
              Step {currentStepNum} of {pathData.steps.length}
              {currentStep && ` - ${currentStep.title}`}
            </p>
          </div>
        </div>

        <div className="text-sm text-gray-400">
          {progress?.stepsCompleted ?? 0}/{pathData.steps.length} completed
        </div>
      </div>

      {/* Progress Bar */}
      <div className="mb-3 flex-shrink-0">
        <PathProgressBar
          steps={stepsForProgressBar}
          currentStep={currentStepNum}
          onStepClick={handleStepChange}
        />
      </div>

      {/* Main Content */}
      <div className="flex-1 min-h-0 grid grid-cols-1 lg:grid-cols-3 gap-3">
        {/* Left: Step Info + Hints */}
        <div className="space-y-3 overflow-y-auto">
          <PathStepCard
            step={{
              stepNumber: currentStep.step_number,
              problemId: currentStep.problem_id,
              title: currentStep.title,
              learningObjectives: currentStep.learning_objectives,
              requiredAccuracy: currentStep.required_accuracy,
              hints: currentStep.hints
            }}
            progress={stepProgress}
            isCurrentStep={true}
          />
          <StepHintPanel
            hints={currentStep.hints}
            attempts={stepProgress?.attempts ?? 0}
          />

          {/* Step Completion Animation */}
          {justCompletedStep && currentStepNum < pathData.steps.length && (
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              className="bg-green-900/50 border border-green-600 rounded-lg p-4 text-center"
            >
              <div className="text-green-400 font-semibold mb-1">Step Complete!</div>
              <div className="text-sm text-gray-400">Moving to next step...</div>
            </motion.div>
          )}
        </div>

        {/* Center: Input/Output + Training */}
        <div className="space-y-3 overflow-y-auto">
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
            onUpdateTarget={() => {/* Not used in path view */}}
            onReset={handleReset}
            currentArchitecture={networkState?.architecture.layer_sizes ?? currentProblem?.default_architecture ?? [2, 4, 1]}
            currentWeightInit={networkState?.architecture.weight_init ?? 'xavier'}
            currentHiddenActivation={networkState?.architecture.hidden_activation ?? 'relu'}
            currentUseBiases={networkState?.architecture.use_biases ?? true}
            isCNN={networkType === 'cnn'}
            currentProblem={currentProblem}
          />
          <LossCurve
            lossHistory={networkState?.loss_history ?? []}
            accuracyHistory={networkState?.accuracy_history ?? []}
            totalEpochs={networkState?.total_epochs}
            trainingComplete={trainingComplete}
          />
        </div>

        {/* Right: Network Visualization */}
        <div className="overflow-y-auto">
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
            />
          ) : (
            <NetworkVisualization
              layerSizes={networkState?.architecture.layer_sizes ?? currentProblem?.default_architecture ?? [2, 4, 1]}
              weights={networkState?.weights ?? []}
              activations={lastPrediction?.activations}
              inputLabels={currentProblem?.input_labels ?? []}
              outputLabels={currentProblem?.output_labels ?? []}
              outputActivation={currentProblem?.output_activation ?? 'sigmoid'}
              trainingInProgress={trainingInProgress}
              currentEpoch={trainingProgress?.epoch ?? 0}
            />
          )}
        </div>
      </div>

      {/* Completion Modal */}
      <PathCompletionModal
        isOpen={showCompletionModal}
        pathName={pathData.name}
        badge={pathData.badge}
        stats={completionStats}
        onReviewPath={() => setShowCompletionModal(false)}
        onBackToPaths={onExitPath}
        onClose={() => setShowCompletionModal(false)}
      />
    </div>
  );
};
