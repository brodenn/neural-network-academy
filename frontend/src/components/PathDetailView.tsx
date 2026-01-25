import { useReducer, useEffect, useCallback, useRef, useState } from 'react';
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
  CNNFeatureMaps,
  TrainingProgress,
  StepProgressData
} from '../types';
import type { Socket } from 'socket.io-client';

const API_URL = 'http://localhost:5000';

// =============================================================================
// TYPES
// =============================================================================

interface PathStep {
  stepNumber: number;
  problemId: string;
  title: string;
  learningObjectives: string[];
  requiredAccuracy: number;
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

// =============================================================================
// REDUCER STATE & ACTIONS
// =============================================================================

interface StepState {
  // Data
  pathData: PathDetailData | null;
  currentStepNum: number;
  stepsProgress: StepProgressData[];

  // UI state
  loading: boolean;
  error: string | null;
  showCompletionBanner: boolean;
  showPathCompleteModal: boolean;

  // Auto-advance timer
  autoAdvanceTimer: ReturnType<typeof setTimeout> | null;
}

type StepAction =
  | { type: 'LOAD_START' }
  | { type: 'LOAD_SUCCESS'; pathData: PathDetailData; stepsProgress: StepProgressData[]; currentStep: number }
  | { type: 'LOAD_ERROR'; error: string }
  | { type: 'NAVIGATE_TO_STEP'; stepNum: number }
  | { type: 'STEP_COMPLETED'; stepNum: number; stepsProgress: StepProgressData[] }
  | { type: 'PATH_COMPLETED' }
  | { type: 'DISMISS_COMPLETION_BANNER' }
  | { type: 'CLOSE_PATH_COMPLETE_MODAL' }
  | { type: 'SET_AUTO_ADVANCE_TIMER'; timer: ReturnType<typeof setTimeout> | null }
  | { type: 'UPDATE_STEPS_PROGRESS'; stepsProgress: StepProgressData[] };

const initialState: StepState = {
  pathData: null,
  currentStepNum: 1,
  stepsProgress: [],
  loading: true,
  error: null,
  showCompletionBanner: false,
  showPathCompleteModal: false,
  autoAdvanceTimer: null,
};

function stepReducer(state: StepState, action: StepAction): StepState {
  switch (action.type) {
    case 'LOAD_START':
      return { ...state, loading: true, error: null };

    case 'LOAD_SUCCESS':
      return {
        ...state,
        loading: false,
        pathData: action.pathData,
        stepsProgress: action.stepsProgress,
        currentStepNum: action.currentStep,
      };

    case 'LOAD_ERROR':
      return { ...state, loading: false, error: action.error };

    case 'NAVIGATE_TO_STEP':
      // Clear any pending auto-advance timer when manually navigating
      if (state.autoAdvanceTimer) {
        clearTimeout(state.autoAdvanceTimer);
      }
      return {
        ...state,
        currentStepNum: action.stepNum,
        showCompletionBanner: false,
        autoAdvanceTimer: null,
      };

    case 'STEP_COMPLETED':
      return {
        ...state,
        stepsProgress: action.stepsProgress,
        showCompletionBanner: true,
      };

    case 'PATH_COMPLETED':
      if (state.autoAdvanceTimer) {
        clearTimeout(state.autoAdvanceTimer);
      }
      return {
        ...state,
        showPathCompleteModal: true,
        showCompletionBanner: false,
        autoAdvanceTimer: null,
      };

    case 'DISMISS_COMPLETION_BANNER':
      return { ...state, showCompletionBanner: false };

    case 'CLOSE_PATH_COMPLETE_MODAL':
      return { ...state, showPathCompleteModal: false };

    case 'SET_AUTO_ADVANCE_TIMER':
      // Clear existing timer before setting new one
      if (state.autoAdvanceTimer) {
        clearTimeout(state.autoAdvanceTimer);
      }
      return { ...state, autoAdvanceTimer: action.timer };

    case 'UPDATE_STEPS_PROGRESS':
      return { ...state, stepsProgress: action.stepsProgress };

    default:
      return state;
  }
}

// =============================================================================
// COMPONENT
// =============================================================================

export const PathDetailView = ({
  pathId,
  socket,
  trainingProgress,
  trainingComplete,
  lastPrediction,
  onExitPath
}: PathDetailViewProps) => {
  // Progress hook for localStorage operations
  const {
    getPathProgress,
    initializePath,
    completeStep: completeStepInStorage,
    recordAttempt,
    setCurrentStep: setCurrentStepInStorage,
  } = usePathProgress();

  // Main state reducer
  const [state, dispatch] = useReducer(stepReducer, initialState);
  const { pathData, currentStepNum, stepsProgress, loading, error, showCompletionBanner, showPathCompleteModal } = state;

  // Network/problem state (separate from step state)
  const [currentProblem, setCurrentProblem] = useState<ProblemInfo | null>(null);
  const [networkState, setNetworkState] = useState<NetworkState | null>(null);
  const [trainingInProgress, setTrainingInProgress] = useState(false);
  const [inputValues, setInputValues] = useState<number[] | number[][]>([]);
  const [featureMaps, setFeatureMaps] = useState<CNNFeatureMaps | null>(null);

  // Track previous training state to detect completion
  const prevTrainingCompleteRef = useRef(trainingComplete);
  const hasProcessedCompletionRef = useRef(false);

  // =============================================================================
  // HELPER FUNCTIONS
  // =============================================================================

  const fetchNetworkState = useCallback(async () => {
    try {
      const res = await fetch(`${API_URL}/api/network`);
      if (!res.ok) return;
      const data = await res.json();
      setNetworkState(data);
    } catch {
      // Silently ignore
    }
  }, []);

  const selectProblem = useCallback(async (problemId: string) => {
    try {
      const res = await fetch(`${API_URL}/api/problems/${problemId}/select`, { method: 'POST' });
      if (!res.ok) throw new Error('Failed to select problem');
      await fetchNetworkState();
    } catch {
      // Silently ignore
    }
  }, [fetchNetworkState]);

  // =============================================================================
  // LOAD PATH DATA
  // =============================================================================

  useEffect(() => {
    const loadPath = async () => {
      dispatch({ type: 'LOAD_START' });

      try {
        const res = await fetch(`${API_URL}/api/paths/${pathId}`);
        if (!res.ok) throw new Error('Failed to load path');
        const data: PathDetailData = await res.json();

        // Initialize progress in localStorage
        const stepConfigs = data.steps.map((s) => ({
          stepNumber: s.stepNumber,
          problemId: s.problemId
        }));
        const progress = initializePath(pathId, stepConfigs);

        dispatch({
          type: 'LOAD_SUCCESS',
          pathData: data,
          stepsProgress: progress.steps,
          currentStep: progress.currentStep,
        });

        // Load the current step's problem
        const currentStepData = data.steps[progress.currentStep - 1];
        if (currentStepData) {
          await selectProblem(currentStepData.problemId);
        }
      } catch (err) {
        dispatch({ type: 'LOAD_ERROR', error: err instanceof Error ? err.message : 'Unknown error' });
      }
    };

    loadPath();
  }, [pathId, initializePath, selectProblem]);

  // =============================================================================
  // SOCKET LISTENERS
  // =============================================================================

  useEffect(() => {
    if (!socket) return;

    const handleProblemChanged = (data: { info: ProblemInfo }) => {
      setCurrentProblem(data.info);
      if (data.info.network_type === 'cnn' && data.info.input_shape) {
        const [h, w] = data.info.input_shape;
        setInputValues(Array.from({ length: h }, () => Array(w).fill(0)));
      } else {
        setInputValues(new Array(data.info.input_labels.length).fill(0));
      }
      setFeatureMaps(null);
      fetchNetworkState();
    };

    const handleTrainingStarted = () => {
      setTrainingInProgress(true);
      hasProcessedCompletionRef.current = false; // Reset for new training
    };

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
  }, [socket, fetchNetworkState]);

  // Track feature maps
  useEffect(() => {
    if (lastPrediction?.feature_maps) {
      setFeatureMaps(lastPrediction.feature_maps);
    }
  }, [lastPrediction]);

  // =============================================================================
  // NAVIGATION (defined before step completion detection that uses it)
  // =============================================================================

  const navigateToStep = useCallback(async (stepNum: number) => {
    if (!pathData) return;

    const stepProgress = stepsProgress.find(s => s.stepNumber === stepNum);
    if (!stepProgress?.unlocked) return;

    // Update localStorage
    setCurrentStepInStorage(pathId, stepNum);

    // Update state
    dispatch({ type: 'NAVIGATE_TO_STEP', stepNum });
    hasProcessedCompletionRef.current = false;

    // Load the new problem
    const step = pathData.steps[stepNum - 1];
    if (step) {
      await selectProblem(step.problemId);
    }
  }, [pathData, stepsProgress, pathId, setCurrentStepInStorage, selectProblem]);

  const advanceToNextStep = useCallback(() => {
    if (!pathData || currentStepNum >= pathData.steps.length) return;
    navigateToStep(currentStepNum + 1);
  }, [pathData, currentStepNum, navigateToStep]);

  const handleContinueNow = useCallback(() => {
    dispatch({ type: 'DISMISS_COMPLETION_BANNER' });
    advanceToNextStep();
  }, [advanceToNextStep]);

  // =============================================================================
  // STEP COMPLETION DETECTION
  // =============================================================================

  useEffect(() => {
    // Detect transition from not-complete to complete
    const justFinished = trainingComplete && !prevTrainingCompleteRef.current;
    prevTrainingCompleteRef.current = trainingComplete;

    // Only process once per training session
    if (!justFinished || hasProcessedCompletionRef.current) return;
    if (!pathData || showCompletionBanner) return;

    const currentStep = pathData.steps[currentStepNum - 1];
    if (!currentStep) return;

    const stepProgress = stepsProgress.find(s => s.stepNumber === currentStepNum);
    if (stepProgress?.completed) return; // Already completed

    const achieved = trainingProgress?.accuracy ?? 0;
    const required = currentStep.requiredAccuracy ?? 0.95;

    // Record the attempt in localStorage
    recordAttempt(pathId, currentStepNum, achieved);

    // Check if accuracy meets requirement
    if (achieved >= required || required === 0.0) {
      hasProcessedCompletionRef.current = true;

      // Complete in localStorage and get updated progress
      completeStepInStorage(pathId, currentStepNum, achieved);
      const updatedProgress = getPathProgress(pathId);

      if (updatedProgress) {
        dispatch({ type: 'STEP_COMPLETED', stepNum: currentStepNum, stepsProgress: updatedProgress.steps });
      }

      // Check if this was the last step
      if (currentStepNum === pathData.steps.length) {
        dispatch({ type: 'PATH_COMPLETED' });
      }
    }
  }, [trainingComplete, trainingProgress, pathData, currentStepNum, stepsProgress, showCompletionBanner, pathId, recordAttempt, completeStepInStorage, getPathProgress]);

  // =============================================================================
  // TRAINING HANDLERS
  // =============================================================================

  const handleStartAdaptive = async (targetAccuracy: number) => {
    setTrainingInProgress(true);
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

  const handleStepTraining = async () => {
    try {
      await fetch(`${API_URL}/api/train/step`, { method: 'POST' });
      await fetchNetworkState();
    } catch {
      // Silently ignore
    }
  };

  const handleResetNetwork = async () => {
    try {
      await fetch(`${API_URL}/api/network/reset`, { method: 'POST' });
      await fetchNetworkState();
    } catch {
      // Silently ignore
    }
  };

  const handleArchitectureChange = async (layers: number[], settings: Record<string, unknown>) => {
    try {
      await fetch(`${API_URL}/api/network/architecture`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ layer_sizes: layers, ...settings })
      });
      await fetchNetworkState();
    } catch {
      // Silently ignore
    }
  };

  const handleInputChange = (values: number[] | number[][]) => {
    setInputValues(values);
    if (socket && trainingComplete) {
      socket.emit('set_inputs', { inputs: values });
    }
  };

  // =============================================================================
  // RENDER
  // =============================================================================

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-gray-400">Loading path...</div>
      </div>
    );
  }

  if (error || !pathData) {
    return (
      <div className="flex flex-col items-center justify-center h-full gap-4">
        <div className="text-red-400">{error || 'Failed to load path'}</div>
        <button onClick={onExitPath} className="px-4 py-2 bg-gray-700 rounded hover:bg-gray-600">
          Go Back
        </button>
      </div>
    );
  }

  const currentStep = pathData.steps[currentStepNum - 1];
  const stepProgress = stepsProgress.find(s => s.stepNumber === currentStepNum);

  return (
    <div className="h-full flex flex-col p-4 gap-4 overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <button
            onClick={onExitPath}
            className="p-2 hover:bg-gray-700 rounded-lg transition-colors"
            title="Exit path"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
            </svg>
          </button>
          <div>
            <h1 className="text-xl font-bold">{pathData.name}</h1>
            <p className="text-sm text-gray-400">
              Step {currentStepNum} of {pathData.steps.length} - {currentStep?.title}
            </p>
          </div>
        </div>
        <div className="text-sm text-gray-400">
          {stepsProgress.filter(s => s.completed).length}/{pathData.steps.length} completed
        </div>
      </div>

      {/* Progress Bar */}
      <PathProgressBar
        steps={stepsProgress}
        currentStep={currentStepNum}
        onStepClick={navigateToStep}
      />

      {/* Main Content Grid */}
      <div className="flex-1 grid grid-cols-[320px_1fr_320px] gap-4 min-h-0 overflow-hidden">
        {/* Left: Step Info */}
        <div className="space-y-3 overflow-y-auto">
          {currentStep && (
            <PathStepCard
              step={currentStep}
              stepProgress={stepProgress}
              isCurrentStep={true}
              bestAccuracy={stepProgress?.bestAccuracy ?? 0}
              attempts={stepProgress?.attempts ?? 0}
            />
          )}

          {currentStep && (
            <StepHintPanel
              hints={currentStep.hints}
              attempts={stepProgress?.attempts ?? 0}
            />
          )}

          {/* Completion Banner */}
          {showCompletionBanner && currentStepNum < pathData.steps.length && (
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              className="bg-green-900/50 border border-green-600 rounded-lg p-4"
            >
              <div className="text-center mb-3">
                <div className="text-green-400 font-semibold">✓ Step Complete!</div>
              </div>
              <button
                onClick={handleContinueNow}
                className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 px-4 rounded-lg transition-colors flex items-center justify-center gap-2"
              >
                <span>Next Step</span>
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
                </svg>
              </button>
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
              trainingComplete={trainingComplete}
            />
          </div>

          {currentProblem?.network_type === 'cnn' && featureMaps && (
            <FeatureMapVisualization featureMaps={featureMaps} />
          )}

          <TrainingPanel
            currentProblem={currentProblem}
            currentEpoch={trainingProgress?.epoch ?? 0}
            currentLoss={trainingProgress?.loss ?? 0}
            currentAccuracy={trainingProgress?.accuracy ?? 0}
            trainingInProgress={trainingInProgress}
            trainingComplete={trainingComplete}
            onStartStatic={handleStartStatic}
            onStartAdaptive={handleStartAdaptive}
            onStep={(lr) => handleStepTraining()}
            onStop={() => {}}
            onUpdateTarget={() => {}}
            onReset={handleResetNetwork}
            onSettingsChange={(settings) => handleArchitectureChange(settings.layers, settings)}
            currentArchitecture={networkState?.architecture?.layer_sizes ?? [2, 4, 1]}
            currentWeightInit={networkState?.architecture?.weight_init ?? 'xavier'}
            currentHiddenActivation={networkState?.architecture?.hidden_activation ?? 'relu'}
            currentUseBiases={networkState?.architecture?.use_biases ?? true}
            isCNN={currentProblem?.network_type === 'cnn'}
          />

          <LossCurve
            lossHistory={networkState?.loss_history ?? []}
            accuracyHistory={networkState?.accuracy_history ?? []}
            totalEpochs={networkState?.total_epochs}
            trainingComplete={trainingComplete}
          />
        </div>

        {/* Right: Network Visualization */}
        <div className="space-y-3 overflow-y-auto">
          <NetworkVisualization
            layerSizes={networkState?.architecture?.layer_sizes ?? currentProblem?.default_architecture ?? [2, 4, 1]}
            weights={networkState?.weights ?? []}
            activations={lastPrediction?.activations}
            inputLabels={currentProblem?.input_labels ?? []}
            outputLabels={currentProblem?.output_labels ?? []}
            outputActivation={currentProblem?.output_activation ?? 'sigmoid'}
            trainingInProgress={trainingInProgress}
            currentEpoch={trainingProgress?.epoch ?? 0}
          />
        </div>
      </div>

      {/* Path Completion Modal */}
      {showPathCompleteModal && (
        <PathCompletionModal
          pathName={pathData.name}
          badge={pathData.badge}
          totalSteps={pathData.steps.length}
          onClose={() => dispatch({ type: 'CLOSE_PATH_COMPLETE_MODAL' })}
          onReview={() => dispatch({ type: 'CLOSE_PATH_COMPLETE_MODAL' })}
          onExit={onExitPath}
        />
      )}
    </div>
  );
};
