import { useReducer, useEffect, useCallback, useRef, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import confetti from 'canvas-confetti';
import { usePathProgress } from '../hooks/usePathProgress';
import { useToast } from '../hooks/useToast';
import { useLearningStore } from '../stores/learningStore';
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
import { TrainingNarrator } from './TrainingNarrator';
import { FailureDramatization, getFailureTypeFromProblemId } from './FailureDramatization';
import { getRichHintsForProblem, getExperimentsForProblem } from '../data/richHints';
// Interactive challenge components
import { NetworkBuilder } from './NetworkBuilder';
import { PredictionQuiz, PREDICTION_QUIZZES } from './PredictionQuiz';
import { DebugChallenge, DEBUG_CHALLENGES } from './DebugChallenge';
import type { StepType, BuildChallengeData, PredictionQuizData, DebugChallengeData } from '../types';

// Milestone thresholds
const MILESTONES = [25, 50, 75] as const;
type MilestoneValue = typeof MILESTONES[number];
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
  // Interactive challenge fields
  stepType?: StepType;
  buildChallenge?: BuildChallengeData;
  predictionQuiz?: PredictionQuizData;
  debugChallenge?: DebugChallengeData;
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
    resetPath,
  } = usePathProgress();

  // Zustand store for achievements
  const storeCompleteStep = useLearningStore(state => state.completeStep);
  const storeRecordAttempt = useLearningStore(state => state.recordAttempt);
  const storeInitializePath = useLearningStore(state => state.initializePath);

  // Reset confirmation state
  const [showResetConfirm, setShowResetConfirm] = useState(false);

  // Main state reducer
  const [state, dispatch] = useReducer(stepReducer, initialState);
  const { pathData, currentStepNum, stepsProgress, loading, error, showCompletionBanner, showPathCompleteModal } = state;

  // Network/problem state (separate from step state)
  const [currentProblem, setCurrentProblem] = useState<ProblemInfo | null>(null);
  const [networkState, setNetworkState] = useState<NetworkState | null>(null);
  const [trainingInProgress, setTrainingInProgress] = useState(false);
  const [inputValues, setInputValues] = useState<number[] | number[][]>([]);
  const [featureMaps, setFeatureMaps] = useState<CNNFeatureMaps | null>(null);

  // Interactive challenge state
  const [, setChallengeCompleted] = useState(false);
  const [showTrainingAfterChallenge, setShowTrainingAfterChallenge] = useState(false);

  // Track previous training state to detect completion
  const prevTrainingCompleteRef = useRef(trainingComplete);
  const hasProcessedCompletionRef = useRef(false);

  // Milestone celebrations
  const [celebratedMilestones, setCelebratedMilestones] = useState<Set<MilestoneValue>>(new Set());
  const [currentMilestone, setCurrentMilestone] = useState<MilestoneValue | null>(null);

  // Toast notifications
  const { showToast } = useToast();

  // =============================================================================
  // HELPER FUNCTIONS
  // =============================================================================

  const fetchNetworkState = useCallback(async () => {
    try {
      const res = await fetch(`${API_URL}/api/network`);
      if (!res.ok) {
        console.warn('Failed to fetch network state');
        return;
      }
      const data = await res.json();
      setNetworkState(data);
    } catch (err) {
      console.error('Network fetch error:', err);
      // Don't show toast for network fetch - it's a background operation
    }
  }, []);

  const selectProblem = useCallback(async (problemId: string) => {
    try {
      const res = await fetch(`${API_URL}/api/problems/${problemId}/select`, { method: 'POST' });
      if (!res.ok) throw new Error('Failed to select problem');
      await fetchNetworkState();
    } catch (err) {
      console.error('Problem selection error:', err);
      showToast('Failed to load problem. Please try again.', 'error');
    }
  }, [fetchNetworkState, showToast]);

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

        // Initialize progress in localStorage and Zustand store
        const stepConfigs = data.steps.map((s) => ({
          stepNumber: s.stepNumber,
          problemId: s.problemId
        }));
        const progress = initializePath(pathId, stepConfigs);
        storeInitializePath(pathId, stepConfigs);

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
  }, [pathId, initializePath, storeInitializePath, selectProblem]);

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

    // Reset challenge state
    setChallengeCompleted(false);
    setShowTrainingAfterChallenge(false);

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

      // Complete in Zustand store (triggers achievement checks)
      storeCompleteStep(pathId, currentStepNum, achieved);

      const updatedProgress = getPathProgress(pathId);

      if (updatedProgress) {
        dispatch({ type: 'STEP_COMPLETED', stepNum: currentStepNum, stepsProgress: updatedProgress.steps });

        // Check for milestone celebrations
        const completedCount = updatedProgress.steps.filter(s => s.completed).length;
        const totalSteps = pathData.steps.length;
        const progressPercent = (completedCount / totalSteps) * 100;

        for (const milestone of MILESTONES) {
          if (progressPercent >= milestone && !celebratedMilestones.has(milestone)) {
            // Trigger milestone celebration
            setCurrentMilestone(milestone);
            setCelebratedMilestones(prev => new Set([...prev, milestone]));

            // Fire confetti!
            confetti({
              particleCount: 100,
              spread: 70,
              origin: { y: 0.6 },
              colors: milestone === 75 ? ['#FFD700', '#FFA500', '#FF6347'] :
                      milestone === 50 ? ['#00CED1', '#20B2AA', '#48D1CC'] :
                                         ['#98FB98', '#90EE90', '#3CB371']
            });
            break; // Only celebrate one milestone at a time
          }
        }
      }

      // Check if this was the last step
      if (currentStepNum === pathData.steps.length) {
        dispatch({ type: 'PATH_COMPLETED' });
      }
    } else {
      // Record failed attempt in Zustand store (for tracking)
      storeRecordAttempt(pathId, currentStepNum, achieved);
    }
  }, [trainingComplete, trainingProgress, pathData, currentStepNum, stepsProgress, showCompletionBanner, pathId, recordAttempt, completeStepInStorage, getPathProgress, celebratedMilestones, storeCompleteStep, storeRecordAttempt]);

  // =============================================================================
  // TRAINING HANDLERS
  // =============================================================================

  const handleStartAdaptive = async (targetAccuracy: number) => {
    setTrainingInProgress(true);
    try {
      const res = await fetch(`${API_URL}/api/train/adaptive`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ target_accuracy: targetAccuracy })
      });
      if (!res.ok) throw new Error('Training request failed');
    } catch (err) {
      console.error('Adaptive training error:', err);
      showToast('Failed to start training. Check if the server is running.', 'error');
      setTrainingInProgress(false);
    }
  };

  const handleStartStatic = async (epochs: number, learningRate: number) => {
    setTrainingInProgress(true);
    try {
      const res = await fetch(`${API_URL}/api/train`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ epochs, learning_rate: learningRate })
      });
      if (!res.ok) throw new Error('Training request failed');
    } catch (err) {
      console.error('Static training error:', err);
      showToast('Failed to start training. Check if the server is running.', 'error');
      setTrainingInProgress(false);
    }
  };

  const handleStepTraining = async () => {
    try {
      const res = await fetch(`${API_URL}/api/train/step`, { method: 'POST' });
      if (!res.ok) throw new Error('Step training failed');
      await fetchNetworkState();
    } catch (err) {
      console.error('Step training error:', err);
      showToast('Failed to run training step.', 'error');
    }
  };

  const handleResetNetwork = async () => {
    try {
      const res = await fetch(`${API_URL}/api/network/reset`, { method: 'POST' });
      if (!res.ok) throw new Error('Reset failed');
      await fetchNetworkState();
      showToast('Network reset successfully!', 'success');
    } catch (err) {
      console.error('Network reset error:', err);
      showToast('Failed to reset network.', 'error');
    }
  };

  const handleArchitectureChange = async (layers: number[], settings: Record<string, unknown>) => {
    try {
      const res = await fetch(`${API_URL}/api/network/architecture`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ layer_sizes: layers, ...settings })
      });
      if (!res.ok) throw new Error('Architecture change failed');
      await fetchNetworkState();
      showToast('Network architecture updated!', 'success');
    } catch (err) {
      console.error('Architecture change error:', err);
      showToast('Failed to update architecture.', 'error');
    }
  };

  const handleInputChange = (values: number[] | number[][]) => {
    setInputValues(values);
    if (socket && trainingComplete) {
      socket.emit('set_inputs', { inputs: values });
    }
  };

  // =============================================================================
  // CHALLENGE HANDLERS
  // =============================================================================

  const handleBuildChallengeSubmit = async (architecture: number[]) => {
    // Apply the architecture
    try {
      await handleArchitectureChange(architecture, {});
      setChallengeCompleted(true);
      setShowTrainingAfterChallenge(true);
      showToast('Architecture submitted! Now train to see the result.', 'success');
    } catch {
      showToast('Failed to apply architecture.', 'error');
    }
  };

  const handlePredictionQuizAnswer = (correct: boolean) => {
    if (correct) {
      showToast('Correct prediction! 🎉', 'success');
    } else {
      showToast('Not quite - let\'s see what actually happens!', 'info');
    }
  };

  const handlePredictionQuizRevealAndTrain = async () => {
    setChallengeCompleted(true);
    setShowTrainingAfterChallenge(true);
    // Mark the step as complete
    if (pathData && currentStepNum <= pathData.steps.length) {
      const currentStep = pathData.steps[currentStepNum - 1];
      if (currentStep?.stepType === 'prediction_quiz') {
        completeStepInStorage(pathId, currentStepNum, 1.0);
        storeCompleteStep(pathId, currentStepNum, 1.0);
        const updatedProgress = getPathProgress(pathId);
        if (updatedProgress) {
          dispatch({ type: 'STEP_COMPLETED', stepNum: currentStepNum, stepsProgress: updatedProgress.steps });
        }
      }
    }
  };

  const handleDebugChallengeSolved = (correct: boolean) => {
    if (correct) {
      showToast('Bug found! Great debugging! 🐛', 'success');
    }
  };

  const handleDebugChallengeTryFix = async (fixId: string) => {
    setChallengeCompleted(true);
    setShowTrainingAfterChallenge(true);
    // Apply the fix based on the bug type
    if (fixId === 'hidden') {
      await handleArchitectureChange([2, 4, 1], {});
    } else if (fixId === 'zeros') {
      await handleArchitectureChange([2, 4, 1], { weight_init: 'xavier' });
    } else if (fixId === 'lr') {
      // Reset for proper LR - will be handled by training panel
    }
    // Mark step as complete
    if (pathData) {
      completeStepInStorage(pathId, currentStepNum, 1.0);
      storeCompleteStep(pathId, currentStepNum, 1.0);
      const updatedProgress = getPathProgress(pathId);
      if (updatedProgress) {
        dispatch({ type: 'STEP_COMPLETED', stepNum: currentStepNum, stepsProgress: updatedProgress.steps });
      }
    }
  };

  const handleResetPath = async () => {
    try {
      // Validate with backend
      const res = await fetch(`${API_URL}/api/paths/${pathId}/reset`, { method: 'POST' });
      if (!res.ok) throw new Error('Reset validation failed');

      // Reset localStorage
      resetPath(pathId);

      showToast('Path progress reset! Reloading...', 'success');

      // Reload the path from scratch after a brief delay
      setTimeout(() => window.location.reload(), 500);
    } catch (err) {
      console.error('Path reset error:', err);
      showToast('Failed to reset path. Please try again.', 'error');
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
        <div className="flex items-center gap-4">
          <div className="text-sm text-gray-400">
            {stepsProgress.filter(s => s.completed).length}/{pathData.steps.length} completed
          </div>
          <button
            onClick={() => setShowResetConfirm(true)}
            className="px-3 py-1.5 text-sm text-gray-400 hover:text-white hover:bg-gray-700 rounded-lg transition-colors"
            title="Reset path progress"
          >
            Reset
          </button>
        </div>
      </div>

      {/* Reset Confirmation Modal */}
      {showResetConfirm && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="bg-gray-800 rounded-lg p-6 max-w-md mx-4 border border-gray-700"
          >
            <h3 className="text-lg font-semibold mb-2">Reset Progress?</h3>
            <p className="text-gray-400 mb-4">
              This will clear all progress for "{pathData.name}" and start from the beginning.
              This action cannot be undone.
            </p>
            <div className="flex gap-3 justify-end">
              <button
                onClick={() => setShowResetConfirm(false)}
                className="px-4 py-2 text-gray-400 hover:text-white transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={() => {
                  setShowResetConfirm(false);
                  handleResetPath();
                }}
                className="px-4 py-2 bg-red-600 hover:bg-red-700 rounded-lg transition-colors"
              >
                Reset Path
              </button>
            </div>
          </motion.div>
        </div>
      )}

      {/* Progress Bar */}
      <PathProgressBar
        steps={stepsProgress}
        stepInfo={pathData?.steps.map(s => ({
          stepNumber: s.stepNumber,
          title: s.title,
          problemId: s.problemId
        }))}
        currentStep={currentStepNum}
        onStepClick={navigateToStep}
      />

      {/* Main Content Grid - Responsive: 1 col mobile, 2 col tablet, 3 col desktop */}
      <div className="flex-1 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-[320px_1fr_320px] gap-4 min-h-0 overflow-hidden">
        {/* Left: Step Info */}
        <div className="space-y-3 overflow-y-auto">
          {currentStep && (
            <PathStepCard
              step={currentStep}
              progress={stepProgress ?? null}
              isCurrentStep={true}
            />
          )}

          {currentStep && (
            <StepHintPanel
              hints={getRichHintsForProblem(currentStep.problemId, currentStep.hints)}
              experiments={getExperimentsForProblem(currentStep.problemId)}
              attempts={stepProgress?.attempts ?? 0}
              accuracy={stepProgress?.bestAccuracy ?? 0}
              requiredAccuracy={currentStep.requiredAccuracy}
              problemId={currentStep.problemId}
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

        {/* Center: Challenge/Training Content - varies by step type */}
        <div className="space-y-3 overflow-y-auto">
          {/* BUILD CHALLENGE STEP */}
          {currentStep?.stepType === 'build_challenge' && currentStep.buildChallenge && !showTrainingAfterChallenge && (
            <NetworkBuilder
              problemId={currentStep.problemId}
              inputSize={currentProblem?.input_labels?.length ?? 2}
              outputSize={currentProblem?.output_labels?.length ?? 1}
              requirements={{
                minLayers: currentStep.buildChallenge.minLayers,
                maxLayers: currentStep.buildChallenge.maxLayers,
                minHiddenNeurons: currentStep.buildChallenge.minHiddenNeurons,
                maxHiddenNeurons: currentStep.buildChallenge.maxHiddenNeurons,
                mustHaveHidden: currentStep.buildChallenge.mustHaveHidden,
              }}
              onSubmit={handleBuildChallengeSubmit}
              onArchitectureChange={(arch) => {
                // Preview architecture in network viz
                if (networkState) {
                  setNetworkState({
                    ...networkState,
                    architecture: { ...networkState.architecture, layer_sizes: arch }
                  });
                }
              }}
            />
          )}

          {/* PREDICTION QUIZ STEP */}
          {currentStep?.stepType === 'prediction_quiz' && currentStep.predictionQuiz && !showTrainingAfterChallenge && (
            (() => {
              const quizId = currentStep.predictionQuiz!.quizId as keyof typeof PREDICTION_QUIZZES;
              const quizData = PREDICTION_QUIZZES[quizId];
              if (!quizData) return null;
              return (
                <PredictionQuiz
                  question={quizData.question}
                  context={quizData.context}
                  options={quizData.options}
                  onAnswer={handlePredictionQuizAnswer}
                  onRevealAndTrain={handlePredictionQuizRevealAndTrain}
                />
              );
            })()
          )}

          {/* DEBUG CHALLENGE STEP */}
          {currentStep?.stepType === 'debug_challenge' && currentStep.debugChallenge && !showTrainingAfterChallenge && (
            (() => {
              const challengeId = currentStep.debugChallenge!.challengeId as keyof typeof DEBUG_CHALLENGES;
              const challengeData = DEBUG_CHALLENGES[challengeId];
              if (!challengeData) return null;
              return (
                <DebugChallenge
                  title={challengeData.title}
                  description={challengeData.description}
                  problem={challengeData.problem}
                  config={challengeData.config}
                  symptoms={challengeData.symptoms}
                  options={challengeData.options}
                  onSolved={handleDebugChallengeSolved}
                  onTryFix={handleDebugChallengeTryFix}
                />
              );
            })()
          )}

          {/* TRAINING STEP (default) or after challenge completion */}
          {(currentStep?.stepType === 'training' || !currentStep?.stepType || showTrainingAfterChallenge) && (
            <>
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

              {currentProblem?.network_type === 'cnn' && featureMaps && (
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
                onStep={() => handleStepTraining()}
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

              {/* Training Narrator - Real-time insights */}
              {trainingInProgress && (
                <TrainingNarrator
                  epoch={trainingProgress?.epoch ?? 0}
                  loss={trainingProgress?.loss ?? 0}
                  accuracy={trainingProgress?.accuracy ?? 0}
                  isTraining={trainingInProgress}
                  isComplete={trainingComplete}
                  targetAccuracy={currentStep?.requiredAccuracy ?? 0.95}
                  isFailureCase={currentStep?.problemId?.startsWith('fail_')}
                />
              )}

              {/* Failure Dramatization for failure case problems */}
              {currentStep?.problemId && getFailureTypeFromProblemId(currentStep.problemId) && (
                <FailureDramatization
                  failureType={getFailureTypeFromProblemId(currentStep.problemId)!}
                  isActive={trainingInProgress || trainingComplete}
                  loss={trainingProgress?.loss ?? 0}
                  accuracy={trainingProgress?.accuracy ?? 0}
                  epoch={trainingProgress?.epoch ?? 0}
                />
              )}

              <LossCurve
                lossHistory={networkState?.loss_history ?? []}
                accuracyHistory={networkState?.accuracy_history ?? []}
                totalEpochs={networkState?.total_epochs}
                trainingComplete={trainingComplete}
                targetAccuracy={currentStep?.requiredAccuracy ?? 0.95}
                isFailureCase={currentStep?.problemId?.startsWith('fail_')}
              />
            </>
          )}
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
          isOpen={showPathCompleteModal}
          pathName={pathData.name}
          badge={pathData.badge}
          stats={{
            totalSteps: pathData.steps.length,
            totalAttempts: stepsProgress.reduce(
              (sum, sp) => sum + (sp?.attempts ?? 0),
              0
            ),
            avgAccuracy: stepsProgress.reduce(
              (sum, sp) => sum + (sp?.bestAccuracy ?? 0),
              0
            ) / Math.max(pathData.steps.length, 1),
          }}
          onClose={() => dispatch({ type: 'CLOSE_PATH_COMPLETE_MODAL' })}
          onReviewPath={() => dispatch({ type: 'CLOSE_PATH_COMPLETE_MODAL' })}
          onBackToPaths={onExitPath}
        />
      )}

      {/* Milestone Celebration Toast */}
      <AnimatePresence>
        {currentMilestone && (
          <motion.div
            initial={{ opacity: 0, y: 50, scale: 0.9 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -20, scale: 0.9 }}
            className="fixed bottom-8 left-1/2 -translate-x-1/2 z-50"
          >
            <motion.div
              className={`px-6 py-4 rounded-xl shadow-2xl flex items-center gap-4 ${
                currentMilestone === 75 ? 'bg-gradient-to-r from-yellow-500 to-orange-500' :
                currentMilestone === 50 ? 'bg-gradient-to-r from-cyan-500 to-teal-500' :
                                          'bg-gradient-to-r from-green-500 to-emerald-500'
              }`}
              initial={{ rotate: -5 }}
              animate={{ rotate: [0, -2, 2, 0] }}
              transition={{ duration: 0.5 }}
            >
              <motion.span
                className="text-4xl"
                animate={{ scale: [1, 1.3, 1], rotate: [0, 10, -10, 0] }}
                transition={{ duration: 0.6, repeat: 2 }}
              >
                {currentMilestone === 75 ? '🏆' : currentMilestone === 50 ? '⭐' : '🎯'}
              </motion.span>
              <div className="text-white">
                <div className="font-bold text-lg">{currentMilestone}% Complete!</div>
                <div className="text-sm opacity-90">
                  {currentMilestone === 75 ? 'Almost there! Final stretch!' :
                   currentMilestone === 50 ? 'Halfway there! Keep going!' :
                                             'Great start! You\'re on a roll!'}
                </div>
              </div>
              <button
                onClick={() => setCurrentMilestone(null)}
                className="ml-2 text-white/80 hover:text-white"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};
