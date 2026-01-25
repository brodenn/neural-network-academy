import { useState, useCallback, useEffect } from 'react';
import type { PathProgressData, StepProgressData } from '../types';

const STORAGE_PREFIX = 'learning_path_';

interface PathStepConfig {
  stepNumber: number;
  problemId: string;
}

interface UsePathProgressReturn {
  // Progress data
  getPathProgress: (pathId: string) => PathProgressData | null;
  getAllPathsProgress: () => Map<string, PathProgressData>;

  // Lifecycle operations
  initializePath: (pathId: string, steps: PathStepConfig[]) => PathProgressData;
  resetPath: (pathId: string) => void;

  // Step operations
  completeStep: (pathId: string, stepNumber: number, accuracy: number) => void;
  recordAttempt: (pathId: string, stepNumber: number, accuracy: number) => void;
  setCurrentStep: (pathId: string, stepNumber: number) => void;

  // Status helpers
  isStepUnlocked: (pathId: string, stepNumber: number) => boolean;
  isStepCompleted: (pathId: string, stepNumber: number) => boolean;
  getStepProgress: (pathId: string, stepNumber: number) => StepProgressData | null;

  // Computed values
  isPathComplete: (pathId: string, totalSteps: number) => boolean;
  getCompletedStepsCount: (pathId: string) => number;
}

export function usePathProgress(): UsePathProgressReturn {
  // Force re-render when localStorage changes
  const [, setUpdateTrigger] = useState(0);

  const forceUpdate = useCallback(() => {
    setUpdateTrigger(prev => prev + 1);
  }, []);

  // Listen for storage events from other tabs
  useEffect(() => {
    const handleStorageChange = (e: StorageEvent) => {
      if (e.key?.startsWith(STORAGE_PREFIX)) {
        forceUpdate();
      }
    };
    window.addEventListener('storage', handleStorageChange);
    return () => window.removeEventListener('storage', handleStorageChange);
  }, [forceUpdate]);

  const getStorageKey = (pathId: string): string => `${STORAGE_PREFIX}${pathId}`;

  const getPathProgress = useCallback((pathId: string): PathProgressData | null => {
    try {
      const stored = localStorage.getItem(getStorageKey(pathId));
      if (!stored) return null;
      return JSON.parse(stored) as PathProgressData;
    } catch (e) {
      console.error('Failed to parse path progress:', e);
      return null;
    }
  }, []);

  const savePathProgress = useCallback((pathId: string, data: PathProgressData): void => {
    try {
      localStorage.setItem(getStorageKey(pathId), JSON.stringify(data));
      forceUpdate();
    } catch (e) {
      console.error('Failed to save path progress:', e);
    }
  }, [forceUpdate]);

  const getAllPathsProgress = useCallback((): Map<string, PathProgressData> => {
    const result = new Map<string, PathProgressData>();
    for (let i = 0; i < localStorage.length; i++) {
      const key = localStorage.key(i);
      if (key?.startsWith(STORAGE_PREFIX)) {
        try {
          const data = localStorage.getItem(key);
          if (data) {
            const progress = JSON.parse(data) as PathProgressData;
            result.set(progress.pathId, progress);
          }
        } catch (e) {
          console.error('Failed to parse progress for key:', key, e);
        }
      }
    }
    return result;
  }, []);

  const initializePath = useCallback((pathId: string, steps: PathStepConfig[]): PathProgressData => {
    // Check if progress already exists
    const existing = getPathProgress(pathId);
    if (existing) {
      // Validate and repair existing data - ensure all steps have proper stepNumber
      const validSteps = existing.steps.filter(s => typeof s.stepNumber === 'number');

      // If data is corrupted (missing stepNumbers) or step count doesn't match, rebuild from config
      if (validSteps.length !== steps.length) {
        // Merge existing progress with new step config
        const repairedSteps = steps.map((stepConfig, index) => {
          // Try to find existing progress for this step
          const existingStep = existing.steps.find(s => s.stepNumber === stepConfig.stepNumber || s.problemId === stepConfig.problemId);
          return {
            stepNumber: stepConfig.stepNumber,
            problemId: stepConfig.problemId,
            unlocked: existingStep?.unlocked ?? (index === 0),
            completed: existingStep?.completed ?? false,
            completedAt: existingStep?.completedAt,
            attempts: existingStep?.attempts ?? 0,
            bestAccuracy: existingStep?.bestAccuracy ?? 0
          };
        });

        const repaired: PathProgressData = {
          ...existing,
          steps: repairedSteps,
          lastActiveAt: new Date().toISOString()
        };
        savePathProgress(pathId, repaired);
        return repaired;
      }

      // Update lastActiveAt and return existing
      const updated = {
        ...existing,
        lastActiveAt: new Date().toISOString()
      };
      savePathProgress(pathId, updated);
      return updated;
    }

    // Create new progress
    const now = new Date().toISOString();
    const newProgress: PathProgressData = {
      pathId,
      currentStep: 1,
      stepsCompleted: 0,
      startedAt: now,
      lastActiveAt: now,
      steps: steps.map((step, index) => ({
        stepNumber: step.stepNumber,
        problemId: step.problemId,
        unlocked: index === 0, // First step unlocked by default
        completed: false,
        attempts: 0,
        bestAccuracy: 0
      }))
    };

    savePathProgress(pathId, newProgress);
    return newProgress;
  }, [getPathProgress, savePathProgress]);

  const resetPath = useCallback((pathId: string): void => {
    localStorage.removeItem(getStorageKey(pathId));
    forceUpdate();
  }, [forceUpdate]);

  const completeStep = useCallback((pathId: string, stepNumber: number, accuracy: number): void => {
    const progress = getPathProgress(pathId);
    if (!progress) return;

    const stepIndex = progress.steps.findIndex(s => s.stepNumber === stepNumber);
    if (stepIndex === -1) return;

    const updatedSteps = [...progress.steps];
    const step = updatedSteps[stepIndex];

    // Mark step as completed
    step.completed = true;
    step.completedAt = new Date().toISOString();
    step.attempts += 1;
    step.bestAccuracy = Math.max(step.bestAccuracy, accuracy);

    // Unlock next step if exists
    if (stepIndex + 1 < updatedSteps.length) {
      updatedSteps[stepIndex + 1].unlocked = true;
    }

    // Update progress
    const completedCount = updatedSteps.filter(s => s.completed).length;
    const updated: PathProgressData = {
      ...progress,
      currentStep: Math.min(stepNumber + 1, progress.steps.length),
      stepsCompleted: completedCount,
      lastActiveAt: new Date().toISOString(),
      steps: updatedSteps
    };

    savePathProgress(pathId, updated);
  }, [getPathProgress, savePathProgress]);

  const recordAttempt = useCallback((pathId: string, stepNumber: number, accuracy: number): void => {
    const progress = getPathProgress(pathId);
    if (!progress) return;

    const stepIndex = progress.steps.findIndex(s => s.stepNumber === stepNumber);
    if (stepIndex === -1) return;

    const updatedSteps = [...progress.steps];
    const step = updatedSteps[stepIndex];

    step.attempts += 1;
    step.bestAccuracy = Math.max(step.bestAccuracy, accuracy);

    const updated: PathProgressData = {
      ...progress,
      lastActiveAt: new Date().toISOString(),
      steps: updatedSteps
    };

    savePathProgress(pathId, updated);
  }, [getPathProgress, savePathProgress]);

  const setCurrentStep = useCallback((pathId: string, stepNumber: number): void => {
    const progress = getPathProgress(pathId);
    if (!progress) return;

    // Verify step is unlocked
    const step = progress.steps.find(s => s.stepNumber === stepNumber);
    if (!step?.unlocked) return;

    const updated: PathProgressData = {
      ...progress,
      currentStep: stepNumber,
      lastActiveAt: new Date().toISOString()
    };

    savePathProgress(pathId, updated);
  }, [getPathProgress, savePathProgress]);

  const isStepUnlocked = useCallback((pathId: string, stepNumber: number): boolean => {
    const progress = getPathProgress(pathId);
    if (!progress) return stepNumber === 1; // First step always unlocked

    const step = progress.steps.find(s => s.stepNumber === stepNumber);
    return step?.unlocked ?? false;
  }, [getPathProgress]);

  const isStepCompleted = useCallback((pathId: string, stepNumber: number): boolean => {
    const progress = getPathProgress(pathId);
    if (!progress) return false;

    const step = progress.steps.find(s => s.stepNumber === stepNumber);
    return step?.completed ?? false;
  }, [getPathProgress]);

  const getStepProgress = useCallback((pathId: string, stepNumber: number): StepProgressData | null => {
    const progress = getPathProgress(pathId);
    if (!progress) return null;

    return progress.steps.find(s => s.stepNumber === stepNumber) ?? null;
  }, [getPathProgress]);

  const isPathComplete = useCallback((pathId: string, totalSteps: number): boolean => {
    const progress = getPathProgress(pathId);
    if (!progress) return false;

    return progress.stepsCompleted === totalSteps;
  }, [getPathProgress]);

  const getCompletedStepsCount = useCallback((pathId: string): number => {
    const progress = getPathProgress(pathId);
    if (!progress) return 0;

    return progress.stepsCompleted;
  }, [getPathProgress]);

  return {
    getPathProgress,
    getAllPathsProgress,
    initializePath,
    resetPath,
    completeStep,
    recordAttempt,
    setCurrentStep,
    isStepUnlocked,
    isStepCompleted,
    getStepProgress,
    isPathComplete,
    getCompletedStepsCount
  };
}
