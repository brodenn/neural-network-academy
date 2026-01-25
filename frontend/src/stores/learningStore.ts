import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { PathProgressData, StepProgressData } from '../types';

// Achievement definitions
export interface Achievement {
  id: string;
  name: string;
  description: string;
  icon: string;
  category: 'milestone' | 'skill' | 'challenge' | 'streak';
  unlockedAt?: string;
}

const ACHIEVEMENTS: Achievement[] = [
  // Milestone achievements
  { id: 'first_step', name: 'First Steps', description: 'Complete your first step', icon: '🎯', category: 'milestone' },
  { id: 'path_complete', name: 'Path Master', description: 'Complete an entire learning path', icon: '🏆', category: 'milestone' },
  { id: 'all_foundations', name: 'Foundation Builder', description: 'Complete the Foundations path', icon: '🏗️', category: 'milestone' },

  // Skill achievements
  { id: 'perfect_accuracy', name: 'Perfectionist', description: 'Achieve 100% accuracy on any step', icon: '💯', category: 'skill' },
  { id: 'speed_learner', name: 'Speed Learner', description: 'Complete a step in under 3 attempts', icon: '⚡', category: 'skill' },
  { id: 'hint_free', name: 'Self Sufficient', description: 'Complete a step without using hints', icon: '🧠', category: 'skill' },

  // Challenge achievements
  { id: 'failure_master', name: 'Learning from Failure', description: 'Complete the Pitfall Prevention path', icon: '📚', category: 'challenge' },
  { id: 'cnn_expert', name: 'Vision Expert', description: 'Complete the Convolutional Vision path', icon: '👁️', category: 'challenge' },

  // Streak achievements
  { id: 'streak_3', name: 'Consistent', description: 'Maintain a 3-day streak', icon: '🔥', category: 'streak' },
  { id: 'streak_7', name: 'Dedicated', description: 'Maintain a 7-day streak', icon: '🔥🔥', category: 'streak' },
  { id: 'streak_30', name: 'Unstoppable', description: 'Maintain a 30-day streak', icon: '🌟', category: 'streak' },
];

// Streak data
interface StreakData {
  lastAccessDate: string;
  currentStreak: number;
  bestStreak: number;
}

// Store state
interface LearningState {
  // Progress data
  pathProgress: Record<string, PathProgressData>;

  // Achievements
  unlockedAchievements: string[];

  // Streak
  streak: StreakData;

  // Stats
  totalStepsCompleted: number;
  totalAttempts: number;

  // Actions
  initializePath: (pathId: string, steps: { stepNumber: number; problemId: string }[]) => PathProgressData;
  resetPath: (pathId: string) => void;
  completeStep: (pathId: string, stepNumber: number, accuracy: number, hintsUsed?: number) => void;
  recordAttempt: (pathId: string, stepNumber: number, accuracy: number) => void;
  setCurrentStep: (pathId: string, stepNumber: number) => void;

  // Getters
  getPathProgress: (pathId: string) => PathProgressData | null;
  getStepProgress: (pathId: string, stepNumber: number) => StepProgressData | null;
  isStepUnlocked: (pathId: string, stepNumber: number) => boolean;
  isStepCompleted: (pathId: string, stepNumber: number) => boolean;
  isPathComplete: (pathId: string, totalSteps: number) => boolean;
  getCompletedStepsCount: (pathId: string) => number;

  // Achievements
  checkAndUnlockAchievements: (context: AchievementContext) => string[];
  getAchievements: () => Achievement[];
  getUnlockedAchievements: () => Achievement[];

  // Streak
  updateStreak: () => StreakData;
}

interface AchievementContext {
  pathId?: string;
  stepNumber?: number;
  accuracy?: number;
  attempts?: number;
  hintsUsed?: number;
}

const getTodayDate = (): string => new Date().toISOString().split('T')[0];
const getYesterdayDate = (): string => {
  const yesterday = new Date();
  yesterday.setDate(yesterday.getDate() - 1);
  return yesterday.toISOString().split('T')[0];
};

export const useLearningStore = create<LearningState>()(
  persist(
    (set, get) => ({
      // Initial state
      pathProgress: {},
      unlockedAchievements: [],
      streak: { lastAccessDate: '', currentStreak: 0, bestStreak: 0 },
      totalStepsCompleted: 0,
      totalAttempts: 0,

      // Initialize a path
      initializePath: (pathId, steps) => {
        const existing = get().pathProgress[pathId];
        if (existing) {
          // Update lastActiveAt and return
          const updated = { ...existing, lastActiveAt: new Date().toISOString() };
          set(state => ({
            pathProgress: { ...state.pathProgress, [pathId]: updated }
          }));
          return updated;
        }

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
            unlocked: index === 0,
            completed: false,
            attempts: 0,
            bestAccuracy: 0
          }))
        };

        set(state => ({
          pathProgress: { ...state.pathProgress, [pathId]: newProgress }
        }));

        return newProgress;
      },

      // Reset a path
      resetPath: (pathId) => {
        set(state => {
          const newProgress = { ...state.pathProgress };
          delete newProgress[pathId];
          return { pathProgress: newProgress };
        });
      },

      // Complete a step
      completeStep: (pathId, stepNumber, accuracy, hintsUsed = 0) => {
        const progress = get().pathProgress[pathId];
        if (!progress) return;

        const stepIndex = progress.steps.findIndex(s => s.stepNumber === stepNumber);
        if (stepIndex === -1) return;

        const updatedSteps = [...progress.steps];
        const step = updatedSteps[stepIndex];
        const wasFirstCompletion = !step.completed;

        step.completed = true;
        step.completedAt = new Date().toISOString();
        step.attempts += 1;
        step.bestAccuracy = Math.max(step.bestAccuracy, accuracy);

        // Unlock next step
        if (stepIndex + 1 < updatedSteps.length) {
          updatedSteps[stepIndex + 1].unlocked = true;
        }

        const completedCount = updatedSteps.filter(s => s.completed).length;
        const updated: PathProgressData = {
          ...progress,
          currentStep: Math.min(stepNumber + 1, progress.steps.length),
          stepsCompleted: completedCount,
          lastActiveAt: new Date().toISOString(),
          steps: updatedSteps
        };

        set(state => ({
          pathProgress: { ...state.pathProgress, [pathId]: updated },
          totalStepsCompleted: wasFirstCompletion
            ? state.totalStepsCompleted + 1
            : state.totalStepsCompleted,
          totalAttempts: state.totalAttempts + 1
        }));

        // Check achievements
        get().checkAndUnlockAchievements({
          pathId,
          stepNumber,
          accuracy,
          attempts: step.attempts,
          hintsUsed
        });
      },

      // Record an attempt
      recordAttempt: (pathId, stepNumber, accuracy) => {
        const progress = get().pathProgress[pathId];
        if (!progress) return;

        const stepIndex = progress.steps.findIndex(s => s.stepNumber === stepNumber);
        if (stepIndex === -1) return;

        const updatedSteps = [...progress.steps];
        updatedSteps[stepIndex].attempts += 1;
        updatedSteps[stepIndex].bestAccuracy = Math.max(
          updatedSteps[stepIndex].bestAccuracy,
          accuracy
        );

        set(state => ({
          pathProgress: {
            ...state.pathProgress,
            [pathId]: { ...progress, steps: updatedSteps, lastActiveAt: new Date().toISOString() }
          },
          totalAttempts: state.totalAttempts + 1
        }));
      },

      // Set current step
      setCurrentStep: (pathId, stepNumber) => {
        const progress = get().pathProgress[pathId];
        if (!progress) return;

        const step = progress.steps.find(s => s.stepNumber === stepNumber);
        if (!step?.unlocked) return;

        set(state => ({
          pathProgress: {
            ...state.pathProgress,
            [pathId]: { ...progress, currentStep: stepNumber, lastActiveAt: new Date().toISOString() }
          }
        }));
      },

      // Getters
      getPathProgress: (pathId) => get().pathProgress[pathId] || null,

      getStepProgress: (pathId, stepNumber) => {
        const progress = get().pathProgress[pathId];
        return progress?.steps.find(s => s.stepNumber === stepNumber) || null;
      },

      isStepUnlocked: (pathId, stepNumber) => {
        const progress = get().pathProgress[pathId];
        if (!progress) return stepNumber === 1;
        return progress.steps.find(s => s.stepNumber === stepNumber)?.unlocked ?? false;
      },

      isStepCompleted: (pathId, stepNumber) => {
        const progress = get().pathProgress[pathId];
        return progress?.steps.find(s => s.stepNumber === stepNumber)?.completed ?? false;
      },

      isPathComplete: (pathId, totalSteps) => {
        const progress = get().pathProgress[pathId];
        return progress?.stepsCompleted === totalSteps;
      },

      getCompletedStepsCount: (pathId) => {
        const progress = get().pathProgress[pathId];
        return progress?.stepsCompleted ?? 0;
      },

      // Achievement system
      checkAndUnlockAchievements: (context) => {
        const state = get();
        const newlyUnlocked: string[] = [];

        const unlock = (id: string) => {
          if (!state.unlockedAchievements.includes(id)) {
            newlyUnlocked.push(id);
          }
        };

        // First step
        if (state.totalStepsCompleted === 1) {
          unlock('first_step');
        }

        // Perfect accuracy
        if (context.accuracy === 1) {
          unlock('perfect_accuracy');
        }

        // Speed learner (completed in under 3 attempts)
        if (context.attempts && context.attempts <= 3) {
          unlock('speed_learner');
        }

        // Hint free (0 hints used)
        if (context.hintsUsed === 0) {
          unlock('hint_free');
        }

        // Path completion achievements
        if (context.pathId) {
          const progress = state.pathProgress[context.pathId];
          if (progress) {
            const allCompleted = progress.steps.every(s => s.completed);
            if (allCompleted) {
              unlock('path_complete');

              if (context.pathId === 'foundations') unlock('all_foundations');
              if (context.pathId === 'pitfall-prevention') unlock('failure_master');
              if (context.pathId === 'convolutional-vision') unlock('cnn_expert');
            }
          }
        }

        // Streak achievements
        const { currentStreak } = state.streak;
        if (currentStreak >= 3) unlock('streak_3');
        if (currentStreak >= 7) unlock('streak_7');
        if (currentStreak >= 30) unlock('streak_30');

        if (newlyUnlocked.length > 0) {
          set(s => ({
            unlockedAchievements: [...s.unlockedAchievements, ...newlyUnlocked]
          }));
        }

        return newlyUnlocked;
      },

      getAchievements: () => ACHIEVEMENTS,

      getUnlockedAchievements: () => {
        const unlocked = get().unlockedAchievements;
        return ACHIEVEMENTS.filter(a => unlocked.includes(a.id)).map(a => ({
          ...a,
          unlockedAt: new Date().toISOString() // Would need to track actual unlock time
        }));
      },

      // Streak management
      updateStreak: () => {
        const today = getTodayDate();
        const yesterday = getYesterdayDate();
        const current = get().streak;

        let newStreak: StreakData;

        if (current.lastAccessDate === today) {
          newStreak = current;
        } else if (current.lastAccessDate === yesterday) {
          const newCurrentStreak = current.currentStreak + 1;
          newStreak = {
            lastAccessDate: today,
            currentStreak: newCurrentStreak,
            bestStreak: Math.max(current.bestStreak, newCurrentStreak)
          };
        } else {
          newStreak = {
            lastAccessDate: today,
            currentStreak: 1,
            bestStreak: Math.max(current.bestStreak, 1)
          };
        }

        set({ streak: newStreak });

        // Check streak achievements
        get().checkAndUnlockAchievements({});

        return newStreak;
      }
    }),
    {
      name: 'learning-progress-store',
      partialize: (state) => ({
        pathProgress: state.pathProgress,
        unlockedAchievements: state.unlockedAchievements,
        streak: state.streak,
        totalStepsCompleted: state.totalStepsCompleted,
        totalAttempts: state.totalAttempts
      })
    }
  )
);

// Computed selectors
export const useAchievementCount = () => useLearningStore(state => state.unlockedAchievements.length);
export const useTotalProgress = () => useLearningStore(state => ({
  stepsCompleted: state.totalStepsCompleted,
  attempts: state.totalAttempts
}));
export const useStreak = () => useLearningStore(state => state.streak);
