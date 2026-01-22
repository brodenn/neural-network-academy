import { useEffect, useCallback } from 'react';

interface KeyboardShortcutsOptions {
  onStartTraining: () => void;
  onStopTraining: () => void;
  onReset: () => void;
  onStep?: () => void;
  trainingInProgress: boolean;
  trainingComplete: boolean;
  disabled?: boolean;
}

/**
 * Hook for keyboard shortcuts in the Neural Network Learning Lab.
 *
 * Shortcuts:
 * - Space: Start/Stop training
 * - R: Reset network
 * - S: Step (single epoch, when not training)
 */
export function useKeyboardShortcuts({
  onStartTraining,
  onStopTraining,
  onReset,
  onStep,
  trainingInProgress,
  trainingComplete,
  disabled = false,
}: KeyboardShortcutsOptions) {
  const handleKeyDown = useCallback(
    (event: KeyboardEvent) => {
      // Don't trigger if typing in an input
      if (
        event.target instanceof HTMLInputElement ||
        event.target instanceof HTMLTextAreaElement
      ) {
        return;
      }

      // Don't trigger if modifier keys are pressed (allow browser shortcuts)
      if (event.ctrlKey || event.metaKey || event.altKey) {
        return;
      }

      if (disabled) return;

      switch (event.code) {
        case 'Space':
          event.preventDefault();
          if (trainingInProgress) {
            onStopTraining();
          } else {
            onStartTraining();
          }
          break;

        case 'KeyR':
          event.preventDefault();
          if (!trainingInProgress) {
            onReset();
          }
          break;

        case 'KeyS':
          event.preventDefault();
          if (!trainingInProgress && trainingComplete && onStep) {
            onStep();
          }
          break;
      }
    },
    [
      onStartTraining,
      onStopTraining,
      onReset,
      onStep,
      trainingInProgress,
      trainingComplete,
      disabled,
    ]
  );

  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleKeyDown]);
}
