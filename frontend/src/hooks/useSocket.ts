import { useEffect, useState, useCallback, useRef } from 'react';
import { io, Socket } from 'socket.io-client';
import type { GPIOState, PredictionResult, TrainingProgress, TrainingResult, SystemStatus } from '../types';

const SOCKET_URL = 'http://localhost:5000';

interface UseSocketReturn {
  connected: boolean;
  gpioState: GPIOState | null;
  trainingProgress: TrainingProgress | null;
  lastPrediction: PredictionResult | null;
  systemStatus: SystemStatus | null;
  trainingComplete: boolean;
  trainingError: string | null;
  toggleButton: (index: number) => void;
  setButtons: (states: number[]) => void;
  socket: Socket | null;
}

export function useSocket(): UseSocketReturn {
  const socketRef = useRef<Socket | null>(null);
  const [connected, setConnected] = useState(false);
  const [gpioState, setGpioState] = useState<GPIOState | null>(null);
  const [trainingProgress, setTrainingProgress] = useState<TrainingProgress | null>(null);
  const [lastPrediction, setLastPrediction] = useState<PredictionResult | null>(null);
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null);
  const [trainingComplete, setTrainingComplete] = useState(false);
  const [trainingError, setTrainingError] = useState<string | null>(null);

  useEffect(() => {
    const socket = io(SOCKET_URL, {
      transports: ['websocket', 'polling'],
    });

    socketRef.current = socket;

    socket.on('connect', () => {
      console.log('Connected to server');
      setConnected(true);
    });

    socket.on('disconnect', () => {
      console.log('Disconnected from server');
      setConnected(false);
    });

    socket.on('status', (status: SystemStatus & { connected: boolean; training_in_progress?: boolean }) => {
      console.log('Received status:', status);
      setSystemStatus(status);
      setTrainingComplete(status.training_complete);
      // Note: training_in_progress is managed by App.tsx, but this provides initial sync
    });

    socket.on('gpio_state', (state: GPIOState) => {
      setGpioState(state);
    });

    socket.on('training_progress', (progress: TrainingProgress) => {
      console.log('Received training_progress:', progress);
      setTrainingProgress(progress);
    });

    socket.on('training_started', (data) => {
      console.log('Received training_started:', data);
      setTrainingComplete(false);
      setTrainingError(null);  // Clear any previous error
    });

    socket.on('training_error', (data: { error: string }) => {
      console.error('Received training_error:', data.error);
      setTrainingError(data.error);
    });

    socket.on('training_complete', (result: TrainingResult) => {
      console.log('Received training_complete:', result);
      setTrainingComplete(true);
      setTrainingError(null);  // Clear any previous error
      setTrainingProgress({
        epoch: result.epochs,
        loss: result.final_loss,
        accuracy: result.final_accuracy,
      });
    });

    socket.on('network_reset', () => {
      // Reset all training-related state
      setTrainingComplete(false);
      setTrainingProgress({ epoch: 0, loss: 0, accuracy: 0 });
      setLastPrediction(null);
    });

    socket.on('prediction', (prediction: PredictionResult) => {
      setLastPrediction(prediction);
    });

    return () => {
      socket.disconnect();
    };
  }, []);

  const toggleButton = useCallback((index: number) => {
    socketRef.current?.emit('toggle_button', { index });
  }, []);

  const setButtons = useCallback((states: number[]) => {
    socketRef.current?.emit('set_buttons', { states });
  }, []);

  return {
    connected,
    gpioState,
    trainingProgress,
    lastPrediction,
    systemStatus,
    trainingComplete,
    trainingError,
    toggleButton,
    setButtons,
    // eslint-disable-next-line react-hooks/refs -- Intentional: socket ref is stable after mount
    socket: socketRef.current,
  };
}
