import { useEffect, useRef, useState, useCallback } from 'react';
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

    socket.on('status', (status: SystemStatus & { connected: boolean }) => {
      setSystemStatus(status);
      setTrainingComplete(status.training_complete);
    });

    socket.on('gpio_state', (state: GPIOState) => {
      setGpioState(state);
    });

    socket.on('training_progress', (progress: TrainingProgress) => {
      setTrainingProgress(progress);
    });

    socket.on('training_started', () => {
      setTrainingComplete(false);
    });

    socket.on('training_complete', (result: TrainingResult) => {
      setTrainingComplete(true);
      setTrainingProgress({
        epoch: result.epochs,
        loss: result.final_loss,
        accuracy: result.final_accuracy,
      });
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
    toggleButton,
    setButtons,
    socket: socketRef.current,
  };
}
