import { useEffect, useState, useRef } from 'react';
import { io, Socket } from 'socket.io-client';
import type { PredictionResult, TrainingProgress, TrainingResult } from '../types';

const SOCKET_URL = 'http://localhost:5000';

interface UseSocketReturn {
  connected: boolean;
  trainingProgress: TrainingProgress | null;
  lastPrediction: PredictionResult | null;
  trainingComplete: boolean;
  trainingError: string | null;
  socket: Socket | null;
}

export function useSocket(): UseSocketReturn {
  const socketRef = useRef<Socket | null>(null);
  const [connected, setConnected] = useState(false);
  const [trainingProgress, setTrainingProgress] = useState<TrainingProgress | null>(null);
  const [lastPrediction, setLastPrediction] = useState<PredictionResult | null>(null);
  const [trainingComplete, setTrainingComplete] = useState(false);
  const [trainingError, setTrainingError] = useState<string | null>(null);

  useEffect(() => {
    const socket = io(SOCKET_URL, {
      transports: ['websocket', 'polling'],
    });

    socketRef.current = socket;

    socket.on('connect', () => {
      setConnected(true);
    });

    socket.on('disconnect', () => {
      setConnected(false);
    });

    socket.on('status', (status: { training_complete: boolean }) => {
      setTrainingComplete(status.training_complete);
    });

    socket.on('training_progress', (progress: TrainingProgress) => {
      setTrainingProgress(progress);
    });

    socket.on('training_started', () => {
      setTrainingComplete(false);
      setTrainingError(null);
    });

    socket.on('training_error', (data: { error: string }) => {
      console.error('Training error:', data.error);
      setTrainingError(data.error);
    });

    socket.on('training_complete', (result: TrainingResult) => {
      setTrainingComplete(true);
      setTrainingError(null);
      setTrainingProgress({
        epoch: result.epochs,
        loss: result.final_loss,
        accuracy: result.final_accuracy,
      });
    });

    socket.on('network_reset', () => {
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

  return {
    connected,
    trainingProgress,
    lastPrediction,
    trainingComplete,
    trainingError,
    // eslint-disable-next-line react-hooks/refs -- Intentional: socket ref is stable after mount
    socket: socketRef.current,
  };
}
