import { useState, useCallback, useRef, useEffect } from 'react';

interface AnimationState {
  isAnimating: boolean;
  activeLayer: number;
  activePulses: Pulse[];
  layerActivations: Map<number, number[]>;
}

interface Pulse {
  id: string;
  fromLayer: number;
  toLayer: number;
  fromNeuron: number;
  toNeuron: number;
  progress: number;
  weight: number;
}

interface UseNetworkAnimationProps {
  layerSizes: number[];
  animationSpeed?: number;
  onAnimationComplete?: () => void;
}

export function useNetworkAnimation({
  layerSizes,
  animationSpeed = 800,
  onAnimationComplete,
}: UseNetworkAnimationProps) {
  const [animationState, setAnimationState] = useState<AnimationState>({
    isAnimating: false,
    activeLayer: -1,
    activePulses: [],
    layerActivations: new Map(),
  });

  const animationRef = useRef<number | null>(null);
  const pulseIdRef = useRef(0);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, []);

  const triggerForwardPass = useCallback(
    (inputActivations?: number[]) => {
      if (animationState.isAnimating) return;

      // Initialize with input layer activations
      const initialActivations = new Map<number, number[]>();
      if (inputActivations) {
        initialActivations.set(0, inputActivations);
      } else {
        // Random activations for demo
        initialActivations.set(
          0,
          Array.from({ length: layerSizes[0] }, () => Math.random())
        );
      }

      setAnimationState({
        isAnimating: true,
        activeLayer: 0,
        activePulses: [],
        layerActivations: initialActivations,
      });

      // Start animation sequence
      let currentLayer = 0;
      const animateLayer = () => {
        if (currentLayer >= layerSizes.length - 1) {
          // Animation complete
          setAnimationState((prev) => ({
            ...prev,
            isAnimating: false,
            activeLayer: -1,
          }));
          onAnimationComplete?.();
          return;
        }

        // Create pulses for connections from current layer to next
        const newPulses: Pulse[] = [];
        for (let from = 0; from < layerSizes[currentLayer]; from++) {
          for (let to = 0; to < layerSizes[currentLayer + 1]; to++) {
            newPulses.push({
              id: `pulse-${pulseIdRef.current++}`,
              fromLayer: currentLayer,
              toLayer: currentLayer + 1,
              fromNeuron: from,
              toNeuron: to,
              progress: 0,
              weight: Math.random() * 2 - 1, // Will be replaced with actual weights
            });
          }
        }

        setAnimationState((prev) => ({
          ...prev,
          activeLayer: currentLayer,
          activePulses: newPulses,
        }));

        // Animate pulses
        const startTime = performance.now();
        const animatePulses = (time: number) => {
          const elapsed = time - startTime;
          const progress = Math.min(elapsed / animationSpeed, 1);

          setAnimationState((prev) => ({
            ...prev,
            activePulses: prev.activePulses.map((pulse) => ({
              ...pulse,
              progress,
            })),
          }));

          if (progress < 1) {
            animationRef.current = requestAnimationFrame(animatePulses);
          } else {
            // Move to next layer
            currentLayer++;
            setAnimationState((prev) => {
              const newActivations = new Map(prev.layerActivations);
              // Generate random activations for the new layer (simulated)
              newActivations.set(
                currentLayer,
                Array.from({ length: layerSizes[currentLayer] }, () =>
                  Math.random()
                )
              );
              return {
                ...prev,
                layerActivations: newActivations,
                activePulses: [],
              };
            });

            // Small delay before next layer
            setTimeout(animateLayer, 150);
          }
        };

        animationRef.current = requestAnimationFrame(animatePulses);
      };

      // Start after a brief delay
      setTimeout(animateLayer, 200);
    },
    [animationState.isAnimating, layerSizes, animationSpeed, onAnimationComplete]
  );

  const resetAnimation = useCallback(() => {
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
    }
    setAnimationState({
      isAnimating: false,
      activeLayer: -1,
      activePulses: [],
      layerActivations: new Map(),
    });
  }, []);

  const getLayerOpacity = useCallback(
    (layerIndex: number): number => {
      if (!animationState.isAnimating) return 1;
      if (layerIndex <= animationState.activeLayer) return 1;
      if (layerIndex === animationState.activeLayer + 1) {
        // Fading in
        const progress = animationState.activePulses[0]?.progress ?? 0;
        return 0.3 + progress * 0.7;
      }
      return 0.3;
    },
    [animationState]
  );

  const getConnectionOpacity = useCallback(
    (fromLayer: number, _toLayer: number): number => {
      if (!animationState.isAnimating) return 1;
      if (fromLayer < animationState.activeLayer) return 1;
      if (fromLayer === animationState.activeLayer) {
        const progress = animationState.activePulses[0]?.progress ?? 0;
        return 0.3 + progress * 0.7;
      }
      return 0.2;
    },
    [animationState]
  );

  return {
    isAnimating: animationState.isAnimating,
    activeLayer: animationState.activeLayer,
    activePulses: animationState.activePulses,
    layerActivations: animationState.layerActivations,
    triggerForwardPass,
    resetAnimation,
    getLayerOpacity,
    getConnectionOpacity,
  };
}
