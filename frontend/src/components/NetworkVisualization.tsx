import { memo, useMemo, useState, useCallback, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import type { LayerWeights } from '../types';
import { DenseNetworkEducationalViz } from './DenseNetworkEducationalViz';

// Animation settings (can be made configurable via props)
const ANIMATION_CONFIG = {
  enablePulse: true,
  enableParticles: true,
  enableWeightSpring: true,
  enableBackprop: true,
  particleCount: 3,
  pulseDuration: 1.5,
  springStiffness: 300,
  springDamping: 20,
};

// Layout constants (moved outside component to avoid recreation on every render)
const VIEWBOX_WIDTH = 800;
const VIEWBOX_HEIGHT = 380;
const MARGIN = { top: 35, right: 50, bottom: 45, left: 50 };
const CONTENT_WIDTH = VIEWBOX_WIDTH - MARGIN.left - MARGIN.right;
const CONTENT_HEIGHT = VIEWBOX_HEIGHT - MARGIN.top - MARGIN.bottom;

interface NetworkVisualizationProps {
  layerSizes: number[];
  weights: LayerWeights[];
  activations?: number[][];
  inputLabels?: string[];
  outputLabels?: string[];
  outputActivation?: 'sigmoid' | 'softmax';
  trainingInProgress?: boolean;
  currentEpoch?: number;
}

interface TooltipData {
  x: number;
  y: number;
  content: string[];
  type: 'neuron' | 'connection';
}

export const NetworkVisualization = memo(function NetworkVisualization({
  layerSizes,
  weights,
  activations,
  inputLabels = [],
  outputLabels = [],
  outputActivation = 'sigmoid',
  trainingInProgress = false,
  currentEpoch = 0,
}: NetworkVisualizationProps) {

  const [tooltip, setTooltip] = useState<TooltipData | null>(null);
  const [hoveredNeuron, setHoveredNeuron] = useState<{
    layer: number;
    neuron: number;
  } | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const [showEducational, setShowEducational] = useState(false);
  const [showLearnHint, setShowLearnHint] = useState(true);

  // Animation state
  const [animationProgress, setAnimationProgress] = useState(1);
  const [prevActivations, setPrevActivations] = useState<number[][] | undefined>();
  const animationRef = useRef<number | null>(null);

  // Backpropagation animation state
  const [backpropActive, setBackpropActive] = useState(false);
  const [backpropLayer, setBackpropLayer] = useState(-1);
  const prevEpochRef = useRef(currentEpoch);

  // Previous weights for spring animation
  const prevWeightsRef = useRef<Map<string, number>>(new Map());
  const [weightDeltas, setWeightDeltas] = useState<Map<string, number>>(new Map());

  // Particle animation state
  const [particles, setParticles] = useState<Array<{
    id: string;
    fromLayer: number;
    toLayer: number;
    fromNeuron: number;
    toNeuron: number;
    progress: number;
    activation: number;
  }>>([]);

  // Trigger forward pass animation when activations change
  useEffect(() => {
    if (activations && JSON.stringify(activations) !== JSON.stringify(prevActivations)) {
      setAnimationProgress(0);
      const startTime = performance.now();
      const duration = 500;

      // Start particle animation if enabled
      if (ANIMATION_CONFIG.enableParticles && activations.length > 1) {
        const newParticles: typeof particles = [];
        for (let layer = 0; layer < layerSizes.length - 1; layer++) {
          for (let from = 0; from < layerSizes[layer]; from++) {
            const fromActivation = activations[layer]?.[from] ?? 0;
            // Only spawn particles for active neurons
            if (fromActivation > 0.2) {
              // Spawn particles to random subset of next layer (limit to avoid too many)
              const targetCount = Math.min(ANIMATION_CONFIG.particleCount, layerSizes[layer + 1]);
              const targets = Array.from({ length: layerSizes[layer + 1] }, (_, i) => i)
                .sort(() => Math.random() - 0.5)
                .slice(0, targetCount);

              for (const to of targets) {
                newParticles.push({
                  id: `p-${layer}-${from}-${to}-${Date.now()}-${Math.random()}`,
                  fromLayer: layer,
                  toLayer: layer + 1,
                  fromNeuron: from,
                  toNeuron: to,
                  progress: 0,
                  activation: fromActivation,
                });
              }
            }
          }
        }
        setParticles(newParticles);
      }

      const animate = (time: number) => {
        const elapsed = time - startTime;
        const progress = Math.min(elapsed / duration, 1);
        // Easing function for smooth animation
        const eased = 1 - Math.pow(1 - progress, 3);
        setAnimationProgress(eased);

        // Update particle progress
        if (ANIMATION_CONFIG.enableParticles) {
          setParticles(prev => prev.map(p => ({ ...p, progress: eased })));
        }

        if (progress < 1) {
          animationRef.current = requestAnimationFrame(animate);
        } else {
          setPrevActivations(activations);
          setParticles([]); // Clear particles when animation completes
        }
      };

      animationRef.current = requestAnimationFrame(animate);

      return () => {
        if (animationRef.current) {
          cancelAnimationFrame(animationRef.current);
        }
      };
    }
  }, [activations, prevActivations, layerSizes]);

  // Trigger backpropagation animation when epoch changes during training
  useEffect(() => {
    if (trainingInProgress && currentEpoch !== prevEpochRef.current && currentEpoch > 0) {
      if (ANIMATION_CONFIG.enableBackprop) {
        setBackpropActive(true);
        setBackpropLayer(layerSizes.length - 1);

        // Animate backward through layers
        let layer = layerSizes.length - 1;
        const backpropInterval = setInterval(() => {
          layer--;
          if (layer < 0) {
            clearInterval(backpropInterval);
            setBackpropActive(false);
            setBackpropLayer(-1);
          } else {
            setBackpropLayer(layer);
          }
        }, 80); // Fast cascade through layers

        return () => clearInterval(backpropInterval);
      }
    }
    prevEpochRef.current = currentEpoch;
  }, [currentEpoch, trainingInProgress, layerSizes.length]);

  // Track weight changes for spring animation
  useEffect(() => {
    if (ANIMATION_CONFIG.enableWeightSpring && weights.length > 0) {
      const newDeltas = new Map<string, number>();

      weights.forEach((layer, layerIdx) => {
        if (!layer?.weights) return;
        layer.weights.forEach((sourceWeights, srcIdx) => {
          sourceWeights.forEach((weight, tgtIdx) => {
            const key = `${layerIdx}-${srcIdx}-${tgtIdx}`;
            const prevWeight = prevWeightsRef.current.get(key);
            if (prevWeight !== undefined) {
              const delta = weight - prevWeight;
              if (Math.abs(delta) > 0.001) {
                newDeltas.set(key, delta);
              }
            }
            prevWeightsRef.current.set(key, weight);
          });
        });
      });

      if (newDeltas.size > 0) {
        setWeightDeltas(newDeltas);
        // Clear deltas after animation completes
        setTimeout(() => setWeightDeltas(new Map()), 300);
      }
    }
  }, [weights]);

  // Calculate neuron radius based on layer count
  const neuronRadius = useMemo(() => {
    const maxNeurons = Math.max(...layerSizes);
    const availableHeight = CONTENT_HEIGHT - 40;
    const maxRadius = Math.min(availableHeight / (maxNeurons * 2.5), 20);
    return Math.max(maxRadius, 8);
  }, [layerSizes]);

  // Calculate positions for neurons
  const neurons = useMemo(() => {
    const result: { x: number; y: number; layer: number; neuron: number }[] = [];
    const layerSpacing = CONTENT_WIDTH / Math.max(layerSizes.length - 1, 1);

    layerSizes.forEach((size, layerIndex) => {
      const x = MARGIN.left + layerIndex * layerSpacing;
      const availableHeight = CONTENT_HEIGHT - 20;
      const neuronSpacing = availableHeight / (size + 1);

      for (let neuronIndex = 0; neuronIndex < size; neuronIndex++) {
        const y = MARGIN.top + (neuronIndex + 1) * neuronSpacing;
        result.push({ x, y, layer: layerIndex, neuron: neuronIndex });
      }
    });

    return result;
  }, [layerSizes]);

  // Calculate connections with Bezier curves
  const connections = useMemo(() => {
    const result: {
      path: string;
      weight: number;
      normalizedWeight: number;
      fromLayer: number;
      toLayer: number;
      fromNeuron: number;
      toNeuron: number;
      isPositive: boolean;
    }[] = [];

    if (weights.length === 0) return result;

    const layerSpacing = CONTENT_WIDTH / Math.max(layerSizes.length - 1, 1);

    weights.forEach((layer, layerIndex) => {
      if (!layer || !layer.weights) return;
      if (!layer.input_size || !layer.output_size) return;

      const x1 = MARGIN.left + layerIndex * layerSpacing;
      const x2 = MARGIN.left + (layerIndex + 1) * layerSpacing;

      const sourceSpacing = (CONTENT_HEIGHT - 20) / (layer.input_size + 1);
      const targetSpacing = (CONTENT_HEIGHT - 20) / (layer.output_size + 1);

      // Skip if spacing would produce NaN
      if (!isFinite(sourceSpacing) || !isFinite(targetSpacing)) return;

      const maxWeight = Math.max(...layer.weights.flat().map(Math.abs), 0.001);

      layer.weights.forEach((sourceWeights, sourceIndex) => {
        if (!sourceWeights) return;
        const y1 = MARGIN.top + (sourceIndex + 1) * sourceSpacing;

        sourceWeights.forEach((weight, targetIndex) => {
          const y2 = MARGIN.top + (targetIndex + 1) * targetSpacing;

          // Skip invalid coordinates
          if (!isFinite(y1) || !isFinite(y2)) return;

          // Bezier control points
          const cpOffset = layerSpacing * 0.4;
          const path = `M ${x1} ${y1} C ${x1 + cpOffset} ${y1}, ${x2 - cpOffset} ${y2}, ${x2} ${y2}`;

          result.push({
            path,
            weight,
            normalizedWeight: weight / maxWeight,
            fromLayer: layerIndex,
            toLayer: layerIndex + 1,
            fromNeuron: sourceIndex,
            toNeuron: targetIndex,
            isPositive: weight >= 0,
          });
        });
      });
    });

    return result;
  }, [weights, layerSizes]);

  // Get activation value
  const getActivation = useCallback(
    (layer: number, neuron: number): number | null => {
      if (!activations?.[layer]) return null;
      return activations[layer][neuron] ?? null;
    },
    [activations]
  );

  // Animation helpers
  const getLayerOpacity = useCallback(
    (layerIndex: number): number => {
      if (animationProgress >= 1 || !activations) return 1;
      const progress = animationProgress * layerSizes.length;
      if (layerIndex <= progress) {
        return 0.5 + Math.min((progress - layerIndex) * 2, 1) * 0.5;
      }
      return 0.5;
    },
    [animationProgress, layerSizes.length, activations]
  );

  const getConnectionOpacity = useCallback(
    (fromLayer: number): number => {
      if (animationProgress >= 1 || !activations) return 1;
      const progress = animationProgress * layerSizes.length;
      if (fromLayer < progress) {
        return 0.3 + Math.min((progress - fromLayer) * 2, 1) * 0.7;
      }
      return 0.3;
    },
    [animationProgress, layerSizes.length, activations]
  );

  // Hover handlers
  const handleNeuronHover = useCallback(
    (e: React.MouseEvent, layer: number, neuron: number, activation: number | null) => {
      const svg = svgRef.current;
      if (!svg) return;

      const point = svg.createSVGPoint();
      point.x = e.clientX;
      point.y = e.clientY;
      const svgPoint = point.matrixTransform(svg.getScreenCTM()?.inverse());

      const layerName =
        layer === 0 ? 'Input' : layer === layerSizes.length - 1 ? 'Output' : `Hidden ${layer}`;

      setTooltip({
        x: svgPoint.x,
        y: svgPoint.y - 15,
        content: [
          `${layerName} #${neuron + 1}`,
          activation !== null ? `Act: ${activation.toFixed(3)}` : 'No data',
        ],
        type: 'neuron',
      });
      setHoveredNeuron({ layer, neuron });
    },
    [layerSizes.length]
  );

  const handleConnectionHover = useCallback(
    (e: React.MouseEvent, conn: (typeof connections)[0]) => {
      const svg = svgRef.current;
      if (!svg) return;

      const point = svg.createSVGPoint();
      point.x = e.clientX;
      point.y = e.clientY;
      const svgPoint = point.matrixTransform(svg.getScreenCTM()?.inverse());

      const sourceAct = getActivation(conn.fromLayer, conn.fromNeuron);
      const signal = sourceAct !== null ? sourceAct * conn.weight : null;

      setTooltip({
        x: svgPoint.x,
        y: svgPoint.y - 15,
        content: [
          `W: ${conn.weight.toFixed(3)}`,
          signal !== null ? `Signal: ${signal.toFixed(3)}` : '',
        ].filter(Boolean),
        type: 'connection',
      });
    },
    [getActivation]
  );

  const handleMouseLeave = useCallback(() => {
    setTooltip(null);
    setHoveredNeuron(null);
  }, []);

  // Check if connection should be highlighted
  const isConnectionHighlighted = useCallback(
    (conn: (typeof connections)[0]): boolean => {
      if (!hoveredNeuron) return false;
      return (
        (conn.fromLayer === hoveredNeuron.layer && conn.fromNeuron === hoveredNeuron.neuron) ||
        (conn.toLayer === hoveredNeuron.layer && conn.toNeuron === hoveredNeuron.neuron)
      );
    },
    [hoveredNeuron]
  );

  // Check if connection is active
  const isConnectionActive = useCallback(
    (conn: (typeof connections)[0]): boolean => {
      if (!activations) return false;
      const sourceAct = getActivation(conn.fromLayer, conn.fromNeuron);
      const targetAct = getActivation(conn.toLayer, conn.toNeuron);
      return sourceAct !== null && targetAct !== null && sourceAct > 0.3 && targetAct > 0.3;
    },
    [activations, getActivation]
  );

  const hasActivations = activations && activations.length > 0;
  const isAnimating = animationProgress < 1 && hasActivations;

  // Helper: calculate position along a cubic Bezier curve
  const getPointOnBezier = useCallback((
    x1: number, y1: number, // start
    cx1: number, cy1: number, // control point 1
    cx2: number, cy2: number, // control point 2
    x2: number, y2: number, // end
    t: number // progress 0-1
  ) => {
    const u = 1 - t;
    const tt = t * t;
    const uu = u * u;
    const uuu = uu * u;
    const ttt = tt * t;

    const x = uuu * x1 + 3 * uu * t * cx1 + 3 * u * tt * cx2 + ttt * x2;
    const y = uuu * y1 + 3 * uu * t * cy1 + 3 * u * tt * cy2 + ttt * y2;
    return { x, y };
  }, []);

  // Helper: get connection endpoints for particles
  const getConnectionEndpoints = useCallback((
    fromLayer: number, fromNeuron: number, toLayer: number, toNeuron: number
  ) => {
    const layerSpacing = CONTENT_WIDTH / Math.max(layerSizes.length - 1, 1);
    const x1 = MARGIN.left + fromLayer * layerSpacing;
    const x2 = MARGIN.left + toLayer * layerSpacing;

    const sourceSpacing = (CONTENT_HEIGHT - 20) / (layerSizes[fromLayer] + 1);
    const targetSpacing = (CONTENT_HEIGHT - 20) / (layerSizes[toLayer] + 1);

    const y1 = MARGIN.top + (fromNeuron + 1) * sourceSpacing;
    const y2 = MARGIN.top + (toNeuron + 1) * targetSpacing;

    const cpOffset = layerSpacing * 0.4;
    return { x1, y1, x2, y2, cx1: x1 + cpOffset, cy1: y1, cx2: x2 - cpOffset, cy2: y2 };
  }, [layerSizes]);

  // Calculate particle positions
  const particlePositions = useMemo(() => {
    return particles.map(p => {
      const endpoints = getConnectionEndpoints(p.fromLayer, p.fromNeuron, p.toLayer, p.toNeuron);
      const pos = getPointOnBezier(
        endpoints.x1, endpoints.y1,
        endpoints.cx1, endpoints.cy1,
        endpoints.cx2, endpoints.cy2,
        endpoints.x2, endpoints.y2,
        p.progress
      );
      return { ...p, ...pos };
    });
  }, [particles, getConnectionEndpoints, getPointOnBezier]);

  return (
    <div className="bg-gray-800/60 backdrop-blur-sm rounded-lg p-2 border border-gray-700/50 overflow-hidden">
      {/* Educational visualization modal */}
      <AnimatePresence>
        {showEducational && (
          <DenseNetworkEducationalViz
            layerSizes={layerSizes}
            inputLabels={inputLabels.length > 0 ? inputLabels : layerSizes[0] > 0 ? Array.from({ length: layerSizes[0] }, (_, i) => `Input ${i + 1}`) : []}
            outputLabels={outputLabels.length > 0 ? outputLabels : ['Output']}
            outputActivation={outputActivation}
            onClose={() => setShowEducational(false)}
          />
        )}
      </AnimatePresence>

      {/* Prominent Learn Banner */}
      <AnimatePresence>
        {showLearnHint && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="mb-3 bg-gradient-to-r from-cyan-900/80 via-purple-900/80 to-cyan-900/80 rounded-lg p-3 border border-cyan-500/30 relative overflow-hidden"
          >
            {/* Animated background shimmer */}
            <motion.div
              className="absolute inset-0 bg-gradient-to-r from-transparent via-white/5 to-transparent"
              animate={{ x: ['-100%', '100%'] }}
              transition={{ duration: 2, repeat: Infinity, ease: 'linear' }}
            />

            <div className="relative flex items-center justify-between gap-3">
              <div className="flex items-center gap-3">
                <motion.div
                  className="w-10 h-10 rounded-full bg-cyan-500/20 flex items-center justify-center text-xl"
                  animate={{ scale: [1, 1.1, 1] }}
                  transition={{ duration: 1.5, repeat: Infinity }}
                >
                  🧠
                </motion.div>
                <div>
                  <div className="text-white font-medium text-sm">New to Neural Networks?</div>
                  <div className="text-cyan-300/80 text-xs">Interactive step-by-step tutorial available</div>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <motion.button
                  onClick={() => {
                    setShowEducational(true);
                    setShowLearnHint(false);
                  }}
                  className="px-4 py-2 bg-cyan-500 hover:bg-cyan-400 text-white font-medium rounded-lg text-sm flex items-center gap-2 shadow-lg shadow-cyan-500/25"
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  <span>📚</span> Start Learning
                </motion.button>
                <button
                  onClick={() => setShowLearnHint(false)}
                  className="text-gray-400 hover:text-white p-1"
                  title="Dismiss"
                >
                  ✕
                </button>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <div className="flex justify-between items-center mb-2">
        <h2 className="text-lg font-semibold text-white truncate">Network Architecture</h2>
        <div className="flex items-center gap-2 flex-shrink-0">
          {!showLearnHint && (
            <motion.button
              onClick={() => setShowEducational(true)}
              className="px-3 py-1.5 text-xs font-medium bg-gradient-to-r from-cyan-600 to-purple-600 hover:from-cyan-500 hover:to-purple-500 text-white rounded-lg transition-all flex items-center gap-1.5 shadow-md"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <span>📚</span> Learn How It Works
            </motion.button>
          )}
          {backpropActive && (
            <motion.span
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              className="text-xs text-orange-400 px-2 py-0.5 bg-orange-900/30 rounded flex items-center gap-1"
            >
              <motion.span
                animate={{ rotate: 180 }}
                transition={{ duration: 0.3 }}
              >
                ←
              </motion.span>
              Backprop
            </motion.span>
          )}
          {isAnimating && !backpropActive && (
            <span className="text-xs text-cyan-400 animate-pulse px-2 py-0.5 bg-cyan-900/30 rounded">
              Forward Pass
            </span>
          )}
          {hasActivations && !isAnimating && !backpropActive && (
            <span className="text-xs text-green-400 px-2 py-0.5 bg-green-900/30 rounded">
              Live
            </span>
          )}
        </div>
      </div>

      <div className="w-full aspect-video min-h-[280px]">
        <svg
          ref={svgRef}
          viewBox={`0 0 ${VIEWBOX_WIDTH} ${VIEWBOX_HEIGHT}`}
          className="w-full h-full"
          preserveAspectRatio="xMidYMid meet"
        >
          <defs>
            {/* Glow filters */}
            <filter id="glowSoft" x="-50%" y="-50%" width="200%" height="200%">
              <feGaussianBlur stdDeviation="3" result="blur" />
              <feMerge>
                <feMergeNode in="blur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>

            <filter id="glowStrong" x="-100%" y="-100%" width="300%" height="300%">
              <feGaussianBlur stdDeviation="6" result="blur" />
              <feMerge>
                <feMergeNode in="blur" />
                <feMergeNode in="blur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>

            <filter id="connectionGlow" x="-20%" y="-20%" width="140%" height="140%">
              <feGaussianBlur stdDeviation="2" result="blur" />
              <feMerge>
                <feMergeNode in="blur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>

            {/* Gradients */}
            <radialGradient id="neuronHigh" cx="35%" cy="35%" r="65%">
              <stop offset="0%" stopColor="#4ade80" />
              <stop offset="100%" stopColor="#16a34a" />
            </radialGradient>

            <radialGradient id="neuronMid" cx="35%" cy="35%" r="65%">
              <stop offset="0%" stopColor="#22d3ee" />
              <stop offset="100%" stopColor="#0891b2" />
            </radialGradient>

            <radialGradient id="neuronLow" cx="35%" cy="35%" r="65%">
              <stop offset="0%" stopColor="#4b5563" />
              <stop offset="100%" stopColor="#374151" />
            </radialGradient>

            <radialGradient id="neuronInput" cx="35%" cy="35%" r="65%">
              <stop offset="0%" stopColor="#a78bfa" />
              <stop offset="100%" stopColor="#7c3aed" />
            </radialGradient>
          </defs>

          {/* Background */}
          <rect
            x="0"
            y="0"
            width={VIEWBOX_WIDTH}
            height={VIEWBOX_HEIGHT}
            fill="url(#bgGradient)"
            rx="8"
          />
          <defs>
            <radialGradient id="bgGradient" cx="50%" cy="50%" r="70%">
              <stop offset="0%" stopColor="rgba(15, 23, 42, 0.6)" />
              <stop offset="100%" stopColor="rgba(15, 23, 42, 0.9)" />
            </radialGradient>
          </defs>

          {/* Connections */}
          <g className="connections">
            {connections.map((conn, i) => {
              const highlighted = isConnectionHighlighted(conn);
              const active = isConnectionActive(conn);
              const opacity = getConnectionOpacity(conn.fromLayer);

              const sourceAct = getActivation(conn.fromLayer, conn.fromNeuron);
              const signalStrength = sourceAct !== null ? Math.abs(sourceAct * conn.normalizedWeight) : 0;

              // Check if this connection is being backprop-updated
              const isBackpropTarget = backpropActive && conn.toLayer === backpropLayer + 1;
              const weightKey = `${conn.fromLayer}-${conn.fromNeuron}-${conn.toNeuron}`;
              const weightDelta = weightDeltas.get(weightKey) ?? 0;
              const hasWeightChange = Math.abs(weightDelta) > 0.001;

              const baseWidth = 0.5 + Math.abs(conn.normalizedWeight) * 1.5;
              let strokeWidth = highlighted ? baseWidth + 2 : active ? baseWidth + signalStrength : baseWidth;

              // Spring effect: temporarily increase width on weight change
              if (ANIMATION_CONFIG.enableWeightSpring && hasWeightChange) {
                strokeWidth += Math.min(Math.abs(weightDelta) * 10, 3);
              }

              // Backprop effect: flash connections during backprop
              const backpropColor = isBackpropTarget ? '#f97316' : null; // Orange for backprop

              const color = backpropColor ?? (conn.isPositive
                ? highlighted || active ? '#22d3ee' : `rgba(34, 211, 238, ${0.2 + Math.abs(conn.normalizedWeight) * 0.4})`
                : highlighted || active ? '#ec4899' : `rgba(236, 72, 153, ${0.2 + Math.abs(conn.normalizedWeight) * 0.4})`);

              return (
                <g key={i}>
                  {/* Backprop glow */}
                  {isBackpropTarget && (
                    <motion.path
                      d={conn.path}
                      fill="none"
                      stroke="#f97316"
                      initial={{ strokeWidth: strokeWidth + 2, opacity: 0.6 }}
                      animate={{ strokeWidth: strokeWidth + 6, opacity: 0 }}
                      transition={{ duration: 0.3 }}
                      filter="url(#connectionGlow)"
                    />
                  )}
                  {(highlighted || (active && signalStrength > 0.2)) && (
                    <path
                      d={conn.path}
                      fill="none"
                      stroke={conn.isPositive ? '#22d3ee' : '#ec4899'}
                      strokeWidth={strokeWidth + 4}
                      opacity={highlighted ? 0.4 : signalStrength * 0.3}
                      filter="url(#connectionGlow)"
                    />
                  )}
                  {/* Weight spring animation */}
                  {ANIMATION_CONFIG.enableWeightSpring && hasWeightChange ? (
                    <motion.path
                      d={conn.path}
                      fill="none"
                      stroke={color}
                      initial={{ strokeWidth: strokeWidth + Math.abs(weightDelta) * 5 }}
                      animate={{ strokeWidth }}
                      transition={{
                        type: "spring",
                        stiffness: ANIMATION_CONFIG.springStiffness,
                        damping: ANIMATION_CONFIG.springDamping,
                      }}
                      opacity={highlighted ? 1 : opacity * 0.8}
                      className="cursor-pointer"
                      onMouseEnter={(e) => handleConnectionHover(e, conn)}
                      onMouseLeave={handleMouseLeave}
                    />
                  ) : (
                    <path
                      d={conn.path}
                      fill="none"
                      stroke={color}
                      strokeWidth={strokeWidth}
                      opacity={highlighted ? 1 : opacity * 0.8}
                      className="cursor-pointer transition-opacity duration-150"
                      onMouseEnter={(e) => handleConnectionHover(e, conn)}
                      onMouseLeave={handleMouseLeave}
                    />
                  )}
                </g>
              );
            })}
          </g>

          {/* Data Flow Particles */}
          {ANIMATION_CONFIG.enableParticles && particlePositions.length > 0 && (
            <g className="particles">
              {particlePositions.map(p => {
                // Color based on activation: blue→green→yellow
                const hue = 200 - p.activation * 140; // 200 (blue) to 60 (yellow)
                const color = `hsl(${hue}, 90%, 60%)`;

                return (
                  <motion.circle
                    key={p.id}
                    cx={p.x}
                    cy={p.y}
                    r={3 + p.activation * 2}
                    fill={color}
                    initial={{ opacity: 0, scale: 0 }}
                    animate={{ opacity: 0.9, scale: 1 }}
                    exit={{ opacity: 0, scale: 0 }}
                    style={{
                      filter: 'blur(0.5px)',
                    }}
                  />
                );
              })}
            </g>
          )}

          {/* Neurons */}
          <g className="neurons">
            {neurons.map((neuron, i) => {
              const activation = getActivation(neuron.layer, neuron.neuron);
              const layerOpacity = getLayerOpacity(neuron.layer);
              const isHovered = hoveredNeuron?.layer === neuron.layer && hoveredNeuron?.neuron === neuron.neuron;
              const isInput = neuron.layer === 0;

              // Check if this neuron is being backprop-updated
              const isBackpropTarget = backpropActive && neuron.layer === backpropLayer;

              // Dynamic sizing
              const baseR = neuronRadius;
              const r = activation !== null ? baseR * (0.85 + activation * 0.35) : baseR;

              // Color based on activation
              let fill = 'url(#neuronLow)';
              let strokeColor = '#6b7280';
              let glowFilter: string | undefined;

              if (isBackpropTarget) {
                // Orange glow during backprop
                strokeColor = '#f97316';
                glowFilter = 'url(#glowStrong)';
              } else if (isInput && activation !== null && activation > 0.5) {
                fill = 'url(#neuronInput)';
                strokeColor = '#a78bfa';
                glowFilter = 'url(#glowSoft)';
              } else if (activation !== null) {
                if (activation > 0.7) {
                  fill = 'url(#neuronHigh)';
                  strokeColor = '#22c55e';
                  glowFilter = 'url(#glowStrong)';
                } else if (activation > 0.3) {
                  fill = 'url(#neuronMid)';
                  strokeColor = '#22d3ee';
                  glowFilter = 'url(#glowSoft)';
                }
              }

              if (isHovered) {
                strokeColor = '#f59e0b';
                glowFilter = 'url(#glowStrong)';
              }

              // Font size based on radius
              const fontSize = Math.max(r * 0.6, 7);
              const showText = r >= 10;

              // Pulse animation for active neurons
              const shouldPulse = ANIMATION_CONFIG.enablePulse && activation !== null && activation > 0.3 && !isAnimating;
              const pulseScale = activation !== null ? 1 + activation * 0.15 : 1;

              return (
                <g key={i} style={{ opacity: layerOpacity }} className="transition-opacity duration-200">
                  {/* Backprop wave ring */}
                  {isBackpropTarget && (
                    <motion.circle
                      cx={neuron.x}
                      cy={neuron.y}
                      r={r + 4}
                      fill="none"
                      stroke="#f97316"
                      strokeWidth={2}
                      initial={{ r: r, opacity: 0.8 }}
                      animate={{ r: r + 15, opacity: 0 }}
                      transition={{ duration: 0.4 }}
                      filter="url(#glowStrong)"
                    />
                  )}

                  {/* Pulsing glow ring for active neurons */}
                  {shouldPulse ? (
                    <motion.circle
                      cx={neuron.x}
                      cy={neuron.y}
                      r={r + 6}
                      fill="none"
                      stroke={strokeColor}
                      strokeWidth={1.5}
                      initial={{ opacity: activation! * 0.2, scale: 1 }}
                      animate={{
                        opacity: [activation! * 0.2, activation! * 0.5, activation! * 0.2],
                        scale: [1, pulseScale, 1],
                      }}
                      transition={{
                        duration: ANIMATION_CONFIG.pulseDuration + (1 - activation!) * 0.3,
                        repeat: Infinity,
                        ease: "easeInOut",
                      }}
                      filter="url(#glowStrong)"
                      style={{ transformOrigin: `${neuron.x}px ${neuron.y}px` }}
                    />
                  ) : (activation !== null && activation > 0.3) || isHovered ? (
                    <circle
                      cx={neuron.x}
                      cy={neuron.y}
                      r={r + 6}
                      fill="none"
                      stroke={strokeColor}
                      strokeWidth={1.5}
                      opacity={isHovered ? 0.6 : activation! * 0.4}
                      filter="url(#glowStrong)"
                    />
                  ) : null}

                  {/* Main neuron with pulse animation */}
                  {shouldPulse ? (
                    <motion.circle
                      cx={neuron.x}
                      cy={neuron.y}
                      r={r}
                      fill={fill}
                      stroke={strokeColor}
                      strokeWidth={isHovered ? 2.5 : 1.5}
                      filter={glowFilter}
                      className="cursor-pointer"
                      animate={{
                        scale: [1, pulseScale * 0.98, 1],
                      }}
                      transition={{
                        duration: ANIMATION_CONFIG.pulseDuration + (1 - activation!) * 0.3,
                        repeat: Infinity,
                        ease: "easeInOut",
                      }}
                      style={{ transformOrigin: `${neuron.x}px ${neuron.y}px` }}
                      onMouseEnter={(e) => handleNeuronHover(e as unknown as React.MouseEvent, neuron.layer, neuron.neuron, activation)}
                      onMouseLeave={handleMouseLeave}
                    />
                  ) : (
                    <circle
                      cx={neuron.x}
                      cy={neuron.y}
                      r={r}
                      fill={fill}
                      stroke={strokeColor}
                      strokeWidth={isHovered ? 2.5 : 1.5}
                      filter={glowFilter}
                      className="cursor-pointer transition-all duration-150"
                      onMouseEnter={(e) => handleNeuronHover(e, neuron.layer, neuron.neuron, activation)}
                      onMouseLeave={handleMouseLeave}
                    />
                  )}

                  {/* Inner highlight */}
                  <circle
                    cx={neuron.x - r * 0.25}
                    cy={neuron.y - r * 0.25}
                    r={r * 0.25}
                    fill="rgba(255,255,255,0.2)"
                    pointerEvents="none"
                  />

                  {/* Activation text */}
                  {showText && activation !== null && (
                    <text
                      x={neuron.x}
                      y={neuron.y + fontSize * 0.35}
                      textAnchor="middle"
                      fontSize={fontSize}
                      fontWeight="600"
                      fill={activation > 0.5 ? '#fff' : '#d1d5db'}
                      pointerEvents="none"
                    >
                      {activation.toFixed(2)}
                    </text>
                  )}
                </g>
              );
            })}
          </g>

          {/* Layer labels */}
          <g className="labels">
            {layerSizes.map((size, layerIndex) => {
              const layerSpacing = CONTENT_WIDTH / Math.max(layerSizes.length - 1, 1);
              const x = MARGIN.left + layerIndex * layerSpacing;
              const label =
                layerIndex === 0
                  ? 'Input'
                  : layerIndex === layerSizes.length - 1
                  ? 'Output'
                  : `H${layerIndex}`;

              return (
                <g key={layerIndex}>
                  {/* Bottom label */}
                  <rect
                    x={x - 28}
                    y={VIEWBOX_HEIGHT - 32}
                    width={56}
                    height={20}
                    rx={4}
                    fill="rgba(17, 24, 39, 0.85)"
                    stroke="rgba(75, 85, 99, 0.4)"
                    strokeWidth={1}
                  />
                  <text
                    x={x}
                    y={VIEWBOX_HEIGHT - 18}
                    textAnchor="middle"
                    fontSize="11"
                    fill="#9ca3af"
                    fontWeight="500"
                  >
                    {label}
                  </text>

                  {/* Top neuron count */}
                  <text x={x} y={MARGIN.top - 12} textAnchor="middle" fontSize="9" fill="#6b7280">
                    {size}n
                  </text>
                </g>
              );
            })}
          </g>

          {/* Tooltip */}
          {tooltip && (
            <g transform={`translate(${tooltip.x}, ${tooltip.y})`} className="pointer-events-none">
              <rect
                x={-50}
                y={-35}
                width={100}
                height={tooltip.content.length * 14 + 10}
                rx={5}
                fill="rgba(17, 24, 39, 0.95)"
                stroke={tooltip.type === 'neuron' ? '#22c55e' : '#22d3ee'}
                strokeWidth={1}
              />
              {tooltip.content.map((line, i) => (
                <text
                  key={i}
                  x={0}
                  y={-20 + i * 14}
                  textAnchor="middle"
                  fontSize="10"
                  fill="#e5e7eb"
                >
                  {line}
                </text>
              ))}
            </g>
          )}
        </svg>
      </div>

      {/* Stats row */}
      <div className="mt-3 flex flex-wrap justify-center gap-2 text-xs">
        <div className="bg-gray-900/60 rounded px-2 py-1 border border-gray-700/50">
          <span className="text-gray-500">Arch:</span>{' '}
          <span className="text-cyan-400 font-mono">[{layerSizes.join('-')}]</span>
        </div>
        <div className="bg-gray-900/60 rounded px-2 py-1 border border-gray-700/50">
          <span className="text-gray-500">W:</span>{' '}
          <span className="text-cyan-400 font-mono">
            {weights.reduce((s, l) => s + (l.weights?.flat()?.length || 0), 0)}
          </span>
        </div>
        <div className="bg-gray-900/60 rounded px-2 py-1 border border-gray-700/50">
          <span className="text-gray-500">B:</span>{' '}
          <span className="text-cyan-400 font-mono">
            {weights.reduce((s, l) => s + (l.biases?.flat()?.length || 0), 0)}
          </span>
        </div>
      </div>

      {/* Legend */}
      <div className="mt-2 border-t border-gray-700/50 pt-2">
        <div className="flex flex-wrap justify-center gap-x-4 gap-y-1 text-xs">
          <div className="text-gray-500 font-medium">Weights:</div>
          <div className="flex items-center gap-1" title="Positive weight: strengthens signal">
            <div className="w-5 h-0.5 bg-gradient-to-r from-cyan-500 to-cyan-400 rounded" />
            <span className="text-cyan-400">+ve (excite)</span>
          </div>
          <div className="flex items-center gap-1" title="Negative weight: inhibits signal">
            <div className="w-5 h-0.5 bg-gradient-to-r from-pink-500 to-pink-400 rounded" />
            <span className="text-pink-400">-ve (inhibit)</span>
          </div>
          <div className="text-gray-600">|</div>
          <div className="text-gray-500 font-medium">Activation:</div>
          <div className="flex items-center gap-1" title="High activation (>70%)">
            <div className="w-2 h-2 rounded-full bg-green-500" />
            <span className="text-green-400">High</span>
          </div>
          <div className="flex items-center gap-1" title="Medium activation (30-70%)">
            <div className="w-2 h-2 rounded-full bg-cyan-500" />
            <span className="text-cyan-400">Med</span>
          </div>
          <div className="flex items-center gap-1" title="Low activation (<30%)">
            <div className="w-2 h-2 rounded-full bg-gray-500" />
            <span className="text-gray-400">Low</span>
          </div>
        </div>
        <div className="text-center text-gray-600 text-xs mt-1">
          Hover over nodes and connections for details
        </div>
      </div>
    </div>
  );
});
