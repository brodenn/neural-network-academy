import { memo, useMemo, useState, useCallback, useEffect, useRef } from 'react';
import type { LayerWeights } from '../types';

interface NetworkVisualizationProps {
  layerSizes: number[];
  weights: LayerWeights[];
  activations?: number[][];
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
}: NetworkVisualizationProps) {
  // Use viewBox for responsive scaling
  const viewBoxWidth = 800;
  const viewBoxHeight = 450;
  const margin = { top: 35, right: 50, bottom: 45, left: 50 };
  const width = viewBoxWidth - margin.left - margin.right;
  const height = viewBoxHeight - margin.top - margin.bottom;

  const [tooltip, setTooltip] = useState<TooltipData | null>(null);
  const [hoveredNeuron, setHoveredNeuron] = useState<{
    layer: number;
    neuron: number;
  } | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);

  // Animation state
  const [animationProgress, setAnimationProgress] = useState(1);
  const [prevActivations, setPrevActivations] = useState<number[][] | undefined>();
  const animationRef = useRef<number | null>(null);

  // Trigger animation when activations change
  useEffect(() => {
    if (activations && JSON.stringify(activations) !== JSON.stringify(prevActivations)) {
      setAnimationProgress(0);
      const startTime = performance.now();
      const duration = 500;

      const animate = (time: number) => {
        const elapsed = time - startTime;
        const progress = Math.min(elapsed / duration, 1);
        // Easing function for smooth animation
        const eased = 1 - Math.pow(1 - progress, 3);
        setAnimationProgress(eased);

        if (progress < 1) {
          animationRef.current = requestAnimationFrame(animate);
        } else {
          setPrevActivations(activations);
        }
      };

      animationRef.current = requestAnimationFrame(animate);

      return () => {
        if (animationRef.current) {
          cancelAnimationFrame(animationRef.current);
        }
      };
    }
  }, [activations, prevActivations]);

  // Calculate neuron radius based on layer count
  const neuronRadius = useMemo(() => {
    const maxNeurons = Math.max(...layerSizes);
    const availableHeight = height - 40;
    const maxRadius = Math.min(availableHeight / (maxNeurons * 2.5), 20);
    return Math.max(maxRadius, 8);
  }, [layerSizes, height]);

  // Calculate positions for neurons
  const neurons = useMemo(() => {
    const result: { x: number; y: number; layer: number; neuron: number }[] = [];
    const layerSpacing = width / Math.max(layerSizes.length - 1, 1);

    layerSizes.forEach((size, layerIndex) => {
      const x = margin.left + layerIndex * layerSpacing;
      const availableHeight = height - 20;
      const neuronSpacing = availableHeight / (size + 1);

      for (let neuronIndex = 0; neuronIndex < size; neuronIndex++) {
        const y = margin.top + (neuronIndex + 1) * neuronSpacing;
        result.push({ x, y, layer: layerIndex, neuron: neuronIndex });
      }
    });

    return result;
  }, [layerSizes, width, height, margin]);

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

    const layerSpacing = width / Math.max(layerSizes.length - 1, 1);

    weights.forEach((layer, layerIndex) => {
      if (!layer || !layer.weights) return;
      if (!layer.input_size || !layer.output_size) return;

      const x1 = margin.left + layerIndex * layerSpacing;
      const x2 = margin.left + (layerIndex + 1) * layerSpacing;

      const sourceSpacing = (height - 20) / (layer.input_size + 1);
      const targetSpacing = (height - 20) / (layer.output_size + 1);

      // Skip if spacing would produce NaN
      if (!isFinite(sourceSpacing) || !isFinite(targetSpacing)) return;

      const maxWeight = Math.max(...layer.weights.flat().map(Math.abs), 0.001);

      layer.weights.forEach((sourceWeights, sourceIndex) => {
        if (!sourceWeights) return;
        const y1 = margin.top + (sourceIndex + 1) * sourceSpacing;

        sourceWeights.forEach((weight, targetIndex) => {
          const y2 = margin.top + (targetIndex + 1) * targetSpacing;

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
  }, [weights, layerSizes, width, height, margin]);

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

  return (
    <div className="bg-gray-800/60 backdrop-blur-sm rounded-xl p-4 border border-gray-700/50 overflow-hidden">
      <div className="flex justify-between items-center mb-2">
        <h2 className="text-lg font-semibold text-white truncate">Network Architecture</h2>
        <div className="flex items-center gap-2 flex-shrink-0">
          {isAnimating && (
            <span className="text-xs text-cyan-400 animate-pulse px-2 py-0.5 bg-cyan-900/30 rounded">
              Forward Pass
            </span>
          )}
          {hasActivations && !isAnimating && (
            <span className="text-xs text-green-400 px-2 py-0.5 bg-green-900/30 rounded">
              Live
            </span>
          )}
        </div>
      </div>

      <div className="w-full aspect-video min-h-[280px]">
        <svg
          ref={svgRef}
          viewBox={`0 0 ${viewBoxWidth} ${viewBoxHeight}`}
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
            width={viewBoxWidth}
            height={viewBoxHeight}
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

              const baseWidth = 0.5 + Math.abs(conn.normalizedWeight) * 1.5;
              const strokeWidth = highlighted ? baseWidth + 2 : active ? baseWidth + signalStrength : baseWidth;

              const color = conn.isPositive
                ? highlighted || active ? '#22d3ee' : `rgba(34, 211, 238, ${0.2 + Math.abs(conn.normalizedWeight) * 0.4})`
                : highlighted || active ? '#ec4899' : `rgba(236, 72, 153, ${0.2 + Math.abs(conn.normalizedWeight) * 0.4})`;

              return (
                <g key={i}>
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
                </g>
              );
            })}
          </g>

          {/* Neurons */}
          <g className="neurons">
            {neurons.map((neuron, i) => {
              const activation = getActivation(neuron.layer, neuron.neuron);
              const layerOpacity = getLayerOpacity(neuron.layer);
              const isHovered = hoveredNeuron?.layer === neuron.layer && hoveredNeuron?.neuron === neuron.neuron;
              const isInput = neuron.layer === 0;

              // Dynamic sizing
              const baseR = neuronRadius;
              const r = activation !== null ? baseR * (0.85 + activation * 0.35) : baseR;

              // Color based on activation
              let fill = 'url(#neuronLow)';
              let strokeColor = '#6b7280';
              let glowFilter: string | undefined;

              if (isInput && activation !== null && activation > 0.5) {
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

              return (
                <g key={i} style={{ opacity: layerOpacity }} className="transition-opacity duration-200">
                  {/* Glow ring for active neurons */}
                  {(activation !== null && activation > 0.3) || isHovered ? (
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

                  {/* Main neuron */}
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
              const layerSpacing = width / Math.max(layerSizes.length - 1, 1);
              const x = margin.left + layerIndex * layerSpacing;
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
                    y={viewBoxHeight - 32}
                    width={56}
                    height={20}
                    rx={4}
                    fill="rgba(17, 24, 39, 0.85)"
                    stroke="rgba(75, 85, 99, 0.4)"
                    strokeWidth={1}
                  />
                  <text
                    x={x}
                    y={viewBoxHeight - 18}
                    textAnchor="middle"
                    fontSize="11"
                    fill="#9ca3af"
                    fontWeight="500"
                  >
                    {label}
                  </text>

                  {/* Top neuron count */}
                  <text x={x} y={margin.top - 12} textAnchor="middle" fontSize="9" fill="#6b7280">
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
      <div className="mt-2 flex flex-wrap justify-center gap-3 text-xs text-gray-500">
        <div className="flex items-center gap-1">
          <div className="w-5 h-0.5 bg-gradient-to-r from-cyan-500 to-cyan-400 rounded" />
          <span>+ve</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-5 h-0.5 bg-gradient-to-r from-pink-500 to-pink-400 rounded" />
          <span>-ve</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-2 h-2 rounded-full bg-green-500" />
          <span>&gt;0.7</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-2 h-2 rounded-full bg-cyan-500" />
          <span>&gt;0.3</span>
        </div>
      </div>
    </div>
  );
});
