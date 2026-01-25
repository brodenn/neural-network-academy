import { useState, useCallback } from 'react';
import {
  DndContext,
  DragOverlay,
  closestCenter,
  KeyboardSensor,
  PointerSensor,
  useSensor,
  useSensors,
} from '@dnd-kit/core';
import type { DragStartEvent, DragEndEvent } from '@dnd-kit/core';
import {
  arrayMove,
  SortableContext,
  sortableKeyboardCoordinates,
  verticalListSortingStrategy,
  useSortable,
} from '@dnd-kit/sortable';
import { CSS } from '@dnd-kit/utilities';
import { motion, AnimatePresence } from 'framer-motion';

// =============================================================================
// TYPES
// =============================================================================

interface LayerConfig {
  id: string;
  type: 'input' | 'hidden' | 'output';
  neurons: number;
  activation?: 'relu' | 'sigmoid' | 'tanh';
}

interface NetworkBuilderProps {
  problemId: string;
  inputSize: number;
  outputSize: number;
  requirements?: {
    minLayers?: number;
    maxLayers?: number;
    minHiddenNeurons?: number;
    maxHiddenNeurons?: number;
    mustHaveHidden?: boolean;
  };
  onSubmit: (architecture: number[]) => void;
  onArchitectureChange?: (architecture: number[]) => void;
}

// =============================================================================
// LAYER PALETTE ITEM (Draggable source)
// =============================================================================

interface PaletteItemProps {
  type: 'hidden';
  defaultNeurons: number;
}

const PaletteItem = ({ type, defaultNeurons }: PaletteItemProps) => {
  const { attributes, listeners, setNodeRef, transform, isDragging } = useSortable({
    id: `palette-${type}`,
    data: { type: 'palette', layerType: type, neurons: defaultNeurons },
  });

  const style = {
    transform: CSS.Transform.toString(transform),
    opacity: isDragging ? 0.5 : 1,
  };

  return (
    <div
      ref={setNodeRef}
      style={style}
      {...attributes}
      {...listeners}
      className="flex items-center gap-2 p-3 bg-blue-600/30 border-2 border-blue-500/50 border-dashed rounded-lg cursor-grab active:cursor-grabbing hover:bg-blue-600/40 transition-colors"
    >
      <div className="w-8 h-8 rounded-full bg-blue-500 flex items-center justify-center text-white font-bold">
        +
      </div>
      <div>
        <p className="text-white font-medium">Hidden Layer</p>
        <p className="text-blue-300 text-xs">Drag to add</p>
      </div>
    </div>
  );
};

// =============================================================================
// SORTABLE LAYER (In the network)
// =============================================================================

interface SortableLayerProps {
  layer: LayerConfig;
  onNeuronsChange: (id: string, neurons: number) => void;
  onRemove: (id: string) => void;
  isFixed?: boolean;
}

const SortableLayer = ({ layer, onNeuronsChange, onRemove, isFixed }: SortableLayerProps) => {
  const {
    attributes,
    listeners,
    setNodeRef,
    transform,
    transition,
    isDragging,
  } = useSortable({ id: layer.id, disabled: isFixed });

  const style = {
    transform: CSS.Transform.toString(transform),
    transition,
    opacity: isDragging ? 0.5 : 1,
  };

  const colors = {
    input: { bg: 'bg-green-600/30', border: 'border-green-500', dot: 'bg-green-500' },
    hidden: { bg: 'bg-blue-600/30', border: 'border-blue-500', dot: 'bg-blue-500' },
    output: { bg: 'bg-red-600/30', border: 'border-red-500', dot: 'bg-red-500' },
  };

  const c = colors[layer.type];

  return (
    <motion.div
      ref={setNodeRef}
      style={style}
      initial={{ opacity: 0, scale: 0.8 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.8 }}
      className={`${c.bg} border-2 ${c.border} rounded-lg p-3 ${!isFixed ? 'cursor-grab active:cursor-grabbing' : ''}`}
      {...(!isFixed ? { ...attributes, ...listeners } : {})}
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className={`w-6 h-6 rounded-full ${c.dot} flex items-center justify-center`}>
            {layer.type === 'input' && '→'}
            {layer.type === 'hidden' && '●'}
            {layer.type === 'output' && '←'}
          </div>
          <span className="text-white font-medium capitalize">{layer.type}</span>
        </div>

        <div className="flex items-center gap-2">
          {layer.type === 'hidden' ? (
            <>
              <button
                onClick={() => onNeuronsChange(layer.id, Math.max(1, layer.neurons - 1))}
                className="w-6 h-6 rounded bg-gray-700 hover:bg-gray-600 text-white text-sm"
              >
                -
              </button>
              <span className="w-8 text-center text-white font-mono">{layer.neurons}</span>
              <button
                onClick={() => onNeuronsChange(layer.id, Math.min(32, layer.neurons + 1))}
                className="w-6 h-6 rounded bg-gray-700 hover:bg-gray-600 text-white text-sm"
              >
                +
              </button>
              <button
                onClick={() => onRemove(layer.id)}
                className="ml-2 w-6 h-6 rounded bg-red-600/50 hover:bg-red-600 text-white text-sm"
              >
                ×
              </button>
            </>
          ) : (
            <span className="text-gray-400 font-mono">{layer.neurons} neurons</span>
          )}
        </div>
      </div>
    </motion.div>
  );
};

// =============================================================================
// LAYER PREVIEW (For drag overlay)
// =============================================================================

const LayerPreview = ({ neurons }: { neurons: number }) => (
  <div className="bg-blue-600/50 border-2 border-blue-500 rounded-lg p-3 shadow-xl">
    <div className="flex items-center gap-2">
      <div className="w-6 h-6 rounded-full bg-blue-500 flex items-center justify-center">●</div>
      <span className="text-white font-medium">Hidden Layer</span>
      <span className="text-blue-300 font-mono ml-2">{neurons} neurons</span>
    </div>
  </div>
);

// =============================================================================
// NETWORK PREVIEW VISUALIZATION
// =============================================================================

const NetworkPreview = ({ layers }: { layers: LayerConfig[] }) => {
  const layerWidth = 60;
  const totalWidth = layers.length * layerWidth + (layers.length - 1) * 40;

  return (
    <div className="flex items-center justify-center py-4 overflow-x-auto">
      <svg width={totalWidth} height={120} className="mx-auto">
        {layers.map((layer, layerIdx) => {
          const x = layerIdx * (layerWidth + 40) + layerWidth / 2;
          const neuronSpacing = Math.min(20, 100 / layer.neurons);
          const startY = 60 - (layer.neurons - 1) * neuronSpacing / 2;

          const colors = {
            input: '#22c55e',
            hidden: '#3b82f6',
            output: '#ef4444',
          };

          return (
            <g key={layer.id}>
              {/* Draw connections to next layer */}
              {layerIdx < layers.length - 1 && (
                <>
                  {Array.from({ length: layer.neurons }).map((_, i) => {
                    const nextLayer = layers[layerIdx + 1];
                    const nextNeuronSpacing = Math.min(20, 100 / nextLayer.neurons);
                    const nextStartY = 60 - (nextLayer.neurons - 1) * nextNeuronSpacing / 2;

                    return Array.from({ length: nextLayer.neurons }).map((_, j) => (
                      <line
                        key={`${i}-${j}`}
                        x1={x + 8}
                        y1={startY + i * neuronSpacing}
                        x2={x + layerWidth + 32}
                        y2={nextStartY + j * nextNeuronSpacing}
                        stroke="#374151"
                        strokeWidth={0.5}
                        opacity={0.5}
                      />
                    ));
                  })}
                </>
              )}

              {/* Draw neurons */}
              {Array.from({ length: layer.neurons }).map((_, i) => (
                <circle
                  key={i}
                  cx={x}
                  cy={startY + i * neuronSpacing}
                  r={6}
                  fill={colors[layer.type]}
                  stroke="white"
                  strokeWidth={1}
                />
              ))}

              {/* Layer label */}
              <text
                x={x}
                y={115}
                textAnchor="middle"
                fill="#9ca3af"
                fontSize={10}
              >
                {layer.neurons}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
};

// =============================================================================
// MAIN COMPONENT
// =============================================================================

export const NetworkBuilder = ({
  problemId: _problemId,
  inputSize,
  outputSize,
  requirements = {},
  onSubmit,
  onArchitectureChange,
}: NetworkBuilderProps) => {
  const [layers, setLayers] = useState<LayerConfig[]>([
    { id: 'input', type: 'input', neurons: inputSize },
    { id: 'output', type: 'output', neurons: outputSize },
  ]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const sensors = useSensors(
    useSensor(PointerSensor, { activationConstraint: { distance: 5 } }),
    useSensor(KeyboardSensor, { coordinateGetter: sortableKeyboardCoordinates })
  );

  // Get current architecture as number array
  const getArchitecture = useCallback(() => {
    return layers.map(l => l.neurons);
  }, [layers]);

  // Validate against requirements
  const validate = useCallback(() => {
    const hiddenLayers = layers.filter(l => l.type === 'hidden');
    const totalLayers = layers.length;
    const totalHiddenNeurons = hiddenLayers.reduce((sum, l) => sum + l.neurons, 0);

    if (requirements.mustHaveHidden && hiddenLayers.length === 0) {
      return 'This problem requires at least one hidden layer!';
    }
    if (requirements.minLayers && totalLayers < requirements.minLayers) {
      return `Need at least ${requirements.minLayers} layers`;
    }
    if (requirements.maxLayers && totalLayers > requirements.maxLayers) {
      return `Maximum ${requirements.maxLayers} layers allowed`;
    }
    if (requirements.minHiddenNeurons && totalHiddenNeurons < requirements.minHiddenNeurons) {
      return `Need at least ${requirements.minHiddenNeurons} hidden neurons`;
    }
    if (requirements.maxHiddenNeurons && totalHiddenNeurons > requirements.maxHiddenNeurons) {
      return `Maximum ${requirements.maxHiddenNeurons} hidden neurons allowed`;
    }
    return null;
  }, [layers, requirements]);

  // Handle drag start
  const handleDragStart = (event: DragStartEvent) => {
    setActiveId(event.active.id as string);
  };

  // Handle drag end
  const handleDragEnd = (event: DragEndEvent) => {
    const { active, over } = event;
    setActiveId(null);

    if (!over) return;

    // Check if dragging from palette
    const activeData = active.data.current;
    if (activeData?.type === 'palette') {
      // Add new layer before output
      const newLayer: LayerConfig = {
        id: `hidden-${Date.now()}`,
        type: 'hidden',
        neurons: activeData.neurons,
        activation: 'relu',
      };

      setLayers(prev => {
        const outputIndex = prev.findIndex(l => l.type === 'output');
        const newLayers = [...prev];
        newLayers.splice(outputIndex, 0, newLayer);
        return newLayers;
      });
    } else if (active.id !== over.id) {
      // Reorder existing layers (but keep input first and output last)
      setLayers(prev => {
        const oldIndex = prev.findIndex(l => l.id === active.id);
        const newIndex = prev.findIndex(l => l.id === over.id);

        // Don't allow moving input or output
        if (prev[oldIndex].type === 'input' || prev[oldIndex].type === 'output') {
          return prev;
        }
        // Don't allow moving before input or after output
        if (newIndex === 0 || newIndex === prev.length - 1) {
          return prev;
        }

        return arrayMove(prev, oldIndex, newIndex);
      });
    }

    // Notify parent of change
    setTimeout(() => {
      onArchitectureChange?.(getArchitecture());
    }, 0);
  };

  // Handle neuron count change
  const handleNeuronsChange = (id: string, neurons: number) => {
    setLayers(prev => prev.map(l => l.id === id ? { ...l, neurons } : l));
    setTimeout(() => {
      onArchitectureChange?.(getArchitecture());
    }, 0);
  };

  // Handle layer removal
  const handleRemove = (id: string) => {
    setLayers(prev => prev.filter(l => l.id !== id));
    setTimeout(() => {
      onArchitectureChange?.(getArchitecture());
    }, 0);
  };

  // Handle submit
  const handleSubmit = () => {
    const validationError = validate();
    if (validationError) {
      setError(validationError);
      return;
    }
    setError(null);
    onSubmit(getArchitecture());
  };

  const architecture = getArchitecture();
  const hiddenCount = layers.filter(l => l.type === 'hidden').length;

  return (
    <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-white">Build Your Network</h3>
        <span className="text-sm text-gray-400 font-mono">
          [{architecture.join(', ')}]
        </span>
      </div>

      <DndContext
        sensors={sensors}
        collisionDetection={closestCenter}
        onDragStart={handleDragStart}
        onDragEnd={handleDragEnd}
      >
        <div className="grid grid-cols-2 gap-4">
          {/* Palette */}
          <div className="space-y-2">
            <p className="text-sm text-gray-400 mb-2">Drag to add layers:</p>
            <SortableContext items={['palette-hidden']} strategy={verticalListSortingStrategy}>
              <PaletteItem type="hidden" defaultNeurons={4} />
            </SortableContext>
          </div>

          {/* Network Layers */}
          <div className="space-y-2">
            <p className="text-sm text-gray-400 mb-2">Your network:</p>
            <SortableContext items={layers.map(l => l.id)} strategy={verticalListSortingStrategy}>
              <AnimatePresence>
                {layers.map(layer => (
                  <SortableLayer
                    key={layer.id}
                    layer={layer}
                    onNeuronsChange={handleNeuronsChange}
                    onRemove={handleRemove}
                    isFixed={layer.type === 'input' || layer.type === 'output'}
                  />
                ))}
              </AnimatePresence>
            </SortableContext>
          </div>
        </div>

        <DragOverlay>
          {activeId?.startsWith('palette-') && <LayerPreview neurons={4} />}
        </DragOverlay>
      </DndContext>

      {/* Network Preview */}
      <div className="mt-4 bg-gray-900/50 rounded-lg p-2">
        <p className="text-xs text-gray-500 text-center mb-2">Preview</p>
        <NetworkPreview layers={layers} />
      </div>

      {/* Stats */}
      <div className="mt-4 flex gap-4 text-sm">
        <div className="px-3 py-1 bg-gray-700 rounded">
          <span className="text-gray-400">Layers: </span>
          <span className="text-white">{layers.length}</span>
        </div>
        <div className="px-3 py-1 bg-gray-700 rounded">
          <span className="text-gray-400">Hidden: </span>
          <span className="text-white">{hiddenCount}</span>
        </div>
        <div className="px-3 py-1 bg-gray-700 rounded">
          <span className="text-gray-400">Total neurons: </span>
          <span className="text-white">{architecture.reduce((a, b) => a + b, 0)}</span>
        </div>
      </div>

      {/* Error */}
      <AnimatePresence>
        {error && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="mt-3 p-2 bg-red-900/50 border border-red-600 rounded text-red-300 text-sm"
          >
            {error}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Submit */}
      <motion.button
        whileHover={{ scale: 1.02 }}
        whileTap={{ scale: 0.98 }}
        onClick={handleSubmit}
        className="mt-4 w-full py-2 bg-green-600 hover:bg-green-500 rounded-lg text-white font-medium transition-colors"
      >
        Use This Architecture
      </motion.button>
    </div>
  );
};

export default NetworkBuilder;
