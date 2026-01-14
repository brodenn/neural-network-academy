import { memo } from 'react';
import type { WeightInit, HiddenActivation } from '../types';

interface NetworkSettingsProps {
  weightInit: WeightInit;
  hiddenActivation: HiddenActivation;
  useBiases: boolean;
  onWeightInitChange: (value: WeightInit) => void;
  onHiddenActivationChange: (value: HiddenActivation) => void;
  onUseBiasesChange: (value: boolean) => void;
  disabled?: boolean;
}

// Simple explanations for beginners
const INIT_HINTS: Record<WeightInit, string> = {
  xavier: 'Balanced start - works well for most cases',
  he: 'Good for ReLU - slightly larger weights',
  random: 'Random [-1,1] - can be unstable',
  zeros: 'All zeros - network cannot learn! (try it)',
};

const ACTIVATION_HINTS: Record<HiddenActivation, string> = {
  relu: 'Fast & popular - outputs 0 or positive',
  sigmoid: 'Smooth 0-1 - can be slow to learn',
  tanh: 'Smooth -1 to 1 - centered around zero',
};

export const NetworkSettings = memo(function NetworkSettings({
  weightInit,
  hiddenActivation,
  useBiases,
  onWeightInitChange,
  onHiddenActivationChange,
  onUseBiasesChange,
  disabled = false,
}: NetworkSettingsProps) {
  return (
    <div className="space-y-4">
      {/* Weight Initialization */}
      <div>
        <label className="block text-gray-400 text-sm mb-1">
          Weight Initialization
        </label>
        <select
          value={weightInit}
          onChange={(e) => onWeightInitChange(e.target.value as WeightInit)}
          disabled={disabled}
          className="w-full bg-gray-700 rounded px-3 py-2 text-white disabled:opacity-50"
        >
          <option value="xavier">Xavier (recommended)</option>
          <option value="he">He (for ReLU)</option>
          <option value="random">Random</option>
          <option value="zeros">Zeros (won't learn!)</option>
        </select>
        <p className="text-xs text-gray-500 mt-1">{INIT_HINTS[weightInit]}</p>
      </div>

      {/* Hidden Activation */}
      <div>
        <label className="block text-gray-400 text-sm mb-1">
          Hidden Layer Activation
        </label>
        <select
          value={hiddenActivation}
          onChange={(e) => onHiddenActivationChange(e.target.value as HiddenActivation)}
          disabled={disabled}
          className="w-full bg-gray-700 rounded px-3 py-2 text-white disabled:opacity-50"
        >
          <option value="relu">ReLU (recommended)</option>
          <option value="sigmoid">Sigmoid</option>
          <option value="tanh">Tanh</option>
        </select>
        <p className="text-xs text-gray-500 mt-1">{ACTIVATION_HINTS[hiddenActivation]}</p>
      </div>

      {/* Biases Toggle */}
      <div className="flex items-center justify-between">
        <div>
          <label className="text-gray-400 text-sm">Use Biases</label>
          <p className="text-xs text-gray-500">Helps shift activation threshold</p>
        </div>
        <button
          onClick={() => onUseBiasesChange(!useBiases)}
          disabled={disabled}
          className={`relative w-12 h-6 rounded-full transition-colors disabled:opacity-50 ${
            useBiases ? 'bg-green-600' : 'bg-gray-600'
          }`}
        >
          <span
            className={`absolute top-1 w-4 h-4 bg-white rounded-full transition-transform ${
              useBiases ? 'left-7' : 'left-1'
            }`}
          />
        </button>
      </div>

      {/* Learning Rate Presets */}
      <div>
        <label className="block text-gray-400 text-sm mb-1">
          Quick Learning Rate
        </label>
        <div className="flex gap-2">
          {[
            { value: 0.01, label: '0.01', hint: 'Slow' },
            { value: 0.1, label: '0.1', hint: 'Safe' },
            { value: 0.5, label: '0.5', hint: 'Fast' },
            { value: 2.0, label: '2.0', hint: 'Chaos!' },
          ].map(({ value, label, hint }) => (
            <button
              key={value}
              onClick={() => {
                // This will be handled by parent to update the LR input
                const event = new CustomEvent('setLearningRate', { detail: value });
                window.dispatchEvent(event);
              }}
              disabled={disabled}
              className="flex-1 bg-gray-700 hover:bg-gray-600 disabled:opacity-50 px-2 py-1 rounded text-sm"
              title={hint}
            >
              {label}
            </button>
          ))}
        </div>
        <p className="text-xs text-gray-500 mt-1">Click to set learning rate</p>
      </div>
    </div>
  );
});
