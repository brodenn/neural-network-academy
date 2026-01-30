import { useState, useEffect } from 'react';
import * as Dialog from '@radix-ui/react-dialog';
import { motion } from 'framer-motion';

const ONBOARDING_KEY = 'learning_paths_onboarding_seen';

interface OnboardingModalProps {
  onSelectPath?: (pathId: string) => void;
}

export const OnboardingModal = ({ onSelectPath }: OnboardingModalProps) => {
  const [isOpen, setIsOpen] = useState(false);
  const [step, setStep] = useState(0);

  // Check if user has seen onboarding
  useEffect(() => {
    const seen = localStorage.getItem(ONBOARDING_KEY);
    if (!seen) {
      // Small delay for better UX
      const timer = setTimeout(() => setIsOpen(true), 500);
      return () => clearTimeout(timer);
    }
  }, []);

  const handleClose = () => {
    localStorage.setItem(ONBOARDING_KEY, 'true');
    setIsOpen(false);
  };

  const handleStartPath = (pathId: string) => {
    handleClose();
    onSelectPath?.(pathId);
  };

  const steps = [
    {
      title: 'Welcome to Learning Paths!',
      content: (
        <div className="space-y-4">
          <p className="text-gray-300">
            Learn neural networks step-by-step with our guided learning paths.
            Each path is designed to build your understanding progressively.
          </p>
          <div className="grid grid-cols-3 gap-4 mt-6">
            <div className="text-center">
              <div className="w-12 h-12 mx-auto bg-blue-600 rounded-full flex items-center justify-center mb-2">
                <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                </svg>
              </div>
              <p className="text-sm text-gray-400">Structured Steps</p>
            </div>
            <div className="text-center">
              <div className="w-12 h-12 mx-auto bg-yellow-600 rounded-full flex items-center justify-center mb-2">
                <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                </svg>
              </div>
              <p className="text-sm text-gray-400">Smart Hints</p>
            </div>
            <div className="text-center">
              <div className="w-12 h-12 mx-auto bg-green-600 rounded-full flex items-center justify-center mb-2">
                <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
              <p className="text-sm text-gray-400">Track Progress</p>
            </div>
          </div>
        </div>
      ),
    },
    {
      title: 'How It Works',
      content: (
        <div className="space-y-4">
          <div className="flex items-start gap-3">
            <div className="w-8 h-8 bg-blue-600 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
              <span className="text-white font-bold text-sm">1</span>
            </div>
            <div>
              <p className="font-medium text-white">Choose a Learning Path</p>
              <p className="text-sm text-gray-400">Select a path that matches your goals</p>
            </div>
          </div>
          <div className="flex items-start gap-3">
            <div className="w-8 h-8 bg-blue-600 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
              <span className="text-white font-bold text-sm">2</span>
            </div>
            <div>
              <p className="font-medium text-white">Train Neural Networks</p>
              <p className="text-sm text-gray-400">Experiment with inputs and watch the network learn</p>
            </div>
          </div>
          <div className="flex items-start gap-3">
            <div className="w-8 h-8 bg-blue-600 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
              <span className="text-white font-bold text-sm">3</span>
            </div>
            <div>
              <p className="font-medium text-white">Reach the Target Accuracy</p>
              <p className="text-sm text-gray-400">Hints unlock if you get stuck</p>
            </div>
          </div>
          <div className="flex items-start gap-3">
            <div className="w-8 h-8 bg-green-600 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
              <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
              </svg>
            </div>
            <div>
              <p className="font-medium text-white">Unlock Next Steps</p>
              <p className="text-sm text-gray-400">Progress through the path and earn badges</p>
            </div>
          </div>
        </div>
      ),
    },
    {
      title: 'Where to Start?',
      content: (
        <div className="space-y-4">
          <p className="text-gray-300 mb-4">
            Choose a path based on your experience level:
          </p>
          <div className="space-y-3">
            <button
              onClick={() => handleStartPath('foundations')}
              className="w-full p-4 bg-gradient-to-r from-green-600 to-green-700 rounded-lg text-left hover:from-green-500 hover:to-green-600 transition-all group"
            >
              <div className="flex items-center justify-between">
                <div>
                  <div className="flex items-center gap-2">
                    <span className="text-lg font-semibold text-white">Foundations</span>
                    <span className="text-xs bg-green-500 text-white px-2 py-0.5 rounded">Recommended</span>
                  </div>
                  <p className="text-sm text-green-200 mt-1">Perfect for beginners - start here!</p>
                </div>
                <svg className="w-5 h-5 text-white group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                </svg>
              </div>
            </button>

            <button
              onClick={() => handleStartPath('training-mastery')}
              className="w-full p-4 bg-gray-700 rounded-lg text-left hover:bg-gray-600 transition-colors group"
            >
              <div className="flex items-center justify-between">
                <div>
                  <span className="text-lg font-semibold text-white">Training Mastery</span>
                  <p className="text-sm text-gray-400 mt-1">For those who know the basics</p>
                </div>
                <svg className="w-5 h-5 text-gray-400 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                </svg>
              </div>
            </button>

            <button
              onClick={() => handleStartPath('advanced-challenges')}
              className="w-full p-4 bg-gray-700 rounded-lg text-left hover:bg-gray-600 transition-colors group"
            >
              <div className="flex items-center justify-between">
                <div>
                  <span className="text-lg font-semibold text-white">Advanced Challenges</span>
                  <p className="text-sm text-gray-400 mt-1">Tackle depth, gradients, and complex boundaries</p>
                </div>
                <svg className="w-5 h-5 text-gray-400 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                </svg>
              </div>
            </button>
          </div>
        </div>
      ),
    },
  ];

  const currentStepData = steps[step];

  return (
    <Dialog.Root open={isOpen} onOpenChange={setIsOpen}>
      <Dialog.Portal>
        <Dialog.Overlay asChild>
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/70 z-50"
          />
        </Dialog.Overlay>
        <Dialog.Content asChild>
          <motion.div
            initial={{ opacity: 0, scale: 0.95, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: 20 }}
            className="fixed top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 bg-gray-800 rounded-xl shadow-2xl border border-gray-700 w-full max-w-lg z-50 overflow-hidden"
          >
            {/* Progress dots */}
            <div className="flex justify-center gap-2 pt-4">
              {steps.map((_, i) => (
                <button
                  key={i}
                  onClick={() => setStep(i)}
                  className={`w-2 h-2 rounded-full transition-colors ${
                    i === step ? 'bg-blue-500' : 'bg-gray-600 hover:bg-gray-500'
                  }`}
                  aria-label={`Go to step ${i + 1}`}
                />
              ))}
            </div>

            {/* Content */}
            <div className="p-6">
              <Dialog.Title className="text-xl font-bold text-white mb-4">
                {currentStepData.title}
              </Dialog.Title>
              <Dialog.Description asChild>
                <div>{currentStepData.content}</div>
              </Dialog.Description>
            </div>

            {/* Footer */}
            <div className="flex items-center justify-between p-4 bg-gray-900/50 border-t border-gray-700">
              <button
                onClick={handleClose}
                className="text-sm text-gray-400 hover:text-white transition-colors"
              >
                Skip tour
              </button>
              <div className="flex gap-2">
                {step > 0 && (
                  <button
                    onClick={() => setStep(step - 1)}
                    className="px-4 py-2 text-sm text-gray-300 hover:text-white transition-colors"
                  >
                    Back
                  </button>
                )}
                {step < steps.length - 1 ? (
                  <button
                    onClick={() => setStep(step + 1)}
                    className="px-4 py-2 bg-blue-600 text-white text-sm rounded-lg hover:bg-blue-500 transition-colors"
                  >
                    Next
                  </button>
                ) : (
                  <button
                    onClick={handleClose}
                    className="px-4 py-2 bg-blue-600 text-white text-sm rounded-lg hover:bg-blue-500 transition-colors"
                  >
                    Get Started
                  </button>
                )}
              </div>
            </div>

            {/* Close button */}
            <Dialog.Close asChild>
              <button
                className="absolute top-4 right-4 text-gray-400 hover:text-white transition-colors"
                aria-label="Close"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </Dialog.Close>
          </motion.div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
};
