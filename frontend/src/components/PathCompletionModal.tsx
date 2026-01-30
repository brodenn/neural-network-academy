import { useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

interface PathCompletionModalProps {
  isOpen: boolean;
  pathName: string;
  badge: {
    icon: string;
    color: string;
    title: string;
  };
  stats: {
    totalSteps: number;
    totalAttempts: number;
    avgAccuracy: number;
    timeSpent?: string;
  };
  onReviewPath: () => void;
  onBackToPaths: () => void;
  onClose: () => void;
}

export const PathCompletionModal = ({
  isOpen,
  pathName,
  badge,
  stats,
  onReviewPath,
  onBackToPaths,
  onClose
}: PathCompletionModalProps) => {
  // Path-themed emoji confetti
  const PATH_EMOJIS: Record<string, string[]> = {
    'Foundations':             ['🏆', '🧱', '🎯', '✨'],
    'Deep Learning Basics':   ['🧠', '📊', '🔬', '💡'],
    'Multi-Class Mastery':    ['🎨', '🌈', '🎯', '✨'],
    'Convolutional Vision':   ['👁️', '📷', '🔍', '🖼️'],
    'Pitfall Prevention':     ['🛡️', '🐛', '🔧', '💪'],
    'Research Frontier':      ['🚀', '🌟', '🔬', '🧪'],
    'Interactive Fundamentals': ['🎮', '🕹️', '⚡', '🎯'],
  };

  // Trigger confetti effect when modal opens
  useEffect(() => {
    if (isOpen) {
      const confettiContainer = document.getElementById('confetti-container');
      if (confettiContainer) {
        confettiContainer.innerHTML = '';

        // Get themed emojis for this path, falling back to generic colors
        const emojis = PATH_EMOJIS[pathName] ?? ['🎉', '✨', '🌟', '🎊'];
        const colors = ['#22c55e', '#3b82f6', '#eab308', '#ec4899', '#8b5cf6'];

        for (let i = 0; i < 60; i++) {
          const piece = document.createElement('div');
          const isEmoji = i < 20; // First 20 pieces are emojis, rest are colored squares
          piece.className = 'confetti-piece';

          if (isEmoji) {
            piece.textContent = emojis[Math.floor(Math.random() * emojis.length)];
            piece.style.cssText = `
              position: absolute;
              font-size: ${14 + Math.random() * 10}px;
              left: ${Math.random() * 100}%;
              top: -30px;
              animation: confetti-fall ${2 + Math.random() * 2}s linear forwards;
              animation-delay: ${Math.random() * 0.8}s;
              transform: rotate(${Math.random() * 360}deg);
              pointer-events: none;
            `;
          } else {
            piece.style.cssText = `
              position: absolute;
              width: ${6 + Math.random() * 8}px;
              height: ${6 + Math.random() * 8}px;
              background: ${colors[Math.floor(Math.random() * colors.length)]};
              left: ${Math.random() * 100}%;
              top: -10px;
              animation: confetti-fall ${2 + Math.random() * 2}s linear forwards;
              animation-delay: ${Math.random() * 0.5}s;
              transform: rotate(${Math.random() * 360}deg);
              border-radius: ${Math.random() > 0.5 ? '50%' : '2px'};
            `;
          }
          confettiContainer.appendChild(piece);
        }
      }
    }
  }, [isOpen, pathName]);

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/70 z-50"
            onClick={onClose}
          />

          {/* Confetti Container */}
          <div
            id="confetti-container"
            className="fixed inset-0 z-50 pointer-events-none overflow-hidden"
          />

          {/* Modal */}
          <motion.div
            initial={{ opacity: 0, scale: 0.8, y: 50 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.8, y: 50 }}
            transition={{ type: 'spring', duration: 0.5 }}
            className="fixed inset-0 z-50 flex items-center justify-center p-4 pointer-events-none"
          >
            <div className="bg-gray-800 rounded-2xl p-8 max-w-md w-full shadow-2xl pointer-events-auto">
              {/* Badge Animation */}
              <motion.div
                initial={{ scale: 0, rotate: -180 }}
                animate={{ scale: 1, rotate: 0 }}
                transition={{ type: 'spring', stiffness: 200, delay: 0.2 }}
                className="flex justify-center mb-6"
              >
                <div className={`w-24 h-24 rounded-full flex items-center justify-center shadow-lg ${badge.color || 'bg-gradient-to-br from-yellow-400 to-yellow-600'}`}>
                  <span className="text-5xl">{badge.icon}</span>
                </div>
              </motion.div>

              {/* Title */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.4 }}
                className="text-center mb-6"
              >
                <h2 className="text-2xl font-bold text-white mb-2">Congratulations!</h2>
                <p className="text-gray-400">
                  You completed <span className="text-green-400 font-semibold">{pathName}</span>
                </p>
                <p className="text-yellow-400 font-medium mt-1">{badge.title}</p>
              </motion.div>

              {/* Stats */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.5 }}
                className="grid grid-cols-4 gap-3 mb-6"
              >
                <div className="bg-gray-900/50 rounded-lg p-3 text-center">
                  <div className="text-2xl font-bold text-green-400">{stats.totalSteps}</div>
                  <div className="text-xs text-gray-500">Steps</div>
                </div>
                <div className="bg-gray-900/50 rounded-lg p-3 text-center">
                  <div className="text-2xl font-bold text-blue-400">{stats.totalAttempts}</div>
                  <div className="text-xs text-gray-500">Attempts</div>
                </div>
                <div className="bg-gray-900/50 rounded-lg p-3 text-center">
                  <div className="text-2xl font-bold text-purple-400">
                    {(stats.avgAccuracy * 100).toFixed(0)}%
                  </div>
                  <div className="text-xs text-gray-500">Avg Accuracy</div>
                </div>
                <div className="bg-gray-900/50 rounded-lg p-3 text-center">
                  <motion.div
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    transition={{ type: 'spring', delay: 0.7 }}
                    className="text-2xl font-bold text-indigo-400"
                  >
                    +200
                  </motion.div>
                  <div className="text-xs text-gray-500">XP Earned</div>
                </div>
              </motion.div>

              {/* Buttons */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.6 }}
                className="space-y-3"
              >
                <button
                  onClick={onReviewPath}
                  className="w-full py-3 bg-gray-700 hover:bg-gray-600 text-white rounded-lg font-medium transition-colors"
                >
                  Review Path
                </button>
                <button
                  onClick={onBackToPaths}
                  className="w-full py-3 bg-blue-600 hover:bg-blue-500 text-white rounded-lg font-medium transition-colors"
                >
                  Back to Learning Paths
                </button>
              </motion.div>
            </div>
          </motion.div>

          {/* Confetti CSS */}
          <style>{`
            @keyframes confetti-fall {
              0% {
                transform: translateY(0) rotate(0deg);
                opacity: 1;
              }
              100% {
                transform: translateY(100vh) rotate(720deg);
                opacity: 0;
              }
            }
          `}</style>
        </>
      )}
    </AnimatePresence>
  );
};
