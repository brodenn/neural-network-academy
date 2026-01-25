import { createContext, useContext, useEffect, useState, useCallback, type ReactNode } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useLearningStore, type Achievement } from '../stores/learningStore';

// Context for achievement notifications
interface AchievementContextValue {
  showAchievement: (achievement: Achievement) => void;
}

const AchievementContext = createContext<AchievementContextValue | null>(null);

export const useAchievementNotifications = () => {
  const context = useContext(AchievementContext);
  if (!context) {
    return { showAchievement: () => {} };
  }
  return context;
};

// Notification component with spring animation
const AchievementNotification = ({
  achievement,
  onDismiss
}: {
  achievement: Achievement;
  onDismiss: () => void;
}) => {
  useEffect(() => {
    const timer = setTimeout(onDismiss, 5000);
    return () => clearTimeout(timer);
  }, [onDismiss]);

  const categoryColors = {
    milestone: 'from-blue-500 to-blue-700',
    skill: 'from-purple-500 to-purple-700',
    challenge: 'from-orange-500 to-orange-700',
    streak: 'from-red-500 to-red-700'
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: -100, scale: 0.8 }}
      animate={{
        opacity: 1,
        y: 0,
        scale: 1,
        transition: {
          type: 'spring',
          stiffness: 300,
          damping: 20
        }
      }}
      exit={{
        opacity: 0,
        y: -50,
        scale: 0.9,
        transition: { duration: 0.2 }
      }}
      className={`fixed top-4 left-1/2 -translate-x-1/2 z-[100] bg-gradient-to-r ${categoryColors[achievement.category]} rounded-2xl shadow-2xl overflow-hidden`}
    >
      {/* Shimmer effect */}
      <motion.div
        className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent"
        initial={{ x: '-100%' }}
        animate={{ x: '100%' }}
        transition={{ duration: 1, delay: 0.3 }}
      />

      <div className="relative px-6 py-4 flex items-center gap-4">
        {/* Animated icon */}
        <motion.div
          className="text-5xl"
          animate={{
            scale: [1, 1.3, 1],
            rotate: [0, 10, -10, 0]
          }}
          transition={{
            duration: 0.6,
            repeat: 2,
            repeatType: 'reverse'
          }}
        >
          {achievement.icon}
        </motion.div>

        <div className="pr-8">
          <motion.p
            className="text-xs text-white/80 uppercase tracking-wider font-medium"
            initial={{ opacity: 0, y: 5 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
          >
            Achievement Unlocked!
          </motion.p>
          <motion.p
            className="text-xl font-bold text-white"
            initial={{ opacity: 0, y: 5 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
          >
            {achievement.name}
          </motion.p>
          <motion.p
            className="text-sm text-white/90"
            initial={{ opacity: 0, y: 5 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
          >
            {achievement.description}
          </motion.p>
        </div>

        <button
          onClick={onDismiss}
          className="absolute top-2 right-2 text-white/60 hover:text-white transition-colors p-1"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>

      {/* Progress bar for auto-dismiss */}
      <motion.div
        className="h-1 bg-white/30"
        initial={{ width: '100%' }}
        animate={{ width: '0%' }}
        transition={{ duration: 5, ease: 'linear' }}
      />
    </motion.div>
  );
};

// Provider component
export const AchievementProvider = ({ children }: { children: ReactNode }) => {
  const [pendingNotifications, setPendingNotifications] = useState<Achievement[]>([]);
  const [previousAchievements, setPreviousAchievements] = useState<string[]>([]);

  const achievements = useLearningStore(state => state.getAchievements());
  const unlockedAchievements = useLearningStore(state => state.unlockedAchievements);

  // Subscribe to achievement changes
  useEffect(() => {
    // Find newly unlocked achievements
    const newlyUnlocked = unlockedAchievements.filter(
      id => !previousAchievements.includes(id)
    );

    if (newlyUnlocked.length > 0 && previousAchievements.length > 0) {
      // Only show notifications after initial load (previousAchievements has been set)
      const newAchievements = newlyUnlocked
        .map(id => achievements.find(a => a.id === id))
        .filter((a): a is Achievement => a !== undefined);

      if (newAchievements.length > 0) {
        setPendingNotifications(prev => [...prev, ...newAchievements]);
      }
    }

    setPreviousAchievements(unlockedAchievements);
  }, [unlockedAchievements, achievements, previousAchievements]);

  const showAchievement = useCallback((achievement: Achievement) => {
    setPendingNotifications(prev => [...prev, achievement]);
  }, []);

  const dismissNotification = useCallback(() => {
    setPendingNotifications(prev => prev.slice(1));
  }, []);

  return (
    <AchievementContext.Provider value={{ showAchievement }}>
      {children}

      {/* Notification container */}
      <AnimatePresence mode="wait">
        {pendingNotifications[0] && (
          <AchievementNotification
            key={pendingNotifications[0].id}
            achievement={pendingNotifications[0]}
            onDismiss={dismissNotification}
          />
        )}
      </AnimatePresence>
    </AchievementContext.Provider>
  );
};
