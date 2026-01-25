import { motion, AnimatePresence } from 'framer-motion';
import { useState, useEffect } from 'react';
import type { Achievement } from '../stores/learningStore';

interface AchievementBadgeProps {
  achievement: Achievement;
  isNew?: boolean;
  size?: 'sm' | 'md' | 'lg';
}

export const AchievementBadge = ({ achievement, isNew = false, size = 'md' }: AchievementBadgeProps) => {
  const sizeClasses = {
    sm: 'w-10 h-10 text-lg',
    md: 'w-14 h-14 text-2xl',
    lg: 'w-20 h-20 text-4xl'
  };

  const categoryColors = {
    milestone: 'from-blue-600 to-blue-800',
    skill: 'from-purple-600 to-purple-800',
    challenge: 'from-orange-600 to-orange-800',
    streak: 'from-red-600 to-red-800'
  };

  return (
    <motion.div
      initial={isNew ? { scale: 0, rotate: -180 } : false}
      animate={{ scale: 1, rotate: 0 }}
      transition={{ type: 'spring', stiffness: 200, damping: 15 }}
      className="flex flex-col items-center gap-1"
    >
      <div
        className={`${sizeClasses[size]} rounded-full bg-gradient-to-br ${categoryColors[achievement.category]} flex items-center justify-center shadow-lg ${isNew ? 'ring-2 ring-yellow-400 ring-offset-2 ring-offset-gray-900' : ''}`}
      >
        <span>{achievement.icon}</span>
      </div>
      {size !== 'sm' && (
        <div className="text-center">
          <p className="text-xs font-medium text-white">{achievement.name}</p>
          {size === 'lg' && (
            <p className="text-xs text-gray-400">{achievement.description}</p>
          )}
        </div>
      )}
    </motion.div>
  );
};

interface AchievementNotificationProps {
  achievement: Achievement;
  onDismiss: () => void;
}

export const AchievementNotification = ({ achievement, onDismiss }: AchievementNotificationProps) => {
  useEffect(() => {
    const timer = setTimeout(onDismiss, 5000);
    return () => clearTimeout(timer);
  }, [onDismiss]);

  return (
    <motion.div
      initial={{ opacity: 0, y: -50, scale: 0.9 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, y: -20, scale: 0.9 }}
      className="fixed top-4 left-1/2 -translate-x-1/2 z-50 bg-gradient-to-r from-yellow-600 to-orange-600 rounded-xl shadow-2xl p-4 flex items-center gap-4"
    >
      <motion.div
        animate={{ scale: [1, 1.2, 1], rotate: [0, 10, -10, 0] }}
        transition={{ duration: 0.5, repeat: 2 }}
        className="text-4xl"
      >
        {achievement.icon}
      </motion.div>
      <div>
        <p className="text-xs text-yellow-200 uppercase tracking-wide">Achievement Unlocked!</p>
        <p className="text-lg font-bold text-white">{achievement.name}</p>
        <p className="text-sm text-yellow-100">{achievement.description}</p>
      </div>
      <button
        onClick={onDismiss}
        className="ml-2 text-white/70 hover:text-white"
      >
        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
        </svg>
      </button>
    </motion.div>
  );
};

interface AchievementListProps {
  achievements: Achievement[];
  unlockedIds: string[];
}

export const AchievementList = ({ achievements, unlockedIds }: AchievementListProps) => {
  const [filter, setFilter] = useState<'all' | 'unlocked' | 'locked'>('all');

  const filteredAchievements = achievements.filter(a => {
    const isUnlocked = unlockedIds.includes(a.id);
    if (filter === 'unlocked') return isUnlocked;
    if (filter === 'locked') return !isUnlocked;
    return true;
  });

  const categories = ['milestone', 'skill', 'challenge', 'streak'] as const;

  return (
    <div className="space-y-4">
      {/* Filter tabs */}
      <div className="flex gap-2">
        {(['all', 'unlocked', 'locked'] as const).map(f => (
          <button
            key={f}
            onClick={() => setFilter(f)}
            className={`px-3 py-1 text-sm rounded-lg transition-colors ${
              filter === f
                ? 'bg-blue-600 text-white'
                : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
            }`}
          >
            {f.charAt(0).toUpperCase() + f.slice(1)}
            {f === 'unlocked' && ` (${unlockedIds.length})`}
          </button>
        ))}
      </div>

      {/* Achievement grid by category */}
      {categories.map(category => {
        const categoryAchievements = filteredAchievements.filter(a => a.category === category);
        if (categoryAchievements.length === 0) return null;

        return (
          <div key={category} className="space-y-2">
            <h3 className="text-sm font-medium text-gray-400 uppercase tracking-wide">
              {category}
            </h3>
            <div className="grid grid-cols-4 gap-4">
              {categoryAchievements.map(achievement => {
                const isUnlocked = unlockedIds.includes(achievement.id);
                return (
                  <div
                    key={achievement.id}
                    className={`relative ${!isUnlocked ? 'opacity-40 grayscale' : ''}`}
                  >
                    <AchievementBadge achievement={achievement} size="md" />
                    {!isUnlocked && (
                      <div className="absolute inset-0 flex items-center justify-center">
                        <svg className="w-6 h-6 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                            d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
                        </svg>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        );
      })}
    </div>
  );
};

// Hook to show achievement notifications
export function useAchievementNotifications() {
  const [pendingAchievements, setPendingAchievements] = useState<Achievement[]>([]);

  const showAchievement = (achievement: Achievement) => {
    setPendingAchievements(prev => [...prev, achievement]);
  };

  const dismissAchievement = () => {
    setPendingAchievements(prev => prev.slice(1));
  };

  const NotificationContainer = () => (
    <AnimatePresence>
      {pendingAchievements[0] && (
        <AchievementNotification
          key={pendingAchievements[0].id}
          achievement={pendingAchievements[0]}
          onDismiss={dismissAchievement}
        />
      )}
    </AnimatePresence>
  );

  return { showAchievement, NotificationContainer };
}
