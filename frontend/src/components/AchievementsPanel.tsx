import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useLearningStore, useLevel, type Achievement } from '../stores/learningStore';

export const AchievementsPanel = () => {
  const [isExpanded, setIsExpanded] = useState(false);
  const achievements = useLearningStore(state => state.getAchievements());
  const unlockedIds = useLearningStore(state => state.unlockedAchievements);
  const totalSteps = useLearningStore(state => state.totalStepsCompleted);
  const streak = useLearningStore(state => state.streak);

  const levelInfo = useLevel();

  const unlockedCount = unlockedIds.length;
  const totalCount = achievements.length;

  const categoryColors: Record<Achievement['category'], string> = {
    milestone: 'from-blue-600 to-blue-800',
    skill: 'from-purple-600 to-purple-800',
    challenge: 'from-orange-600 to-orange-800',
    streak: 'from-red-600 to-red-800'
  };

  const categoryIcons: Record<Achievement['category'], string> = {
    milestone: '🎯',
    skill: '⭐',
    challenge: '🏆',
    streak: '🔥'
  };

  // Group achievements by category
  const groupedAchievements = achievements.reduce((acc, achievement) => {
    if (!acc[achievement.category]) {
      acc[achievement.category] = [];
    }
    acc[achievement.category].push(achievement);
    return acc;
  }, {} as Record<Achievement['category'], Achievement[]>);

  return (
    <div className="bg-gray-800 rounded-xl border border-gray-700 overflow-hidden">
      {/* Header - Always visible */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full px-4 py-3 flex items-center justify-between hover:bg-gray-700/50 transition-colors"
      >
        <div className="flex items-center gap-3">
          <div className="flex -space-x-2">
            {/* Show first 3 unlocked achievement icons */}
            {achievements
              .filter(a => unlockedIds.includes(a.id))
              .slice(0, 3)
              .map((a, i) => (
                <motion.div
                  key={a.id}
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  transition={{ delay: i * 0.1 }}
                  className={`w-8 h-8 rounded-full bg-gradient-to-br ${categoryColors[a.category]} flex items-center justify-center text-sm border-2 border-gray-800`}
                >
                  {a.icon}
                </motion.div>
              ))}
            {unlockedCount === 0 && (
              <div className="w-8 h-8 rounded-full bg-gray-700 flex items-center justify-center text-sm border-2 border-gray-800">
                🔒
              </div>
            )}
          </div>
          <div>
            <span className="font-medium text-white">Achievements</span>
            <span className="text-sm text-gray-400 ml-2">
              {unlockedCount}/{totalCount}
            </span>
          </div>
        </div>

        <div className="flex items-center gap-4">
          {/* Level & XP */}
          <div className="hidden sm:flex items-center gap-3">
            <div className="flex items-center gap-2">
              <div className="w-7 h-7 rounded-full bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center text-xs font-bold text-white shadow">
                {levelInfo.level}
              </div>
              <div className="text-sm">
                <span className="text-indigo-300 font-medium">{levelInfo.title}</span>
                <div className="flex items-center gap-1">
                  <div className="w-16 h-1.5 bg-gray-700 rounded-full overflow-hidden">
                    <motion.div
                      className="h-full bg-gradient-to-r from-indigo-500 to-purple-500 rounded-full"
                      initial={{ width: 0 }}
                      animate={{ width: `${levelInfo.progress * 100}%` }}
                      transition={{ duration: 0.5 }}
                    />
                  </div>
                  <span className="text-[10px] text-gray-500">{levelInfo.xp} XP</span>
                </div>
              </div>
            </div>
            <span className="text-gray-600">|</span>
            <span className="text-sm text-gray-400">{totalSteps} steps</span>
            {streak.currentStreak > 0 && (
              <span className="flex items-center gap-1 text-sm text-gray-400">
                <span>🔥</span>
                {streak.currentStreak}d
              </span>
            )}
          </div>

          <motion.svg
            className="w-5 h-5 text-gray-400"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
            animate={{ rotate: isExpanded ? 180 : 0 }}
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </motion.svg>
        </div>
      </button>

      {/* Expanded content */}
      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="px-4 pb-4 space-y-4">
              {/* Progress bar */}
              <div className="space-y-1">
                <div className="flex justify-between text-xs text-gray-400">
                  <span>Progress</span>
                  <span>{Math.round((unlockedCount / totalCount) * 100)}%</span>
                </div>
                <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
                  <motion.div
                    className="h-full bg-gradient-to-r from-blue-500 to-purple-500"
                    initial={{ width: 0 }}
                    animate={{ width: `${(unlockedCount / totalCount) * 100}%` }}
                    transition={{ duration: 0.5, ease: 'easeOut' }}
                  />
                </div>
              </div>

              {/* Achievement categories */}
              {(Object.keys(groupedAchievements) as Achievement['category'][]).map(category => (
                <div key={category} className="space-y-2">
                  <div className="flex items-center gap-2 text-sm">
                    <span>{categoryIcons[category]}</span>
                    <span className="text-gray-300 capitalize">{category}</span>
                    <span className="text-gray-500">
                      ({groupedAchievements[category].filter(a => unlockedIds.includes(a.id)).length}/
                      {groupedAchievements[category].length})
                    </span>
                  </div>

                  <div className="grid grid-cols-4 sm:grid-cols-6 md:grid-cols-8 gap-2">
                    {groupedAchievements[category].map(achievement => {
                      const isUnlocked = unlockedIds.includes(achievement.id);

                      return (
                        <motion.div
                          key={achievement.id}
                          className="relative group"
                          whileHover={{ scale: 1.1 }}
                        >
                          <div
                            className={`w-10 h-10 rounded-full flex items-center justify-center text-lg transition-all ${
                              isUnlocked
                                ? `bg-gradient-to-br ${categoryColors[category]} shadow-lg`
                                : 'bg-gray-700 grayscale opacity-40'
                            }`}
                          >
                            {isUnlocked ? achievement.icon : '🔒'}
                          </div>

                          {/* Tooltip */}
                          <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-10">
                            <div className="bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 shadow-xl whitespace-nowrap">
                              <p className="font-medium text-white text-sm">{achievement.name}</p>
                              <p className="text-xs text-gray-400">{achievement.description}</p>
                            </div>
                          </div>
                        </motion.div>
                      );
                    })}
                  </div>
                </div>
              ))}

              {/* Encouragement message */}
              {unlockedCount === 0 && (
                <div className="text-center text-gray-400 text-sm py-2">
                  Complete steps to unlock achievements! Start with the Foundations path.
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};
