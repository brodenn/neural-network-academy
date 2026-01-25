import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { LearningPathCard } from './LearningPathCard';
import { OnboardingModal } from './OnboardingModal';
import { AchievementsPanel } from './AchievementsPanel';
import { usePathProgress } from '../hooks/usePathProgress';
import type { LearningPath } from '../types';

interface LearningPathSelectorProps {
  onSelectPath: (pathId: string) => void;
}

export const LearningPathSelector = ({ onSelectPath }: LearningPathSelectorProps) => {
  const [paths, setPaths] = useState<LearningPath[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const { getCompletedStepsCount, isPathComplete, updateStreak } = usePathProgress();

  // Update streak on component mount
  const [streak, setStreak] = useState({ currentStreak: 0, bestStreak: 0 });

  useEffect(() => {
    const updatedStreak = updateStreak();
    setStreak({ currentStreak: updatedStreak.currentStreak, bestStreak: updatedStreak.bestStreak });
  }, [updateStreak]);

  // Check if all prerequisites for a path are completed
  const arePrerequisitesMet = (path: LearningPath): boolean => {
    if (!path.prerequisites || path.prerequisites.length === 0) return true;
    return path.prerequisites.every(prereqId => {
      const prereqPath = paths.find(p => p.id === prereqId);
      return prereqPath && isPathComplete(prereqId, prereqPath.steps);
    });
  };

  // Get names of incomplete prerequisites
  const getIncompletePrerequisites = (path: LearningPath): string[] => {
    if (!path.prerequisites || path.prerequisites.length === 0) return [];
    return path.prerequisites
      .filter(prereqId => {
        const prereqPath = paths.find(p => p.id === prereqId);
        return !prereqPath || !isPathComplete(prereqId, prereqPath.steps);
      })
      .map(prereqId => {
        const prereqPath = paths.find(p => p.id === prereqId);
        return prereqPath?.name || prereqId;
      });
  };

  useEffect(() => {
    // Fetch learning paths from backend
    const fetchPaths = async () => {
      try {
        const response = await fetch('http://localhost:5000/api/paths');
        if (!response.ok) {
          throw new Error('Failed to fetch learning paths');
        }
        const data = await response.json();
        setPaths(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unknown error');
      } finally {
        setLoading(false);
      }
    };

    fetchPaths();
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="text-lg text-gray-600">Loading learning paths...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="text-lg text-red-600">Error: {error}</div>
      </div>
    );
  }

  return (
    <>
      {/* Onboarding Modal for first-time users */}
      <OnboardingModal onSelectPath={onSelectPath} />

      <div className="container mx-auto p-6">
        <div className="mb-8 flex items-start justify-between">
          <div>
            <h1 className="text-3xl font-bold mb-2">Learning Paths</h1>
            <p className="text-gray-600">
              Choose your journey through neural network fundamentals
            </p>
          </div>

          {/* Streak Display */}
          {streak.currentStreak > 0 && (
            <motion.div
              initial={{ scale: 0, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              className="flex items-center gap-3 bg-gradient-to-r from-orange-500 to-red-500 text-white px-4 py-2 rounded-lg shadow-lg"
            >
              <motion.span
                className="text-2xl"
                animate={{ scale: [1, 1.2, 1] }}
                transition={{ duration: 0.5, repeat: Infinity, repeatDelay: 2 }}
              >
                🔥
              </motion.span>
              <div className="text-sm">
                <div className="font-bold">{streak.currentStreak} Day Streak!</div>
                {streak.bestStreak > streak.currentStreak && (
                  <div className="text-orange-200 text-xs">Best: {streak.bestStreak} days</div>
                )}
              </div>
            </motion.div>
          )}
        </div>

        {/* Achievements Panel */}
        <div className="mb-6">
          <AchievementsPanel />
        </div>

        {paths.length === 0 ? (
          <div className="text-center text-gray-600 py-12">
            <p className="text-lg">No learning paths available yet.</p>
            <p className="text-sm mt-2">Check back soon!</p>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {paths.map(path => (
              <LearningPathCard
                key={path.id}
                path={path}
                completed={getCompletedStepsCount(path.id)}
                isComplete={isPathComplete(path.id, path.steps)}
                isLocked={!arePrerequisitesMet(path)}
                missingPrerequisites={getIncompletePrerequisites(path)}
                onSelect={() => onSelectPath(path.id)}
              />
            ))}
          </div>
        )}
      </div>
    </>
  );
};
