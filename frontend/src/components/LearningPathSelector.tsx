import { useState, useEffect } from 'react';
import { LearningPathCard } from './LearningPathCard';
import { usePathProgress } from '../hooks/usePathProgress';
import type { LearningPath } from '../types';

interface LearningPathSelectorProps {
  onSelectPath: (pathId: string) => void;
}

export const LearningPathSelector = ({ onSelectPath }: LearningPathSelectorProps) => {
  const [paths, setPaths] = useState<LearningPath[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const { getCompletedStepsCount, isPathComplete } = usePathProgress();

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
    <div className="container mx-auto p-6">
      <div className="mb-8">
        <h1 className="text-3xl font-bold mb-2">Learning Paths</h1>
        <p className="text-gray-600">
          Choose your journey through neural network fundamentals
        </p>
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
              onSelect={() => onSelectPath(path.id)}
            />
          ))}
        </div>
      )}
    </div>
  );
};
