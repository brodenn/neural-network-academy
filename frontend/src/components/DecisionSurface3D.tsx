import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import * as THREE from 'three';
import { motion } from 'framer-motion';

interface DecisionSurfaceData {
  surface: number[][];
  classes: number[][];
  x_range: [number, number];
  y_range: [number, number];
  resolution: number;
  problem_id: string;
  category: string;
  output_labels: string[];
  training_data: {
    inputs: number[][];
    labels: number[][];
  };
}

interface DecisionSurface3DProps {
  problemId: string;
  trainingComplete: boolean;
  currentEpoch: number;
}

// Contour points component
function ContourPoints({ points, level }: { points: THREE.Vector3[]; level: number }) {
  const geometry = useMemo(() => {
    const geo = new THREE.BufferGeometry();
    const positions = new Float32Array(points.flatMap(p => [p.x, p.y, p.z]));
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    return geo;
  }, [points]);

  return (
    <points geometry={geometry} rotation={[-Math.PI / 2, 0, 0]}>
      <pointsMaterial
        color={level === 0.5 ? '#ffffff' : '#cccccc'}
        size={0.05}
        transparent
        opacity={level === 0.5 ? 1 : 0.6}
      />
    </points>
  );
}

// 3D Surface mesh component
function SurfaceMesh({ data, showContours }: { data: DecisionSurfaceData; showContours: boolean }) {
  const meshRef = useRef<THREE.Mesh>(null);

  const { geometry } = useMemo(() => {
    const { surface, classes, resolution, category } = data;
    const size = 4; // World units
    const geo = new THREE.PlaneGeometry(size, size, resolution - 1, resolution - 1);
    const positions = geo.attributes.position.array as Float32Array;
    const colorArray = new Float32Array(positions.length);

    // Update vertex positions (Z = probability) and colors
    for (let i = 0; i < resolution; i++) {
      for (let j = 0; j < resolution; j++) {
        const idx = (i * resolution + j) * 3;
        const prob = surface[resolution - 1 - i][j]; // Flip Y for correct orientation

        // Z position = probability (scaled)
        positions[idx + 2] = prob * 2 - 1; // Scale to -1 to 1

        // Color based on prediction
        if (category === 'multi-class') {
          const classIdx = classes[resolution - 1 - i][j];
          const numClasses = data.output_labels.length;
          const hue = (classIdx / numClasses);
          const color = new THREE.Color().setHSL(hue, 0.8, 0.4 + prob * 0.2);
          colorArray[idx] = color.r;
          colorArray[idx + 1] = color.g;
          colorArray[idx + 2] = color.b;
        } else {
          // Binary: blue (0) to red (1)
          colorArray[idx] = prob; // R
          colorArray[idx + 1] = 0.2; // G
          colorArray[idx + 2] = 1 - prob; // B
        }
      }
    }

    geo.setAttribute('color', new THREE.BufferAttribute(colorArray, 3));
    geo.computeVertexNormals();
    return { geometry: geo };
  }, [data]);

  // Contour lines at 0.25, 0.5, 0.75
  const contourLines = useMemo(() => {
    if (!showContours) return [];

    const { surface, resolution } = data;
    const lines: { points: THREE.Vector3[]; level: number }[] = [];
    const levels = [0.25, 0.5, 0.75];
    const size = 4;
    const step = size / (resolution - 1);

    for (const level of levels) {
      const points: THREE.Vector3[] = [];
      for (let i = 0; i < resolution - 1; i++) {
        for (let j = 0; j < resolution - 1; j++) {
          const p00 = surface[resolution - 1 - i][j];
          const p01 = surface[resolution - 1 - i][j + 1];
          const p10 = surface[resolution - 1 - (i + 1)][j];

          // Check if level crosses this cell
          if ((p00 < level && p01 >= level) || (p00 >= level && p01 < level)) {
            const t = (level - p00) / (p01 - p00);
            const x = -size / 2 + j * step + t * step;
            const y = -size / 2 + i * step;
            points.push(new THREE.Vector3(x, y, level * 2 - 1));
          }
          if ((p00 < level && p10 >= level) || (p00 >= level && p10 < level)) {
            const t = (level - p00) / (p10 - p00);
            const x = -size / 2 + j * step;
            const y = -size / 2 + i * step + t * step;
            points.push(new THREE.Vector3(x, y, level * 2 - 1));
          }
        }
      }
      if (points.length > 0) {
        lines.push({ points, level });
      }
    }
    return lines;
  }, [data, showContours]);

  return (
    <group>
      <mesh ref={meshRef} geometry={geometry} rotation={[-Math.PI / 2, 0, 0]}>
        <meshStandardMaterial
          vertexColors
          side={THREE.DoubleSide}
          transparent
          opacity={0.9}
        />
      </mesh>

      {/* Contour lines */}
      {contourLines.map((line, idx) => (
        <ContourPoints key={idx} points={line.points} level={line.level} />
      ))}
    </group>
  );
}

// Training data points as spheres
function DataPoints({ data }: { data: DecisionSurfaceData }) {
  const points = useMemo(() => {
    const { training_data, x_range, y_range, category, output_labels } = data;
    const [xMin, xMax] = x_range;
    const [yMin, yMax] = y_range;

    return training_data.inputs.map((input, i) => {
      const [x, y] = input;
      const label = training_data.labels[i];

      // Normalize to -2 to 2 world space
      const worldX = ((x - xMin) / (xMax - xMin) - 0.5) * 4;
      const worldY = ((y - yMin) / (yMax - yMin) - 0.5) * 4;

      // Determine class and color
      let classIdx: number;
      let color: THREE.Color;

      if (category === 'multi-class') {
        classIdx = label.indexOf(Math.max(...label));
        const hue = classIdx / output_labels.length;
        color = new THREE.Color().setHSL(hue, 0.9, 0.5);
      } else {
        classIdx = label[0] >= 0.5 ? 1 : 0;
        color = new THREE.Color(classIdx === 1 ? '#ef4444' : '#3b82f6');
      }

      // Z position based on class (float above or below surface)
      const worldZ = classIdx === 1 ? 1.3 : -1.3;

      return { x: worldX, y: worldY, z: worldZ, color, classIdx };
    });
  }, [data]);

  return (
    <group rotation={[-Math.PI / 2, 0, 0]}>
      {points.map((point, i) => (
        <mesh key={i} position={[point.x, point.y, point.z]}>
          <sphereGeometry args={[0.08, 12, 12]} />
          <meshStandardMaterial
            color={point.color}
            emissive={point.color}
            emissiveIntensity={0.3}
          />
        </mesh>
      ))}
    </group>
  );
}

// Animated camera for auto-rotation
function CameraController({ autoRotate }: { autoRotate: boolean }) {
  const controlsRef = useRef<any>(null);

  useFrame(() => {
    if (controlsRef.current && autoRotate) {
      controlsRef.current.update();
    }
  });

  return (
    <OrbitControls
      ref={controlsRef}
      enablePan={false}
      maxPolarAngle={Math.PI / 2.1}
      minPolarAngle={Math.PI / 6}
      autoRotate={autoRotate}
      autoRotateSpeed={0.5}
    />
  );
}

export function DecisionSurface3D({
  problemId,
  trainingComplete,
  currentEpoch,
}: DecisionSurface3DProps) {
  const [data, setData] = useState<DecisionSurfaceData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showDataPoints, setShowDataPoints] = useState(true);
  const [showContours, setShowContours] = useState(true);
  const [autoRotate, setAutoRotate] = useState(false);

  // Only show for 2D problems
  const is2DProblem = [
    'two_blobs', 'moons', 'circle', 'donut', 'spiral',
    'fail_underfit', 'quadrants', 'blobs', 'colors',
  ].includes(problemId);

  const fetchSurface = useCallback(async () => {
    if (!is2DProblem || !trainingComplete) return;

    setLoading(true);
    setError(null);

    try {
      const res = await fetch('http://localhost:5000/api/decision-surface?resolution=40');
      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.error || 'Failed to fetch');
      }
      const surfaceData = await res.json();
      setData(surfaceData);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  }, [is2DProblem, trainingComplete]);

  // Fetch when training completes
  useEffect(() => {
    if (trainingComplete && is2DProblem) {
      fetchSurface();
    }
  }, [trainingComplete, is2DProblem, fetchSurface]);

  // Refresh during training (every 200 epochs)
  useEffect(() => {
    if (currentEpoch > 0 && currentEpoch % 200 === 0 && is2DProblem) {
      fetchSurface();
    }
  }, [currentEpoch, is2DProblem, fetchSurface]);

  if (!is2DProblem) {
    return null;
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-gray-800 rounded-lg p-3"
    >
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-lg font-semibold flex items-center gap-2">
          <span className="text-xl">🏔️</span>
          3D Decision Surface
        </h3>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowDataPoints(!showDataPoints)}
            className={`px-2 py-1 text-xs rounded ${
              showDataPoints ? 'bg-blue-600 text-white' : 'bg-gray-700 text-gray-400'
            }`}
          >
            Points
          </button>
          <button
            onClick={() => setShowContours(!showContours)}
            className={`px-2 py-1 text-xs rounded ${
              showContours ? 'bg-purple-600 text-white' : 'bg-gray-700 text-gray-400'
            }`}
          >
            Contours
          </button>
          <button
            onClick={() => setAutoRotate(!autoRotate)}
            className={`px-2 py-1 text-xs rounded ${
              autoRotate ? 'bg-cyan-600 text-white' : 'bg-gray-700 text-gray-400'
            }`}
          >
            Rotate
          </button>
          <button
            onClick={fetchSurface}
            disabled={loading}
            className="px-2 py-1 text-xs rounded bg-gray-700 text-gray-300 hover:bg-gray-600 disabled:opacity-50"
          >
            {loading ? '...' : 'Refresh'}
          </button>
        </div>
      </div>

      {/* Educational explanation */}
      <div className="mb-3 p-2 bg-gray-700/50 rounded text-xs text-gray-400">
        <p>
          The 3D surface shows prediction probability as height.{' '}
          <span className="text-red-400">Red peaks</span> = high probability of class 1,{' '}
          <span className="text-blue-400">blue valleys</span> = class 0.
          The <span className="text-white">white contour</span> at height 0 is the decision boundary.
        </p>
      </div>

      {!trainingComplete ? (
        <div className="h-64 bg-gray-700/50 rounded flex items-center justify-center text-gray-500">
          <p>Train the network to see the 3D surface</p>
        </div>
      ) : loading && !data ? (
        <div className="h-64 bg-gray-700/50 rounded flex items-center justify-center text-gray-500">
          <p>Loading decision surface...</p>
        </div>
      ) : error ? (
        <div className="h-64 bg-gray-700/50 rounded flex items-center justify-center text-red-400">
          <p>Error: {error}</p>
        </div>
      ) : data ? (
        <div className="h-64 bg-gray-900 rounded-lg overflow-hidden">
          <Canvas camera={{ position: [4, 4, 4], fov: 45 }}>
            <ambientLight intensity={0.4} />
            <directionalLight position={[5, 5, 5]} intensity={0.8} />
            <directionalLight position={[-5, -5, 5]} intensity={0.3} />

            <SurfaceMesh data={data} showContours={showContours} />
            {showDataPoints && <DataPoints data={data} />}

            {/* Reference grid */}
            <gridHelper args={[4, 20, '#374151', '#1f2937']} position={[0, -1.5, 0]} />

            {/* Axis indicators */}
            <group>
              {/* X axis label */}
              <mesh position={[2.5, -1.5, 0]}>
                <boxGeometry args={[0.3, 0.02, 0.02]} />
                <meshBasicMaterial color="#ef4444" />
              </mesh>
              {/* Y axis label */}
              <mesh position={[0, -1.5, 2.5]}>
                <boxGeometry args={[0.02, 0.02, 0.3]} />
                <meshBasicMaterial color="#22c55e" />
              </mesh>
            </group>

            <CameraController autoRotate={autoRotate} />
          </Canvas>
        </div>
      ) : null}

      {/* Legend */}
      {data && (
        <div className="mt-2 flex justify-center gap-4 text-xs">
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 rounded-full bg-blue-500" />
            <span className="text-gray-400">Class 0 (Low)</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 rounded-full bg-red-500" />
            <span className="text-gray-400">Class 1 (High)</span>
          </div>
        </div>
      )}
    </motion.div>
  );
}
