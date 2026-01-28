import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import { useMemo, useRef, useState, useEffect, useCallback } from 'react';
import * as THREE from 'three';

interface RealisticLossData {
  losses: number[][];
  resolution: number;
  range: number;
  center_loss: number;
  alphas: number[];
  betas: number[];
}

interface LossLandscape3DProps {
  lossHistory: number[];
  currentEpoch?: number;
  trainingInProgress?: boolean;
  totalEpochs?: number;
  networkType?: string;
}

// Synthetic terrain mesh (original bowl shape)
function SyntheticTerrainMesh({ lossHistory }: { lossHistory: number[] }) {
  const geometry = useMemo(() => {
    const size = 32;
    const geo = new THREE.PlaneGeometry(4, 4, size - 1, size - 1);
    const positions = geo.attributes.position.array as Float32Array;

    const maxLoss = Math.max(...lossHistory, 0.5);

    for (let i = 0; i < size; i++) {
      for (let j = 0; j < size; j++) {
        const idx = (i * size + j) * 3 + 2; // Z component
        const x = (i / size - 0.5) * 2;
        const y = (j / size - 0.5) * 2;

        // Quadratic bowl + noise for local minima
        const bowl = (x * x + y * y) * 0.8;
        const noise = Math.sin(x * 4) * Math.cos(y * 4) * 0.15;
        positions[idx] = (bowl + noise + 0.1) * maxLoss;
      }
    }

    geo.computeVertexNormals();
    return geo;
  }, [lossHistory]);

  return (
    <mesh geometry={geometry} rotation={[-Math.PI / 2, 0, 0]} position={[0, 0, 0]}>
      <meshStandardMaterial
        color="#0891b2"
        wireframe={false}
        side={THREE.DoubleSide}
        transparent
        opacity={0.85}
      />
    </mesh>
  );
}

// Realistic terrain mesh from actual loss samples
function RealisticTerrainMesh({ data }: { data: RealisticLossData }) {
  const { geometry } = useMemo(() => {
    const { losses, resolution } = data;
    const geo = new THREE.PlaneGeometry(4, 4, resolution - 1, resolution - 1);
    const positions = geo.attributes.position.array as Float32Array;
    const colorArray = new Float32Array(positions.length);

    // Find min/max for normalization
    const flatLosses = losses.flat();
    const minLoss = Math.min(...flatLosses);
    const maxLoss = Math.max(...flatLosses);
    const lossRange = maxLoss - minLoss || 1;

    // Update vertex positions and colors
    for (let i = 0; i < resolution; i++) {
      for (let j = 0; j < resolution; j++) {
        const idx = (i * resolution + j) * 3;
        const loss = losses[i][j];

        // Z position = loss value (normalized to reasonable height)
        const normalizedLoss = (loss - minLoss) / lossRange;
        positions[idx + 2] = normalizedLoss * 2; // Scale to 0-2 range

        // Color: blue (low loss) -> cyan -> yellow -> red (high loss)
        const t = normalizedLoss;
        let color: THREE.Color;
        if (t < 0.33) {
          // Blue to cyan
          const localT = t / 0.33;
          color = new THREE.Color().setHSL(0.55 - localT * 0.05, 0.8, 0.4 + localT * 0.1);
        } else if (t < 0.66) {
          // Cyan to yellow
          const localT = (t - 0.33) / 0.33;
          color = new THREE.Color().setHSL(0.5 - localT * 0.35, 0.9, 0.5);
        } else {
          // Yellow to red
          const localT = (t - 0.66) / 0.34;
          color = new THREE.Color().setHSL(0.15 - localT * 0.15, 0.9, 0.45 + localT * 0.1);
        }

        colorArray[idx] = color.r;
        colorArray[idx + 1] = color.g;
        colorArray[idx + 2] = color.b;
      }
    }

    geo.setAttribute('color', new THREE.BufferAttribute(colorArray, 3));
    geo.computeVertexNormals();
    return { geometry: geo };
  }, [data]);

  return (
    <mesh geometry={geometry} rotation={[-Math.PI / 2, 0, 0]} position={[0, 0, 0]}>
      <meshStandardMaterial
        vertexColors
        wireframe={false}
        side={THREE.DoubleSide}
        transparent
        opacity={0.9}
      />
    </mesh>
  );
}

interface GradientBallProps {
  epoch: number;
  totalEpochs: number;
  currentLoss: number;
}

function GradientBall({ epoch, totalEpochs, currentLoss }: GradientBallProps) {
  const meshRef = useRef<THREE.Mesh>(null);

  useFrame(() => {
    if (!meshRef.current) return;

    const progress = Math.min(epoch / Math.max(totalEpochs, 1), 1);
    const radius = (1 - progress) * 1.8 + 0.1;
    const angle = progress * Math.PI * 6; // Spiral

    const targetX = Math.cos(angle) * radius;
    const targetZ = Math.sin(angle) * radius;
    const targetY = currentLoss * 0.8 + 0.15;

    // Smooth lerp to target
    meshRef.current.position.x += (targetX - meshRef.current.position.x) * 0.1;
    meshRef.current.position.y += (targetY - meshRef.current.position.y) * 0.1;
    meshRef.current.position.z += (targetZ - meshRef.current.position.z) * 0.1;
  });

  return (
    <mesh ref={meshRef} position={[1.5, 0.8, 0]}>
      <sphereGeometry args={[0.08, 16, 16]} />
      <meshStandardMaterial color="#ef4444" emissive="#ef4444" emissiveIntensity={0.5} />
    </mesh>
  );
}

export function LossLandscape3D({
  lossHistory,
  currentEpoch,
  totalEpochs,
  networkType = 'dense',
}: LossLandscape3DProps) {
  const [mode, setMode] = useState<'synthetic' | 'realistic'>('synthetic');
  const [realisticData, setRealisticData] = useState<RealisticLossData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const epoch = currentEpoch ?? lossHistory.length - 1;
  const currentLoss = lossHistory[epoch] ?? 0.5;
  const epochs = totalEpochs ?? lossHistory.length;

  // Only allow realistic mode for dense networks
  const canUseRealistic = networkType === 'dense';

  const fetchRealisticLandscape = useCallback(async () => {
    if (!canUseRealistic) return;

    setLoading(true);
    setError(null);

    try {
      const res = await fetch('http://localhost:5000/api/loss-landscape', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ resolution: 25, range: 1.5 }),
      });

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.error || 'Failed to fetch');
      }

      const data = await res.json();
      setRealisticData(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
      setMode('synthetic'); // Fall back to synthetic on error
    } finally {
      setLoading(false);
    }
  }, [canUseRealistic]);

  // Fetch realistic data when mode changes or training completes
  useEffect(() => {
    if (mode === 'realistic' && !realisticData && lossHistory.length > 0) {
      fetchRealisticLandscape();
    }
  }, [mode, realisticData, lossHistory.length, fetchRealisticLandscape]);

  // Clear realistic data when network changes (detected by loss history reset)
  useEffect(() => {
    if (lossHistory.length === 0) {
      setRealisticData(null);
    }
  }, [lossHistory.length]);

  return (
    <div className="space-y-2">
      {/* Mode toggle */}
      {canUseRealistic && (
        <div className="flex items-center justify-between">
          <div className="flex gap-0.5 bg-gray-900 rounded p-0.5">
            <button
              onClick={() => setMode('synthetic')}
              className={`px-2 py-1 text-xs rounded transition-colors ${
                mode === 'synthetic'
                  ? 'bg-cyan-600 text-white'
                  : 'text-gray-400 hover:text-white'
              }`}
            >
              Synthetic
            </button>
            <button
              onClick={() => setMode('realistic')}
              disabled={loading}
              className={`px-2 py-1 text-xs rounded transition-colors ${
                mode === 'realistic'
                  ? 'bg-orange-600 text-white'
                  : 'text-gray-400 hover:text-white'
              } disabled:opacity-50`}
            >
              {loading ? 'Loading...' : 'Realistic'}
            </button>
          </div>

          {mode === 'realistic' && realisticData && (
            <button
              onClick={fetchRealisticLandscape}
              disabled={loading}
              className="px-2 py-1 text-xs rounded bg-gray-700 text-gray-300 hover:bg-gray-600 disabled:opacity-50"
            >
              Refresh
            </button>
          )}
        </div>
      )}

      {/* Mode explanation */}
      {mode === 'realistic' && (
        <div className="p-2 bg-gray-800/50 rounded text-xs text-gray-400">
          <p>
            <span className="text-orange-400 font-medium">Realistic mode</span> samples actual loss values
            along random directions in weight space. Shows true landscape topology: saddle points, local minima, and plateaus.
          </p>
        </div>
      )}

      {error && (
        <div className="p-2 bg-red-900/30 rounded text-xs text-red-400">
          Error loading realistic landscape: {error}
        </div>
      )}

      <div className="w-full h-64 bg-gray-900 rounded-lg overflow-hidden">
        <Canvas camera={{ position: [3, 3, 3], fov: 50 }}>
          <ambientLight intensity={0.4} />
          <directionalLight position={[5, 5, 5]} intensity={0.8} />

          {mode === 'realistic' && realisticData ? (
            <RealisticTerrainMesh data={realisticData} />
          ) : (
            <SyntheticTerrainMesh lossHistory={lossHistory} />
          )}

          <GradientBall
            epoch={epoch}
            totalEpochs={epochs}
            currentLoss={currentLoss}
          />

          <gridHelper args={[4, 20, '#374151', '#1f2937']} rotation={[0, 0, 0]} />
          <OrbitControls enablePan={false} maxPolarAngle={Math.PI / 2.1} />
        </Canvas>
      </div>

      {/* Color legend for realistic mode */}
      {mode === 'realistic' && realisticData && (
        <div className="flex justify-center gap-4 text-xs">
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 rounded bg-blue-500" />
            <span className="text-gray-400">Low loss</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 rounded bg-yellow-500" />
            <span className="text-gray-400">Medium</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 rounded bg-red-500" />
            <span className="text-gray-400">High loss</span>
          </div>
        </div>
      )}
    </div>
  );
}
