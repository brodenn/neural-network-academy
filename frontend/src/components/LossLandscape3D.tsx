import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import { useMemo, useRef } from 'react';
import * as THREE from 'three';

interface LossLandscape3DProps {
  lossHistory: number[];
  currentEpoch?: number;
  trainingInProgress?: boolean;
  totalEpochs?: number;
}

function TerrainMesh({ lossHistory }: { lossHistory: number[] }) {
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
}: LossLandscape3DProps) {
  const epoch = currentEpoch ?? lossHistory.length - 1;
  const currentLoss = lossHistory[epoch] ?? 0.5;
  const epochs = totalEpochs ?? lossHistory.length;

  return (
    <div className="w-full h-64 bg-gray-900 rounded-lg overflow-hidden">
      <Canvas camera={{ position: [3, 3, 3], fov: 50 }}>
        <ambientLight intensity={0.4} />
        <directionalLight position={[5, 5, 5]} intensity={0.8} />

        <TerrainMesh lossHistory={lossHistory} />
        <GradientBall
          epoch={epoch}
          totalEpochs={epochs}
          currentLoss={currentLoss}
        />

        <gridHelper args={[4, 20, '#374151', '#1f2937']} rotation={[0, 0, 0]} />
        <OrbitControls enablePan={false} maxPolarAngle={Math.PI / 2.1} />
      </Canvas>
    </div>
  );
}
