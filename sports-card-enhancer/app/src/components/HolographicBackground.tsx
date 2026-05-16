import { useRef, useMemo, useCallback } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import * as THREE from 'three';

// Advanced holographic grid with wave distortion
function HolographicGrid() {
  const meshRef = useRef<THREE.Mesh>(null);
  const materialRef = useRef<THREE.ShaderMaterial>(null);

  const uniforms = useMemo(
    () => ({
      uTime: { value: 0 },
      uColor1: { value: new THREE.Color('#00ffff') },
      uColor2: { value: new THREE.Color('#ff00ff') },
      uColor3: { value: new THREE.Color('#0080ff') },
      uOpacity: { value: 0.2 },
      uMouse: { value: new THREE.Vector2(0.5, 0.5) }
    }),
    []
  );

  const vertexShader = `
    varying vec2 vUv;
    varying vec3 vPosition;
    uniform float uTime;
    
    void main() {
      vUv = uv;
      vPosition = position;
      
      // Wave displacement
      vec3 pos = position;
      float wave = sin(pos.x * 2.0 + uTime * 0.5) * cos(pos.y * 2.0 + uTime * 0.3) * 0.15;
      pos.z += wave;
      
      gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
    }
  `;

  const fragmentShader = `
    uniform float uTime;
    uniform vec3 uColor1;
    uniform vec3 uColor2;
    uniform vec3 uColor3;
    uniform float uOpacity;
    uniform vec2 uMouse;
    
    varying vec2 vUv;
    varying vec3 vPosition;
    
    // Smooth noise function
    float hash(vec2 p) {
      return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
    }
    
    float noise(vec2 p) {
      vec2 i = floor(p);
      vec2 f = fract(p);
      f = f * f * (3.0 - 2.0 * f);
      
      float a = hash(i);
      float b = hash(i + vec2(1.0, 0.0));
      float c = hash(i + vec2(0.0, 1.0));
      float d = hash(i + vec2(1.0, 1.0));
      
      return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
    }
    
    void main() {
      // Create advanced grid pattern
      float gridSize = 25.0;
      vec2 grid = fract(vUv * gridSize);
      
      // Line thickness with anti-aliasing
      float lineWidth = 0.03;
      float lineX = smoothstep(lineWidth, 0.0, abs(grid.x - 0.5) * 2.0 - (1.0 - lineWidth * 2.0));
      float lineY = smoothstep(lineWidth, 0.0, abs(grid.y - 0.5) * 2.0 - (1.0 - lineWidth * 2.0));
      float lines = max(lineX, lineY);
      
      // Create intersection highlights
      float intersect = lineX * lineY * 3.0;
      
      // Animated energy waves
      float wave1 = sin(vUv.x * 15.0 + uTime * 0.8) * 0.5 + 0.5;
      float wave2 = sin(vUv.y * 12.0 + uTime * 0.6) * 0.5 + 0.5;
      float wave3 = sin(length(vUv - 0.5) * 20.0 - uTime * 1.2) * 0.5 + 0.5;
      float energy = wave1 * wave2 * wave3;
      
      // Add noise for organic feel
      float n = noise(vUv * 10.0 + uTime * 0.1) * 0.3;
      
      // Color mixing with three colors
      vec3 color = mix(uColor1, uColor2, wave1 * 0.5);
      color = mix(color, uColor3, wave2 * 0.3);
      color += vec3(intersect * 0.3);
      
      // Mouse interaction glow
      float mouseGlow = 1.0 - smoothstep(0.0, 0.4, length(vUv - uMouse));
      color += uColor1 * mouseGlow * 0.5;
      
      // Radial fade from center
      float dist = length(vUv - 0.5);
      float radialFade = 1.0 - smoothstep(0.2, 0.8, dist);
      
      // Edge fade
      float fadeX = smoothstep(0.0, 0.2, vUv.x) * smoothstep(1.0, 0.8, vUv.x);
      float fadeY = smoothstep(0.0, 0.2, vUv.y) * smoothstep(1.0, 0.8, vUv.y);
      float fade = fadeX * fadeY;
      
      // Final composition
      float alpha = (lines + intersect * 0.5 + energy * 0.1 + n) * uOpacity * fade * radialFade;
      
      gl_FragColor = vec4(color, alpha);
    }
  `;

  useFrame((state) => {
    if (materialRef.current) {
      materialRef.current.uniforms.uTime.value = state.clock.elapsedTime;
    }
  });

  return (
    <mesh ref={meshRef} rotation={[-Math.PI / 2.2, 0, 0]} position={[0, -2.5, -2]}>
      <planeGeometry args={[30, 30, 64, 64]} />
      <shaderMaterial
        ref={materialRef}
        uniforms={uniforms}
        vertexShader={vertexShader}
        fragmentShader={fragmentShader}
        transparent={true}
        side={THREE.DoubleSide}
        depthWrite={false}
        blending={THREE.AdditiveBlending}
      />
    </mesh>
  );
}

// Enhanced floating particles with trails
function FloatingParticles() {
  const pointsRef = useRef<THREE.Points>(null);
  const particleCount = 150;

  const [positions, colors, sizes] = useMemo(() => {
    const pos = new Float32Array(particleCount * 3);
    const col = new Float32Array(particleCount * 3);
    const siz = new Float32Array(particleCount);
    
    for (let i = 0; i < particleCount; i++) {
      pos[i * 3] = (Math.random() - 0.5) * 25;
      pos[i * 3 + 1] = (Math.random() - 0.5) * 15;
      pos[i * 3 + 2] = (Math.random() - 0.5) * 15;
      
      // Random color between cyan and magenta
      const t = Math.random();
      col[i * 3] = t < 0.5 ? 0 : 1; // R
      col[i * 3 + 1] = t < 0.5 ? 1 : 0; // G
      col[i * 3 + 2] = 1; // B
      
      siz[i] = Math.random() * 0.08 + 0.02;
    }
    
    return [pos, col, siz];
  }, []);

  const velocities = useMemo(() => {
    const vel = new Float32Array(particleCount * 3);
    for (let i = 0; i < particleCount; i++) {
      vel[i * 3] = (Math.random() - 0.5) * 0.008;
      vel[i * 3 + 1] = Math.random() * 0.015 + 0.005;
      vel[i * 3 + 2] = (Math.random() - 0.5) * 0.008;
    }
    return vel;
  }, []);

  useFrame((state) => {
    if (pointsRef.current) {
      const positionAttribute = pointsRef.current.geometry.attributes.position;
      const posArray = positionAttribute.array as Float32Array;
      
      for (let i = 0; i < particleCount; i++) {
        // Add sine wave motion
        const wave = Math.sin(state.clock.elapsedTime * 0.5 + i * 0.1) * 0.002;
        
        posArray[i * 3] += velocities[i * 3] + wave;
        posArray[i * 3 + 1] += velocities[i * 3 + 1];
        posArray[i * 3 + 2] += velocities[i * 3 + 2];
        
        // Reset particles
        if (posArray[i * 3 + 1] > 8) {
          posArray[i * 3 + 1] = -8;
          posArray[i * 3] = (Math.random() - 0.5) * 25;
          posArray[i * 3 + 2] = (Math.random() - 0.5) * 15;
        }
      }
      
      positionAttribute.needsUpdate = true;
      
      // Rotate the particle system slowly
      pointsRef.current.rotation.y = state.clock.elapsedTime * 0.02;
    }
  });

  return (
    <points ref={pointsRef}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[positions, 3]} />
        <bufferAttribute attach="attributes-color" args={[colors, 3]} />
        <bufferAttribute attach="attributes-size" args={[sizes, 1]} />
      </bufferGeometry>
      <pointsMaterial
        size={0.06}
        vertexColors
        transparent
        opacity={0.7}
        sizeAttenuation
        blending={THREE.AdditiveBlending}
        depthWrite={false}
      />
    </points>
  );
}

// Animated energy rings with pulsing effect
function EnergyRings() {
  const groupRef = useRef<THREE.Group>(null);
  const ringsRef = useRef<THREE.Mesh[]>([]);

  useFrame((state) => {
    if (groupRef.current) {
      groupRef.current.rotation.z = state.clock.elapsedTime * 0.08;
      
      // Pulse each ring
      ringsRef.current.forEach((ring, i) => {
        if (ring) {
          const scale = 1 + Math.sin(state.clock.elapsedTime * 0.5 + i * 0.5) * 0.05;
          ring.scale.set(scale, scale, 1);
          
          // Update opacity
          const material = ring.material as THREE.MeshBasicMaterial;
          material.opacity = (0.15 / (i + 1)) * (0.8 + Math.sin(state.clock.elapsedTime * 0.8 + i) * 0.2);
        }
      });
    }
  });

  return (
    <group ref={groupRef}>
      {[1, 1.8, 2.8, 4].map((radius, i) => (
        <mesh
          key={i}
          ref={(el) => { if (el) ringsRef.current[i] = el; }}
          rotation={[Math.PI / 2, 0, 0]}
          position={[0, -1.5, 0]}
        >
          <ringGeometry args={[radius, radius + 0.04, 128]} />
          <meshBasicMaterial
            color={i % 2 === 0 ? '#00ffff' : '#ff00ff'}
            transparent
            opacity={0.15 / (i + 1)}
            side={THREE.DoubleSide}
            blending={THREE.AdditiveBlending}
            depthWrite={false}
          />
        </mesh>
      ))}
    </group>
  );
}

// Holographic prism effect
function HolographicPrism() {
  const meshRef = useRef<THREE.Mesh>(null);
  const materialRef = useRef<THREE.ShaderMaterial>(null);

  const uniforms = useMemo(
    () => ({
      uTime: { value: 0 }
    }),
    []
  );

  const vertexShader = `
    varying vec3 vNormal;
    varying vec3 vPosition;
    
    void main() {
      vNormal = normalize(normalMatrix * normal);
      vPosition = position;
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
  `;

  const fragmentShader = `
    uniform float uTime;
    varying vec3 vNormal;
    varying vec3 vPosition;
    
    void main() {
      // Fresnel effect
      vec3 viewDir = normalize(cameraPosition - vPosition);
      float fresnel = pow(1.0 - abs(dot(vNormal, viewDir)), 3.0);
      
      // Rainbow shift based on angle
      float hue = vNormal.y * 0.5 + 0.5 + uTime * 0.1;
      vec3 color = vec3(
        sin(hue * 6.28) * 0.5 + 0.5,
        sin((hue + 0.33) * 6.28) * 0.5 + 0.5,
        sin((hue + 0.66) * 6.28) * 0.5 + 0.5
      );
      
      gl_FragColor = vec4(color * fresnel, fresnel * 0.3);
    }
  `;

  useFrame((state) => {
    if (meshRef.current && materialRef.current) {
      meshRef.current.rotation.y = state.clock.elapsedTime * 0.1;
      meshRef.current.rotation.x = Math.sin(state.clock.elapsedTime * 0.2) * 0.1;
      materialRef.current.uniforms.uTime.value = state.clock.elapsedTime;
    }
  });

  return (
    <mesh ref={meshRef} position={[0, 0, -5]} scale={[2, 2, 2]}>
      <icosahedronGeometry args={[1, 1]} />
      <shaderMaterial
        ref={materialRef}
        uniforms={uniforms}
        vertexShader={vertexShader}
        fragmentShader={fragmentShader}
        transparent
        side={THREE.DoubleSide}
        blending={THREE.AdditiveBlending}
        depthWrite={false}
      />
    </mesh>
  );
}

// Data streams effect
function DataStreams() {
  const linesRef = useRef<THREE.Group>(null);
  const lineCount = 20;

  const lines = useMemo(() => {
    return Array.from({ length: lineCount }, (_, i) => ({
      x: (Math.random() - 0.5) * 20,
      z: (Math.random() - 0.5) * 10 - 5,
      speed: Math.random() * 0.05 + 0.02,
      length: Math.random() * 3 + 1,
      delay: Math.random() * 10
    }));
  }, []);

  useFrame((state) => {
    if (linesRef.current) {
      linesRef.current.children.forEach((line, i) => {
        const data = lines[i];
        const y = ((state.clock.elapsedTime * data.speed + data.delay) % 2 - 1) * 10;
        line.position.y = y;
      });
    }
  });

  return (
    <group ref={linesRef}>
      {lines.map((line, i) => (
        <mesh key={i} position={[line.x, 0, line.z]}>
          <boxGeometry args={[0.02, line.length, 0.02]} />
          <meshBasicMaterial
            color={i % 2 === 0 ? '#00ffff' : '#ff00ff'}
            transparent
            opacity={0.3}
            blending={THREE.AdditiveBlending}
          />
        </mesh>
      ))}
    </group>
  );
}

// Main scene composition
function Scene() {
  return (
    <>
      <ambientLight intensity={0.05} />
      <HolographicGrid />
      <FloatingParticles />
      <EnergyRings />
      <HolographicPrism />
      <DataStreams />
      <fog attach="fog" args={['#000000', 5, 25]} />
    </>
  );
}

// Main component with overlay effects
export function HolographicBackground() {
  return (
    <div className="fixed inset-0 z-0">
      <Canvas
        camera={{ position: [0, 4, 12], fov: 55 }}
        gl={{ 
          antialias: true, 
          alpha: true,
          powerPreference: 'high-performance',
          stencil: false,
          depth: true
        }}
        dpr={[1, 1.5]}
      >
        <Scene />
      </Canvas>
      
      {/* Radial gradient overlay */}
      <div className="absolute inset-0 pointer-events-none"
        style={{
          background: `
            radial-gradient(ellipse at 50% 0%, rgba(0, 255, 255, 0.08) 0%, transparent 50%),
            radial-gradient(ellipse at 80% 80%, rgba(255, 0, 255, 0.05) 0%, transparent 40%),
            radial-gradient(ellipse at 20% 90%, rgba(0, 128, 255, 0.05) 0%, transparent 40%),
            linear-gradient(to bottom, rgba(0, 0, 0, 0.3) 0%, transparent 20%, transparent 80%, rgba(0, 0, 0, 0.8) 100%)
          `
        }}
      />
      
      {/* Scan lines overlay */}
      <div className="absolute inset-0 scan-lines pointer-events-none opacity-40" />
      
      {/* Vignette */}
      <div className="absolute inset-0 pointer-events-none"
        style={{
          background: 'radial-gradient(ellipse at center, transparent 0%, rgba(0, 0, 0, 0.4) 100%)'
        }}
      />
      
      {/* Corner accents */}
      <div className="absolute top-0 left-0 w-32 h-32 pointer-events-none"
        style={{
          borderLeft: '2px solid rgba(0, 255, 255, 0.3)',
          borderTop: '2px solid rgba(0, 255, 255, 0.3)',
          background: 'linear-gradient(135deg, rgba(0, 255, 255, 0.05) 0%, transparent 50%)'
        }}
      />
      <div className="absolute top-0 right-0 w-32 h-32 pointer-events-none"
        style={{
          borderRight: '2px solid rgba(255, 0, 255, 0.3)',
          borderTop: '2px solid rgba(255, 0, 255, 0.3)',
          background: 'linear-gradient(-135deg, rgba(255, 0, 255, 0.05) 0%, transparent 50%)'
        }}
      />
      <div className="absolute bottom-0 left-0 w-32 h-32 pointer-events-none"
        style={{
          borderLeft: '2px solid rgba(0, 255, 255, 0.3)',
          borderBottom: '2px solid rgba(0, 255, 255, 0.3)',
          background: 'linear-gradient(45deg, rgba(0, 255, 255, 0.05) 0%, transparent 50%)'
        }}
      />
      <div className="absolute bottom-0 right-0 w-32 h-32 pointer-events-none"
        style={{
          borderRight: '2px solid rgba(255, 0, 255, 0.3)',
          borderBottom: '2px solid rgba(255, 0, 255, 0.3)',
          background: 'linear-gradient(-45deg, rgba(255, 0, 255, 0.05) 0%, transparent 50%)'
        }}
      />
    </div>
  );
}
