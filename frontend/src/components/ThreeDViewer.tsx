import React, { useRef, useEffect, useState } from 'react';
import * as THREE from 'three';

interface ThreeDViewerProps {
  imageUrl?: string;
  width?: number;
  height?: number;
  autoRotate?: boolean;
  enableZoom?: boolean;
}

const ThreeDViewer: React.FC<ThreeDViewerProps> = ({
  imageUrl = '',
  width = 400,
  height = 500,
  autoRotate = true,
  enableZoom = true,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [rotation, setRotation] = useState(0);
  const [scale, setScale] = useState(1);
  const scaleRef = useRef(scale);
  const rotationDisplayRef = useRef<HTMLSpanElement>(null);
  const isDragging = useRef(false);
  const previousMousePosition = useRef({ x: 0, y: 0 });

  // Keep scaleRef in sync with scale state for use inside the animation loop
  useEffect(() => {
    scaleRef.current = scale;
  }, [scale]);

  useEffect(() => {
    if (!containerRef.current) return;

    // Scene setup
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xfafafa);

    // Camera
    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
    camera.position.z = 4;

    // Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    containerRef.current.appendChild(renderer.domElement);

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.9);
    scene.add(ambientLight);

    const mainLight = new THREE.DirectionalLight(0xffffff, 1.2);
    mainLight.position.set(5, 5, 5);
    mainLight.castShadow = true;
    scene.add(mainLight);

    const fillLight = new THREE.DirectionalLight(0xffffff, 0.4);
    fillLight.position.set(-5, 0, -5);
    scene.add(fillLight);

    const rimLight = new THREE.DirectionalLight(0x88ccff, 0.6);
    rimLight.position.set(0, 5, -5);
    scene.add(rimLight);

    // Create group for rotation
    const group = new THREE.Group();

    // Create elegant display stand
    const standGeometry = new THREE.CylinderGeometry(0.3, 0.5, 0.1, 32);
    const standMaterial = new THREE.MeshStandardMaterial({
      color: 0x333333,
      metalness: 0.8,
      roughness: 0.2,
    });
    const stand = new THREE.Mesh(standGeometry, standMaterial);
    stand.position.y = -1.8;
    stand.receiveShadow = true;
    group.add(stand);

    // Create pole
    const poleGeometry = new THREE.CylinderGeometry(0.03, 0.03, 0.8, 16);
    const poleMaterial = new THREE.MeshStandardMaterial({
      color: 0xcccccc,
      metalness: 0.9,
      roughness: 0.1,
    });
    const pole = new THREE.Mesh(poleGeometry, poleMaterial);
    pole.position.y = -1.4;
    pole.castShadow = true;
    group.add(pole);

    // Create hanger (arc shape using torus)
    const hangerGeometry = new THREE.TorusGeometry(0.25, 0.015, 8, 32, Math.PI);
    const hangerMaterial = new THREE.MeshStandardMaterial({
      color: 0xd4af37, // gold color
      metalness: 0.9,
      roughness: 0.1,
    });
    const hanger = new THREE.Mesh(hangerGeometry, hangerMaterial);
    hanger.position.y = -0.95;
    hanger.rotation.z = Math.PI;
    hanger.castShadow = true;
    group.add(hanger);

    // Load clothing image as texture
    let clothingMesh: THREE.Mesh | null = null;

    const loadTexture = () => {
      if (!imageUrl) {
        setIsLoading(false);
        return;
      }

      const textureLoader = new THREE.TextureLoader();
      textureLoader.load(
        imageUrl,
        (texture) => {
          texture.colorSpace = THREE.SRGBColorSpace;

          // Calculate aspect ratio to fit the image properly
          const aspectRatio = texture.image ? texture.image.width / texture.image.height : 0.8;
          const clothingWidth = 1.6;
          const clothingHeight = clothingWidth / aspectRatio;

          // Create plane for clothing
          const clothingGeometry = new THREE.PlaneGeometry(clothingWidth, clothingHeight);
          const clothingMaterial = new THREE.MeshStandardMaterial({
            map: texture,
            transparent: true,
            side: THREE.DoubleSide,
            alphaTest: 0.1,
          });

          clothingMesh = new THREE.Mesh(clothingGeometry, clothingMaterial);
          clothingMesh.position.y = -0.2;
          clothingMesh.castShadow = true;
          group.add(clothingMesh);

          // Add subtle shadow plane
          const shadowGeometry = new THREE.PlaneGeometry(2, 2.5);
          const shadowMaterial = new THREE.ShadowMaterial({ opacity: 0.15 });
          const shadowPlane = new THREE.Mesh(shadowGeometry, shadowMaterial);
          shadowPlane.position.y = -1.75;
          shadowPlane.receiveShadow = true;
          group.add(shadowPlane);

          setIsLoading(false);
        },
        undefined,
        (err) => {
          console.error('Error loading texture:', err);
          setError('图片加载失败');
          setIsLoading(false);
        }
      );
    };

    loadTexture();

    // Add group to scene
    scene.add(group);

    // Animation
    let animationId: number;
    const animate = () => {
      animationId = requestAnimationFrame(animate);

      if (autoRotate && !isDragging.current) {
        group.rotation.y += 0.008;
      }

      const s = scaleRef.current;
      group.scale.set(s, s, s);
      // Update rotation display via DOM ref to avoid per-frame React re-renders
      if (rotationDisplayRef.current) {
        rotationDisplayRef.current.textContent = `${(group.rotation.y * 180 / Math.PI).toFixed(0)}°`;
      }
      renderer.render(scene, camera);
    };
    animate();

    // Mouse controls
    const handleMouseDown = (e: MouseEvent) => {
      isDragging.current = true;
      previousMousePosition.current = { x: e.clientX, y: e.clientY };
    };

    const handleMouseMove = (e: MouseEvent) => {
      if (!isDragging.current) return;

      const deltaX = e.clientX - previousMousePosition.current.x;
      group.rotation.y += deltaX * 0.01;
      if (rotationDisplayRef.current) {
        rotationDisplayRef.current.textContent = `${(group.rotation.y * 180 / Math.PI).toFixed(0)}°`;
      }
      previousMousePosition.current = { x: e.clientX, y: e.clientY };
    };

    const handleMouseUp = () => {
      isDragging.current = false;
    };

    const handleWheel = (e: WheelEvent) => {
      if (!enableZoom) return;
      e.preventDefault();
      const delta = e.deltaY > 0 ? 0.9 : 1.1;
      setScale((prev) => Math.max(0.5, Math.min(2.5, prev * delta)));
    };

    const container = containerRef.current;
    container.addEventListener('mousedown', handleMouseDown);
    container.addEventListener('mousemove', handleMouseMove);
    container.addEventListener('mouseup', handleMouseUp);
    container.addEventListener('mouseleave', handleMouseUp);
    container.addEventListener('wheel', handleWheel, { passive: false });

    // Double click to reset
    const handleDoubleClick = () => {
      group.rotation.set(0, 0, 0);
      setScale(1);
      setRotation(0);
    };
    container.addEventListener('dblclick', handleDoubleClick);

    // Cleanup
    return () => {
      cancelAnimationFrame(animationId);
      container.removeEventListener('mousedown', handleMouseDown);
      container.removeEventListener('mousemove', handleMouseMove);
      container.removeEventListener('mouseup', handleMouseUp);
      container.removeEventListener('mouseleave', handleMouseUp);
      container.removeEventListener('wheel', handleWheel);
      container.removeEventListener('dblclick', handleDoubleClick);
      // Dispose all Three.js geometries, materials, and textures in the scene
      scene.traverse((obj: any) => {
        if (obj.isMesh) {
          obj.geometry?.dispose();
          const material = obj.material;
          if (Array.isArray(material)) {
            material.forEach((m: THREE.Material) => {
              const mat = m as any;
              mat.map?.dispose?.();
              m.dispose();
            });
          } else if (material) {
            const mat = material as any;
            mat.map?.dispose?.();
            material.dispose();
          }
        }
      });
      container.removeChild(renderer.domElement);
      renderer.dispose();
    };
  }, [imageUrl, width, height, autoRotate, enableZoom]);

  return (
    <div style={{ position: 'relative', width, height }}>
      <div
        ref={containerRef}
        style={{
          width: '100%',
          height: '100%',
          cursor: isDragging.current ? 'grabbing' : 'grab',
          borderRadius: '12px',
          overflow: 'hidden',
          boxShadow: '0 8px 32px rgba(0,0,0,0.1)',
          background: 'linear-gradient(180deg, #fafafa 0%, #f0f0f0 100%)',
        }}
      />

      {isLoading && (
        <div
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            width: '100%',
            height: '100%',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            background: 'linear-gradient(180deg, #fafafa 0%, #f0f0f0 100%)',
          }}
        >
          <div
            style={{
              width: 40,
              height: 40,
              border: '3px solid #e0e0e0',
              borderTopColor: '#1890ff',
              borderRadius: '50%',
              animation: 'spin 1s linear infinite',
            }}
          />
          <span style={{ marginTop: 12, color: '#666' }}>加载中...</span>
        </div>
      )}

      {error && (
        <div
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            width: '100%',
            height: '100%',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            background: '#fff0f0',
            color: '#f5222d',
          }}
        >
          {error}
        </div>
      )}

      {/* Controls overlay */}
      <div
        style={{
          position: 'absolute',
          bottom: '12px',
          left: '50%',
          transform: 'translateX(-50%)',
          display: 'flex',
          gap: '12px',
          background: 'rgba(255,255,255,0.95)',
          padding: '8px 16px',
          borderRadius: '20px',
          boxShadow: '0 2px 12px rgba(0,0,0,0.1)',
        }}
      >
        <span ref={rotationDisplayRef} style={{ color: '#333', fontSize: '12px', fontWeight: 500 }}>
          {(rotation * 180 / Math.PI).toFixed(0)}°
        </span>
        <span style={{ color: '#888', fontSize: '12px' }}>|</span>
        <span style={{ color: '#333', fontSize: '12px', fontWeight: 500 }}>
          {(scale * 100).toFixed(0)}%
        </span>
      </div>

      {/* Hint */}
      <div
        style={{
          position: 'absolute',
          top: '10px',
          right: '10px',
          background: 'rgba(0,0,0,0.6)',
          padding: '4px 10px',
          borderRadius: '12px',
          fontSize: '11px',
          color: '#fff',
        }}
      >
        拖动旋转 · 滚轮缩放 · 双击重置
      </div>

      <style>
        {`
          @keyframes spin {
            to { transform: rotate(360deg); }
          }
        `}
      </style>
    </div>
  );
};

export default ThreeDViewer;
