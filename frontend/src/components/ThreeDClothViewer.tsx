import React, { useRef, useEffect, useState, useCallback } from 'react'
import { Spin, Radio } from 'antd'
import * as THREE from 'three'
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js'

interface ClothingSet {
  top?: string
  bottom?: string
  dress?: string
  outerwear?: string
}

interface Props {
  clothing: ClothingSet
  width?: number
  height?: number
  autoRotate?: boolean
  gender?: 'male' | 'female'
  onGenderChange?: (gender: 'male' | 'female') => void
}

// 模型URL - 都用VRM女性模型，男性只是缩小一点
const MODEL_URLS = {
  female: 'https://raw.githubusercontent.com/pixiv/three-vrm/dev/packages/three-vrm/examples/models/VRM1_Constraint_Twist_Sample.vrm',
  male: 'https://raw.githubusercontent.com/pixiv/three-vrm/dev/packages/three-vrm/examples/models/VRM1_Constraint_Twist_Sample.vrm',
}

const ThreeDClothViewer: React.FC<Props> = ({
  clothing,
  width = 400,
  height = 550,
  autoRotate = true,
  gender = 'female',
  onGenderChange,
}) => {
  const containerRef = useRef<HTMLDivElement>(null)
  const sceneRef = useRef<THREE.Scene | null>(null)
  const cameraRef = useRef<THREE.Camera | null>(null)
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null)
  const modelRef = useRef<THREE.Object3D | null>(null)
  const clothingGroupRef = useRef<THREE.Group | null>(null)
  const mixerRef = useRef<THREE.AnimationMixer | null>(null)
  const clockRef = useRef<THREE.Clock | null>(null)
  const animationIdRef = useRef<number>(0)

  const [isLoading, setIsLoading] = useState(true)
  const [loadProgress, setLoadProgress] = useState(0)
  const isDraggingRef = useRef(false)
  const previousMouseRef = useRef({ x: 0, y: 0 })
  const targetRotationRef = useRef(0)
  const currentRotationRef = useRef(0)

  const cleanup = useCallback(() => {
    if (animationIdRef.current) {
      cancelAnimationFrame(animationIdRef.current)
    }
    // Dispose the AnimationMixer
    if (mixerRef.current) {
      mixerRef.current.uncacheRoot(mixerRef.current.getRoot())
      mixerRef.current = null
    }
    // Dispose old model geometries/materials/textures before removing
    const disposeObject = (obj: THREE.Object3D | null) => {
      if (!obj) return
      obj.traverse((node: any) => {
        if (node.isMesh) {
          node.geometry?.dispose()
          const material = node.material
          if (Array.isArray(material)) {
            material.forEach((m: any) => {
              m.map?.dispose?.()
              m.dispose()
            })
          } else if (material) {
            material.map?.dispose?.()
            material.dispose()
          }
        }
      })
    }
    if (clothingGroupRef.current && sceneRef.current) {
      sceneRef.current.remove(clothingGroupRef.current)
      disposeObject(clothingGroupRef.current)
      clothingGroupRef.current = null
    }
    if (modelRef.current && sceneRef.current) {
      sceneRef.current.remove(modelRef.current)
      disposeObject(modelRef.current)
      modelRef.current = null
    }
    if (rendererRef.current && containerRef.current) {
      if (rendererRef.current.domElement.parentNode === containerRef.current) {
        containerRef.current.removeChild(rendererRef.current.domElement)
      }
      rendererRef.current.dispose()
      rendererRef.current = null
    }
  }, [])

  const loadModel = useCallback((url: string, isMale: boolean, cancelled: () => boolean) => {
    if (!sceneRef.current) return

    setIsLoading(true)
    setLoadProgress(0)

    // Remove and dispose old model
    if (modelRef.current && sceneRef.current) {
      sceneRef.current.remove(modelRef.current)
      modelRef.current.traverse((obj: any) => {
        if (obj.isMesh) {
          obj.geometry?.dispose()
          const material = obj.material
          if (Array.isArray(material)) {
            material.forEach((m: any) => {
              m.map?.dispose?.()
              m.dispose()
            })
          } else if (material) {
            material.map?.dispose?.()
            material.dispose()
          }
        }
      })
      modelRef.current = null
    }
    // Dispose old mixer
    if (mixerRef.current) {
      mixerRef.current.uncacheRoot(mixerRef.current.getRoot())
      mixerRef.current = null
    }

    console.log('Loading model:', url, 'isMale:', isMale)

    const loader = new GLTFLoader()

    loader.load(
      url,
      (gltf) => {
        // Race-condition guard: ignore result if the effect was cancelled
        if (cancelled()) {
          // Dispose the late-arriving model to avoid leaking it
          gltf.scene.traverse((obj: any) => {
            if (obj.isMesh) {
              obj.geometry?.dispose()
              const material = obj.material
              if (Array.isArray(material)) {
                material.forEach((m: any) => {
                  m.map?.dispose?.()
                  m.dispose()
                })
              } else if (material) {
                material.map?.dispose?.()
                material.dispose()
              }
            }
          })
          return
        }
        console.log('Model loaded successfully', isMale ? 'Male' : 'Female')
        setLoadProgress(90)
        const model = gltf.scene

        // 根据模型类型调整大小和位置
        if (isMale) {
          // VRM 男性 - 缩小一点
          model.scale.set(0.9, 0.9, 0.9)
          model.position.y = -0.7
        } else {
          // VRM 女性
          model.scale.set(1.0, 1.0, 1.0)
          model.position.y = -0.65
        }

        console.log('Model position:', model.position.y, 'scale:', model.scale.y)

        // Enable shadows
        model.traverse((obj: any) => {
          if (obj.isMesh) {
            obj.castShadow = true
            obj.receiveShadow = true
          }
        })

        modelRef.current = model
        sceneRef.current?.add(model)

        // Play animation if available (for Astronaut)
        if (gltf.animations && gltf.animations.length > 0 && isMale) {
          const mixer = new THREE.AnimationMixer(model)
          const action = mixer.clipAction(gltf.animations[0])
          action.play()
          mixerRef.current = mixer
        }

        setLoadProgress(100)
        setIsLoading(false)
      },
      (xhr) => {
        if (xhr.lengthComputable && xhr.total > 0) {
          const percent = Math.round((xhr.loaded / xhr.total) * 90)
          setLoadProgress(Math.min(percent, 95))
        }
      },
      (error) => {
        console.error('Error loading model:', error)
        setIsLoading(false)
      }
    )
  }, [])

  useEffect(() => {
    if (!containerRef.current) return

    // Scene
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0xf0f0f0)
    sceneRef.current = scene

    // Camera
    const camera = new THREE.PerspectiveCamera(30, width / height, 0.1, 1000)
    camera.position.set(0, 1.2, 3.5)
    camera.lookAt(0, 0.2, 0)
    cameraRef.current = camera

    // Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true })
    renderer.setSize(width, height)
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.shadowMap.enabled = true
    renderer.shadowMap.type = THREE.PCFSoftShadowMap
    renderer.outputColorSpace = THREE.SRGBColorSpace
    containerRef.current.appendChild(renderer.domElement)
    rendererRef.current = renderer

    // Lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.8)
    scene.add(ambientLight)

    const mainLight = new THREE.DirectionalLight(0xffffff, 1.0)
    mainLight.position.set(3, 10, 5)
    mainLight.castShadow = true
    scene.add(mainLight)

    const fillLight = new THREE.DirectionalLight(0xffffff, 0.4)
    fillLight.position.set(-3, 5, -3)
    scene.add(fillLight)

    // Ground
    const groundGeo = new THREE.CircleGeometry(2.5, 64)
    const groundMat = new THREE.ShadowMaterial({ opacity: 0.1 })
    const ground = new THREE.Mesh(groundGeo, groundMat)
    ground.rotation.x = -Math.PI / 2
    ground.position.y = -0.01
    ground.receiveShadow = true
    scene.add(ground)

    // Load initial model
    const isMale = gender === 'male'
    let cancelled = false
    loadModel(MODEL_URLS[gender], isMale, () => cancelled)

    // Animation loop
    const clock = new THREE.Clock()
    clockRef.current = clock
    const animate = () => {
      animationIdRef.current = requestAnimationFrame(animate)

      // Smooth rotation
      if (autoRotate && !isDraggingRef.current) {
        targetRotationRef.current += 0.006
      }
      currentRotationRef.current += (targetRotationRef.current - currentRotationRef.current) * 0.1

      if (modelRef.current) {
        modelRef.current.rotation.y = currentRotationRef.current
      }
      if (clothingGroupRef.current) {
        clothingGroupRef.current.rotation.y = currentRotationRef.current
      }

      // Advance animations
      if (mixerRef.current && clockRef.current) {
        mixerRef.current.update(clockRef.current.getDelta())
      }

      renderer.render(scene, camera)
    }
    animate()

    // Mouse controls
    const handleMouseDown = (e: MouseEvent) => {
      isDraggingRef.current = true
      previousMouseRef.current = { x: e.clientX, y: e.clientY }
    }

    const handleMouseMove = (e: MouseEvent) => {
      if (!isDraggingRef.current) return
      const deltaX = e.clientX - previousMouseRef.current.x
      targetRotationRef.current += deltaX * 0.02
      previousMouseRef.current = { x: e.clientX, y: e.clientY }
    }

    const handleMouseUp = () => {
      isDraggingRef.current = false
    }

    const handleDoubleClick = () => {
      targetRotationRef.current = 0
      currentRotationRef.current = 0
    }

    const container = containerRef.current
    container.addEventListener('mousedown', handleMouseDown)
    container.addEventListener('mousemove', handleMouseMove)
    container.addEventListener('mouseup', handleMouseUp)
    container.addEventListener('mouseleave', handleMouseUp)
    container.addEventListener('dblclick', handleDoubleClick)

    return () => {
      cancelled = true
      cleanup()
      container.removeEventListener('mousedown', handleMouseDown)
      container.removeEventListener('mousemove', handleMouseMove)
      container.removeEventListener('mouseup', handleMouseUp)
      container.removeEventListener('mouseleave', handleMouseUp)
      container.removeEventListener('dblclick', handleDoubleClick)
    }
  }, [width, height, autoRotate, gender, cleanup, loadModel])

  // Update clothing when changed
  useEffect(() => {
    if (!sceneRef.current) return

    // Remove old clothing
    if (clothingGroupRef.current) {
      sceneRef.current.remove(clothingGroupRef.current)
      clothingGroupRef.current.traverse((obj: any) => {
        if (obj.isMesh) {
          obj.geometry?.dispose()
          if (Array.isArray(obj.material)) {
            obj.material.forEach((m: any) => m.dispose())
          } else if (obj.material) {
            obj.material.dispose()
          }
        }
      })
    }

    if (Object.keys(clothing).length === 0) return

    // Create clothing group
    const clothingGroup = new THREE.Group()
    const textureLoader = new THREE.TextureLoader()

    const createClothingMesh = (
      imgUrl: string,
      geometry: THREE.BufferGeometry,
      position: THREE.Vector3,
      rotation?: THREE.Euler
    ) => {
      textureLoader.load(
        imgUrl,
        (texture) => {
          texture.colorSpace = THREE.SRGBColorSpace
          const material = new THREE.MeshStandardMaterial({
            map: texture,
            transparent: true,
            side: THREE.DoubleSide,
            roughness: 0.8,
          })
          const mesh = new THREE.Mesh(geometry, material)
          mesh.position.copy(position)
          if (rotation) mesh.rotation.copy(rotation)
          mesh.castShadow = true
          clothingGroup.add(mesh)
        },
        undefined,
        () => {
          // Fallback colored material
          const material = new THREE.MeshStandardMaterial({
            color: 0x888888,
            transparent: true,
            side: THREE.DoubleSide,
            roughness: 0.8,
          })
          const mesh = new THREE.Mesh(geometry, material)
          mesh.position.copy(position)
          if (rotation) mesh.rotation.copy(rotation)
          mesh.castShadow = true
          clothingGroup.add(mesh)
        }
      )
    }

    // Top (shirt) - 根据性别调整位置
    if (clothing.top) {
      const isMale = gender === 'male'
      const topGeo = new THREE.BoxGeometry(0.35, 0.3, 0.2)
      const topPos = new THREE.Vector3(0, isMale ? 0.3 : 0.25, 0)
      createClothingMesh(clothing.top, topGeo, topPos)

      const sleeveGeo = new THREE.CylinderGeometry(0.06, 0.08, 0.18, 12)
      const leftSleevePos = new THREE.Vector3(isMale ? -0.28 : -0.24, isMale ? 0.35 : 0.3, 0)
      const leftSleeveRot = new THREE.Euler(0, 0, 0.5)
      createClothingMesh(clothing.top, sleeveGeo, leftSleevePos, leftSleeveRot)

      const rightSleevePos = new THREE.Vector3(isMale ? 0.28 : 0.24, isMale ? 0.35 : 0.3, 0)
      const rightSleeveRot = new THREE.Euler(0, 0, -0.5)
      createClothingMesh(clothing.top, sleeveGeo, rightSleevePos, rightSleeveRot)
    }

    // Bottom
    if (clothing.bottom) {
      const isMale = gender === 'male'
      const bottomGeo = new THREE.BoxGeometry(0.3, 0.4, 0.2)
      const bottomPos = new THREE.Vector3(0, isMale ? -0.25 : -0.25, 0)
      createClothingMesh(clothing.bottom, bottomGeo, bottomPos)
    }

    // Dress
    if (clothing.dress) {
      const dressGeo = new THREE.CylinderGeometry(0.18, 0.3, 0.6, 12)
      const dressPos = new THREE.Vector3(0, 0, 0)
      createClothingMesh(clothing.dress, dressGeo, dressPos)
    }

    // Outerwear
    if (clothing.outerwear) {
      const isMale = gender === 'male'
      const outerGeo = new THREE.BoxGeometry(0.4, 0.32, 0.25)
      const outerPos = new THREE.Vector3(0, isMale ? 0.25 : 0.22, 0)
      createClothingMesh(clothing.outerwear, outerGeo, outerPos)
    }

    clothingGroupRef.current = clothingGroup
    sceneRef.current.add(clothingGroup)
  }, [clothing, gender])

  return (
    <div
      style={{
        width,
        height,
        borderRadius: 20,
        overflow: 'hidden',
        boxShadow: '0 12px 40px rgba(0,0,0,0.15)',
        background: 'linear-gradient(180deg, #fafafa 0%, #f0f0f0 100%)',
        position: 'relative',
      }}
    >
      <div
        ref={containerRef}
        style={{
          width: '100%',
          height: '100%',
          cursor: isDraggingRef.current ? 'grabbing' : 'grab',
        }}
      />

      {/* 性别切换 */}
      <div
        style={{
          position: 'absolute',
          top: 16,
          left: 16,
          background: 'rgba(255,255,255,0.95)',
          padding: '8px 12px',
          borderRadius: 20,
          boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
          zIndex: 10,
        }}
      >
        <Radio.Group
          value={gender}
          onChange={(e) => onGenderChange?.(e.target.value)}
          buttonStyle="solid"
          size="small"
        >
          <Radio.Button value="female">👩 女</Radio.Button>
          <Radio.Button value="male">👨 男</Radio.Button>
        </Radio.Group>
      </div>

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
            background: 'rgba(250,250,250,0.98)',
          }}
        >
          <Spin size="large" />
          <div
            style={{
              marginTop: 20,
              width: 200,
              height: 6,
              background: '#e0e0e0',
              borderRadius: 3,
              overflow: 'hidden',
            }}
          >
            <div
              style={{
                width: `${loadProgress}%`,
                height: '100%',
                background: 'linear-gradient(90deg, #1890ff, #40a9ff)',
                borderRadius: 3,
                transition: 'width 0.3s',
              }}
            />
          </div>
          <div style={{ marginTop: 12, color: '#666', fontSize: 14 }}>
            加载中... {loadProgress}%
          </div>
        </div>
      )}

      <div
        style={{
          position: 'absolute',
          bottom: 16,
          left: '50%',
          transform: 'translateX(-50%)',
          background: 'rgba(255,255,255,0.95)',
          padding: '8px 20px',
          borderRadius: 20,
          fontSize: 13,
          color: '#666',
          boxShadow: '0 4px 12px rgba(0,0,0,0.1)',
        }}
      >
        拖动旋转 · 双击重置
      </div>
    </div>
  )
}

export default ThreeDClothViewer
