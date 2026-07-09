import React, { useRef, useEffect, useState, useCallback } from 'react'
import { Spin, Radio } from 'antd'
import * as THREE from 'three'
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js'

// === Programmatic clothing geometry (used when no GLB provided) ===
// 用 Three.js 几何体程序化生成精细的 T 恤 + 裤子(比 BoxGeometry 强 10 倍)
// 仅生成 1 套 T 恤 + 1 条裤子(固定,用于演示 + 论文截图)
function buildProgrammaticClothing(
  category: 'top' | 'bottom',
  color: number = 0x4a90e2,
): THREE.Group {
  const group = new THREE.Group()
  const material = new THREE.MeshStandardMaterial({
    color,
    roughness: 0.75,
    metalness: 0.0,
    side: THREE.DoubleSide,
  })
  if (category === 'top') {
    // T 恤:身体 = 略收口的圆柱(用 LatheGeometry 沿垂直轴旋转一个轮廓,模拟衣身)
    // + 两个袖子 = 圆柱
    // 衣身:用 LatheGeometry 模拟一个轮廓(衣领到衣摆)
    const points: THREE.Vector2[] = []
    // 上窄下宽,模拟 T 恤轮廓(从 y=0.3 衣领到 y=-0.3 衣摆)
    points.push(new THREE.Vector2(0.05, 0.32))  // 衣领
    points.push(new THREE.Vector2(0.18, 0.28))  // 肩
    points.push(new THREE.Vector2(0.22, 0.10))  // 胸
    points.push(new THREE.Vector2(0.20, -0.10)) // 腰
    points.push(new THREE.Vector2(0.24, -0.30)) // 衣摆(略宽)
    const body = new THREE.Mesh(
      new THREE.LatheGeometry(points, 24),
      material,
    )
    body.castShadow = true
    group.add(body)
    // 衣领开口(顶部小圆环)
    const collarRing = new THREE.Mesh(
      new THREE.TorusGeometry(0.08, 0.015, 8, 24),
      new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.7 }),
    )
    collarRing.position.y = 0.32
    collarRing.rotation.x = Math.PI / 2
    group.add(collarRing)
    // 袖子:两个略斜的圆柱
    const sleeveGeo = new THREE.CylinderGeometry(0.06, 0.07, 0.28, 12)
    const sleeveL = new THREE.Mesh(sleeveGeo, material)
    sleeveL.position.set(-0.26, 0.18, 0)
    sleeveL.rotation.z = 0.3
    sleeveL.castShadow = true
    group.add(sleeveL)
    const sleeveR = new THREE.Mesh(sleeveGeo, material)
    sleeveR.position.set(0.26, 0.18, 0)
    sleeveR.rotation.z = -0.3
    sleeveR.castShadow = true
    group.add(sleeveR)
    // 位置:对齐 VRM 躯干
    group.position.set(0, 0.05, 0)
  } else if (category === 'bottom') {
    // 裤子:两个分腿的圆柱 + 腰带
    // 腰带
    const belt = new THREE.Mesh(
      new THREE.CylinderGeometry(0.20, 0.20, 0.05, 16),
      new THREE.MeshStandardMaterial({ color: 0x2c3e50, roughness: 0.6 }),
    )
    belt.position.set(0, -0.30, 0)
    belt.castShadow = true
    group.add(belt)
    // 腿:两个圆柱(略张开)
    const legGeo = new THREE.CylinderGeometry(0.08, 0.07, 0.45, 12)
    const legL = new THREE.Mesh(legGeo, material)
    legL.position.set(-0.07, -0.55, 0)
    legL.castShadow = true
    group.add(legL)
    const legR = new THREE.Mesh(legGeo, material)
    legR.position.set(0.07, -0.55, 0)
    legR.castShadow = true
    group.add(legR)
    // 裤脚
    const cuffGeo = new THREE.CylinderGeometry(0.085, 0.085, 0.04, 12)
    const cuffL = new THREE.Mesh(cuffGeo, material)
    cuffL.position.set(-0.07, -0.78, 0)
    group.add(cuffL)
    const cuffR = new THREE.Mesh(cuffGeo, material)
    cuffR.position.set(0.07, -0.78, 0)
    group.add(cuffR)
    // 位置:对齐 VRM 腿部
    group.position.set(0, 0.0, 0)
  }
  return group
}

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

// VRM 模型 URL - 本地化,加载快且不依赖外网
// (原 GitHub raw URL 慢且易失败,改用 public/models/ 本地资源)
const MODEL_URLS = {
  female: '/models/vrm_sample.vrm',
  male: '/models/vrm_sample.vrm',
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

    // Create clothing group (支持 3 种来源:程序化生成 / GLB 加载 / 老图片贴图)
    const clothingGroup = new THREE.Group()
    const glbLoader = new GLTFLoader()

    // 判断 clothing 值类型
    const isGlb = (url: string) => url.endsWith('.glb') || url.endsWith('.gltf')
    const isProgrammatic = (url: string) => url.startsWith('__programmatic_')

    const loadClothingGlb = (glbUrl: string, category: string) => {
      glbLoader.load(
        glbUrl,
        (gltf) => {
          const clothModel = gltf.scene
          // 启用阴影 + 调整比例(根据人偶身材,约 0.8-1.0)
          clothModel.scale.set(1.0, 1.0, 1.0)
          clothModel.traverse((obj: any) => {
            if (obj.isMesh) {
              obj.castShadow = true
              obj.receiveShadow = true
              // 提升材质质感
              if (obj.material) {
                if (Array.isArray(obj.material)) {
                  obj.material.forEach((m: any) => {
                    m.roughness = 0.7
                    m.metalness = 0.0
                  })
                } else {
                  obj.material.roughness = 0.7
                  obj.material.metalness = 0.0
                }
              }
            }
          })
          // 类别位置微调(对齐人偶的不同身体部位)
          const isMale = gender === 'male'
          const posMap: Record<string, THREE.Vector3> = {
            top: new THREE.Vector3(0, isMale ? 0.3 : 0.25, 0),
            bottom: new THREE.Vector3(0, -0.25, 0),
            dress: new THREE.Vector3(0, 0, 0),
            outerwear: new THREE.Vector3(0, isMale ? 0.25 : 0.22, 0),
          }
          if (posMap[category]) {
            clothModel.position.copy(posMap[category])
          }
          clothingGroup.add(clothModel)
        },
        undefined,
        (err) => {
          console.error(`[clothing ${category}] GLB 加载失败:`, glbUrl, err)
          // 加载失败:占位 BoxGeometry + 灰色(避免空白)
          const fallback = new THREE.Mesh(
            new THREE.BoxGeometry(0.4, 0.4, 0.2),
            new THREE.MeshStandardMaterial({ color: 0xcccccc, roughness: 0.8 })
          )
          fallback.castShadow = true
          clothingGroup.add(fallback)
        }
      )
    }

    // 按类别加载(优先级:程序化生成 / GLB / 老图片贴图)
    ;(['top', 'bottom', 'dress', 'outerwear'] as const).forEach((cat) => {
      const url = clothing[cat]
      if (!url) return
      if (isProgrammatic(url)) {
        // 程序化生成(默认演示用,无外部资源依赖)
        const color = url.includes('pants') ? 0x2c3e50 : 0x4a90e2
        const clothModel = buildProgrammaticClothing(
          (cat === 'top' || cat === 'outerwear') ? 'top' : 'bottom',
          color,
        )
        clothingGroup.add(clothModel)
      } else if (isGlb(url)) {
        loadClothingGlb(url, cat)
      } else {
        // 老格式(图片 URL)用 BoxGeometry + 贴图,作为 fallback
        const isMale = gender === 'male'
        const geoMap: Record<string, THREE.BufferGeometry> = {
          top: new THREE.BoxGeometry(0.35, 0.3, 0.2),
          bottom: new THREE.BoxGeometry(0.3, 0.4, 0.2),
          dress: new THREE.CylinderGeometry(0.18, 0.3, 0.6, 12),
          outerwear: new THREE.BoxGeometry(0.4, 0.32, 0.25),
        }
        const posMap: Record<string, THREE.Vector3> = {
          top: new THREE.Vector3(0, isMale ? 0.3 : 0.25, 0),
          bottom: new THREE.Vector3(0, -0.25, 0),
          dress: new THREE.Vector3(0, 0, 0),
          outerwear: new THREE.Vector3(0, isMale ? 0.25 : 0.22, 0),
        }
        const texLoader = new THREE.TextureLoader()
        texLoader.load(
          url,
          (tex) => {
            tex.colorSpace = THREE.SRGBColorSpace
            const mat = new THREE.MeshStandardMaterial({
              map: tex, transparent: true, side: THREE.DoubleSide, roughness: 0.8,
            })
            const mesh = new THREE.Mesh(geoMap[cat], mat)
            mesh.position.copy(posMap[cat])
            mesh.castShadow = true
            clothingGroup.add(mesh)
          },
          undefined,
          () => {
            const mat = new THREE.MeshStandardMaterial({
              color: 0x888888, transparent: true, side: THREE.DoubleSide, roughness: 0.8,
            })
            const mesh = new THREE.Mesh(geoMap[cat], mat)
            mesh.position.copy(posMap[cat])
            mesh.castShadow = true
            clothingGroup.add(mesh)
          }
        )
      }
    })

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
