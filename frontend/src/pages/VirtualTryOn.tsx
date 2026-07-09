import React, { useState, useEffect, useRef } from 'react'
import {
  Card,
  Row,
  Col,
  Select,
  Spin,
  message,
  Typography,
  Space,
  Divider,
  Button,
  Input,
  Modal,
  Tabs,
  Upload,
  Alert,
  Tag,
  Empty,
} from 'antd'
import {
  ReloadOutlined,
  CheckCircleOutlined,
  UserOutlined,
  LinkOutlined,
  UploadOutlined,
  PictureOutlined,
} from '@ant-design/icons'
import ThreeDClothViewer from '../components/ThreeDClothViewer'
import { tryonAPI, tryonApi } from '../api'

const { Title, Text, Paragraph } = Typography

interface ClothingSet {
  top?: string
  bottom?: string
  dress?: string
  outerwear?: string
}

// Categories accepted by the backend (drives placement on the body)
const CATEGORY_OPTIONS = [
  { value: 'top', label: '👕 上衣' },
  { value: 'bottom', label: '👖 下装' },
  { value: 'dress', label: '👗 连衣裙' },
  { value: 'outerwear', label: '🧥 外套' },
  { value: 'shoes', label: '👟 鞋子' },
  { value: 'accessories', label: '👒 配饰' },
]

// 服装图片URL（仅用于 3D 形象预览的次要 Tab）
const CLOTHING_IMAGES: Record<string, string[]> = {
  top: [
    'https://images.unsplash.com/photo-1521572163474-6864f9cf17ab?w=300&h=400&fit=crop',
    'https://images.unsplash.com/photo-1618354691373-d851c5c3a990?w=300&h=400&fit=crop',
    'https://images.unsplash.com/photo-1583743814966-8936f5b7be1a?w=300&h=400&fit=crop',
  ],
  bottom: [
    'https://images.unsplash.com/photo-1541099649105-f69ad21f3246?w=300&h=400&fit=crop',
    'https://images.unsplash.com/photo-1594938298603-c8148c4dae35?w=300&h=400&fit=crop',
    'https://images.unsplash.com/photo-1473966968600-fa801b869a1a?w=300&h=400&fit=crop',
  ],
  dress: [
    'https://images.unsplash.com/photo-1595777457583-95e059d581b8?w=300&h=400&fit=crop',
    'https://images.unsplash.com/photo-1572804013309-59a88b7e92f1?w=300&h=400&fit=crop',
  ],
  outerwear: [
    'https://images.unsplash.com/photo-1591047139829-d91aecb6caea?w=300&h=400&fit=crop',
    'https://images.unsplash.com/photo-1544022613-e87ca75a784a?w=300&h=400&fit=crop',
  ],
}

const THREE_D_CATEGORIES = [
  { value: 'top', label: '👕 上衣' },
  { value: 'bottom', label: '👖 下装' },
  { value: 'dress', label: '👗 连衣裙' },
  { value: 'outerwear', label: '🧥 外套' },
]

const TRYON_BASE = tryonApi.defaults.baseURL || 'http://localhost:8001'

/** Fetch a URL (same-origin to the try-on service) into a File for upload. */
async function urlToFile(url: string, name: string): Promise<File> {
  const res = await fetch(url)
  if (!res.ok) throw new Error(`下载失败: ${res.status}`)
  const blob = await res.blob()
  return new File([blob], name, { type: blob.type || 'image/jpeg' })
}

const VirtualTryOn: React.FC = () => {
  // ---- Active tab ----
  const [activeTab, setActiveTab] = useState<'real' | '3d'>('real')

  // ---- Real try-on state ----
  const [personImageId, setPersonImageId] = useState<string>('')
  const [personPreview, setPersonPreview] = useState<string>('')
  const [clothImageId, setClothImageId] = useState<string>('')
  const [clothPreview, setClothPreview] = useState<string>('')
  const [category, setCategory] = useState<string>('top')
  const [resultUrl, setResultUrl] = useState<string>('')
  const [resultBackend, setResultBackend] = useState<string>('')
  const [realLoading, setRealLoading] = useState(false)
  const [realError, setRealError] = useState<string>('')

  // ---- 3D preview state (secondary) ----
  const [selectedCategory, setSelectedCategory] = useState<string>('top')
  const [selectedIndex, setSelectedIndex] = useState<number>(0)
  const [isLoading, setIsLoading] = useState(false)
  // 默认就用程序化生成的 1 套 T 恤 + 裤子(无需用户选择,打开 3D tab 即可看)
  // 用 '__programmatic_*' 标记,ThreeDClothViewer 识别后程序化生成衣服几何
  const [clothing, setClothing] = useState<ClothingSet>({
    top: '__programmatic_top__',
    bottom: '__programmatic_bottom__',
  })
  const [tryOnComplete, setTryOnComplete] = useState(false)
  const [gender, setGender] = useState<'male' | 'female'>('female')
  const [urlModalVisible, setUrlModalVisible] = useState(false)
  const [customUrl, setCustomUrl] = useState<string>('')
  const timeoutsRef = useRef<ReturnType<typeof setTimeout>[]>([])

  useEffect(() => {
    return () => {
      timeoutsRef.current.forEach((t) => clearTimeout(t))
      timeoutsRef.current = []
    }
  }, [])

  // ===== Real try-on handlers =====

  const uploadPerson = async (file: File) => {
    setRealLoading(true)
    setRealError('')
    try {
      setPersonPreview(URL.createObjectURL(file))
      const res = await tryonAPI.uploadPersonImage(file)
      setPersonImageId(res.image_id)
      // clear stale result when inputs change
      setResultUrl('')
      message.success('人像上传成功')
    } catch (e: any) {
      setRealError(e?.response?.data?.detail || '人像上传失败')
      message.error('人像上传失败')
    } finally {
      setRealLoading(false)
    }
  }

  const uploadCloth = async (file: File) => {
    setRealLoading(true)
    setRealError('')
    try {
      setClothPreview(URL.createObjectURL(file))
      const res = await tryonAPI.uploadClothImage(file)
      setClothImageId(res.image_id)
      setResultUrl('')
      message.success('服装上传成功')
    } catch (e: any) {
      setRealError(e?.response?.data?.detail || '服装上传失败')
      message.error('服装上传失败')
    } finally {
      setRealLoading(false)
    }
  }

  const useSamplePerson = async () => {
    setRealLoading(true)
    setRealError('')
    try {
      const file = await urlToFile(`${TRYON_BASE}/static/test/person.jpg`, 'person.jpg')
      setPersonPreview(URL.createObjectURL(file))
      const res = await tryonAPI.uploadPersonImage(file)
      setPersonImageId(res.image_id)
      setResultUrl('')
      message.success('已使用示例人像')
    } catch (e: any) {
      setRealError('示例人像加载失败：' + (e?.message || ''))
      message.error('示例人像加载失败')
    } finally {
      setRealLoading(false)
    }
  }

  const useSampleCloth = async () => {
    setRealLoading(true)
    setRealError('')
    try {
      const file = await urlToFile(`${TRYON_BASE}/static/test/cloth.jpg`, 'cloth.jpg')
      setClothPreview(URL.createObjectURL(file))
      const res = await tryonAPI.uploadClothImage(file)
      setClothImageId(res.image_id)
      setResultUrl('')
      message.success('已使用示例服装')
    } catch (e: any) {
      setRealError('示例服装加载失败：' + (e?.message || ''))
      message.error('示例服装加载失败')
    } finally {
      setRealLoading(false)
    }
  }

  const handleTryOn = async () => {
    if (!personImageId || !clothImageId) {
      message.warning('请先上传人像和服装（或使用示例）')
      return
    }
    setRealLoading(true)
    setRealError('')
    setResultUrl('')
    try {
      const res = await tryonAPI.generateTryOn(personImageId, clothImageId, category)
      setResultUrl(`${TRYON_BASE}${res.result_url}`)
      setResultBackend(res.backend || '')
      message.success('试穿完成！')
    } catch (e: any) {
      const detail = e?.response?.data?.detail || '试穿失败，请重试'
      setRealError(detail)
      message.error('试穿失败')
    } finally {
      setRealLoading(false)
    }
  }

  const handleRealReset = () => {
    setPersonImageId('')
    setPersonPreview('')
    setClothImageId('')
    setClothPreview('')
    setResultUrl('')
    setResultBackend('')
    setRealError('')
  }

  // ===== 3D preview handlers (secondary) =====

  const currentClothingList = CLOTHING_IMAGES[selectedCategory] || []
  const currentClothingImage = currentClothingList[selectedIndex] || ''

  const applyTryOn = () => {
    setIsLoading(true)
    setTryOnComplete(false)

    const t = setTimeout(() => {
      const newClothing: ClothingSet = {}
      switch (selectedCategory) {
        case 'top':
          newClothing.top = currentClothingImage
          break
        case 'bottom':
          newClothing.bottom = currentClothingImage
          break
        case 'dress':
          newClothing.dress = currentClothingImage
          break
        case 'outerwear':
          newClothing.outerwear = currentClothingImage
          break
      }
      setClothing(newClothing)
      setIsLoading(false)
      setTryOnComplete(true)
      message.success('3D 试穿预览完成！')
    }, 800)
    timeoutsRef.current.push(t)
  }

  const applyFullOutfit = () => {
    setIsLoading(true)
    setTryOnComplete(false)
    const t = setTimeout(() => {
      setClothing({
        top: CLOTHING_IMAGES.top[selectedIndex % CLOTHING_IMAGES.top.length],
        bottom: CLOTHING_IMAGES.bottom[selectedIndex % CLOTHING_IMAGES.bottom.length],
      })
      setIsLoading(false)
      setTryOnComplete(true)
      message.success('整套搭配试穿成功！')
    }, 1000)
    timeoutsRef.current.push(t)
  }

  const handleReset = () => {
    setClothing({})
    setTryOnComplete(false)
  }

  const handleChangeAvatar = () => {
    if (customUrl.trim()) {
      setUrlModalVisible(false)
      setCustomUrl('')
      message.success('请刷新页面使用新形象！（当前版本需手动刷新）')
    } else {
      message.warning('请输入有效的URL')
    }
  }

  const handleUseDefault = () => {
    setUrlModalVisible(false)
    message.info('已使用默认虚拟形象')
  }

  // ===== Render =====

  return (
    <div style={{ padding: 24 }}>
      <Title level={2} style={{ textAlign: 'center', marginBottom: 24 }}>
        👗 虚拟试穿
      </Title>

      <Tabs
        activeKey={activeTab}
        onChange={(k) => setActiveTab(k as 'real' | '3d')}
        centered
        items={[
          {
            key: 'real',
            label: '📷 真人试穿（AI）',
            children: (
              <Row gutter={32}>
                {/* 左侧：试穿结果 */}
                <Col span={14}>
                  <Card
                    title={
                      <Space>
                        <UserOutlined />
                        <span>试穿结果</span>
                        {resultBackend && (
                          <Tag color={resultBackend === 'mediapipe' ? 'green' : 'orange'}>
                            {resultBackend === 'mediapipe' ? '姿态感知' : '固定比例'}
                          </Tag>
                        )}
                      </Space>
                    }
                  >
                    <div
                      style={{
                        display: 'flex',
                        justifyContent: 'center',
                        alignItems: 'center',
                        minHeight: 500,
                        background: '#f5f5f5',
                        borderRadius: 12,
                        padding: 16,
                      }}
                    >
                      {realLoading ? (
                        <Spin size="large" tip="正在生成试穿效果..." />
                      ) : resultUrl ? (
                        <img
                          src={resultUrl}
                          alt="试穿结果"
                          style={{
                            maxWidth: '100%',
                            maxHeight: 600,
                            objectFit: 'contain',
                            borderRadius: 8,
                          }}
                        />
                      ) : (
                        <Empty description="上传人像与服装后点击「生成试穿」" />
                      )}
                    </div>

                    {realError && (
                      <Alert
                        type="error"
                        message={realError}
                        style={{ marginTop: 12 }}
                        showIcon
                        closable
                        onClose={() => setRealError('')}
                      />
                    )}

                    <Paragraph type="secondary" style={{ marginTop: 12, fontSize: 12 }}>
                      💡 后端使用 MediaPipe 人体姿态关键点，按肩宽自适应缩放、躯干倾角旋转、
                      GrabCut 抠图并 alpha 羽化贴图；检测不到人体时回退固定比例。
                    </Paragraph>
                  </Card>
                </Col>

                {/* 右侧：人像 + 服装 + 试穿 */}
                <Col span={10}>
                  <Card title="① 上传人像">
                    <Space direction="vertical" style={{ width: '100%' }} size="middle">
                      <div style={{ display: 'flex', gap: 12 }}>
                        <Upload
                          accept="image/jpeg,image/png,image/jpg"
                          showUploadList={false}
                          beforeUpload={(file) => {
                            uploadPerson(file)
                            return false
                          }}
                        >
                          <Button icon={<UploadOutlined />} block>
                            上传人像
                          </Button>
                        </Upload>
                        <Button icon={<PictureOutlined />} onClick={useSamplePerson} block>
                          示例人像
                        </Button>
                      </div>
                      {personPreview && (
                        <img
                          src={personPreview}
                          alt="人像预览"
                          style={{
                            width: '100%',
                            maxHeight: 200,
                            objectFit: 'contain',
                            borderRadius: 8,
                            border: '1px solid #e0e0e0',
                          }}
                        />
                      )}
                    </Space>
                  </Card>

                  <Card title="② 选择服装" style={{ marginTop: 16 }}>
                    <Space direction="vertical" style={{ width: '100%' }} size="middle">
                      <div style={{ display: 'flex', gap: 12 }}>
                        <Upload
                          accept="image/jpeg,image/png,image/jpg"
                          showUploadList={false}
                          beforeUpload={(file) => {
                            uploadCloth(file)
                            return false
                          }}
                        >
                          <Button icon={<UploadOutlined />} block>
                            上传服装
                          </Button>
                        </Upload>
                        <Button icon={<PictureOutlined />} onClick={useSampleCloth} block>
                          示例服装
                        </Button>
                      </div>
                      {clothPreview && (
                        <img
                          src={clothPreview}
                          alt="服装预览"
                          style={{
                            width: '100%',
                            maxHeight: 200,
                            objectFit: 'contain',
                            borderRadius: 8,
                            border: '1px solid #e0e0e0',
                          }}
                        />
                      )}
                    </Space>
                  </Card>

                  <Card title="③ 服装类别与试穿" style={{ marginTop: 16 }}>
                    <Space direction="vertical" style={{ width: '100%' }} size="middle">
                      <div>
                        <Text strong style={{ display: 'block', marginBottom: 8 }}>
                          类别（决定贴图位置）
                        </Text>
                        <Select
                          value={category}
                          onChange={setCategory}
                          style={{ width: '100%' }}
                          size="large"
                          options={CATEGORY_OPTIONS}
                        />
                      </div>
                      <Button
                        type="primary"
                        size="large"
                        block
                        onClick={handleTryOn}
                        loading={realLoading}
                        disabled={!personImageId || !clothImageId}
                        style={{ height: 48, fontSize: 16 }}
                      >
                        ✨ 生成试穿
                      </Button>
                      <Button icon={<ReloadOutlined />} onClick={handleRealReset} block>
                        重置
                      </Button>
                    </Space>
                  </Card>
                </Col>
              </Row>
            ),
          },
          {
            key: '3d',
            label: '🧍 3D 形象预览（次要）',
            children: (
              <Row gutter={32}>
                <Col span={14}>
                  <Card
                    title={
                      <Space>
                        <UserOutlined />
                        <span>3D 虚拟人物</span>
                      </Space>
                    }
                    extra={
                      <Button
                        icon={<LinkOutlined />}
                        onClick={() => setUrlModalVisible(true)}
                        size="small"
                      >
                        更换形象
                      </Button>
                    }
                  >
                    <div style={{ display: 'flex', justifyContent: 'center', padding: '20px 0' }}>
                      {isLoading ? (
                        <div
                          style={{
                            width: 400,
                            height: 550,
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            background: '#f5f5f5',
                            borderRadius: 16,
                          }}
                        >
                          <Spin size="large" tip="正在穿戴..." />
                        </div>
                      ) : (
                        <ThreeDClothViewer
                          clothing={clothing}
                          width={400}
                          height={550}
                          gender={gender}
                          onGenderChange={setGender}
                        />
                      )}
                    </div>

                    {tryOnComplete && Object.keys(clothing).length > 0 && (
                      <div
                        style={{
                          marginTop: 16,
                          padding: 12,
                          background: '#f6ffed',
                          border: '1px solid #b7eb8f',
                          borderRadius: 8,
                          textAlign: 'center',
                        }}
                      >
                        <CheckCircleOutlined style={{ color: '#52c41a', marginRight: 8 }} />
                        <Text style={{ color: '#52c41a' }}>试穿完成！拖动旋转查看效果</Text>
                      </div>
                    )}
                  </Card>
                </Col>

                <Col span={10}>
                  <Card title="选择服装进行 3D 试穿">
                    <div style={{ marginBottom: 20 }}>
                      <Text strong style={{ display: 'block', marginBottom: 8 }}>
                        服装分类
                      </Text>
                      <Select
                        value={selectedCategory}
                        onChange={(v) => {
                          setSelectedCategory(v)
                          setSelectedIndex(0)
                          setTryOnComplete(false)
                        }}
                        style={{ width: '100%' }}
                        size="large"
                        options={THREE_D_CATEGORIES.map((c) => ({
                          value: c.value,
                          label: c.label,
                        }))}
                      />
                    </div>

                    <Divider />

                    <div style={{ marginBottom: 20 }}>
                      <Text strong style={{ display: 'block', marginBottom: 8 }}>
                        选择款式 ({selectedIndex + 1}/{currentClothingList.length})
                      </Text>
                      <div
                        style={{
                          display: 'grid',
                          gridTemplateColumns: 'repeat(3, 1fr)',
                          gap: 12,
                        }}
                      >
                        {currentClothingList.map((img, index) => (
                          <div
                            key={index}
                            onClick={() => {
                              setSelectedIndex(index)
                              setTryOnComplete(false)
                            }}
                            style={{
                              cursor: 'pointer',
                              borderRadius: 8,
                              overflow: 'hidden',
                              border:
                                selectedIndex === index
                                  ? '3px solid #1890ff'
                                  : '2px solid #e0e0e0',
                              boxShadow:
                                selectedIndex === index
                                  ? '0 4px 12px rgba(24,144,255,0.3)'
                                  : 'none',
                              transition: 'all 0.2s',
                            }}
                          >
                            <img
                              src={img}
                              alt={`款式${index + 1}`}
                              style={{ width: '100%', height: 100, objectFit: 'cover' }}
                            />
                          </div>
                        ))}
                      </div>
                    </div>

                    <Divider />

                    <Space direction="vertical" style={{ width: '100%' }} size="middle">
                      <Button
                        type="primary"
                        size="large"
                        block
                        onClick={applyTryOn}
                        loading={isLoading}
                        style={{ height: 48, fontSize: 16 }}
                      >
                        👔 试穿这件{' '}
                        {THREE_D_CATEGORIES.find((c) => c.value === selectedCategory)?.label}
                      </Button>

                      {selectedCategory === 'top' && (
                        <Button size="large" block onClick={applyFullOutfit} loading={isLoading}>
                          👗 试穿整套搭配（上衣+下装）
                        </Button>
                      )}

                      <Button icon={<ReloadOutlined />} onClick={handleReset} block>
                        清除服装
                      </Button>
                    </Space>
                  </Card>
                </Col>
              </Row>
            ),
          },
        ]}
      />

      {/* 更换虚拟形象弹窗 */}
      <Modal
        title="更换虚拟形象"
        open={urlModalVisible}
        onCancel={() => setUrlModalVisible(false)}
        onOk={handleChangeAvatar}
        okText="确认更换"
        cancelText="取消"
      >
        <div style={{ marginBottom: 16 }}>
          <Text strong style={{ display: 'block', marginBottom: 8 }}>
            输入你的 ReadyPlayerMe 虚拟形象 URL：
          </Text>
          <Input
            placeholder="粘贴 .glb 文件URL"
            value={customUrl}
            onChange={(e) => setCustomUrl(e.target.value)}
            prefix={<LinkOutlined />}
          />
          <Text type="secondary" style={{ fontSize: 12, display: 'block', marginTop: 8 }}>
            提示：去{' '}
            <a href="https://readyplayer.me/" target="_blank" rel="noreferrer">
              readyplayer.me
            </a>{' '}
            创建一个免费账号，选择你喜欢的虚拟形象，导出为.glb格式，复制链接粘贴到这里。
          </Text>
        </div>

        <Divider>或者使用默认形象</Divider>

        <Button onClick={handleUseDefault} block>
          使用默认卡通形象
        </Button>
      </Modal>
    </div>
  )
}

export default VirtualTryOn
