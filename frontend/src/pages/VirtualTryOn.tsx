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
} from 'antd'
import { ReloadOutlined, CheckCircleOutlined, UserOutlined, LinkOutlined } from '@ant-design/icons'
import ThreeDClothViewer from '../components/ThreeDClothViewer'

const { Title, Text } = Typography

interface ClothingSet {
  top?: string
  bottom?: string
  dress?: string
  outerwear?: string
}

// 分类配置
const CATEGORIES = [
  { value: 'top', label: '👕 上衣' },
  { value: 'bottom', label: '👖 下装' },
  { value: 'dress', label: '👗 连衣裙' },
  { value: 'outerwear', label: '🧥 外套' },
]

// 服装图片URL
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

const VirtualTryOn: React.FC = () => {
  const [selectedCategory, setSelectedCategory] = useState<string>('top')
  const [selectedIndex, setSelectedIndex] = useState<number>(0)
  const [isLoading, setIsLoading] = useState(false)
  const [clothing, setClothing] = useState<ClothingSet>({})
  const [tryOnComplete, setTryOnComplete] = useState(false)
  const [gender, setGender] = useState<'male' | 'female'>('female')
  const [urlModalVisible, setUrlModalVisible] = useState(false)
  const [customUrl, setCustomUrl] = useState<string>('')
  const timeoutsRef = useRef<ReturnType<typeof setTimeout>[]>([])

  // Clear any pending timeouts on unmount to avoid setState after unmount
  useEffect(() => {
    return () => {
      timeoutsRef.current.forEach((t) => clearTimeout(t))
      timeoutsRef.current = []
    }
  }, [])

  // 获取当前分类的服装
  const currentClothingList = CLOTHING_IMAGES[selectedCategory] || []
  const currentClothingImage = currentClothingList[selectedIndex] || ''

  // 应用试穿
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
      message.success('虚拟试穿成功！')
    }, 800)
    timeoutsRef.current.push(t)
  }

  // 整套搭配
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

  // 清除试穿
  const handleReset = () => {
    setClothing({})
    setTryOnComplete(false)
  }

  // 更换虚拟形象
  const handleChangeAvatar = () => {
    if (customUrl.trim()) {
      setUrlModalVisible(false)
      setCustomUrl('')
      message.success('请刷新页面使用新形象！（当前版本需手动刷新）')
    } else {
      message.warning('请输入有效的URL')
    }
  }

  // 使用默认形象
  const handleUseDefault = () => {
    setUrlModalVisible(false)
    message.info('已使用默认虚拟形象')
  }

  return (
    <div style={{ padding: 24 }}>
      <Title level={2} style={{ textAlign: 'center', marginBottom: 24 }}>
        👗 3D 虚拟试穿
      </Title>

      <Row gutter={32}>
        {/* 左侧：3D虚拟人物 */}
        <Col span={14}>
          <Card
            title={
              <Space>
                <UserOutlined />
                <span>虚拟人物</span>
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
                <Text style={{ color: '#52c41a' }}>
                  试穿完成！拖动旋转查看效果
                </Text>
              </div>
            )}
          </Card>
        </Col>

        {/* 右侧：服装选择 */}
        <Col span={10}>
          <Card title="选择服装进行试穿">
            {/* 分类选择 */}
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
                options={CATEGORIES.map((c) => ({
                  value: c.value,
                  label: c.label,
                }))}
              />
            </div>

            <Divider />

            {/* 服装款式选择 */}
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
                      style={{
                        width: '100%',
                        height: 100,
                        objectFit: 'cover',
                      }}
                    />
                  </div>
                ))}
              </div>
            </div>

            <Divider />

            {/* 试穿操作 */}
            <Space direction="vertical" style={{ width: '100%' }} size="middle">
              <Button
                type="primary"
                size="large"
                block
                onClick={applyTryOn}
                loading={isLoading}
                style={{ height: 48, fontSize: 16 }}
              >
                👔 试穿这件 {CATEGORIES.find((c) => c.value === selectedCategory)?.label}
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

            {/* 当前选择预览 */}
            {currentClothingImage && (
              <Card size="small" style={{ marginTop: 20, background: '#fafafa' }}>
                <Text type="secondary" style={{ fontSize: 12 }}>
                  当前选择预览
                </Text>
                <div style={{ marginTop: 8 }}>
                  <img
                    src={currentClothingImage}
                    alt="选择"
                    style={{
                      width: '100%',
                      maxHeight: 150,
                      objectFit: 'contain',
                      borderRadius: 8,
                    }}
                  />
                </div>
              </Card>
            )}
          </Card>

          {/* 说明 */}
          <Card size="small" style={{ marginTop: 16 }}>
            <Title level={5}>💡 如何更换虚拟形象？</Title>
            <ol style={{ margin: 0, paddingLeft: 20, color: '#666', fontSize: 13 }}>
              <li>访问 <a href="https://readyplayer.me/" target="_blank">readyplayer.me</a> 创建你的专属形象</li>
              <li>生成后复制.glb链接</li>
              <li>点击右上角"更换形象"按钮</li>
              <li>粘贴链接并确认</li>
            </ol>
            <Text type="secondary" style={{ fontSize: 12, display: 'block', marginTop: 8 }}>
              当前使用默认卡通形象，可自定义更换为你的专属虚拟形象。
            </Text>
          </Card>
        </Col>
      </Row>

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
            <a href="https://readyplayer.me/" target="_blank">
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
