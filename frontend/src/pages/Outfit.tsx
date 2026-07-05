import React, { useState, useEffect } from 'react'
import {
  Card,
  Row,
  Col,
  Typography,
  Tag,
  Button,
  Spin,
  Empty,
  Space,
  Badge,
  Tabs,
  Modal,
  message,
  Divider,
  Avatar,
} from 'antd'
import {
  ShoppingOutlined,
  StarOutlined,
  FireOutlined,
  BulbOutlined,
  HeartOutlined,
  CheckCircleOutlined,
  SendOutlined,
  PlusOutlined,
  ScissorOutlined,
} from '@ant-design/icons'
import { useNavigate } from 'react-router-dom'
import { userAPI } from '../api'

const { Title, Text } = Typography

interface ClothingItem {
  id: number
  name: string
  category: string
  price: number
  style?: string
  brand?: string
}

interface OutfitItem {
  id: number
  name: string
  category: string
  emoji: string
  item?: ClothingItem
}

interface Outfit {
  id: number
  name: string
  scene: string
  style: string
  items: OutfitItem[]
  score?: number
  liked?: boolean
}

interface FavoriteOutfit {
  id: number
  name: string
  scene: string
  items: OutfitItem[]
  created_at: string
}

// 场景定义
const scenes = [
  { key: 'office', label: '上班', icon: '💼', color: '#1890ff' },
  { key: 'date', label: '约会', icon: '💕', color: '#eb2f96' },
  { key: 'casual', label: '休闲', icon: '☕', color: '#52c41a' },
  { key: 'sports', label: '运动', icon: '🏃', color: '#f5222d' },
]

const Outfit: React.FC = () => {
  const navigate = useNavigate()
  const userId = 1 // 模拟当前用户

  const [loading, setLoading] = useState(true)
  const [generating, setGenerating] = useState(false)
  const [outfits, setOutfits] = useState<Outfit[]>([])
  const [favoriteOutfits, setFavoriteOutfits] = useState<FavoriteOutfit[]>([])
  const [selectedScene, setSelectedScene] = useState<string>('office')
  const [activeTab, setActiveTab] = useState<string>('recommend')
  const [modalVisible, setModalVisible] = useState(false)
  const [selectedOutfit, setSelectedOutfit] = useState<Outfit | null>(null)

  // 获取搭配数据
  useEffect(() => {
    fetchOutfits()
    fetchFavorites()
  }, [selectedScene])

  const fetchOutfits = async () => {
    setLoading(true)
    try {
      // 获取该场景的搭配推荐（实际项目中应使用API返回数据）
      await userAPI.getOutfitRecommendations(userId, 0, 6)

      // 模拟搭配数据（实际项目中应使用API返回数据）
      const mockOutfits: Outfit[] = generateMockOutfits(selectedScene)
      setOutfits(mockOutfits)
    } catch (error) {
      console.error('Failed to fetch outfits:', error)
      // 即使API失败也显示模拟数据
      setOutfits(generateMockOutfits(selectedScene))
    } finally {
      setLoading(false)
    }
  }

  const fetchFavorites = async () => {
    // 模拟收藏的搭配
    setFavoriteOutfits([
      {
        id: 101,
        name: '我的通勤穿搭',
        scene: 'office',
        items: [
          { id: 1, name: '白色衬衫', category: 'top', emoji: '👕' },
          { id: 2, name: '黑色西裤', category: 'bottom', emoji: '👖' },
          { id: 3, name: '乐福鞋', category: 'shoes', emoji: '👟' },
        ],
        created_at: '2026-06-28',
      },
      {
        id: 102,
        name: '周末约会装',
        scene: 'date',
        items: [
          { id: 4, name: '碎花裙', category: 'dress', emoji: '👗' },
          { id: 5, name: '小手提包', category: 'accessories', emoji: '👜' },
        ],
        created_at: '2026-06-25',
      },
    ])
  }

  // 生成模拟搭配数据
  const generateMockOutfits = (scene: string): Outfit[] => {
    const outfitTemplates: Record<string, Outfit[]> = {
      office: [
        {
          id: 1,
          name: '简约职业风',
          scene: 'office',
          style: '简约干练',
          items: [
            { id: 1, name: '白色衬衫', category: 'top', emoji: '👕' },
            { id: 2, name: '黑色西裤', category: 'bottom', emoji: '👖' },
            { id: 3, name: '尖头高跟鞋', category: 'shoes', emoji: '👟' },
          ],
          score: 95,
        },
        {
          id: 2,
          name: '干练套装',
          scene: 'office',
          style: '精致优雅',
          items: [
            { id: 4, name: '修身西装', category: 'outerwear', emoji: '🧥' },
            { id: 5, name: '包臀裙', category: 'bottom', emoji: '👖' },
            { id: 6, name: '商务皮鞋', category: 'shoes', emoji: '👟' },
          ],
          score: 92,
        },
        {
          id: 3,
          name: '轻商务风格',
          scene: 'office',
          style: '舒适得体',
          items: [
            { id: 7, name: '针织衫', category: 'top', emoji: '👕' },
            { id: 8, name: '阔腿裤', category: 'bottom', emoji: '👖' },
            { id: 9, name: '乐福鞋', category: 'shoes', emoji: '👟' },
          ],
          score: 88,
        },
      ],
      date: [
        {
          id: 4,
          name: '浪漫约会',
          scene: 'date',
          style: '优雅迷人',
          items: [
            { id: 10, name: '蕾丝上衣', category: 'top', emoji: '👕' },
            { id: 11, name: '半身裙', category: 'bottom', emoji: '👖' },
            { id: 12, name: '细跟高跟鞋', category: 'shoes', emoji: '👟' },
          ],
          score: 96,
        },
        {
          id: 5,
          name: '法式风情',
          scene: 'date',
          style: '复古浪漫',
          items: [
            { id: 13, name: '波点连衣裙', category: 'dress', emoji: '👗' },
            { id: 14, name: '复古手提包', category: 'accessories', emoji: '👜' },
            { id: 15, name: '玛丽珍鞋', category: 'shoes', emoji: '👟' },
          ],
          score: 94,
        },
        {
          id: 6,
          name: '精致小香风',
          scene: 'date',
          style: '高贵精致',
          items: [
            { id: 16, name: '粗花呢外套', category: 'outerwear', emoji: '🧥' },
            { id: 17, name: '小黑裙', category: 'dress', emoji: '👗' },
            { id: 18, name: '珍珠配饰', category: 'accessories', emoji: '👜' },
          ],
          score: 93,
        },
      ],
      casual: [
        {
          id: 7,
          name: '日常休闲',
          scene: 'casual',
          style: '随性舒适',
          items: [
            { id: 19, name: '纯棉T恤', category: 'top', emoji: '👕' },
            { id: 20, name: '牛仔裤', category: 'bottom', emoji: '👖' },
            { id: 21, name: '运动鞋', category: 'shoes', emoji: '👟' },
          ],
          score: 90,
        },
        {
          id: 8,
          name: '文艺清新',
          scene: 'casual',
          style: '自然舒适',
          items: [
            { id: 22, name: '条纹衬衫', category: 'top', emoji: '👕' },
            { id: 23, name: '棉麻长裤', category: 'bottom', emoji: '👖' },
            { id: 24, name: '帆布鞋', category: 'shoes', emoji: '👟' },
          ],
          score: 87,
        },
        {
          id: 9,
          name: '慵懒周末',
          scene: 'casual',
          style: '放松惬意',
          items: [
            { id: 25, name: '卫衣', category: 'top', emoji: '👕' },
            { id: 26, name: '运动裤', category: 'bottom', emoji: '👖' },
            { id: 27, name: '拖鞋', category: 'shoes', emoji: '👟' },
          ],
          score: 85,
        },
      ],
      sports: [
        {
          id: 10,
          name: '健身训练',
          scene: 'sports',
          style: '专业运动',
          items: [
            { id: 28, name: '运动背心', category: 'top', emoji: '👕' },
            { id: 29, name: '运动短裤', category: 'bottom', emoji: '👖' },
            { id: 30, name: '训练鞋', category: 'shoes', emoji: '👟' },
          ],
          score: 94,
        },
        {
          id: 11,
          name: '跑步装备',
          scene: 'sports',
          style: '轻盈透气',
          items: [
            { id: 31, name: '跑步T恤', category: 'top', emoji: '👕' },
            { id: 32, name: '跑步短裤', category: 'bottom', emoji: '👖' },
            { id: 33, name: '跑鞋', category: 'shoes', emoji: '👟' },
          ],
          score: 92,
        },
        {
          id: 12,
          name: '瑜伽穿着',
          scene: 'sports',
          style: '柔软舒适',
          items: [
            { id: 34, name: '瑜伽上衣', category: 'top', emoji: '👕' },
            { id: 35, name: '瑜伽裤', category: 'bottom', emoji: '👖' },
            { id: 36, name: '瑜伽垫', category: 'accessories', emoji: '👜' },
          ],
          score: 89,
        },
      ],
    }
    return outfitTemplates[scene] || outfitTemplates.office
  }

  // 智能生成搭配
  const handleGenerateOutfit = async () => {
    setGenerating(true)
    try {
      // 模拟AI生成延迟
      await new Promise(resolve => setTimeout(resolve, 1500))

      // 随机生成一个新搭配
      const newOutfit: Outfit = {
        id: Date.now(),
        name: `AI智能搭配-${Math.floor(Math.random() * 100)}`,
        scene: selectedScene,
        style: 'AI推荐',
        items: [
          { id: 100 + Math.random(), name: 'AI生成上衣', category: 'top', emoji: '👕' },
          { id: 200 + Math.random(), name: 'AI生成下装', category: 'bottom', emoji: '👖' },
          { id: 300 + Math.random(), name: 'AI生成鞋子', category: 'shoes', emoji: '👟' },
        ],
        score: Math.floor(Math.random() * 20) + 80,
      }

      setOutfits(prev => [newOutfit, ...prev])
      message.success('AI已为您生成新搭配！')
    } catch (error) {
      message.error('生成失败，请重试')
    } finally {
      setGenerating(false)
    }
  }

  // 收藏搭配
  const handleFavorite = (outfit: Outfit) => {
    const isFavorited = favoriteOutfits.some(f => f.id === outfit.id)
    if (isFavorited) {
      setFavoriteOutfits(prev => prev.filter(f => f.id !== outfit.id))
      message.success('已取消收藏')
    } else {
      const newFavorite: FavoriteOutfit = {
        id: outfit.id,
        name: outfit.name,
        scene: outfit.scene,
        items: outfit.items,
        created_at: new Date().toISOString().split('T')[0],
      }
      setFavoriteOutfits(prev => [newFavorite, ...prev])
      message.success('已添加到收藏')
    }
  }

  // 查看搭配详情
  const handleViewDetail = (outfit: Outfit) => {
    setSelectedOutfit(outfit)
    setModalVisible(true)
  }

  // 应用搭配
  const handleApplyOutfit = (outfit: Outfit) => {
    message.success(`已应用「${outfit.name}」到您的衣橱`)
    setModalVisible(false)
  }

  // 商品点击
  const handleItemClick = (item: OutfitItem) => {
    if (item.item?.id) {
      navigate(`/clothing/${item.item.id}`)
    }
  }

  // 渲染搭配卡片
  const renderOutfitCard = (outfit: Outfit) => {
    const isFavorited = favoriteOutfits.some(f => f.id === outfit.id)
    const sceneInfo = scenes.find(s => s.key === outfit.scene)

    return (
      <Card
        key={outfit.id}
        hoverable
        onClick={() => handleViewDetail(outfit)}
        cover={
          <div
            style={{
              background: `linear-gradient(135deg, ${sceneInfo?.color}15 0%, ${sceneInfo?.color}30 100%)`,
              padding: '24px 16px 16px',
              minHeight: 140,
            }}
          >
            <Space direction="vertical" size="middle" style={{ width: '100%' }}>
              <Row gutter={[8, 8]} justify="center">
                {outfit.items.map((item, idx) => (
                  <Col key={item.id}>
                    <div
                      style={{
                        width: 60,
                        height: 60,
                        background: '#fff',
                        borderRadius: 12,
                        display: 'flex',
                        flexDirection: 'column',
                        alignItems: 'center',
                        justifyContent: 'center',
                        boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
                        cursor: 'pointer',
                      }}
                      onClick={(e) => {
                        e.stopPropagation()
                        handleItemClick(item)
                      }}
                    >
                      <span style={{ fontSize: 24 }}>{item.emoji}</span>
                    </div>
                    {idx < outfit.items.length - 1 && (
                      <span style={{ position: 'absolute', right: -10, top: '50%', color: '#ccc' }}>+</span>
                    )}
                  </Col>
                ))}
              </Row>
              {outfit.score && (
                <Tag color={outfit.score >= 90 ? 'green' : 'blue'} icon={<StarOutlined />}>
                  匹配度 {outfit.score}%
                </Tag>
              )}
            </Space>
          </div>
        }
        actions={[
          <StarOutlined
            key="收藏"
            onClick={(e) => {
              e.stopPropagation()
              handleFavorite(outfit)
            }}
            style={{ color: isFavorited ? '#f5222d' : undefined }}
          />,
          <ShoppingOutlined key="应用" onClick={(e) => {
            e.stopPropagation()
            handleApplyOutfit(outfit)
          }} />,
        ]}
      >
        <Card.Meta
          title={<Text strong>{outfit.name}</Text>}
          description={
            <Space direction="vertical" size="small" style={{ width: '100%' }}>
              <Row justify="space-between">
                <Tag color={sceneInfo?.color}>{sceneInfo?.label}</Tag>
                <Text type="secondary" style={{ fontSize: 12 }}>{outfit.style}</Text>
              </Row>
              {isFavorited && <Tag color="red" icon={<HeartOutlined />}>已收藏</Tag>}
            </Space>
          }
        />
      </Card>
    )
  }

  return (
    <div style={{ padding: '0 0 24px 0' }}>
      {/* 标题 */}
      <div style={{ marginBottom: 24 }}>
        <Title level={3} style={{ margin: 0 }}>
          <Badge color="purple" />
          搭配推荐
        </Title>
        <Text type="secondary">智能穿搭，告别搭配烦恼</Text>
      </div>

      {/* 场景选择 */}
      <Card style={{ marginBottom: 24 }}>
        <Row gutter={[16, 16]} align="middle">
          <Col>
            <Text strong>选择场景：</Text>
          </Col>
          {scenes.map((scene) => (
            <Col key={scene.key}>
              <Tag
                color={selectedScene === scene.key ? scene.color : 'default'}
                style={{
                  padding: '8px 16px',
                  fontSize: 14,
                  cursor: 'pointer',
                  borderRadius: 20,
                }}
                onClick={() => setSelectedScene(scene.key)}
                icon={<span>{scene.icon}</span>}
              >
                {scene.label}
              </Tag>
            </Col>
          ))}
          <Col push={1}>
            <Button
              type="primary"
              icon={<BulbOutlined />}
              onClick={handleGenerateOutfit}
              loading={generating}
            >
              AI智能生成
            </Button>
          </Col>
        </Row>
      </Card>

      {/* Tab切换 */}
      <Tabs
        activeKey={activeTab}
        onChange={setActiveTab}
        items={[
          {
            key: 'recommend',
            label: (
              <Space>
                <FireOutlined />
                推荐搭配
              </Space>
            ),
            children: (
              <div>
                {loading ? (
                  <div style={{ textAlign: 'center', padding: '60px 0' }}>
                    <Spin size="large" />
                    <div style={{ marginTop: 16 }}>加载中...</div>
                  </div>
                ) : outfits.length > 0 ? (
                  <Row gutter={[16, 16]}>
                    {outfits.map((outfit) => (
                      <Col xs={12} sm={12} md={8} lg={6} key={outfit.id}>
                        {renderOutfitCard(outfit)}
                      </Col>
                    ))}
                  </Row>
                ) : (
                  <Empty description="暂无搭配推荐" />
                )}
              </div>
            ),
          },
          {
            key: 'favorites',
            label: (
              <Space>
                <HeartOutlined />
                我的收藏
                {favoriteOutfits.length > 0 && (
                  <Badge count={favoriteOutfits.length} size="small" />
                )}
              </Space>
            ),
            children: (
              <div>
                {favoriteOutfits.length > 0 ? (
                  <Row gutter={[16, 16]}>
                    {favoriteOutfits.map((outfit) => (
                      <Col xs={12} sm={12} md={8} lg={6} key={outfit.id}>
                        <Card
                          hoverable
                          onClick={() => handleViewDetail(outfit as unknown as Outfit)}
                          cover={
                            <div
                              style={{
                                background: 'linear-gradient(135deg, #f0f0f0 0%, #e8e8e8 100%)',
                                padding: '24px 16px 16px',
                                minHeight: 140,
                              }}
                            >
                              <Space direction="vertical" size="middle" style={{ width: '100%' }}>
                                <Row gutter={[8, 8]} justify="center">
                                  {outfit.items.map((item, idx) => (
                                    <Col key={item.id}>
                                      <div
                                        style={{
                                          width: 60,
                                          height: 60,
                                          background: '#fff',
                                          borderRadius: 12,
                                          display: 'flex',
                                          flexDirection: 'column',
                                          alignItems: 'center',
                                          justifyContent: 'center',
                                          boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
                                        }}
                                      >
                                        <span style={{ fontSize: 24 }}>{item.emoji}</span>
                                      </div>
                                      {idx < outfit.items.length - 1 && (
                                        <span style={{ position: 'absolute', right: -10, top: '50%', color: '#ccc' }}>+</span>
                                      )}
                                    </Col>
                                  ))}
                                </Row>
                              </Space>
                            </div>
                          }
                          actions={[
                            <HeartOutlined key="收藏" style={{ color: '#f5222d' }} />,
                            <SendOutlined key="应用" onClick={(e) => {
                              e.stopPropagation()
                              handleApplyOutfit(outfit as unknown as Outfit)
                            }} />,
                          ]}
                        >
                          <Card.Meta
                            title={<Text strong>{outfit.name}</Text>}
                            description={
                              <Space direction="vertical" size="small" style={{ width: '100%' }}>
                                <Tag color="purple">{scenes.find(s => s.key === outfit.scene)?.label}</Tag>
                                <Text type="secondary" style={{ fontSize: 12 }}>
                                  收藏于 {outfit.created_at}
                                </Text>
                              </Space>
                            }
                          />
                        </Card>
                      </Col>
                    ))}
                  </Row>
                ) : (
                  <Empty description="暂无收藏搭配">
                    <Button type="primary" onClick={() => setActiveTab('recommend')}>
                      去逛逛
                    </Button>
                  </Empty>
                )}
              </div>
            ),
          },
          {
            key: 'create',
            label: (
              <Space>
                <PlusOutlined />
                创建搭配
              </Space>
            ),
            children: (
              <Card>
                <Row gutter={[24, 24]} align="middle" justify="center">
                  <Col span={24} style={{ textAlign: 'center' }}>
                    <div style={{ fontSize: 64, marginBottom: 16 }}>
                      <ScissorOutlined style={{ color: '#1890ff' }} />
                    </div>
                    <Title level={4}>自由创建搭配</Title>
                    <Text type="secondary">
                      从商品列表中选择单品，自由组合成您想要的搭配方案
                    </Text>
                    <div style={{ marginTop: 24 }}>
                      <Button
                        type="primary"
                        size="large"
                        icon={<ShoppingOutlined />}
                        onClick={() => navigate('/clothing')}
                      >
                        浏览商品
                      </Button>
                    </div>
                  </Col>
                </Row>
              </Card>
            ),
          },
        ]}
      />

      {/* 搭配详情弹窗 */}
      <Modal
        title={
          <Space>
            <CheckCircleOutlined style={{ color: '#52c41a' }} />
            搭配详情
          </Space>
        }
        open={modalVisible}
        onCancel={() => setModalVisible(false)}
        footer={[
          <Button key="close" onClick={() => setModalVisible(false)}>
            关闭
          </Button>,
          <Button
            key="apply"
            type="primary"
            icon={<CheckCircleOutlined />}
            onClick={() => selectedOutfit && handleApplyOutfit(selectedOutfit)}
          >
            应用此搭配
          </Button>,
        ]}
        width={600}
      >
        {selectedOutfit && (
          <div>
            <Row gutter={[16, 16]} align="middle" style={{ marginBottom: 24 }}>
              <Col>
                <Avatar size="large" style={{ backgroundColor: '#1890ff' }}>
                  {selectedOutfit.name.charAt(0)}
                </Avatar>
              </Col>
              <Col flex={1}>
                <Title level={4} style={{ margin: 0 }}>{selectedOutfit.name}</Title>
                <Space>
                  <Tag color={scenes.find(s => s.key === selectedOutfit.scene)?.color}>
                    {scenes.find(s => s.key === selectedOutfit.scene)?.label}
                  </Tag>
                  <Text type="secondary">{selectedOutfit.style}</Text>
                  {selectedOutfit.score && (
                    <Tag icon={<StarOutlined />} color="green">
                      匹配度 {selectedOutfit.score}%
                    </Tag>
                  )}
                </Space>
              </Col>
            </Row>

            <Divider>搭配单品</Divider>

            <Row gutter={[16, 16]}>
              {selectedOutfit.items.map((item) => (
                <Col xs={12} sm={8} key={item.id}>
                  <Card
                    hoverable
                    size="small"
                    onClick={() => handleItemClick(item)}
                    cover={
                      <div
                        style={{
                          height: 80,
                          background: `linear-gradient(180deg, #f5f5f5 0%, #e8e8e8 100%)`,
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          fontSize: 36,
                        }}
                      >
                        {item.emoji}
                      </div>
                    }
                  >
                    <Card.Meta
                      title={<Text style={{ fontSize: 12 }}>{item.name}</Text>}
                      description={
                        <Tag style={{ fontSize: 11 }}>{item.category}</Tag>
                      }
                    />
                  </Card>
                </Col>
              ))}
            </Row>

            <Divider>搭配说明</Divider>

            <Card size="small" style={{ background: '#f5f5f5' }}>
              <Text type="secondary">
                此搭配方案基于您的风格偏好和当前流行趋势智能生成，
                适合{scenes.find(s => s.key === selectedOutfit.scene)?.label}场景穿着。
              </Text>
            </Card>
          </div>
        )}
      </Modal>
    </div>
  )
}

export default Outfit
