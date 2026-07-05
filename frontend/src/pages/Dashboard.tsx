import React, { useState, useEffect } from 'react'
import {
  Card,
  Row,
  Col,
  Carousel,
  Grid,
  Typography,
  Badge,
  Spin,
  Empty,
  Space,
} from 'antd'
import {
  ShopOutlined,
  FireOutlined,
  SkinOutlined,
  ManOutlined,
  ThunderboltOutlined,
  HeartOutlined,
  HistoryOutlined,
  RightOutlined,
  CrownOutlined,
  StarOutlined,
} from '@ant-design/icons'
import { useNavigate } from 'react-router-dom'
import { clothingAPI } from '../api'

const { Title, Text } = Typography
const { useBreakpoint } = Grid

interface ClothingItem {
  id: number
  name: string
  category: string
  price: number
  style?: string
  image_url?: string
}

interface RecommendItem {
  id: number
  name: string
  category: string
  price: number
  reason?: string
  image_url?: string
}

const Dashboard: React.FC = () => {
  const navigate = useNavigate()
  const screens = useBreakpoint()
  const [loading, setLoading] = useState(true)
  const [recommendations, setRecommendations] = useState<RecommendItem[]>([])
  const [hotItems, setHotItems] = useState<ClothingItem[]>([])
  const [newItems, setNewItems] = useState<ClothingItem[]>([])

  useEffect(() => {
    fetchData()
  }, [])

  const fetchData = async () => {
    setLoading(true)
    try {
      // 获取猜你喜欢推荐
      const recData = await clothingAPI.getList({ limit: 6 })
      setRecommendations(
        recData.map((item: any) => ({
          ...item,
          reason: getRecommendReason(item),
        }))
      )

      // 获取热门商品（复用数据，实际项目中应调用专门的热门接口）
      setHotItems(recData.slice(0, 6))

      // 获取新品（复用数据，实际项目中应调用专门的新品接口）
      setNewItems(recData.slice(0, 6))
    } catch (error) {
      console.error('Failed to fetch data:', error)
    } finally {
      setLoading(false)
    }
  }

  const getRecommendReason = (_item: any) => {
    const reasons = [
      '根据您的风格偏好推荐',
      '热门款式，销量领先',
      '搭配神器，百搭之选',
      '今季流行趋势',
      '高性价比之选',
    ]
    return reasons[Math.floor(Math.random() * reasons.length)]
  }

  const categories = [
    { key: 'top', label: '上衣', icon: <ManOutlined />, color: '#1890ff' },
    { key: 'bottom', label: '下装', icon: <ShopOutlined />, color: '#eb2f96' },
    { key: 'dress', label: '连衣裙', icon: <SkinOutlined />, color: '#52c41a' },
    { key: 'outerwear', label: '外套', icon: <ThunderboltOutlined />, color: '#faad14' },
    { key: 'shoes', label: '鞋子', icon: <CrownOutlined />, color: '#f5222d' },
    { key: 'accessories', label: '配饰', icon: <HeartOutlined />, color: '#722ed1' },
  ]

  const banners = [
    {
      title: '夏季新品上市',
      subtitle: '清凉一夏，时尚出行',
      gradient: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
      emoji: '☀️',
    },
    {
      title: '搭配指南',
      subtitle: '智能穿搭，轻松出门',
      gradient: 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)',
      emoji: '👗',
    },
    {
      title: '会员专享优惠',
      subtitle: '限时折扣，抢购中',
      gradient: 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)',
      emoji: '🎉',
    },
  ]

  const handleCategoryClick = (category: string) => {
    navigate(`/clothing?category=${category}`)
  }

  const handleItemClick = (item: any) => {
    navigate(`/clothing/${item.id}`)
  }

  const ProductCard: React.FC<{ item: any; showReason?: boolean }> = ({ item, showReason }) => {
    const categoryEmojis: Record<string, string> = {
      top: '👕',
      bottom: '👖',
      dress: '👗',
      outerwear: '🧥',
      shoes: '👟',
      accessories: '👜',
    }

    return (
    <Card
      hoverable
      onClick={() => handleItemClick(item)}
      cover={
        <div
          style={{
            height: 180,
            background: item.image_url ? '#f5f5f5' : `linear-gradient(180deg, #f5f5f5 0%, #e8e8e8 100%)`,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontSize: 48,
            overflow: 'hidden',
          }}
        >
          {item.image_url ? (
            <img
              alt={item.name}
              src={item.image_url}
              style={{
                width: '100%',
                height: '100%',
                objectFit: 'cover',
              }}
              onError={(e) => {
                const target = e.target as HTMLImageElement
                target.style.display = 'none'
              }}
            />
          ) : (
            categoryEmojis[item.category] || '👕'
          )}
        </div>
      }
      styles={{ body: { padding: 12 } }}
    >
      <Text
        style={{
          display: 'block',
          fontSize: 14,
          marginBottom: 4,
          overflow: 'hidden',
          textOverflow: 'ellipsis',
          whiteSpace: 'nowrap',
        }}
      >
        {item.name}
      </Text>
      <Row justify="space-between" align="middle">
        <Text strong style={{ color: '#f5222d', fontSize: 16 }}>
          ¥{item.price?.toFixed(2) || '0.00'}
        </Text>
        <Text type="secondary" style={{ fontSize: 12 }}>
          {item.category}
        </Text>
      </Row>
      {showReason && item.reason && (
        <Text
          type="secondary"
          style={{ fontSize: 11, display: 'block', marginTop: 4 }}
        >
          💡 {item.reason}
        </Text>
      )}
    </Card>
  )
  }

  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: '100px 0' }}>
        <Spin size="large" />
        <div style={{ marginTop: 16 }}>加载中...</div>
      </div>
    )
  }

  return (
    <div style={{ padding: '0 0 24px 0' }}>
      {/* 轮播Banner */}
      <Carousel autoplay dots style={{ marginBottom: 24 }}>
        {banners.map((banner, index) => (
          <div key={index}>
            <div
              style={{
                height: screens.lg ? 200 : 150,
                background: banner.gradient,
                borderRadius: 12,
                padding: screens.lg ? '40px 50px' : '24px 30px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                color: 'white',
              }}
            >
              <div>
                <Title level={screens.lg ? 2 : 4} style={{ color: 'white', margin: 0 }}>
                  {banner.emoji} {banner.title}
                </Title>
                <Text style={{ color: 'rgba(255,255,255,0.9)', fontSize: screens.lg ? 16 : 14 }}>
                  {banner.subtitle}
                </Text>
                <div style={{ marginTop: 16 }}>
                  <span
                    style={{
                      background: 'rgba(255,255,255,0.2)',
                      padding: '8px 20px',
                      borderRadius: 20,
                      cursor: 'pointer',
                      fontSize: 14,
                    }}
                  >
                    立即选购 <RightOutlined />
                  </span>
                </div>
              </div>
              <div style={{ fontSize: screens.lg ? 100 : 60 }}>{banner.emoji}</div>
            </div>
          </div>
        ))}
      </Carousel>

      {/* 分类入口 */}
      <Card
        title="热门分类"
        extra={
          <Text type="secondary" style={{ cursor: 'pointer' }}>
            查看全部 <RightOutlined />
          </Text>
        }
        style={{ marginBottom: 24 }}
      >
        <Row gutter={[16, 16]}>
          {categories.map((cat) => (
            <Col xs={8} sm={4} key={cat.key}>
              <div
                onClick={() => handleCategoryClick(cat.key)}
                style={{
                  textAlign: 'center',
                  padding: '16px 8px',
                  borderRadius: 12,
                  background: `${cat.color}15`,
                  cursor: 'pointer',
                  transition: 'all 0.3s',
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.background = `${cat.color}30`
                  e.currentTarget.style.transform = 'translateY(-2px)'
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.background = `${cat.color}15`
                  e.currentTarget.style.transform = 'translateY(0)'
                }}
              >
                <div style={{ fontSize: 32, marginBottom: 8 }}>{cat.icon}</div>
                <Text style={{ fontSize: 13, color: cat.color }}>{cat.label}</Text>
              </div>
            </Col>
          ))}
        </Row>
      </Card>

      {/* 猜你喜欢 */}
      <Card
        title={
          <Space>
            <Badge color="blue" />
            猜你喜欢
          </Space>
        }
        extra={
          <Text
            type="secondary"
            style={{ cursor: 'pointer' }}
            onClick={() => navigate('/recommend')}
          >
            查看更多 <RightOutlined />
          </Text>
        }
        style={{ marginBottom: 24 }}
      >
        {recommendations.length > 0 ? (
          <Row gutter={[12, 12]}>
            {recommendations.map((item) => (
              <Col xs={12} sm={8} md={4} key={item.id}>
                <ProductCard item={item} showReason />
              </Col>
            ))}
          </Row>
        ) : (
          <Empty description="暂无推荐数据" />
        )}
      </Card>

      {/* 热门榜单 & 新品上市 */}
      <Row gutter={24}>
        <Col xs={24} lg={12}>
          <Card
            title={
              <Space>
                <FireOutlined style={{ color: '#f5222d' }} />
                热门榜单
              </Space>
            }
            extra={
              <Text type="secondary" style={{ cursor: 'pointer' }}>
                查看更多 <RightOutlined />
              </Text>
            }
            style={{ marginBottom: 24 }}
          >
            {hotItems.length > 0 ? (
              <Row gutter={[12, 12]}>
                {hotItems.slice(0, 3).map((item, index) => (
                  <Col span={8} key={item.id}>
                    <Card
                      size="small"
                      hoverable
                      onClick={() => handleItemClick(item)}
                      cover={
                        <div
                          style={{
                            height: 100,
                            background: `linear-gradient(135deg, #ff6b6b ${index * 30}%, #feca57 100%)`,
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            fontSize: 36,
                          }}
                        >
                          #{index + 1}
                        </div>
                      }
                    >
                      <Text
                        style={{
                          display: 'block',
                          fontSize: 12,
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                          whiteSpace: 'nowrap',
                        }}
                      >
                        {item.name}
                      </Text>
                      <Text strong style={{ color: '#f5222d' }}>
                        ¥{item.price?.toFixed(2)}
                      </Text>
                    </Card>
                  </Col>
                ))}
              </Row>
            ) : (
              <Empty description="暂无数据" />
            )}
          </Card>
        </Col>

        <Col xs={24} lg={12}>
          <Card
            title={
              <Space>
                <StarOutlined style={{ color: '#52c41a' }} />
                新品上市
              </Space>
            }
            extra={
              <Text type="secondary" style={{ cursor: 'pointer' }}>
                查看更多 <RightOutlined />
              </Text>
            }
            style={{ marginBottom: 24 }}
          >
            {newItems.length > 0 ? (
              <Row gutter={[12, 12]}>
                {newItems.slice(0, 3).map((item, index) => (
                  <Col span={8} key={item.id}>
                    <Card
                      size="small"
                      hoverable
                      onClick={() => handleItemClick(item)}
                      cover={
                        <div
                          style={{
                            height: 100,
                            background: `linear-gradient(135deg, #a8edea ${index * 20}%, #fed6e3 100%)`,
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            fontSize: 36,
                          }}
                        >
                          <StarOutlined style={{ color: '#52c41a' }} />
                        </div>
                      }
                    >
                      <Text
                        style={{
                          display: 'block',
                          fontSize: 12,
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                          whiteSpace: 'nowrap',
                        }}
                      >
                        {item.name}
                      </Text>
                      <Text strong style={{ color: '#f5222d' }}>
                        ¥{item.price?.toFixed(2)}
                      </Text>
                    </Card>
                  </Col>
                ))}
              </Row>
            ) : (
              <Empty description="暂无数据" />
            )}
          </Card>
        </Col>
      </Row>

      {/* 快捷入口 */}
      <Card title="快捷功能">
        <Row gutter={[16, 16]}>
          <Col xs={12} sm={6}>
            <div
              onClick={() => navigate('/tryon')}
              style={{
                padding: 20,
                borderRadius: 12,
                background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                color: 'white',
                textAlign: 'center',
                cursor: 'pointer',
              }}
            >
              <div style={{ fontSize: 28, marginBottom: 8 }}>👗</div>
              <Text style={{ color: 'white' }}>虚拟试穿</Text>
            </div>
          </Col>
          <Col xs={12} sm={6}>
            <div
              onClick={() => navigate('/recommend')}
              style={{
                padding: 20,
                borderRadius: 12,
                background: 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)',
                color: 'white',
                textAlign: 'center',
                cursor: 'pointer',
              }}
            >
              <div style={{ fontSize: 28, marginBottom: 8 }}>✨</div>
              <Text style={{ color: 'white' }}>智能推荐</Text>
            </div>
          </Col>
          <Col xs={12} sm={6}>
            <div
              onClick={() => navigate('/clothing')}
              style={{
                padding: 20,
                borderRadius: 12,
                background: 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)',
                color: 'white',
                textAlign: 'center',
                cursor: 'pointer',
              }}
            >
              <div style={{ fontSize: 28, marginBottom: 8 }}>🔍</div>
              <Text style={{ color: 'white' }}>浏览全部</Text>
            </div>
          </Col>
          <Col xs={12} sm={6}>
            <div
              onClick={() => navigate('/knowledge-graph')}
              style={{
                padding: 20,
                borderRadius: 12,
                background: 'linear-gradient(135deg, #fa709a 0%, #fee140 100%)',
                color: 'white',
                textAlign: 'center',
                cursor: 'pointer',
              }}
            >
              <div style={{ fontSize: 28, marginBottom: 8 }}>🧠</div>
              <Text style={{ color: 'white' }}>知识图谱</Text>
            </div>
          </Col>
        </Row>
      </Card>

      {/* 浏览历史入口 */}
      <Card style={{ marginTop: 24 }}>
        <Row gutter={24} align="middle">
          <Col>
            <HistoryOutlined style={{ fontSize: 24, color: '#1890ff' }} />
          </Col>
          <Col flex="auto">
            <Title level={5} style={{ margin: 0 }}>浏览历史</Title>
            <Text type="secondary">记录您的浏览足迹</Text>
          </Col>
          <Col>
            <Text style={{ cursor: 'pointer', color: '#1890ff' }}>
              查看全部 <RightOutlined />
            </Text>
          </Col>
        </Row>
      </Card>
    </div>
  )
}

export default Dashboard
