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
  Avatar,
  Timeline,
  Divider,
  message,
} from 'antd'
import {
  LikeOutlined,
  StarOutlined,
  HistoryOutlined,
  FireOutlined,
  UserOutlined,
  CheckCircleOutlined,
} from '@ant-design/icons'
import { useNavigate } from 'react-router-dom'
import { userAPI } from '../api'
import { useAuth } from '../context/AuthContext'

const { Title, Text } = Typography

interface ClothingItem {
  id: number
  name: string
  category: string
  price: number
  style?: string
  brand?: string
  image_url?: string
}

interface Recommendation {
  clothing_id: number
  score: number
  reason: string
  item: ClothingItem
}

interface BrowseHistory {
  id: number
  clothing_id: number
  item: ClothingItem
  viewed_at: string
}

const Recommendations: React.FC = () => {
  const navigate = useNavigate()
  const { user, isLoading: authLoading } = useAuth()
  const userId = user?.id

  const [loading, setLoading] = useState(true)
  const [recommendations, setRecommendations] = useState<Recommendation[]>([])
  const [similarUsers, setSimilarUsers] = useState<any[]>([])
  const [browseHistory, setBrowseHistory] = useState<BrowseHistory[]>([])
  const [outfits, setOutfits] = useState<any[]>([])

  // 分类 emoji 映射
  const categoryEmojis: Record<string, string> = {
    top: '👕',
    bottom: '👖',
    dress: '👗',
    outerwear: '🧥',
    shoes: '👟',
    accessories: '👜',
  }

  // 获取推荐数据
  useEffect(() => {
    if (authLoading) return
    if (!userId) return
    const fetchData = async () => {
      setLoading(true)
      try {
        // 获取猜你喜欢推荐
        const recData = await userAPI.getRecommendations(userId, undefined, 8)
        setRecommendations(
          recData.map((r: any) => ({
            clothing_id: r.clothing_id,
            score: r.score,
            reason: r.reason || getDefaultReason(r),
            item: r.item || r,
          }))
        )

        // 模拟相似用户数据
        setSimilarUsers([
          { id: 2, name: '时尚达人', avatar: '👩', match: 95 },
          { id: 3, name: '潮流教主', avatar: '👨', match: 88 },
          { id: 4, name: '品质生活', avatar: '👩', match: 82 },
        ])

        // 模拟浏览历史
        setBrowseHistory([
          {
            id: 1,
            clothing_id: 101,
            item: { id: 101, name: '简约白色T恤', category: 'top', price: 129 },
            viewed_at: '2026-06-29 15:30',
          },
          {
            id: 2,
            clothing_id: 102,
            item: { id: 102, name: '破洞牛仔裤', category: 'bottom', price: 299 },
            viewed_at: '2026-06-29 14:20',
          },
          {
            id: 3,
            clothing_id: 103,
            item: { id: 103, name: '运动休闲鞋', category: 'shoes', price: 459 },
            viewed_at: '2026-06-28 20:15',
          },
        ])

        // 模拟搭配套装
        setOutfits([
          {
            id: 1,
            name: '简约日常风',
            items: [
              { id: 101, name: '白色T恤', category: 'top' },
              { id: 201, name: '牛仔裤', category: 'bottom' },
              { id: 301, name: '小白鞋', category: 'shoes' },
            ],
          },
          {
            id: 2,
            name: '商务休闲风',
            items: [
              { id: 102, name: '衬衫', category: 'top' },
              { id: 202, name: '西裤', category: 'bottom' },
              { id: 302, name: '皮鞋', category: 'shoes' },
            ],
          },
        ])
      } catch (error) {
        console.error('Failed to fetch recommendations:', error)
      } finally {
        setLoading(false)
      }
    }
    fetchData()
  }, [userId, authLoading])

  // 获取默认推荐理由
  const getDefaultReason = (_item: any): string => {
    const reasons = [
      '根据您的风格偏好推荐',
      '与您浏览过的商品相似',
      '热门款式，销量领先',
      '搭配神器，百搭之选',
      '今季流行趋势',
    ]
    return reasons[Math.floor(Math.random() * reasons.length)]
  }

  // 处理商品点击
  const handleItemClick = (item: ClothingItem) => {
    navigate(`/clothing/${item.id}`)
  }

  // 记录收藏
  const handleFavorite = (_clothingId: number) => {
    message.success('已添加到收藏')
  }

  // 商品卡片
  const RecommendationCard = ({ rec }: { rec: Recommendation }) => (
    <Card
      hoverable
      onClick={() => handleItemClick(rec.item)}
      cover={
        <div
          style={{
            height: 160,
            background: rec.item.image_url ? '#f5f5f5' : `linear-gradient(180deg, #f5f5f5 0%, #e8e8e8 100%)`,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontSize: 48,
            overflow: 'hidden',
          }}
        >
          {rec.item.image_url ? (
            <img
              alt={rec.item.name}
              src={rec.item.image_url}
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
            categoryEmojis[rec.item.category] || '👕'
          )}
        </div>
      }
      actions={[
        <StarOutlined key="收藏" onClick={(e) => { e.stopPropagation(); handleFavorite(rec.clothing_id) }} />,
        <LikeOutlined key="喜欢" />,
      ]}
    >
      <Card.Meta
        title={
          <Text
            style={{ fontSize: 14 }}
            ellipsis={{ tooltip: rec.item.name }}
          >
            {rec.item.name}
          </Text>
        }
        description={
          <>
            <Text strong style={{ color: '#f5222d', fontSize: 16 }}>
              ¥{rec.item.price?.toFixed(2) || '0.00'}
            </Text>
            <div style={{ marginTop: 8 }}>
              <Tag color="blue">{rec.item.category}</Tag>
              {rec.item.style && <Tag color="green">{rec.item.style}</Tag>}
            </div>
            <div
              style={{
                marginTop: 8,
                fontSize: 12,
                color: '#1890ff',
                background: '#e6f7ff',
                padding: '4px 8px',
                borderRadius: 4,
              }}
            >
              💡 {rec.reason}
            </div>
          </>
        }
      />
    </Card>
  )

  if (!authLoading && !userId) {
    return (
      <div style={{ textAlign: 'center', padding: '100px 0' }}>
        <Empty description="请先登录以查看个性化推荐">
          <Button type="primary" onClick={() => navigate('/login')}>
            去登录
          </Button>
        </Empty>
      </div>
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
      {/* 标题 */}
      <div style={{ marginBottom: 24 }}>
        <Title level={3} style={{ margin: 0 }}>
          <Badge color="blue" />
          个性化推荐
        </Title>
        <Text type="secondary">为您精选符合您风格的商品</Text>
      </div>

      <Row gutter={24}>
        {/* 左侧主体 */}
        <Col xs={24} lg={16}>
          {/* 推荐理由说明 */}
          <Card style={{ marginBottom: 24 }}>
            <Row gutter={[24, 16]} align="middle">
              <Col>
                <div style={{ fontSize: 32 }}>🎯</div>
              </Col>
              <Col flex={1}>
                <Title level={5} style={{ margin: 0 }}>推荐算法说明</Title>
                <Text type="secondary">
                  基于您的浏览历史、收藏记录和风格偏好，采用 KG-HybridRec 混合推荐算法为您推荐。
                </Text>
              </Col>
              <Col>
                <Button type="link" onClick={() => navigate('/knowledge-graph')}>
                  查看知识图谱
                </Button>
              </Col>
            </Row>
          </Card>

          {/* 猜你喜欢 */}
          <Card
            title={
              <Space>
                <LikeOutlined style={{ color: '#1890ff' }} />
                猜你喜欢
              </Space>
            }
            extra={
              <Text type="secondary">
                基于您的行为推荐
              </Text>
            }
            style={{ marginBottom: 24 }}
          >
            {recommendations.length > 0 ? (
              <Row gutter={[12, 12]}>
                {recommendations.map((rec) => (
                  <Col xs={12} sm={12} md={8} lg={6} key={rec.clothing_id}>
                    <RecommendationCard rec={rec} />
                  </Col>
                ))}
              </Row>
            ) : (
              <Empty description="暂无推荐数据" />
            )}
          </Card>

          {/* 搭配套装推荐 */}
          <Card
            title={
              <Space>
                <FireOutlined style={{ color: '#f5222d' }} />
                搭配套装
              </Space>
            }
            style={{ marginBottom: 24 }}
          >
            <Row gutter={[16, 16]}>
              {outfits.map((outfit) => (
                <Col xs={24} sm={12} key={outfit.id}>
                  <Card size="small" hoverable>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                      {outfit.items.map((item: any, idx: number) => (
                        <React.Fragment key={item.id}>
                          <div
                            style={{
                              width: 60,
                              height: 60,
                              background: '#f5f5f5',
                              borderRadius: 8,
                              display: 'flex',
                              alignItems: 'center',
                              justifyContent: 'center',
                              fontSize: 24,
                              cursor: 'pointer',
                            }}
                            onClick={() => navigate(`/clothing/${item.id}`)}
                          >
                            {categoryEmojis[item.category] || '👕'}
                          </div>
                          {idx < outfit.items.length - 1 && <span style={{ color: '#ccc' }}>+</span>}
                        </React.Fragment>
                      ))}
                    </div>
                    <Text
                      style={{ display: 'block', marginTop: 8, textAlign: 'center' }}
                    >
                      {outfit.name}
                    </Text>
                  </Card>
                </Col>
              ))}
            </Row>
          </Card>
        </Col>

        {/* 右侧边栏 */}
        <Col xs={24} lg={8}>
          {/* 相似用户推荐 */}
          <Card
            title={
              <Space>
                <UserOutlined />
                相似用户
              </Space>
            }
            style={{ marginBottom: 24 }}
          >
            <Space direction="vertical" style={{ width: '100%' }}>
              {similarUsers.map((user) => (
                <Card key={user.id} size="small">
                  <Row align="middle" gutter={12}>
                    <Col>
                      <Avatar size="large" style={{ backgroundColor: '#1890ff' }}>
                        {user.avatar}
                      </Avatar>
                    </Col>
                    <Col flex={1}>
                      <Text strong>{user.name}</Text>
                      <div>
                        <Text type="secondary" style={{ fontSize: 12 }}>
                          喜好匹配度
                        </Text>
                      </div>
                    </Col>
                    <Col>
                      <Tag color="green">{user.match}%</Tag>
                    </Col>
                  </Row>
                </Card>
              ))}
            </Space>
          </Card>

          {/* 浏览历史 */}
          <Card
            title={
              <Space>
                <HistoryOutlined />
                浏览历史
              </Space>
            }
            extra={
              <Button type="link" size="small" onClick={() => navigate('/users')}>
                查看全部
              </Button>
            }
          >
            {browseHistory.length > 0 ? (
              <Timeline
                items={browseHistory.map((history) => ({
                  color: 'blue',
                  children: (
                    <div
                      style={{ cursor: 'pointer' }}
                      onClick={() => navigate(`/clothing/${history.clothing_id}`)}
                    >
                      <Text style={{ display: 'block' }}>
                        {history.item.name}
                      </Text>
                      <Row justify="space-between" align="middle">
                        <Text type="secondary" style={{ fontSize: 12 }}>
                          ¥{history.item.price}
                        </Text>
                        <Text type="secondary" style={{ fontSize: 11 }}>
                          {history.viewed_at}
                        </Text>
                      </Row>
                    </div>
                  ),
                }))}
              />
            ) : (
              <Empty description="暂无浏览记录" image={Empty.PRESENTED_IMAGE_SIMPLE} />
            )}
          </Card>

          {/* 风格偏好 */}
          <Card
            title={
              <Space>
                <CheckCircleOutlined style={{ color: '#52c41a' }} />
                我的风格
              </Space>
            }
            style={{ marginTop: 24 }}
          >
            <Space wrap>
              <Tag color="blue">简约</Tag>
              <Tag color="green">休闲</Tag>
              <Tag color="orange">运动</Tag>
              <Tag color="purple">百搭</Tag>
            </Space>
            <Divider style={{ margin: '12px 0' }} />
            <Button type="link" block onClick={() => navigate('/users')}>
              编辑偏好设置
            </Button>
          </Card>
        </Col>
      </Row>
    </div>
  )
}

export default Recommendations
