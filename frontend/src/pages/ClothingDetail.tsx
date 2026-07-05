import React, { useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import {
  Card,
  Row,
  Col,
  Typography,
  Button,
  Spin,
  Empty,
  Breadcrumb,
  Rate,
  Avatar,
  Tag,
  Space,
  Divider,
  message,
} from 'antd'
import {
  ShoppingCartOutlined,
  HeartOutlined,
  StarOutlined,
  LeftOutlined,
  FireOutlined,
  SyncOutlined,
} from '@ant-design/icons'
import { clothingAPI, interactionAPI } from '../api'
import { useAuth } from '../context/AuthContext'

const { Title, Text, Paragraph } = Typography

interface ClothingItem {
  id: number
  name: string
  category: string
  price: number
  style?: string
  brand?: string
  color?: string
  material?: string
  season?: string
  occasion?: string
  description?: string
  sales_count?: number
  rating?: number
  image_url?: string
}

interface Review {
  id: number
  user_id: number
  user_name: string
  rating: number
  comment: string
  timestamp: string
}

const ClothingDetail: React.FC = () => {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()
  const { user } = useAuth()
  const [loading, setLoading] = useState(true)
  const [item, setItem] = useState<ClothingItem | null>(null)
  const [similarItems, setSimilarItems] = useState<ClothingItem[]>([])
  const [reviews, setReviews] = useState<Review[]>([])
  const [selectedColor, setSelectedColor] = useState('')
  const [selectedSize, setSelectedSize] = useState('')
  const [quantity, setQuantity] = useState(1)
  const [favorited, setFavorited] = useState(false)

  // 分类 emoji 映射
  const categoryEmojis: Record<string, string> = {
    top: '👕',
    bottom: '👖',
    dress: '👗',
    outerwear: '🧥',
    shoes: '👟',
    accessories: '👜',
  }

  // 颜色选项
  const colorOptions = ['黑色', '白色', '灰色', '蓝色', '红色', '粉色', '绿色', '米色']
  // 尺码选项
  const sizeOptions = ['XS', 'S', 'M', 'L', 'XL', 'XXL']

  // 获取商品详情
  useEffect(() => {
    const fetchData = async () => {
      if (!id) return
      setLoading(true)
      try {
        const clothingId = parseInt(id)
        const [itemData, similarData] = await Promise.all([
          clothingAPI.getById(clothingId),
          clothingAPI.getSimilar(clothingId, 4),
        ])
        setItem(itemData)
        setSimilarItems(similarData)

        // 记录浏览
        try {
          if (user) {
            await interactionAPI.recordView(user.id, clothingId, 30)
          }
        } catch (e) {
          // 忽略记录错误
        }

        // 模拟评价数据
        setReviews([
          {
            id: 1,
            user_id: 1,
            user_name: '时尚达人',
            rating: 5,
            comment: '质量很好，穿着舒适，款式也很时尚！',
            timestamp: '2026-06-28',
          },
          {
            id: 2,
            user_id: 2,
            user_name: '潮流买家',
            rating: 4,
            comment: '整体不错，就是尺码稍微偏大。',
            timestamp: '2026-06-25',
          },
          {
            id: 3,
            user_id: 3,
            user_name: '品质生活',
            rating: 5,
            comment: '非常满意，会回购的！',
            timestamp: '2026-06-20',
          },
        ])
      } catch (error) {
        console.error('Failed to fetch item:', error)
      } finally {
        setLoading(false)
      }
    }
    fetchData()
  }, [id, user])

  // 处理收藏
  const handleFavorite = () => {
    setFavorited(!favorited)
    message.success(favorited ? '已取消收藏' : '已添加到收藏')
  }

  // 处理购买
  const handleBuy = () => {
    if (!selectedColor || !selectedSize) {
      message.warning('请选择颜色和尺码')
      return
    }
    message.success('已添加到购物车')
  }

  // 处理加入购物车
  const handleAddToCart = () => {
    if (!selectedColor || !selectedSize) {
      message.warning('请选择颜色和尺码')
      return
    }
    message.success('已添加到购物车')
  }

  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: '100px 0' }}>
        <Spin size="large" />
        <div style={{ marginTop: 16 }}>加载中...</div>
      </div>
    )
  }

  if (!item) {
    return (
      <Empty description="商品不存在">
        <Button type="primary" onClick={() => navigate('/clothing')}>
          返回商品列表
        </Button>
      </Empty>
    )
  }

  return (
    <div style={{ padding: '0 0 24px 0' }}>
      {/* 面包屑 */}
      <Breadcrumb
        style={{ marginBottom: 16 }}
        items={[
          { title: <a onClick={() => navigate('/')}>首页</a> },
          { title: <a onClick={() => navigate('/clothing')}>商品列表</a> },
          { title: item.name },
        ]}
      />

      {/* 返回按钮 */}
      <Button
        type="link"
        icon={<LeftOutlined />}
        onClick={() => navigate('/clothing')}
        style={{ marginBottom: 16, padding: 0 }}
      >
        返回列表
      </Button>

      {/* 商品主体 */}
      <Card style={{ marginBottom: 24 }}>
        <Row gutter={[24, 24]}>
          {/* 图片画廊 */}
          <Col xs={24} md={10}>
            <div
              style={{
                background: item.image_url ? '#f5f5f5' : `linear-gradient(180deg, #f5f5f5 0%, #e8e8e8 100%)`,
                borderRadius: 12,
                height: 400,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: 120,
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
                    objectFit: 'contain',
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
            {/* 缩略图 */}
            <Row gutter={[8, 8]} style={{ marginTop: 12 }}>
              {[1, 2, 3, 4].map((i) => (
                <Col span={6} key={i}>
                  <div
                    style={{
                      background: '#f5f5f5',
                      borderRadius: 8,
                      height: 80,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      fontSize: 32,
                      cursor: 'pointer',
                      border: i === 1 ? '2px solid #1890ff' : '1px solid #e8e8e8',
                      overflow: 'hidden',
                    }}
                  >
                    {item.image_url ? (
                      <img
                        alt={item.name}
                        src={item.image_url}
                        style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                        onError={(e) => {
                          const target = e.target as HTMLImageElement
                          target.style.display = 'none'
                        }}
                      />
                    ) : (
                      categoryEmojis[item.category] || '👕'
                    )}
                  </div>
                </Col>
              ))}
            </Row>
          </Col>

          {/* 商品信息 */}
          <Col xs={24} md={14}>
            <Space direction="vertical" size="small" style={{ width: '100%' }}>
              <Title level={3} style={{ margin: 0 }}>
                {item.name}
              </Title>

              <Row align="middle" gutter={16}>
                <Col>
                  <Text strong style={{ color: '#f5222d', fontSize: 28 }}>
                    ¥{item.price?.toFixed(2) || '0.00'}
                  </Text>
                </Col>
                <Col>
                  <Tag color="red" icon={<FireOutlined />}>
                    热销中
                  </Tag>
                </Col>
                {item.sales_count && (
                  <Col>
                    <Text type="secondary">销量: {item.sales_count}</Text>
                  </Col>
                )}
              </Row>

              <Row align="middle" gutter={16}>
                <Col>
                  <Rate disabled defaultValue={item.rating || 4.5} allowHalf />
                </Col>
                <Col>
                  <Text type="secondary">({reviews.length} 条评价)</Text>
                </Col>
              </Row>

              <Divider style={{ margin: '12px 0' }} />

              {/* 分类和风格 */}
              <Row gutter={[16, 8]}>
                <Col>
                  <Text type="secondary">分类：</Text>
                  <Tag color="blue">{item.category}</Tag>
                </Col>
                <Col>
                  <Text type="secondary">风格：</Text>
                  <Tag color="green">{item.style || 'casual'}</Tag>
                </Col>
                {item.brand && (
                  <Col>
                    <Text type="secondary">品牌：</Text>
                    <Tag>{item.brand}</Tag>
                  </Col>
                )}
              </Row>

              {/* 颜色选择 */}
              <div>
                <Text strong style={{ marginRight: 16 }}>颜色：</Text>
                <Space>
                  {colorOptions.map((color) => (
                    <Tag
                      key={color}
                      color={selectedColor === color ? 'blue' : 'default'}
                      style={{ cursor: 'pointer', padding: '4px 12px' }}
                      onClick={() => setSelectedColor(color)}
                    >
                      {color}
                    </Tag>
                  ))}
                </Space>
              </div>

              {/* 尺码选择 */}
              <div>
                <Text strong style={{ marginRight: 16 }}>尺码：</Text>
                <Space>
                  {sizeOptions.map((size) => (
                    <Tag
                      key={size}
                      color={selectedSize === size ? 'blue' : 'default'}
                      style={{
                        cursor: 'pointer',
                        padding: '4px 16px',
                        fontSize: 14,
                      }}
                      onClick={() => setSelectedSize(size)}
                    >
                      {size}
                    </Tag>
                  ))}
                </Space>
              </div>

              {/* 数量选择 */}
              <div>
                <Text strong style={{ marginRight: 16 }}>数量：</Text>
                <Space>
                  <Button
                    size="small"
                    disabled={quantity <= 1}
                    onClick={() => setQuantity(q => q - 1)}
                  >
                    -
                  </Button>
                  <Text style={{ minWidth: 40, textAlign: 'center', display: 'inline-block' }}>
                    {quantity}
                  </Text>
                  <Button size="small" onClick={() => setQuantity(q => q + 1)}>
                    +
                  </Button>
                </Space>
              </div>

              <Divider style={{ margin: '12px 0' }} />

              {/* 操作按钮 */}
              <Row gutter={12}>
                <Col>
                  <Button
                    type="primary"
                    size="large"
                    icon={<ShoppingCartOutlined />}
                    onClick={handleAddToCart}
                  >
                    加入购物车
                  </Button>
                </Col>
                <Col>
                  <Button size="large" type="primary" danger onClick={handleBuy}>
                    立即购买
                  </Button>
                </Col>
                <Col>
                  <Button
                    size="large"
                    icon={<HeartOutlined />}
                    onClick={handleFavorite}
                    type={favorited ? 'primary' : 'default'}
                  >
                    {favorited ? '已收藏' : '收藏'}
                  </Button>
                </Col>
              </Row>
            </Space>
          </Col>
        </Row>
      </Card>

      {/* 商品详情和评价 */}
      <Row gutter={24}>
        <Col xs={24} lg={16}>
          {/* 商品详情 */}
          <Card title="商品详情" style={{ marginBottom: 24 }}>
            <Paragraph>
              {item.description || `这是一款优质的${item.category}，采用优质面料制作，舒适透气，款式时尚。`}
            </Paragraph>
            <Row gutter={[16, 8]}>
              {item.material && (
                <Col span={12}>
                  <Text type="secondary">材质：</Text> {item.material}
                </Col>
              )}
              {item.season && (
                <Col span={12}>
                  <Text type="secondary">季节：</Text> {item.season}
                </Col>
              )}
              {item.occasion && (
                <Col span={12}>
                  <Text type="secondary">场合：</Text> {item.occasion}
                </Col>
              )}
              {item.color && (
                <Col span={12}>
                  <Text type="secondary">颜色：</Text> {item.color}
                </Col>
              )}
            </Row>
          </Card>

          {/* 评价列表 */}
          <Card title="商品评价">
            {reviews.length > 0 ? (
              <Space direction="vertical" style={{ width: '100%' }} size="middle">
                {reviews.map((review) => (
                  <Card size="small" key={review.id}>
                    <Row justify="space-between" align="top">
                      <Col>
                        <Space>
                          <Avatar style={{ backgroundColor: '#1890ff' }}>
                            {review.user_name.charAt(0)}
                          </Avatar>
                          <div>
                            <Text strong>{review.user_name}</Text>
                            <div>
                              <Rate disabled value={review.rating} style={{ fontSize: 12 }} />
                            </div>
                          </div>
                        </Space>
                      </Col>
                      <Col>
                        <Text type="secondary">{review.timestamp}</Text>
                      </Col>
                    </Row>
                    <Paragraph style={{ marginTop: 8, marginBottom: 0 }}>
                      {review.comment}
                    </Paragraph>
                  </Card>
                ))}
              </Space>
            ) : (
              <Empty description="暂无评价" />
            )}
          </Card>
        </Col>

        <Col xs={24} lg={8}>
          {/* 相似商品 */}
          <Card
            title={
              <Space>
                <SyncOutlined />
                相似商品
              </Space>
            }
            extra={
              <a onClick={() => navigate('/clothing')}>查看更多</a>
            }
          >
            {similarItems.length > 0 ? (
              <Space direction="vertical" style={{ width: '100%' }} size="middle">
                {similarItems.map((simItem) => (
                  <Card
                    key={simItem.id}
                    size="small"
                    hoverable
                    onClick={() => navigate(`/clothing/${simItem.id}`)}
                    cover={
                      <div
                        style={{
                          height: 100,
                          background: '#f5f5f5',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          fontSize: 36,
                          overflow: 'hidden',
                        }}
                      >
                        {simItem.image_url ? (
                          <img
                            alt={simItem.name}
                            src={simItem.image_url}
                            style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                            onError={(e) => {
                              const target = e.target as HTMLImageElement
                              target.style.display = 'none'
                            }}
                          />
                        ) : (
                          categoryEmojis[simItem.category] || '👕'
                        )}
                      </div>
                    }
                  >
                    <Text
                      style={{
                        display: 'block',
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        whiteSpace: 'nowrap',
                      }}
                    >
                      {simItem.name}
                    </Text>
                    <Text strong style={{ color: '#f5222d' }}>
                      ¥{simItem.price?.toFixed(2)}
                    </Text>
                  </Card>
                ))}
              </Space>
            ) : (
              <Empty description="暂无相似商品" />
            )}
          </Card>

          {/* 搭配推荐 */}
          <Card
            title={
              <Space>
                <StarOutlined />
                搭配推荐
              </Space>
            }
            style={{ marginTop: 24 }}
          >
            <Space direction="vertical" style={{ width: '100%' }}>
              <div
                style={{
                  padding: 16,
                  background: '#f5f5f5',
                  borderRadius: 8,
                  textAlign: 'center',
                }}
              >
                <Text type="secondary">根据当前商品智能搭配</Text>
                <Button type="link" block onClick={() => navigate('/recommend')}>
                  查看完整搭配方案
                </Button>
              </div>
            </Space>
          </Card>
        </Col>
      </Row>
    </div>
  )
}

export default ClothingDetail
