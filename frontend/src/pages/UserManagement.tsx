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
  Table,
  Avatar,
  Divider,
  Tabs,
  Progress,
} from 'antd'
import {
  UserOutlined,
  EditOutlined,
  HistoryOutlined,
  HeartOutlined,
  SettingOutlined,
  LockOutlined,
  SafetyOutlined,
} from '@ant-design/icons'
import type { ColumnsType } from 'antd/es/table'
import { userAPI } from '../api'

const { Title, Text } = Typography

interface User {
  id: number
  username: string
  email?: string
  gender?: string
  age?: number
  height?: number
  weight?: number
  body_shape?: string
  skin_tone?: string
  preferred_style?: string
  created_at?: string
}

interface UserProfile {
  user_id: number
  static_info: User
  style_distribution: Record<string, number>
  color_preferences: Record<string, number>
  price_range: { min: number; max: number; avg: number }
  interaction_count: number
}

interface BrowseHistoryItem {
  id: number
  clothing_id: number
  clothing_name: string
  category: string
  viewed_at: string
}

interface FavoriteItem {
  id: number
  clothing_id: number
  clothing_name: string
  category: string
  price: number
  favorited_at: string
}

const UserManagement: React.FC = () => {
  const currentUserId = 1 // 模拟当前用户

  const [loading, setLoading] = useState(true)
  const [profile, setProfile] = useState<UserProfile | null>(null)
  const [browseHistory, setBrowseHistory] = useState<BrowseHistoryItem[]>([])
  const [favorites, setFavorites] = useState<FavoriteItem[]>([])
  const [activeTab, setActiveTab] = useState<string>('profile')

  // 体型标签颜色
  const bodyShapeColors: Record<string, string> = {
    slim: 'cyan',
    athletic: 'blue',
    average: 'green',
    curvy: 'orange',
    plus_size: 'red',
  }

  // 分类 emoji 映射
  const categoryEmojis: Record<string, string> = {
    top: '👕',
    bottom: '👖',
    dress: '👗',
    outerwear: '🧥',
    shoes: '👟',
    accessories: '👜',
  }

  // 获取用户资料
  useEffect(() => {
    fetchUserProfile()
    fetchBrowseHistory()
    fetchFavorites()
  }, [currentUserId])

  const fetchUserProfile = async () => {
    setLoading(true)
    try {
      const data = await userAPI.getProfile(currentUserId)
      // 模拟完整的用户资料
      setProfile({
        user_id: currentUserId,
        static_info: data,
        style_distribution: {
          '简约': 0.35,
          '休闲': 0.25,
          '运动': 0.20,
          '商务': 0.15,
          '复古': 0.05,
        },
        color_preferences: {
          '黑色': 28,
          '白色': 22,
          '蓝色': 18,
          '灰色': 12,
          '红色': 8,
          '其他': 12,
        },
        price_range: { min: 100, max: 800, avg: 380 },
        interaction_count: 156,
      })
    } catch (error) {
      console.error('Failed to fetch profile:', error)
      // 即使API失败也显示模拟数据
      setProfile({
        user_id: currentUserId,
        static_info: {
          id: currentUserId,
          username: '时尚达人',
          email: 'user@example.com',
          gender: '女',
          age: 28,
          height: 165,
          weight: 52,
          body_shape: 'average',
          skin_tone: '白皙',
          preferred_style: '简约休闲',
        },
        style_distribution: {
          '简约': 0.35,
          '休闲': 0.25,
          '运动': 0.20,
          '商务': 0.15,
          '复古': 0.05,
        },
        color_preferences: {
          '黑色': 28,
          '白色': 22,
          '蓝色': 18,
          '灰色': 12,
          '红色': 8,
          '其他': 12,
        },
        price_range: { min: 100, max: 800, avg: 380 },
        interaction_count: 156,
      })
    } finally {
      setLoading(false)
    }
  }

  const fetchBrowseHistory = async () => {
    // 模拟浏览历史数据
    setBrowseHistory([
      {
        id: 1,
        clothing_id: 101,
        clothing_name: '简约白色T恤',
        category: 'top',
        viewed_at: '2026-06-29 15:30',
      },
      {
        id: 2,
        clothing_id: 102,
        clothing_name: '破洞牛仔裤',
        category: 'bottom',
        viewed_at: '2026-06-29 14:20',
      },
      {
        id: 3,
        clothing_id: 103,
        clothing_name: '运动休闲鞋',
        category: 'shoes',
        viewed_at: '2026-06-28 20:15',
      },
      {
        id: 4,
        clothing_id: 104,
        clothing_name: '轻薄风衣',
        category: 'outerwear',
        viewed_at: '2026-06-28 16:45',
      },
      {
        id: 5,
        clothing_id: 105,
        clothing_name: '碎花连衣裙',
        category: 'dress',
        viewed_at: '2026-06-27 18:00',
      },
    ])
  }

  const fetchFavorites = async () => {
    // 模拟收藏数据
    setFavorites([
      {
        id: 1,
        clothing_id: 201,
        clothing_name: '韩版衬衫',
        category: 'top',
        price: 199,
        favorited_at: '2026-06-28',
      },
      {
        id: 2,
        clothing_id: 202,
        clothing_name: '高腰牛仔裤',
        category: 'bottom',
        price: 299,
        favorited_at: '2026-06-27',
      },
      {
        id: 3,
        clothing_id: 203,
        clothing_name: '复古手提包',
        category: 'accessories',
        price: 459,
        favorited_at: '2026-06-25',
      },
    ])
  }

  // 风格偏好进度条
  const renderStylePreferences = () => {
    if (!profile?.style_distribution) return null

    return (
      <Card title="风格偏好" style={{ marginBottom: 24 }}>
        <Space direction="vertical" style={{ width: '100%' }} size="middle">
          {Object.entries(profile.style_distribution)
            .sort(([, a], [, b]) => b - a)
            .map(([style, ratio]) => (
              <div key={style}>
                <Row justify="space-between" align="middle">
                  <Text>{style}</Text>
                  <Text type="secondary">{(ratio * 100).toFixed(0)}%</Text>
                </Row>
                <Progress
                  percent={Math.round(ratio * 100)}
                  showInfo={false}
                  strokeColor="#1890ff"
                  trailColor="#f0f0f0"
                />
              </div>
            ))}
        </Space>
      </Card>
    )
  }

  // 颜色偏好
  const renderColorPreferences = () => {
    if (!profile?.color_preferences) return null

    const total = Object.values(profile.color_preferences).reduce((a, b) => a + b, 0)

    return (
      <Card title="颜色偏好" style={{ marginBottom: 24 }}>
        <Row gutter={[12, 12]}>
          {Object.entries(profile.color_preferences)
            .sort(([, a], [, b]) => b - a)
            .map(([color, count]) => (
              <Col span={12} key={color}>
                <Row justify="space-between" align="middle">
                  <Space>
                    <div
                      style={{
                        width: 16,
                        height: 16,
                        borderRadius: 4,
                        background: getColorHex(color),
                        border: '1px solid #d9d9d9',
                      }}
                    />
                    <Text>{color}</Text>
                  </Space>
                  <Text type="secondary">{((count / total) * 100).toFixed(0)}%</Text>
                </Row>
              </Col>
            ))}
        </Row>
      </Card>
    )
  }

  // 获取颜色对应的十六进制值
  const getColorHex = (color: string): string => {
    const colorMap: Record<string, string> = {
      '黑色': '#000000',
      '白色': '#FFFFFF',
      '蓝色': '#1890ff',
      '灰色': '#8c8c8c',
      '红色': '#f5222d',
      '绿色': '#52c41a',
      '黄色': '#faad14',
      '紫色': '#722ed1',
      '粉色': '#eb2f96',
      '橙色': '#fa8c16',
    }
    return colorMap[color] || '#d9d9d9'
  }

  // 浏览历史表格列
  const historyColumns: ColumnsType<BrowseHistoryItem> = [
    {
      title: '商品',
      key: 'clothing',
      render: (_, record) => (
        <Space>
          <div
            style={{
              width: 40,
              height: 40,
              background: '#f5f5f5',
              borderRadius: 8,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: 20,
            }}
          >
            {categoryEmojis[record.category] || '👕'}
          </div>
          <Text>{record.clothing_name}</Text>
        </Space>
      ),
    },
    {
      title: '分类',
      dataIndex: 'category',
      key: 'category',
      render: (cat) => <Tag>{cat}</Tag>,
    },
    {
      title: '浏览时间',
      dataIndex: 'viewed_at',
      key: 'viewed_at',
      render: (time) => <Text type="secondary">{time}</Text>,
    },
    {
      title: '操作',
      key: 'action',
      render: () => (
        <Button type="link" size="small">
          查看详情
        </Button>
      ),
    },
  ]

  // 收藏表格列
  const favoriteColumns: ColumnsType<FavoriteItem> = [
    {
      title: '商品',
      key: 'clothing',
      render: (_, record) => (
        <Space>
          <div
            style={{
              width: 40,
              height: 40,
              background: '#f5f5f5',
              borderRadius: 8,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: 20,
            }}
          >
            {categoryEmojis[record.category] || '👕'}
          </div>
          <Text>{record.clothing_name}</Text>
        </Space>
      ),
    },
    {
      title: '分类',
      dataIndex: 'category',
      key: 'category',
      render: (cat) => <Tag color="blue">{cat}</Tag>,
    },
    {
      title: '价格',
      dataIndex: 'price',
      key: 'price',
      render: (price) => <Text strong style={{ color: '#f5222d' }}>¥{price}</Text>,
    },
    {
      title: '收藏时间',
      dataIndex: 'favorited_at',
      key: 'favorited_at',
      render: (time) => <Text type="secondary">{time}</Text>,
    },
    {
      title: '操作',
      key: 'action',
      render: () => (
        <Space>
          <Button type="link" size="small" danger>
            取消收藏
          </Button>
        </Space>
      ),
    },
  ]

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
          <Badge color="green" />
          个人中心
        </Title>
        <Text type="secondary">管理您的个人信息和偏好设置</Text>
      </div>

      <Row gutter={24}>
        {/* 左侧 - 用户信息 */}
        <Col xs={24} lg={8}>
          <Card>
            <div style={{ textAlign: 'center', marginBottom: 24 }}>
              <Avatar
                size={80}
                icon={<UserOutlined />}
                style={{ backgroundColor: '#1890ff', marginBottom: 16 }}
              />
              <Title level={4} style={{ margin: 0 }}>
                {profile?.static_info?.username || '用户'}
              </Title>
              <Text type="secondary">{profile?.static_info?.email}</Text>
              <div style={{ marginTop: 8 }}>
                <Tag color="blue">{profile?.static_info?.preferred_style || '简约休闲'}</Tag>
              </div>
            </div>

            <Divider />

            <Space direction="vertical" style={{ width: '100%' }} size="small">
              <Row justify="space-between">
                <Text type="secondary">性别</Text>
                <Text>{profile?.static_info?.gender || '-'}</Text>
              </Row>
              <Row justify="space-between">
                <Text type="secondary">年龄</Text>
                <Text>{profile?.static_info?.age || '-'} 岁</Text>
              </Row>
              <Row justify="space-between">
                <Text type="secondary">身高</Text>
                <Text>{profile?.static_info?.height || '-'} cm</Text>
              </Row>
              <Row justify="space-between">
                <Text type="secondary">体重</Text>
                <Text>{profile?.static_info?.weight || '-'} kg</Text>
              </Row>
              <Row justify="space-between">
                <Text type="secondary">体型</Text>
                <Tag color={bodyShapeColors[profile?.static_info?.body_shape || ''] || 'default'}>
                  {profile?.static_info?.body_shape || '未知'}
                </Tag>
              </Row>
              <Row justify="space-between">
                <Text type="secondary">肤质</Text>
                <Text>{profile?.static_info?.skin_tone || '-'}</Text>
              </Row>
            </Space>

            <Divider />

            <Space style={{ width: '100%' }}>
              <Button icon={<EditOutlined />} block>
                编辑资料
              </Button>
            </Space>
          </Card>

          {/* 消费偏好 */}
          <Card title="消费偏好" style={{ marginTop: 24 }}>
            <Space direction="vertical" style={{ width: '100%' }} size="middle">
              <Row justify="space-between">
                <Text type="secondary">价格区间</Text>
                <Text>
                  ¥{profile?.price_range?.min || 0} - ¥{profile?.price_range?.max || 0}
                </Text>
              </Row>
              <Row justify="space-between">
                <Text type="secondary">平均消费</Text>
                <Text strong style={{ color: '#f5222d' }}>
                  ¥{profile?.price_range?.avg?.toFixed(0) || 0}
                </Text>
              </Row>
              <Row justify="space-between">
                <Text type="secondary">交互次数</Text>
                <Badge count={profile?.interaction_count || 0} showZero />
              </Row>
            </Space>
          </Card>
        </Col>

        {/* 右侧 - Tab切换 */}
        <Col xs={24} lg={16}>
          <Tabs
            activeKey={activeTab}
            onChange={setActiveTab}
            items={[
              {
                key: 'profile',
                label: (
                  <Space>
                    <UserOutlined />
                    偏好设置
                  </Space>
                ),
                children: (
                  <div>
                    {renderStylePreferences()}
                    {renderColorPreferences()}

                    <Card title="账号安全" style={{ marginBottom: 24 }}>
                      <Space direction="vertical" style={{ width: '100%' }}>
                        <Row align="middle" gutter={12}>
                          <LockOutlined style={{ fontSize: 20, color: '#1890ff' }} />
                          <div style={{ flex: 1 }}>
                            <Text strong>登录密码</Text>
                            <br />
                            <Text type="secondary" style={{ fontSize: 12 }}>
                              已设置
                            </Text>
                          </div>
                          <Button type="link" icon={<EditOutlined />}>
                            修改
                          </Button>
                        </Row>
                        <Divider style={{ margin: '12px 0' }} />
                        <Row align="middle" gutter={12}>
                          <SafetyOutlined style={{ fontSize: 20, color: '#52c41a' }} />
                          <div style={{ flex: 1 }}>
                            <Text strong>绑定手机</Text>
                            <br />
                            <Text type="secondary" style={{ fontSize: 12 }}>
                              138****8888
                            </Text>
                          </div>
                          <Button type="link" icon={<EditOutlined />}>
                            修改
                          </Button>
                        </Row>
                      </Space>
                    </Card>
                  </div>
                ),
              },
              {
                key: 'history',
                label: (
                  <Space>
                    <HistoryOutlined />
                    浏览历史
                    <Badge count={browseHistory.length} size="small" />
                  </Space>
                ),
                children: (
                  <Card>
                    {browseHistory.length > 0 ? (
                      <Table
                        columns={historyColumns}
                        dataSource={browseHistory}
                        rowKey="id"
                        pagination={{ pageSize: 5 }}
                        size="small"
                      />
                    ) : (
                      <Empty description="暂无浏览记录" />
                    )}
                  </Card>
                ),
              },
              {
                key: 'favorites',
                label: (
                  <Space>
                    <HeartOutlined />
                    我的收藏
                    <Badge count={favorites.length} size="small" />
                  </Space>
                ),
                children: (
                  <Card>
                    {favorites.length > 0 ? (
                      <Table
                        columns={favoriteColumns}
                        dataSource={favorites}
                        rowKey="id"
                        pagination={{ pageSize: 5 }}
                        size="small"
                      />
                    ) : (
                      <Empty description="暂无收藏">
                        <Button type="primary" onClick={() => window.location.href = '/clothing'}>
                          去逛逛
                        </Button>
                      </Empty>
                    )}
                  </Card>
                ),
              },
              {
                key: 'settings',
                label: (
                  <Space>
                    <SettingOutlined />
                    设置
                  </Space>
                ),
                children: (
                  <Card title="消息通知">
                    <Space direction="vertical" style={{ width: '100%' }}>
                      <Row justify="space-between" align="middle">
                        <div>
                          <Text strong>推荐通知</Text>
                          <br />
                          <Text type="secondary" style={{ fontSize: 12 }}>
                            接收个性化商品推荐
                          </Text>
                        </div>
                        <Tag color="green">已开启</Tag>
                      </Row>
                      <Divider style={{ margin: '12px 0' }} />
                      <Row justify="space-between" align="middle">
                        <div>
                          <Text strong>活动通知</Text>
                          <br />
                          <Text type="secondary" style={{ fontSize: 12 }}>
                            接收促销活动和优惠信息
                          </Text>
                        </div>
                        <Tag color="green">已开启</Tag>
                      </Row>
                      <Divider style={{ margin: '12px 0' }} />
                      <Row justify="space-between" align="middle">
                        <div>
                          <Text strong>物流通知</Text>
                          <br />
                          <Text type="secondary" style={{ fontSize: 12 }}>
                            订单发货和收货提醒
                          </Text>
                        </div>
                        <Tag color="green">已开启</Tag>
                      </Row>
                    </Space>
                  </Card>
                ),
              },
            ]}
          />
        </Col>
      </Row>
    </div>
  )
}

export default UserManagement
