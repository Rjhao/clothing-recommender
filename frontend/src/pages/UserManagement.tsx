import React, { useState, useEffect, useCallback } from 'react'
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
  Alert,
  Modal,
  Form,
  Select,
  InputNumber,
  message,
  Tooltip,
} from 'antd'
import {
  UserOutlined,
  EditOutlined,
  HistoryOutlined,
  SettingOutlined,
  LockOutlined,
  SafetyOutlined,
  ReloadOutlined,
  HeartOutlined,
  BulbOutlined,
  FireOutlined,
  EyeOutlined,
  ArrowRightOutlined,
} from '@ant-design/icons'
import type { ColumnsType } from 'antd/es/table'
import { userAPI } from '../api'
import type { User, UserProfile, HistoryEntry, ClothingItem } from '../api'
import { useAuth } from '../context/AuthContext'
import {
  GENDER_LABELS,
  BODY_SHAPE_LABELS,
  BODY_SHAPE_COLORS,
  SKIN_TONE_LABELS,
  SKIN_TONE_SWATCHES,
  CATEGORY_EMOJIS,
  RELATION_COLORS,
  genderLabel,
  bodyShapeLabel,
  skinToneLabel,
  styleLabel,
  categoryLabel,
  relationLabel,
  colorLabel,
  colorHex,
} from '../constants/labels'

const { Title, Text } = Typography

interface HistoryRow {
  key: string
  clothing_id: number
  clothing_name: string
  category?: string
  relation_type: string
  score?: number
  timestamp?: string
}

/** Fallback profile shown when the backend is unreachable — clearly flagged in the UI. */
const DEMO_PROFILE: UserProfile = {
  user_id: 0,
  static_info: {
    username: '演示用户',
    email: 'demo@example.com',
    gender: 'female',
    age: 28,
    height: 165,
    weight: 52,
    body_shape: 'average',
    skin_tone: 'fair',
  },
  style_distribution: {
    minimalist: 0.35,
    casual: 0.25,
    sporty: 0.2,
    business: 0.15,
    vintage: 0.05,
  },
  color_preferences: { black: 28, white: 22, blue: 18, grey: 12, red: 8 },
  price_range: { min: 100, max: 800, avg: 380 },
  interaction_count: 156,
}

const formatTimestamp = (raw?: string | number): string => {
  if (raw === undefined || raw === null || raw === '') return '-'
  // Neo4j may hand back an epoch (seconds or milliseconds) or an ISO string.
  const asNumber = typeof raw === 'number' ? raw : Number(raw)
  if (!Number.isNaN(asNumber) && asNumber > 0) {
    const ms = asNumber > 1e12 ? asNumber : asNumber * 1000
    const d = new Date(ms)
    if (!Number.isNaN(d.getTime())) return d.toLocaleString('zh-CN', { hour12: false })
  }
  const parsed = new Date(String(raw))
  if (!Number.isNaN(parsed.getTime())) return parsed.toLocaleString('zh-CN', { hour12: false })
  return String(raw)
}

const UserManagement: React.FC = () => {
  const { user: authUser, refreshUser } = useAuth()
  const currentUserId = authUser?.id

  const [loading, setLoading] = useState(true)
  const [profile, setProfile] = useState<UserProfile | null>(null)
  const [staticInfo, setStaticInfo] = useState<Partial<User> | null>(null)
  const [history, setHistory] = useState<HistoryRow[]>([])
  const [recommendations, setRecommendations] = useState<ClothingItem[]>([])
  const [recommendedStyles, setRecommendedStyles] = useState<string[]>([])
  const [recsLoading, setRecsLoading] = useState(false)
  const [stylesLoading, setStylesLoading] = useState(false)
  const [recsError, setRecsError] = useState<string | null>(null)
  const [stylesError, setStylesError] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState<string>('profile')
  const [profileError, setProfileError] = useState<string | null>(null)
  const [historyError, setHistoryError] = useState<string | null>(null)
  const [isDemo, setIsDemo] = useState(false)

  const [editOpen, setEditOpen] = useState(false)
  const [saving, setSaving] = useState(false)
  const [form] = Form.useForm()

  const loadProfile = useCallback(async () => {
    if (!currentUserId) return
    setLoading(true)
    setProfileError(null)
    setIsDemo(false)

    // The two sources are independent: /users/{id} is the authoritative account
    // record (MySQL), /recommend/{id}/profile is the behaviour-derived profile
    // (Neo4j). One failing must not blank out the other.
    const [accountResult, behaviourResult] = await Promise.allSettled([
      userAPI.getProfile(currentUserId),
      userAPI.getFullProfile(currentUserId),
    ])

    const account =
      accountResult.status === 'fulfilled' ? accountResult.value : null
    const behaviour =
      behaviourResult.status === 'fulfilled' ? behaviourResult.value : null

    if (!account && !behaviour) {
      setProfileError('无法连接后端服务，以下为演示数据。')
      setIsDemo(true)
      setProfile(DEMO_PROFILE)
      setStaticInfo(DEMO_PROFILE.static_info)
      setLoading(false)
      return
    }

    if (!account) {
      setProfileError('账号资料加载失败，仅显示行为偏好数据。')
    } else if (!behaviour) {
      setProfileError('偏好画像加载失败（Neo4j 可能未启动），仅显示账号资料。')
    }

    // Prefer the MySQL account record for demographics; the graph copy can be stale.
    setStaticInfo({ ...(behaviour?.static_info ?? {}), ...(account ?? {}) })
    setProfile(
      behaviour ?? {
        user_id: currentUserId,
        static_info: account ?? {},
        style_distribution: {},
        color_preferences: {},
        price_range: { min: 0, max: 0, avg: 0 },
        interaction_count: 0,
      }
    )
    setLoading(false)
  }, [currentUserId])

  const loadHistory = useCallback(async () => {
    if (!currentUserId) return
    setHistoryError(null)
    try {
      const entries: HistoryEntry[] = await userAPI.getHistory(currentUserId, 50)
      setHistory(
        entries.map((entry, index) => ({
          key: `${entry.c?.id ?? 'x'}-${entry.relation_type}-${index}`,
          clothing_id: entry.c?.id,
          clothing_name: entry.c?.name ?? `商品 #${entry.c?.id ?? '?'}`,
          category: entry.c?.category,
          relation_type: entry.relation_type,
          score: entry.score,
          timestamp: entry.timestamp,
        }))
      )
    } catch {
      setHistory([])
      setHistoryError('浏览历史加载失败，请确认后端与 Neo4j 已启动。')
    }
  }, [currentUserId])

  const loadRecommendations = useCallback(async () => {
    if (!currentUserId) return
    setRecsError(null)
    setRecsLoading(true)
    try {
      const list: any[] = await userAPI.getRecommendations(currentUserId, undefined, 12)
      // 后端返回的 item 字段或平铺字段统一为 ClothingItem
      const normalized: ClothingItem[] = list.map((r) => ({
        id: r.clothing_id ?? r.id,
        name: r.name ?? r.item?.name ?? `商品 #${r.clothing_id ?? r.id}`,
        category: r.category ?? r.item?.category ?? '',
        brand: r.brand ?? r.item?.brand,
        price: r.price ?? r.item?.price ?? 0,
        image_url: r.image_url ?? r.item?.image_url,
      }))
      setRecommendations(normalized)
    } catch {
      setRecommendations([])
      setRecsError('推荐加载失败，请稍后重试。')
    } finally {
      setRecsLoading(false)
    }
  }, [currentUserId])

  const loadRecommendedStyles = useCallback(async () => {
    if (!currentUserId) return
    setStylesError(null)
    setStylesLoading(true)
    try {
      const styles: string[] = await userAPI.getRecommendedStyles(currentUserId)
      setRecommendedStyles(styles)
    } catch {
      setRecommendedStyles([])
      setStylesError('推荐风格加载失败。')
    } finally {
      setStylesLoading(false)
    }
  }, [currentUserId])

  useEffect(() => {
    if (!currentUserId) {
      setLoading(false)
      return
    }
    loadProfile()
    loadHistory()
    loadRecommendations()
    loadRecommendedStyles()
  }, [currentUserId, loadProfile, loadHistory, loadRecommendations, loadRecommendedStyles])

  const openEdit = () => {
    form.setFieldsValue({
      gender: staticInfo?.gender,
      age: staticInfo?.age,
      height: staticInfo?.height,
      weight: staticInfo?.weight,
      body_shape: staticInfo?.body_shape,
      skin_tone: staticInfo?.skin_tone,
    })
    setEditOpen(true)
  }

  const submitEdit = async () => {
    if (!currentUserId) return
    const values = await form.validateFields()
    setSaving(true)
    try {
      const updated = await userAPI.updateProfile(currentUserId, values)
      setStaticInfo((prev) => ({ ...(prev ?? {}), ...updated }))
      await refreshUser()
      message.success('资料已更新')
      setEditOpen(false)
    } catch (error: any) {
      message.error(error?.response?.data?.detail || '保存失败，请稍后再试')
    } finally {
      setSaving(false)
    }
  }

  const renderStylePreferences = () => {
    const dist = profile?.style_distribution ?? {}
    const entries = Object.entries(dist).sort(([, a], [, b]) => b - a)

    return (
      <Card title="风格偏好" style={{ marginBottom: 24 }}>
        {entries.length > 0 ? (
          <Space direction="vertical" style={{ width: '100%' }} size="middle">
            {entries.map(([style, ratio]) => (
              <div key={style}>
                <Row justify="space-between" align="middle">
                  <Text>{styleLabel(style)}</Text>
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
        ) : (
          <Empty
            image={Empty.PRESENTED_IMAGE_SIMPLE}
            description="暂无风格偏好数据，多浏览一些商品后即可生成"
          />
        )}
      </Card>
    )
  }

  const renderColorPreferences = () => {
    const prefs = profile?.color_preferences ?? {}
    const entries = Object.entries(prefs).sort(([, a], [, b]) => b - a)
    const total = entries.reduce((sum, [, count]) => sum + count, 0)

    return (
      <Card title="颜色偏好" style={{ marginBottom: 24 }}>
        {total > 0 ? (
          <Row gutter={[12, 12]}>
            {entries.map(([color, count]) => (
              <Col span={12} key={color}>
                <Row justify="space-between" align="middle">
                  <Space>
                    <div
                      style={{
                        width: 16,
                        height: 16,
                        borderRadius: 4,
                        background: colorHex(color),
                        border: '1px solid #d9d9d9',
                      }}
                    />
                    <Text>{colorLabel(color)}</Text>
                  </Space>
                  <Text type="secondary">{((count / total) * 100).toFixed(0)}%</Text>
                </Row>
              </Col>
            ))}
          </Row>
        ) : (
          <Empty
            image={Empty.PRESENTED_IMAGE_SIMPLE}
            description="暂无颜色偏好数据"
          />
        )}
      </Card>
    )
  }

  const renderRecommendations = () => {
    if (recsLoading) {
      return (
        <div style={{ textAlign: 'center', padding: '60px 0' }}>
          <Spin size="large" />
          <div style={{ marginTop: 16 }}>正在为您挑选商品...</div>
        </div>
      )
    }

    if (recsError) {
      return (
        <Alert
          type="error"
          showIcon
          message={recsError}
          action={
            <Button size="small" onClick={loadRecommendations}>
              重试
            </Button>
          }
        />
      )
    }

    if (recommendations.length === 0) {
      return (
        <Empty description="暂无推荐，多浏览一些商品后系统会为您生成专属推荐">
          <Button type="primary" onClick={() => window.location.assign('/clothing')}>
            去逛逛
          </Button>
        </Empty>
      )
    }

    return (
      <Card>
        <Row gutter={[16, 16]}>
          {recommendations.map((item) => (
            <Col xs={12} sm={12} md={8} lg={8} xl={6} key={item.id}>
              <Card
                hoverable
                size="small"
                cover={
                  item.image_url ? (
                    <div
                      style={{
                        height: 200,
                        background: `#f5f5f5 url(${item.image_url}) center/cover no-repeat`,
                      }}
                    />
                  ) : (
                    <div
                      style={{
                        height: 200,
                        background: '#f5f5f5',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        fontSize: 56,
                      }}
                    >
                      {CATEGORY_EMOJIS[item.category] || '👕'}
                    </div>
                  )
                }
                onClick={() => window.location.assign(`/clothing/${item.id}`)}
              >
                <Card.Meta
                  title={
                    <Text ellipsis style={{ fontSize: 13 }}>
                      {item.name}
                    </Text>
                  }
                  description={
                    <Space direction="vertical" size={4} style={{ width: '100%' }}>
                      <Space size={4}>
                        <Tag color="blue">{categoryLabel(item.category) || '其他'}</Tag>
                        {item.brand && <Tag>{item.brand}</Tag>}
                      </Space>
                      <Row justify="space-between" align="middle">
                        <Text strong style={{ color: '#f5222d' }}>
                          ¥{item.price.toFixed(0)}
                        </Text>
                        <Button
                          type="link"
                          size="small"
                          icon={<EyeOutlined />}
                          onClick={(e) => {
                            e.stopPropagation()
                            window.location.assign(`/clothing/${item.id}`)
                          }}
                        >
                          查看
                        </Button>
                      </Row>
                    </Space>
                  }
                />
              </Card>
            </Col>
          ))}
        </Row>
      </Card>
    )
  }

  const renderRecommendedStyles = () => {
    if (stylesLoading) {
      return (
        <div style={{ textAlign: 'center', padding: '60px 0' }}>
          <Spin size="large" />
        </div>
      )
    }

    if (stylesError) {
      return (
        <Alert
          type="error"
          showIcon
          message={stylesError}
          action={
            <Button size="small" onClick={loadRecommendedStyles}>
              重试
            </Button>
          }
        />
      )
    }

    if (recommendedStyles.length === 0) {
      return (
        <Empty description="暂无可推荐风格，完善身材资料后即可生成">
          <Button type="primary" onClick={openEdit}>
            去完善资料
          </Button>
        </Empty>
      )
    }

    const styleColors: Record<string, string> = {
      minimalist: '#1890ff',
      casual: '#52c41a',
      sporty: '#f5222d',
      business: '#722ed1',
      vintage: '#fa8c16',
      streetwear: '#eb2f96',
      romantic: '#eb2f96',
      bohemian: '#13c2c2',
    }

    return (
      <Card>
        <Space direction="vertical" size="middle" style={{ width: '100%' }}>
          <Text type="secondary">
            基于您的「体型」与「风格偏好」综合推断，以下风格可能适合您：
          </Text>
          <Row gutter={[12, 12]}>
            {recommendedStyles.map((style) => (
              <Col key={style}>
                <Tag
                  color={styleColors[style] || 'blue'}
                  style={{
                    padding: '8px 16px',
                    fontSize: 14,
                    borderRadius: 20,
                  }}
                  icon={<HeartOutlined />}
                >
                  {styleLabel(style)}
                </Tag>
              </Col>
            ))}
          </Row>
          <Divider style={{ margin: '12px 0' }} />
          <Button
            type="link"
            icon={<ArrowRightOutlined />}
            onClick={() => window.location.assign('/clothing')}
          >
            按这些风格浏览更多商品 →
          </Button>
        </Space>
      </Card>
    )
  }

  const historyColumns: ColumnsType<HistoryRow> = [
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
            {CATEGORY_EMOJIS[record.category ?? ''] || '👕'}
          </div>
          <Text>{record.clothing_name}</Text>
        </Space>
      ),
    },
    {
      title: '分类',
      dataIndex: 'category',
      key: 'category',
      render: (cat?: string) => <Tag>{categoryLabel(cat)}</Tag>,
    },
    {
      title: '行为',
      dataIndex: 'relation_type',
      key: 'relation_type',
      render: (type: string, record) => (
        <Space size={4}>
          <Tag color={RELATION_COLORS[type] ?? 'default'}>{relationLabel(type)}</Tag>
          {record.score !== undefined && record.score !== null && (
            <Text type="secondary">{record.score} 分</Text>
          )}
        </Space>
      ),
    },
    {
      title: '时间',
      dataIndex: 'timestamp',
      key: 'timestamp',
      render: (time?: string) => <Text type="secondary">{formatTimestamp(time)}</Text>,
    },
    {
      title: '操作',
      key: 'action',
      render: (_, record) =>
        record.clothing_id ? (
          <Button
            type="link"
            size="small"
            onClick={() => window.location.assign(`/clothing/${record.clothing_id}`)}
          >
            查看详情
          </Button>
        ) : null,
    },
  ]

  if (!currentUserId) {
    return (
      <div style={{ padding: '80px 0', textAlign: 'center' }}>
        <Empty description="请先登录后查看个人中心">
          <Button type="primary" onClick={() => window.location.assign('/login')}>
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

  const priceRange = profile?.price_range
  const hasPriceData = !!priceRange && priceRange.max > 0

  return (
    <div style={{ padding: '0 0 24px 0' }}>
      <div style={{ marginBottom: 24 }}>
        <Row justify="space-between" align="middle">
          <div>
            <Title level={3} style={{ margin: 0 }}>
              <Badge color="green" />
              个人中心
            </Title>
            <Text type="secondary">管理您的个人信息和偏好设置</Text>
          </div>
          <Button
            icon={<ReloadOutlined />}
            onClick={() => {
              loadProfile()
              loadHistory()
              loadRecommendations()
              loadRecommendedStyles()
            }}
          >
            刷新
          </Button>
        </Row>
      </div>

      {profileError && (
        <Alert
          type={isDemo ? 'warning' : 'info'}
          showIcon
          style={{ marginBottom: 16 }}
          message={profileError}
          action={
            <Button size="small" onClick={loadProfile}>
              重试
            </Button>
          }
        />
      )}

      <Row gutter={24}>
        <Col xs={24} lg={8}>
          <Card>
            <div style={{ textAlign: 'center', marginBottom: 24 }}>
              <Avatar
                size={80}
                icon={<UserOutlined />}
                style={{ backgroundColor: '#1890ff', marginBottom: 16 }}
              />
              <Title level={4} style={{ margin: 0 }}>
                {staticInfo?.username || authUser?.username || '用户'}
              </Title>
              <Text type="secondary">{staticInfo?.email || authUser?.email || '-'}</Text>
            </div>

            <Divider />

            <Space direction="vertical" style={{ width: '100%' }} size="small">
              <Row justify="space-between">
                <Text type="secondary">性别</Text>
                <Text>{genderLabel(staticInfo?.gender)}</Text>
              </Row>
              <Row justify="space-between">
                <Text type="secondary">年龄</Text>
                <Text>{staticInfo?.age ? `${staticInfo.age} 岁` : '-'}</Text>
              </Row>
              <Row justify="space-between">
                <Text type="secondary">身高</Text>
                <Text>{staticInfo?.height ? `${staticInfo.height} cm` : '-'}</Text>
              </Row>
              <Row justify="space-between">
                <Text type="secondary">体重</Text>
                <Text>{staticInfo?.weight ? `${staticInfo.weight} kg` : '-'}</Text>
              </Row>
              <Row justify="space-between">
                <Text type="secondary">体型</Text>
                <Tag color={BODY_SHAPE_COLORS[staticInfo?.body_shape ?? ''] || 'default'}>
                  {bodyShapeLabel(staticInfo?.body_shape)}
                </Tag>
              </Row>
              <Row justify="space-between" align="middle">
                <Tooltip title="用于色彩搭配推荐，非肤质">
                  <Text type="secondary">肤色</Text>
                </Tooltip>
                <Space size={6}>
                  <div
                    style={{
                      width: 14,
                      height: 14,
                      borderRadius: '50%',
                      background: SKIN_TONE_SWATCHES[staticInfo?.skin_tone ?? ''] || '#f0f0f0',
                      border: '1px solid #d9d9d9',
                    }}
                  />
                  <Text>{skinToneLabel(staticInfo?.skin_tone)}</Text>
                </Space>
              </Row>
            </Space>

            <Divider />

            <Button icon={<EditOutlined />} block onClick={openEdit} disabled={isDemo}>
              编辑资料
            </Button>
          </Card>

          <Card title="消费偏好" style={{ marginTop: 24 }}>
            {hasPriceData ? (
              <Space direction="vertical" style={{ width: '100%' }} size="middle">
                <Row justify="space-between">
                  <Text type="secondary">价格区间</Text>
                  <Text>
                    ¥{priceRange!.min.toFixed(0)} - ¥{priceRange!.max.toFixed(0)}
                  </Text>
                </Row>
                <Row justify="space-between">
                  <Text type="secondary">平均消费</Text>
                  <Text strong style={{ color: '#f5222d' }}>
                    ¥{priceRange!.avg.toFixed(0)}
                  </Text>
                </Row>
                <Row justify="space-between">
                  <Text type="secondary">交互次数</Text>
                  <Badge count={profile?.interaction_count || 0} showZero />
                </Row>
              </Space>
            ) : (
              <Empty
                image={Empty.PRESENTED_IMAGE_SIMPLE}
                description="暂无消费数据"
              />
            )}
          </Card>
        </Col>

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
                          <div style={{ flex: 1, marginLeft: 12 }}>
                            <Text strong>登录密码</Text>
                            <br />
                            <Text type="secondary" style={{ fontSize: 12 }}>
                              已设置
                            </Text>
                          </div>
                          <Tag color="default">近期上线</Tag>
                        </Row>
                        <Divider style={{ margin: '12px 0' }} />
                        <Row align="middle" gutter={12}>
                          <SafetyOutlined style={{ fontSize: 20, color: '#52c41a' }} />
                          <div style={{ flex: 1, marginLeft: 12 }}>
                            <Text strong>绑定手机</Text>
                            <br />
                            <Text type="secondary" style={{ fontSize: 12 }}>
                              未绑定
                            </Text>
                          </div>
                          <Tag color="default">近期上线</Tag>
                        </Row>
                      </Space>
                      <Alert
                        type="info"
                        showIcon
                        style={{ marginTop: 16 }}
                        message="账号安全相关操作(改密 / 绑手机)正在完善中，敬请期待。"
                      />
                    </Card>
                  </div>
                ),
              },
              {
                key: 'history',
                label: (
                  <Space>
                    <HistoryOutlined />
                    行为记录
                    <Badge count={history.length} size="small" />
                  </Space>
                ),
                children: (
                  <Card>
                    {historyError && (
                      <Alert
                        type="info"
                        showIcon
                        style={{ marginBottom: 16 }}
                        message={historyError}
                        action={
                          <Button size="small" onClick={loadHistory}>
                            重试
                          </Button>
                        }
                      />
                    )}
                    {history.length > 0 ? (
                      <Table
                        columns={historyColumns}
                        dataSource={history}
                        rowKey="key"
                        pagination={{ pageSize: 10 }}
                        size="small"
                      />
                    ) : (
                      <Empty description="暂无行为记录">
                        <Button type="primary" onClick={() => window.location.assign('/clothing')}>
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
                    <Alert
                      type="info"
                      showIcon
                      message="通知偏好(推荐通知 / 活动通知)正在完善中，敬请期待。"
                      style={{ marginBottom: 16 }}
                    />
                    <Space direction="vertical" style={{ width: '100%' }}>
                      <Row justify="space-between" align="middle">
                        <div>
                          <Text strong>推荐通知</Text>
                          <br />
                          <Text type="secondary" style={{ fontSize: 12 }}>
                            接收个性化商品推荐
                          </Text>
                        </div>
                        <Tag color="default">近期上线</Tag>
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
                        <Tag color="default">近期上线</Tag>
                      </Row>
                    </Space>
                  </Card>
                ),
              },
              {
                key: 'recommend',
                label: (
                  <Space>
                    <FireOutlined />
                    为我推荐
                    <Badge count={recommendations.length} size="small" showZero={false} />
                  </Space>
                ),
                children: renderRecommendations(),
              },
              {
                key: 'styles',
                label: (
                  <Space>
                    <BulbOutlined />
                    推荐风格
                    <Badge count={recommendedStyles.length} size="small" showZero={false} />
                  </Space>
                ),
                children: renderRecommendedStyles(),
              },
            ]}
          />
        </Col>
      </Row>

      <Modal
        title="编辑资料"
        open={editOpen}
        onCancel={() => setEditOpen(false)}
        onOk={submitEdit}
        confirmLoading={saving}
        okText="保存"
        cancelText="取消"
        destroyOnClose
      >
        <Form form={form} layout="vertical" preserve={false}>
          <Form.Item name="gender" label="性别">
            <Select allowClear placeholder="请选择">
              {Object.entries(GENDER_LABELS).map(([value, label]) => (
                <Select.Option key={value} value={value}>
                  {label}
                </Select.Option>
              ))}
            </Select>
          </Form.Item>
          <Form.Item
            name="age"
            label="年龄"
            rules={[{ type: 'number', min: 10, max: 100, message: '年龄需在 10-100 之间' }]}
          >
            <InputNumber style={{ width: '100%' }} addonAfter="岁" />
          </Form.Item>
          <Form.Item
            name="height"
            label="身高"
            rules={[{ type: 'number', min: 100, max: 250, message: '身高需在 100-250cm 之间' }]}
          >
            <InputNumber style={{ width: '100%' }} addonAfter="cm" />
          </Form.Item>
          <Form.Item
            name="weight"
            label="体重"
            rules={[{ type: 'number', min: 25, max: 250, message: '体重需在 25-250kg 之间' }]}
          >
            <InputNumber style={{ width: '100%' }} addonAfter="kg" />
          </Form.Item>
          <Form.Item name="body_shape" label="体型">
            <Select allowClear placeholder="请选择">
              {Object.entries(BODY_SHAPE_LABELS).map(([value, label]) => (
                <Select.Option key={value} value={value}>
                  {label}
                </Select.Option>
              ))}
            </Select>
          </Form.Item>
          <Form.Item
            name="skin_tone"
            label="肤色"
            tooltip="用于色彩搭配推荐"
          >
            <Select allowClear placeholder="请选择">
              {Object.entries(SKIN_TONE_LABELS).map(([value, label]) => (
                <Select.Option key={value} value={value}>
                  <Space>
                    <span
                      style={{
                        display: 'inline-block',
                        width: 12,
                        height: 12,
                        borderRadius: '50%',
                        background: SKIN_TONE_SWATCHES[value],
                        border: '1px solid #d9d9d9',
                      }}
                    />
                    {label}
                  </Space>
                </Select.Option>
              ))}
            </Select>
          </Form.Item>
        </Form>
      </Modal>
    </div>
  )
}

export default UserManagement
