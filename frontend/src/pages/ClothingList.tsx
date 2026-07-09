import React, { useState, useEffect, useCallback, useRef } from 'react'
import {
  Card,
  Row,
  Col,
  Input,
  Select,
  Radio,
  Spin,
  Empty,
  Typography,
  Badge,
  Button,
  Drawer,
  Affix,
} from 'antd'
import {
  SearchOutlined,
  FilterOutlined,
  SortAscendingOutlined,
  AppstoreOutlined,
  UnorderedListOutlined,
  CloseOutlined,
} from '@ant-design/icons'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { clothingAPI } from '../api'

const { Title, Text } = Typography
const { Search } = Input

interface ClothingItem {
  id: number
  name: string
  category: string
  price: number
  style?: string
  brand?: string
  color?: string
  image_url?: string
}

type ViewMode = 'grid' | 'list'
type SortType = 'recommend' | 'sales' | 'price_asc' | 'price_desc' | 'newest'

const ClothingList: React.FC = () => {
  const navigate = useNavigate()
  const [searchParams] = useSearchParams()
  const observerRef = useRef<IntersectionObserver | null>(null)
  const lastItemRef = useRef<HTMLDivElement | null>(null)
  const offsetRef = useRef(0)

  // 状态
  const [items, setItems] = useState<ClothingItem[]>([])
  const [loading, setLoading] = useState(true)
  const [loadingMore, setLoadingMore] = useState(false)
  const [hasMore, setHasMore] = useState(true)

  // 筛选状态
  const [viewMode, setViewMode] = useState<ViewMode>('grid')
  const [searchQuery, setSearchQuery] = useState('')
  const [category, setCategory] = useState<string>(searchParams.get('category') || '')
  const [style, setStyle] = useState<string>('')
  const [priceRange, setPriceRange] = useState<[number, number] | null>(null)
  const [sortType, setSortType] = useState<SortType>('recommend')
  const [drawerVisible, setDrawerVisible] = useState(false)

  // 分页
  const limit = 20

  // 分类选项
  const categoryOptions = [
    { value: '', label: '全部' },
    { value: 'top', label: '上衣' },
    { value: 'bottom', label: '下装' },
    { value: 'dress', label: '连衣裙' },
    { value: 'outerwear', label: '外套' },
    { value: 'shoes', label: '鞋子' },
    { value: 'accessories', label: '配饰' },
  ]

  // 风格选项
  const styleOptions = [
    { value: '', label: '全部风格' },
    { value: 'minimalist', label: '简约' },
    { value: 'casual', label: '休闲' },
    { value: 'bohemian', label: '波西米亚' },
    { value: 'vintage', label: '复古' },
    { value: 'streetwear', label: '街头' },
    { value: 'business', label: '商务' },
    { value: 'sporty', label: '运动' },
    { value: 'romantic', label: '浪漫' },
  ]

  // 价格区间选项
  const priceOptions = [
    { value: 'all', label: '全部价格' },
    { value: '0-100', label: '100元以下' },
    { value: '100-300', label: '100-300元' },
    { value: '300-500', label: '300-500元' },
    { value: '500-1000', label: '500-1000元' },
    { value: '1000+', label: '1000元以上' },
  ]

  // 获取价格区间
  const getPriceRange = (value: string): [number, number] | null => {
    switch (value) {
      case '0-100': return [0, 100]
      case '100-300': return [100, 300]
      case '300-500': return [300, 500]
      case '500-1000': return [500, 1000]
      case '1000+': return [1000, 99999]
      default: return null
    }
  }

  // 获取商品列表
  const fetchItems = useCallback(async (isLoadMore = false) => {
    if (isLoadMore) {
      setLoadingMore(true)
    } else {
      setLoading(true)
      offsetRef.current = 0
    }

    const offset = isLoadMore ? items.length : 0
    offsetRef.current = offset

    try {
      const data = await clothingAPI.getList({ category, style, limit, offset })
      console.log('API返回数据:', data, '长度:', data?.length, 'offset:', offset)

      let filteredData = [...data]

      // 搜索过滤
      if (searchQuery) {
        filteredData = filteredData.filter(item =>
          item.name.toLowerCase().includes(searchQuery.toLowerCase())
        )
      }

      // 价格过滤
      if (priceRange) {
        filteredData = filteredData.filter(item =>
          item.price >= priceRange[0] && item.price <= priceRange[1]
        )
      }

      // 排序
      switch (sortType) {
        case 'price_asc':
          filteredData.sort((a, b) => a.price - b.price)
          break
        case 'price_desc':
          filteredData.sort((a, b) => b.price - a.price)
          break
        case 'newest':
          filteredData.reverse()
          break
        // recommend 和 sales 暂时保持原顺序
      }

      if (isLoadMore) {
        setItems(prev => [...prev, ...filteredData])
      } else {
        setItems(filteredData)
      }

      // 后端返回不足一页时表示没有更多
      setHasMore(data.length >= limit)
    } catch (error) {
      console.error('Failed to fetch items:', error)
    } finally {
      setLoading(false)
      setLoadingMore(false)
    }
  }, [category, style, priceRange, sortType, searchQuery, items.length])

  // 初始加载和筛选变化时重新加载
  useEffect(() => {
    setItems([])
    fetchItems()
  }, [category, style, sortType, searchQuery])

  // 价格变化时重新筛选
  useEffect(() => {
    setItems([])
    fetchItems()
  }, [priceRange])

  // 无限滚动
  useEffect(() => {
    if (!hasMore || loadingMore) return

    observerRef.current = new IntersectionObserver(
      (entries) => {
        if (entries[0].isIntersecting && hasMore && !loadingMore) {
          fetchItems(true)
        }
      },
      { threshold: 0.1 }
    )

    if (lastItemRef.current) {
      observerRef.current.observe(lastItemRef.current)
    }

    return () => {
      if (observerRef.current) {
        observerRef.current.disconnect()
      }
    }
  }, [hasMore, loadingMore, fetchItems])

  // 处理商品点击
  const handleItemClick = (item: ClothingItem) => {
    navigate(`/clothing/${item.id}`)
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

  // 占位图(与后端 kg.image_url.resolve_image 一致):图片加载失败时回退到
  // loremflickr 真实服装图(按类目关键词),避免空白方块。
  const categoryPlaceholder = (category: string, id: number): string => {
    const map: Record<string, string> = {
      top: 'tshirt',
      bottom: 'jeans',
      dress: 'dress',
      outerwear: 'jacket',
      shoes: 'sneakers',
      accessories: 'handbag',
    }
    const kw = map[category] || 'fashion'
    return `https://loremflickr.com/400/500/${kw}?lock=${id}`
  }

  // 价格区间标签
  const priceValue = priceRange
    ? priceRange[1] >= 99999
      ? `${priceRange[0]}+`
      : `${priceRange[0]}-${priceRange[1]}`
    : 'all'

  // 筛选栏内容
  const FilterContent = () => (
    <div>
      <div style={{ marginBottom: 16 }}>
        <Text strong style={{ display: 'block', marginBottom: 8 }}>分类</Text>
        <Select
          value={category}
          onChange={setCategory}
          style={{ width: '100%' }}
          options={categoryOptions}
        />
      </div>

      <div style={{ marginBottom: 16 }}>
        <Text strong style={{ display: 'block', marginBottom: 8 }}>风格</Text>
        <Select
          value={style}
          onChange={setStyle}
          style={{ width: '100%' }}
          options={styleOptions}
        />
      </div>

      <div style={{ marginBottom: 16 }}>
        <Text strong style={{ display: 'block', marginBottom: 8 }}>价格</Text>
        <Select
          value={priceValue}
          onChange={(val) => setPriceRange(getPriceRange(val))}
          style={{ width: '100%' }}
          options={priceOptions}
        />
      </div>

      <Button
        type="link"
        onClick={() => {
          setCategory('')
          setStyle('')
          setPriceRange(null)
          setSearchQuery('')
        }}
      >
        重置筛选
      </Button>
    </div>
  )

  // 网格视图商品卡片
  const GridCard = ({ item }: { item: ClothingItem }) => (
    <Card
      hoverable
      onClick={() => handleItemClick(item)}
      cover={
        <div
          style={{
            height: viewMode === 'grid' ? 180 : 120,
            background: item.image_url ? '#f5f5f5' : `linear-gradient(180deg, #f5f5f5 0%, #e8e8e8 100%)`,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontSize: viewMode === 'grid' ? 48 : 32,
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
                target.onerror = null
                target.src = categoryPlaceholder(item.category, item.id)
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
    </Card>
  )

  // 列表视图商品卡片
  const ListCard = ({ item }: { item: ClothingItem }) => (
    <Card hoverable onClick={() => handleItemClick(item)} style={{ marginBottom: 12 }}>
      <Row gutter={16} align="middle">
        <Col>
          <div
            style={{
              width: 100,
              height: 100,
              background: '#f5f5f5',
              borderRadius: 8,
              overflow: 'hidden',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: 36,
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
                  target.onerror = null
                  target.src = categoryPlaceholder(item.category, item.id)
                }}
              />
            ) : (
              categoryEmojis[item.category] || '👕'
            )}
          </div>
        </Col>
        <Col flex={1}>
          <Text strong style={{ fontSize: 16, display: 'block', marginBottom: 8 }}>
            {item.name}
          </Text>
          <Text type="secondary" style={{ fontSize: 12, display: 'block', marginBottom: 4 }}>
            分类: {item.category} | 风格: {item.style || 'casual'}
          </Text>
          {item.brand && (
            <Text type="secondary" style={{ fontSize: 12, display: 'block', marginBottom: 4 }}>
              品牌: {item.brand}
            </Text>
          )}
        </Col>
        <Col>
          <Text strong style={{ color: '#f5222d', fontSize: 18 }}>
            ¥{item.price?.toFixed(2) || '0.00'}
          </Text>
        </Col>
      </Row>
    </Card>
  )

  return (
    <div style={{ padding: '0 0 24px 0' }}>
      {/* 顶部标题 */}
      <div style={{ marginBottom: 24 }}>
        <Title level={3} style={{ margin: 0 }}>
          <Badge color="blue" />
          商品列表
        </Title>
        <Text type="secondary">发现更多好货</Text>
      </div>

      {/* 搜索和工具栏 */}
      <Card style={{ marginBottom: 16 }}>
        <Row gutter={[12, 12]} align="middle">
          <Col xs={24} sm={16}>
            <Search
              placeholder="搜索商品名称..."
              allowClear
              enterButton={<SearchOutlined />}
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              onSearch={setSearchQuery}
            />
          </Col>
          <Col xs={24} sm={8}>
            <Row gutter={8} align="middle">
              <Col>
                <Button
                  icon={<FilterOutlined />}
                  onClick={() => setDrawerVisible(true)}
                >
                  筛选
                </Button>
              </Col>
              <Col>
                <Radio.Group value={sortType} onChange={(e) => setSortType(e.target.value)}>
                  <Radio.Button value="recommend">推荐</Radio.Button>
                  <Radio.Button value="sales">销量</Radio.Button>
                  <Radio.Button value="price_asc">价格↑</Radio.Button>
                  <Radio.Button value="price_desc">价格↓</Radio.Button>
                </Radio.Group>
              </Col>
            </Row>
          </Col>
        </Row>

        {/* 视图切换 */}
        <Row justify="end" style={{ marginTop: 12 }}>
          <Radio.Group value={viewMode} onChange={(e) => setViewMode(e.target.value)}>
            <Radio.Button value="grid">
              <AppstoreOutlined />
            </Radio.Button>
            <Radio.Button value="list">
              <UnorderedListOutlined />
            </Radio.Button>
          </Radio.Group>
        </Row>

        {/* 移动端筛选按钮 */}
        <Row justify="space-between" align="middle" style={{ marginTop: 12 }} className="mobile-filter-row">
          <Col>
            <Text type="secondary">
              {items.length} 件商品
            </Text>
          </Col>
          <Col>
            <Button
              type="link"
              icon={<FilterOutlined />}
              onClick={() => setDrawerVisible(true)}
            >
              筛选
            </Button>
          </Col>
        </Row>
      </Card>

      {/* 商品列表 */}
      {loading ? (
        <div style={{ textAlign: 'center', padding: '60px 0' }}>
          <Spin size="large" />
          <div style={{ marginTop: 16 }}>加载中...</div>
        </div>
      ) : items.length === 0 ? (
        <Empty description="暂无商品" />
      ) : viewMode === 'grid' ? (
        <Row gutter={[12, 12]}>
          {items.map((item) => (
            <Col xs={12} sm={12} md={8} lg={6} xl={4} key={item.id}>
              <GridCard item={item} />
            </Col>
          ))}
        </Row>
      ) : (
        <div>
          {items.map((item) => (
            <ListCard key={item.id} item={item} />
          ))}
        </div>
      )}

      {/* 加载更多 */}
      {hasMore && !loading && (
        <div ref={lastItemRef} style={{ textAlign: 'center', padding: '24px 0' }}>
          <Spin size="small" spinning={loadingMore}>
            <Text type="secondary">加载更多...</Text>
          </Spin>
        </div>
      )}

      {!hasMore && items.length > 0 && (
        <div style={{ textAlign: 'center', padding: '24px 0' }}>
          <Text type="secondary">— 没有更多了 —</Text>
        </div>
      )}

      {/* 筛选抽屉 */}
      <Drawer
        title="商品筛选"
        placement="right"
        onClose={() => setDrawerVisible(false)}
        open={drawerVisible}
        extra={
          <Button
            type="link"
            icon={<CloseOutlined />}
            onClick={() => setDrawerVisible(false)}
          />
        }
      >
        <FilterContent />
      </Drawer>

      {/* 移动端底部筛选条 */}
      <Affix offsetBottom={0}>
        <div
          style={{
            display: 'none',
            background: '#fff',
            padding: '12px 16px',
            borderTop: '1px solid #f0f0f0',
          }}
          className="mobile-filter-bar"
        >
          <Row gutter={12}>
            <Col span={12}>
              <Button block onClick={() => setDrawerVisible(true)}>
                <FilterOutlined /> 筛选
              </Button>
            </Col>
            <Col span={12}>
              <Button block onClick={() => setDrawerVisible(true)}>
                <SortAscendingOutlined /> 排序
              </Button>
            </Col>
          </Row>
        </div>
      </Affix>
    </div>
  )
}

export default ClothingList
