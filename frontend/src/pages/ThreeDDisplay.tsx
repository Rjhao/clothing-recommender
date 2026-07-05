import React, { useState, useEffect } from 'react';
import { Card, Row, Col, Select, Button, Spin, Empty, Modal, message } from 'antd';
import { ReloadOutlined, ExpandOutlined, SwapOutlined, CheckCircleOutlined } from '@ant-design/icons';
import ThreeDViewer from '../components/ThreeDViewer';

interface ClothingItem {
  id: number;
  name: string;
  category: string;
  brand: string;
  color: string;
  price: number;
  style: string;
  season: string;
  imageUrl?: string;
}

const ThreeDDisplay: React.FC = () => {
  const [loading, setLoading] = useState(false);
  const [clothingList, setClothingList] = useState<ClothingItem[]>([]);
  const [selectedItem, setSelectedItem] = useState<ClothingItem | null>(null);
  const [compareItem, setCompareItem] = useState<ClothingItem | null>(null);
  const [viewMode, setViewMode] = useState<'single' | 'compare'>('single');
  const [selectedCategory, setSelectedCategory] = useState<string | undefined>();
  const [selectedStyle, setSelectedStyle] = useState<string | undefined>();
  const [previewVisible, setPreviewVisible] = useState(false);

  const fetchClothing = async () => {
    setLoading(true);
    try {
      const params = new URLSearchParams({
        limit: '50',
        ...(selectedCategory && { category: selectedCategory }),
        ...(selectedStyle && { style: selectedStyle }),
      });

      const response = await fetch(`http://localhost:8000/clothing/?${params}`);
      const data = await response.json();
      setClothingList(data || []);
    } catch (error) {
      console.error('Failed to fetch clothing:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchClothing();
  }, [selectedCategory, selectedStyle]);

  // Mock image URLs for demo (since we don't have real product images)
  const getDemoImageUrl = (category: string) => {
    const images: Record<string, string> = {
      top: 'https://images.unsplash.com/photo-1521572163474-6864f9cf17ab?w=400&h=500&fit=crop',
      bottom: 'https://images.unsplash.com/photo-1541099649105-f69ad21f3246?w=400&h=500&fit=crop',
      dress: 'https://images.unsplash.com/photo-1595777457583-95e059d581b8?w=400&h=500&fit=crop',
      outerwear: 'https://images.unsplash.com/photo-1591047139829-d91aecb6caea?w=400&h=500&fit=crop',
      shoes: 'https://images.unsplash.com/photo-1542291026-7eec264c27ff?w=400&h=500&fit=crop',
      accessories: 'https://images.unsplash.com/photo-1523170335258-f5ed11844a49?w=400&h=500&fit=crop',
    };
    return images[category] || images.top;
  };

  const handleSelectItem = (item: ClothingItem) => {
    if (viewMode === 'compare') {
      if (!selectedItem) {
        setSelectedItem(item);
      } else if (selectedItem.id !== item.id) {
        setCompareItem(item);
      } else {
        message.warning('请选择不同的服装进行对比');
      }
    } else {
      setSelectedItem(item);
    }
  };

  const handleReset = () => {
    setSelectedItem(null);
    setCompareItem(null);
  };

  return (
    <div style={{ padding: '24px' }}>
      <h1 style={{ fontSize: '24px', marginBottom: '24px' }}>3D 服装展示</h1>

      {/* Filter Bar */}
      <Card style={{ marginBottom: '24px' }}>
        <Row gutter={16} align="middle">
          <Col>
            <Select
              placeholder="选择分类"
              style={{ width: 150 }}
              allowClear
              value={selectedCategory}
              onChange={setSelectedCategory}
              options={[
                { value: 'top', label: '上衣' },
                { value: 'bottom', label: '下装' },
                { value: 'dress', label: '连衣裙' },
                { value: 'outerwear', label: '外套' },
                { value: 'shoes', label: '鞋子' },
                { value: 'accessories', label: '配饰' },
              ]}
            />
          </Col>
          <Col>
            <Select
              placeholder="选择风格"
              style={{ width: 150 }}
              allowClear
              value={selectedStyle}
              onChange={setSelectedStyle}
              options={[
                { value: 'minimalist', label: '简约' },
                { value: 'casual', label: '休闲' },
                { value: 'bohemian', label: '波西米亚' },
                { value: 'vintage', label: '复古' },
                { value: 'streetwear', label: '街头' },
                { value: 'business', label: '商务' },
                { value: 'sporty', label: '运动' },
                { value: 'romantic', label: '浪漫' },
              ]}
            />
          </Col>
          <Col>
            <Button icon={<ReloadOutlined />} onClick={handleReset}>
              重置选择
            </Button>
          </Col>
          <Col>
            <Button.Group>
              <Button
                type={viewMode === 'single' ? 'primary' : 'default'}
                onClick={() => setViewMode('single')}
              >
                单品展示
              </Button>
              <Button
                type={viewMode === 'compare' ? 'primary' : 'default'}
                onClick={() => setViewMode('compare')}
                icon={<SwapOutlined />}
              >
                对比模式
              </Button>
            </Button.Group>
          </Col>
        </Row>
      </Card>

      <Row gutter={24}>
        {/* 3D Viewer Area */}
        <Col span={viewMode === 'compare' ? 12 : 16}>
          <Card
            title={viewMode === 'compare' ? '3D 对比展示' : '3D 展示'}
            extra={
              <Button icon={<ExpandOutlined />} onClick={() => setPreviewVisible(true)}>
                全屏预览
              </Button>
            }
          >
            {viewMode === 'compare' ? (
              <Row gutter={16}>
                <Col span={12}>
                  <div style={{ textAlign: 'center', marginBottom: '16px' }}>
                    <span style={{ fontWeight: 'bold' }}>
                      {selectedItem ? selectedItem.name : '请选择左侧服装'}
                    </span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'center' }}>
                    {selectedItem ? (
                      <ThreeDViewer
                        imageUrl={getDemoImageUrl(selectedItem.category)}
                        width={300}
                        height={400}
                      />
                    ) : (
                      <Empty description="请从右侧列表选择服装" />
                    )}
                  </div>
                </Col>
                <Col span={12}>
                  <div style={{ textAlign: 'center', marginBottom: '16px' }}>
                    <span style={{ fontWeight: 'bold' }}>
                      {compareItem ? compareItem.name : '请选择右侧服装'}
                    </span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'center' }}>
                    {compareItem ? (
                      <ThreeDViewer
                        imageUrl={getDemoImageUrl(compareItem.category)}
                        width={300}
                        height={400}
                      />
                    ) : (
                      <Empty description="请从右侧列表选择服装" />
                    )}
                  </div>
                </Col>
              </Row>
            ) : (
              <div style={{ display: 'flex', justifyContent: 'center' }}>
                {selectedItem ? (
                  <ThreeDViewer
                    imageUrl={getDemoImageUrl(selectedItem.category)}
                    width={400}
                    height={500}
                    autoRotate
                  />
                ) : (
                  <Empty description="请从右侧列表选择服装" style={{ padding: '100px' }} />
                )}
              </div>
            )}

            {/* Selected Item Info */}
            {selectedItem && (
              <Card
                size="small"
                style={{ marginTop: '24px', background: '#fafafa' }}
              >
                <Row gutter={16}>
                  <Col span={12}>
                    <p><strong>服装:</strong> {selectedItem.name}</p>
                    <p><strong>分类:</strong> {selectedItem.category}</p>
                    <p><strong>风格:</strong> {selectedItem.style}</p>
                  </Col>
                  <Col span={12}>
                    <p><strong>价格:</strong> ¥{selectedItem.price?.toFixed(2)}</p>
                    <p><strong>季节:</strong> {selectedItem.season}</p>
                    <p><strong>品牌:</strong> {selectedItem.brand || '-'}</p>
                  </Col>
                </Row>
                <Button type="primary" style={{ marginTop: '12px' }} block>
                  加入搭配方案
                </Button>
              </Card>
            )}
          </Card>
        </Col>

        {/* Clothing List */}
        <Col span={viewMode === 'compare' ? 12 : 8}>
          <Card title="选择服装">
            {loading ? (
              <div style={{ textAlign: 'center', padding: '40px' }}>
                <Spin />
              </div>
            ) : (
              <div style={{ maxHeight: '600px', overflowY: 'auto' }}>
                {clothingList.map((item) => (
                  <Card
                    key={item.id}
                    size="small"
                    hoverable
                    onClick={() => handleSelectItem(item)}
                    style={{
                      marginBottom: '12px',
                      cursor: 'pointer',
                      borderColor:
                        selectedItem?.id === item.id || compareItem?.id === item.id
                          ? '#1890ff'
                          : '#f0f0f0',
                    }}
                    cover={
                      <div
                        style={{
                          height: '120px',
                          background: `url(${getDemoImageUrl(item.category)}) center/cover no-repeat`,
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                        }}
                      >
                        {(selectedItem?.id === item.id || compareItem?.id === item.id) && (
                          <CheckCircleOutlined
                            style={{
                              fontSize: '32px',
                              color: '#1890ff',
                              background: '#fff',
                              borderRadius: '50%',
                            }}
                          />
                        )}
                      </div>
                    }
                  >
                    <Card.Meta
                      title={item.name}
                      description={
                        <div>
                          <span style={{ color: '#f5222d', fontWeight: 'bold' }}>
                            ¥{item.price?.toFixed(2)}
                          </span>
                          <br />
                          <span
                            style={{
                              fontSize: '12px',
                              color:
                                selectedItem?.id === item.id
                                  ? '#1890ff'
                                  : '#888',
                            }}
                          >
                            {item.category} | {item.style}
                          </span>
                        </div>
                      }
                    />
                  </Card>
                ))}
              </div>
            )}
          </Card>
        </Col>
      </Row>

      {/* Full Screen Preview Modal */}
      <Modal
        title="3D 全屏预览"
        open={previewVisible}
        onCancel={() => setPreviewVisible(false)}
        footer={null}
        width="80%"
        style={{ top: 20 }}
      >
        {selectedItem && (
          <div style={{ display: 'flex', justifyContent: 'center', padding: '20px' }}>
            <ThreeDViewer
              imageUrl={getDemoImageUrl(selectedItem.category)}
              width={600}
              height={750}
              autoRotate
            />
          </div>
        )}
      </Modal>
    </div>
  );
};

export default ThreeDDisplay;
