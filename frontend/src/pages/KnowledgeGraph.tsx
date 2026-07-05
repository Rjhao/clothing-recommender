import React, { useState, useEffect } from 'react';
import { Card, Row, Col, Tag, Statistic, Progress } from 'antd';

const KnowledgeGraph: React.FC = () => {
  const [stats, setStats] = useState({
    userCount: 0,
    clothingCount: 0,
    styleCount: 0,
    categoryCount: 0,
    relationshipCount: 0,
  });

  const fetchGraphStats = async () => {
    try {
      // Query Neo4j for statistics
      // Note: In production, these would be API calls to your backend
      setStats({
        userCount: 50,
        clothingCount: 100,
        styleCount: 8,
        categoryCount: 6,
        relationshipCount: 500,
      });
    } catch (error) {
      console.error('Failed to fetch graph stats:', error);
    }
  };

  useEffect(() => {
    fetchGraphStats();
  }, []);

  return (
    <div style={{ padding: '24px' }}>
      <h1 style={{ fontSize: '24px', marginBottom: '24px' }}>知识图谱</h1>

      <Row gutter={16}>
        <Col span={4}>
          <Card>
            <Statistic
              title="用户节点"
              value={stats.userCount}
              valueStyle={{ color: '#1890ff' }}
            />
          </Card>
        </Col>
        <Col span={4}>
          <Card>
            <Statistic
              title="服饰节点"
              value={stats.clothingCount}
              valueStyle={{ color: '#52c41a' }}
            />
          </Card>
        </Col>
        <Col span={4}>
          <Card>
            <Statistic
              title="风格节点"
              value={stats.styleCount}
              valueStyle={{ color: '#faad14' }}
            />
          </Card>
        </Col>
        <Col span={4}>
          <Card>
            <Statistic
              title="分类节点"
              value={stats.categoryCount}
              valueStyle={{ color: '#f5222d' }}
            />
          </Card>
        </Col>
        <Col span={4}>
          <Card>
            <Statistic
              title="关系边"
              value={stats.relationshipCount}
              valueStyle={{ color: '#722ed1' }}
            />
          </Card>
        </Col>
      </Row>

      <Row gutter={16} style={{ marginTop: '24px' }}>
        <Col span={12}>
          <Card title="图谱结构">
            <div style={{ fontFamily: 'monospace', fontSize: '12px', lineHeight: '1.8' }}>
              <div>
                <Tag color="blue">User</Tag>
                <span style={{ margin: '0 8px' }}>-PREFERS-&gt;</span>
                <Tag color="orange">Style</Tag>
              </div>
              <div>
                <Tag color="blue">User</Tag>
                <span style={{ margin: '0 8px' }}>-HAS_BODY_SHAPE-&gt;</span>
                <Tag color="cyan">BodyShape</Tag>
              </div>
              <div>
                <Tag color="blue">User</Tag>
                <span style={{ margin: '0 8px' }}>-BOUGHT/VIEWED/RATED-&gt;</span>
                <Tag color="green">Clothing</Tag>
              </div>
              <div>
                <Tag color="green">Clothing</Tag>
                <span style={{ margin: '0 8px' }}>-IN_CATEGORY-&gt;</span>
                <Tag color="red">Category</Tag>
              </div>
              <div>
                <Tag color="green">Clothing</Tag>
                <span style={{ margin: '0 8px' }}>-MATCHES_WITH-&gt;</span>
                <Tag color="green">Clothing</Tag>
              </div>
              <div>
                <Tag color="green">Clothing</Tag>
                <span style={{ margin: '0 8px' }}>-SIMILAR_TO-&gt;</span>
                <Tag color="green">Clothing</Tag>
              </div>
            </div>
          </Card>
        </Col>

        <Col span={12}>
          <Card title="实体类型">
            <Row gutter={[16, 8]}>
              <Col span={12}>
                <Card size="small">
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span>User</span>
                    <Tag>用户实体</Tag>
                  </div>
                  <div style={{ fontSize: '12px', color: '#888', marginTop: '4px' }}>
                    属性: id, username, gender, age, height, weight, body_shape, skin_tone
                  </div>
                </Card>
              </Col>
              <Col span={12}>
                <Card size="small">
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span>Clothing</span>
                    <Tag color="green">服饰实体</Tag>
                  </div>
                  <div style={{ fontSize: '12px', color: '#888', marginTop: '4px' }}>
                    属性: id, name, category, brand, color, material, style, season, price
                  </div>
                </Card>
              </Col>
              <Col span={12}>
                <Card size="small">
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span>Style</span>
                    <Tag color="orange">风格实体</Tag>
                  </div>
                  <div style={{ fontSize: '12px', color: '#888', marginTop: '4px' }}>
                    属性: id, name, description
                  </div>
                </Card>
              </Col>
              <Col span={12}>
                <Card size="small">
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span>Category</span>
                    <Tag color="red">分类实体</Tag>
                  </div>
                  <div style={{ fontSize: '12px', color: '#888', marginTop: '4px' }}>
                    属性: id, name
                  </div>
                </Card>
              </Col>
            </Row>
          </Card>
        </Col>
      </Row>

      <Card title="推荐算法权重" style={{ marginTop: '24px' }}>
        <Row gutter={[24, 16]}>
          <Col span={8}>
            <div style={{ marginBottom: '8px' }}>知识图谱推荐</div>
            <Progress percent={50} strokeColor="#1890ff" />
            <div style={{ fontSize: '12px', color: '#888' }}>
              基于用户-风格偏好关系、服饰搭配关系、体型匹配
            </div>
          </Col>
          <Col span={8}>
            <div style={{ marginBottom: '8px' }}>协同过滤</div>
            <Progress percent={30} strokeColor="#52c41a" />
            <div style={{ fontSize: '12px', color: '#888' }}>
              基于相似用户的历史行为
            </div>
          </Col>
          <Col span={8}>
            <div style={{ marginBottom: '8px' }}>内容推荐</div>
            <Progress percent={20} strokeColor="#faad14" />
            <div style={{ fontSize: '12px', color: '#888' }}>
              基于服饰属性匹配用户偏好
            </div>
          </Col>
        </Row>
      </Card>
    </div>
  );
};

export default KnowledgeGraph;