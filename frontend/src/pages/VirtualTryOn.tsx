/**
 * VirtualTryOn — 3D-only clothing preview.
 *
 * Simplified page that shows the ThreeDClothViewer with a row of preset
 * GLB clothing models (and a couple of texture-mapped fallbacks).
 * Real-photo tryon / category selectors / upload UI removed on purpose.
 */
import React, { useState } from 'react'
import { Card, Row, Col, Button, Space, Typography, message } from 'antd'
import { UserOutlined, LinkOutlined } from '@ant-design/icons'
import ThreeDClothViewer from '../components/ThreeDClothViewer'

const { Title, Text } = Typography

interface ClothingSet {
  top?: string
  bottom?: string
  dress?: string
  outerwear?: string
}

// Real 3D garment presets — load user-supplied .glb clothing models
const REAL_CLOTH_PRESETS: Array<{
  key: string
  label: string
  emoji: string
  clothing: ClothingSet
}> = [
  {
    key: 'balmain',
    label: 'Balmain T 恤',
    emoji: '👕',
    clothing: { top: '/models/clothes/balmain_t-shirt.glb' },
  },
  {
    key: 'skirt_top',
    label: '半身裙+上衣',
    emoji: '👗',
    clothing: { dress: '/models/clothes/skirt_and_t-shirt.glb' },
  },
]

const VirtualTryOn: React.FC = () => {
  const [clothing, setClothing] = useState<ClothingSet>({})
  const [gender, setGender] = useState<'male' | 'female'>('female')

  return (
    <div style={{ padding: 24 }}>
      <Title level={2} style={{ textAlign: 'center', marginBottom: 24 }}>
        👗 虚拟试穿
      </Title>

      <Row gutter={32} justify="center">
        <Col span={14}>
          <Card
            title={
              <Space>
                <UserOutlined />
                <span>3D 虚拟人物</span>
              </Space>
            }
            extra={
              <Button
                icon={<LinkOutlined />}
                onClick={() => message.info('刷新页面使用新形象')}
                size="small"
              >
                更换形象
              </Button>
            }
          >
            {/* Real-cloth presets — tap to swap outfit instantly */}
            <div style={{ marginBottom: 12 }}>
              <Text type="secondary" style={{ fontSize: 12 }}>
                真实衣服 (1-click):
              </Text>
              <div style={{ marginTop: 6 }}>
                <Space wrap>
                  {REAL_CLOTH_PRESETS.map((p) => (
                    <Button
                      key={p.key}
                      size="small"
                      onClick={() => {
                        setClothing(p.clothing)
                        message.success(`已切换到: ${p.label}`)
                      }}
                    >
                      {p.emoji} {p.label}
                    </Button>
                  ))}
                </Space>
              </div>
            </div>
            <div style={{ display: 'flex', justifyContent: 'center', padding: '20px 0' }}>
              <ThreeDClothViewer
                clothing={clothing}
                width={400}
                height={550}
                gender={gender}
                onGenderChange={setGender}
                autoRotate={false}
              />
            </div>
          </Card>
        </Col>
      </Row>
    </div>
  )
}

export default VirtualTryOn