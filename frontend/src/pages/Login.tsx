import { useEffect } from 'react'
import { Card, Typography, Spin } from 'antd'
import { useNavigate } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'

const { Title, Text } = Typography

export default function Login() {
  const { login, isAuthenticated } = useAuth()
  const navigate = useNavigate()

  useEffect(() => {
    // 直接写死登录 admin/admin123
    login('admin', 'admin123')
      .then(() => navigate('/'))
      .catch(() => {})
  }, [])

  // 已登录直接跳转
  if (isAuthenticated) {
    navigate('/')
    return null
  }

  return (
    <div style={{
      minHeight: '100vh',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    }}>
      <Card style={{ width: 400, boxShadow: '0 14px 40px rgba(0,0,0,0.2)' }}>
        <div style={{ textAlign: 'center', marginBottom: 24 }}>
          <Title level={2}>服装推荐系统</Title>
          <Text type="secondary">基于知识图谱的智能推荐</Text>
        </div>
        <div style={{ textAlign: 'center', padding: '40px 0' }}>
          <Spin size="large" />
          <div style={{ marginTop: 16 }}>正在登录...</div>
        </div>
      </Card>
    </div>
  )
}
