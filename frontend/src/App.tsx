import React from 'react'
import { Routes, Route, Navigate, useNavigate, useLocation } from 'react-router-dom'
import { Layout, Menu, Button } from 'antd'
import {
  HomeOutlined,
  ShoppingOutlined,
  UserOutlined,
  AppstoreOutlined,
  LogoutOutlined,
} from '@ant-design/icons'
import { AuthProvider, useAuth } from './context/AuthContext'

import Dashboard from './pages/Dashboard'
import Recommendations from './pages/Recommendations'
import ClothingList from './pages/ClothingList'
import ClothingManagement from './pages/ClothingManagement'
import ClothingDetail from './pages/ClothingDetail'
import Outfit from './pages/Outfit'
import UserManagement from './pages/UserManagement'
import KnowledgeGraph from './pages/KnowledgeGraph'
import VirtualTryOn from './pages/VirtualTryOn'
import Login from './pages/Login'
import Register from './pages/Register'
import TestAPI from './pages/TestAPI'

const { Header, Content, Footer } = Layout

const AppLayout: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const navigate = useNavigate()
  const location = useLocation()
  const { user, isAuthenticated, logout } = useAuth()

  const menuItems = [
    { key: '/', icon: <HomeOutlined />, label: '首页' },
    { key: '/clothing', icon: <AppstoreOutlined />, label: '商品列表' },
    { key: '/recommend', icon: <ShoppingOutlined />, label: '推荐' },
    { key: '/outfit', icon: <AppstoreOutlined />, label: '搭配' },
    { key: '/tryon', icon: <ShoppingOutlined />, label: '虚拟试穿' },
    { key: '/knowledge-graph', icon: <AppstoreOutlined />, label: '知识图谱' },
    { key: '/users', icon: <UserOutlined />, label: '用户中心' },
    { key: '/admin/clothing', icon: <AppstoreOutlined />, label: '服装管理' },
    { key: '/test-api', icon: <AppstoreOutlined />, label: 'API测试' },
  ]

  const handleLogout = () => {
    logout()
    navigate('/login')
  }

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Header style={{ display: 'flex', alignItems: 'center' }}>
        <div
          style={{
            color: 'white',
            fontSize: 20,
            fontWeight: 'bold',
            marginRight: 40,
          }}
        >
          👗 服装推荐系统
        </div>
        <Menu
          theme="dark"
          mode="horizontal"
          selectedKeys={[location.pathname]}
          items={menuItems}
          onClick={({ key }) => navigate(key)}
          style={{ flex: 1 }}
        />
        <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 16 }}>
          {isAuthenticated ? (
            <>
              <span style={{ color: '#fff' }}>欢迎，{user?.username}</span>
              <Button
                type="text"
                icon={<LogoutOutlined />}
                onClick={handleLogout}
                style={{ color: '#fff' }}
              >
                退出
              </Button>
            </>
          ) : (
            <Button type="primary" onClick={() => navigate('/login')}>
              登录
            </Button>
          )}
        </div>
      </Header>
      <Content style={{ padding: '24px 50px' }}>
        <div style={{ background: '#fff', padding: 24, minHeight: '85vh' }}>
          {children}
        </div>
      </Content>
      <Footer style={{ textAlign: 'center' }}>
        基于知识图谱的智能个性化服装推荐系统 © 2026
      </Footer>
    </Layout>
  )
}

const App: React.FC = () => {
  return (
    <AuthProvider>
      <AppLayout>
        <Routes>
          <Route path="/login" element={<Login />} />
          <Route path="/register" element={<Register />} />
          <Route path="/" element={<Dashboard />} />
          <Route path="/clothing" element={<ClothingList />} />
          <Route path="/clothing/:id" element={<ClothingDetail />} />
          <Route path="/recommend" element={<Recommendations />} />
          <Route path="/outfit" element={<Outfit />} />
          <Route path="/tryon" element={<VirtualTryOn />} />
          <Route path="/knowledge-graph" element={<KnowledgeGraph />} />
          <Route path="/users" element={<UserManagement />} />
          <Route path="/admin/clothing" element={<ClothingManagement />} />
          <Route path="/test-api" element={<TestAPI />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </AppLayout>
    </AuthProvider>
  )
}

export default App
