import { useState } from 'react'
import { Button, Card, List, message } from 'antd'
import axios from 'axios'

const API_URL = 'http://localhost:8000'

export default function TestAPI() {
  const [data, setData] = useState<any[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const testDirectFetch = async () => {
    setLoading(true)
    setError(null)
    try {
      const response = await axios.get(`${API_URL}/clothing/`, { params: { limit: 5 } })
      console.log('Direct fetch success:', response.data)
      setData(response.data)
      message.success(`成功获取 ${response.data.length} 条数据!`)
    } catch (err: any) {
      console.error('Direct fetch error:', err)
      setError(err.message)
      message.error('请求失败: ' + err.message)
    } finally {
      setLoading(false)
    }
  }

  const testViaApiClient = async () => {
    setLoading(true)
    setError(null)
    try {
      const api = axios.create({ baseURL: API_URL })
      const response = await api.get('/clothing/', { params: { limit: 5 } })
      console.log('Via api client success:', response.data)
      setData(response.data)
      message.success(`成功获取 ${response.data.length} 条数据!`)
    } catch (err: any) {
      console.error('Via api client error:', err)
      setError(err.message)
      message.error('请求失败: ' + err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={{ padding: 24 }}>
      <h1>API测试页面</h1>
      <Card style={{ marginBottom: 16 }}>
        <p><strong>测试目标:</strong> {API_URL}</p>
        <p><strong>当前状态:</strong> {loading ? '加载中...' : (error ? '错误: ' + error : (data.length > 0 ? '成功!' : '未测试'))}</p>
      </Card>

      <div style={{ display: 'flex', gap: 8, marginBottom: 16 }}>
        <Button type="primary" onClick={testDirectFetch} loading={loading}>
          直接Fetch测试
        </Button>
        <Button onClick={testViaApiClient} loading={loading}>
          API Client测试
        </Button>
      </div>

      {data.length > 0 && (
        <List
          header={<div>获取到的数据 ({data.length}条)</div>}
          bordered
          dataSource={data}
          renderItem={(item: any) => (
            <List.Item>
              <List.Item.Meta
                title={item.name}
                description={`ID: ${item.id} | 分类: ${item.category} | 价格: ¥${item.price}`}
              />
            </List.Item>
          )}
        />
      )}

      <div style={{ marginTop: 16 }}>
        <h3>调试信息:</h3>
        <pre style={{ background: '#f5f5f5', padding: 8, fontSize: 12 }}>
          API_URL = {API_URL}
          {'\n'}
          VITE_API_URL = {import.meta.env.VITE_API_URL || '未设置(使用默认值)'}
          {'\n'}
          当前页面URL = {typeof window !== 'undefined' ? window.location.href : 'N/A'}
        </pre>
      </div>
    </div>
  )
}
