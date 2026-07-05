import React, { useState, useEffect } from 'react';
import { Table, Tag, Button, Input, Select, Space, Modal, Form, message } from 'antd';
import { PlusOutlined, EditOutlined, DeleteOutlined } from '@ant-design/icons';
import type { ColumnsType } from 'antd/es/table';

interface ClothingItem {
  id: number;
  name: string;
  category: string;
  brand: string;
  color: string;
  material: string;
  style: string;
  season: string;
  occasion: string;
  price: number;
}

const ClothingManagement: React.FC = () => {
  const [clothing, setClothing] = useState<ClothingItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [pagination, setPagination] = useState({ current: 1, pageSize: 20, total: 0 });
  const [filters, setFilters] = useState<{ category?: string; style?: string }>({});
  const [modalVisible, setModalVisible] = useState(false);
  const [editingItem, setEditingItem] = useState<ClothingItem | null>(null);
  const [form] = Form.useForm();

  const fetchClothing = async () => {
    setLoading(true);
    try {
      const params = new URLSearchParams({
        limit: String(pagination.pageSize),
        offset: String((pagination.current - 1) * pagination.pageSize),
        ...(filters.category && { category: filters.category }),
        ...(filters.style && { style: filters.style }),
      });

      const response = await fetch(`http://localhost:8000/clothing/?${params}`);
      const data = await response.json();
      setClothing(data || []);
      // 假设后端返回数组长度就是总数（实际应返回 total 字段）
      setPagination(prev => ({ ...prev, total: data.length >= pagination.pageSize ? (pagination.current + 1) * pagination.pageSize : data.length }));
    } catch (error) {
      console.error('Failed to fetch clothing:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchClothing();
  }, [pagination.current, filters]);

  const handleDelete = async (id: number) => {
    try {
      await fetch(`http://localhost:8000/clothing/${id}`, { method: 'DELETE' });
      message.success('删除成功');
      fetchClothing();
    } catch (error) {
      message.error('删除失败');
    }
  };

  const handleEdit = (record: ClothingItem) => {
    setEditingItem(record);
    form.setFieldsValue(record);
    setModalVisible(true);
  };

  const handleSave = async () => {
    try {
      const values = form.getFieldsValue();
      if (editingItem) {
        await fetch(`http://localhost:8000/clothing/${editingItem.id}`, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(values),
        });
        message.success('更新成功');
      } else {
        await fetch('http://localhost:8000/clothing/', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(values),
        });
        message.success('创建成功');
      }
      setModalVisible(false);
      fetchClothing();
    } catch (error) {
      message.error('保存失败');
    }
  };

  const columns: ColumnsType<ClothingItem> = [
    { title: 'ID', dataIndex: 'id', key: 'id', width: 80, render: (id) => String(id).slice(0, 8) + '...' },
    { title: '名称', dataIndex: 'name', key: 'name', ellipsis: true },
    {
      title: '分类',
      dataIndex: 'category',
      key: 'category',
      render: (category) => <Tag color="blue">{category || '未分类'}</Tag>,
    },
    { title: '品牌', dataIndex: 'brand', key: 'brand', render: (brand) => brand || '-' },
    { title: '颜色', dataIndex: 'color', key: 'color', render: (color) => color || '-' },
    {
      title: '风格',
      dataIndex: 'style',
      key: 'style',
      render: (style) => <Tag color="green">{style || 'casual'}</Tag>,
    },
    {
      title: '季节',
      dataIndex: 'season',
      key: 'season',
      render: (season) => <Tag color="orange">{season || 'all'}</Tag>,
    },
    {
      title: '价格',
      dataIndex: 'price',
      key: 'price',
      render: (price) => `¥${price?.toFixed(2) || '0.00'}`,
    },
    {
      title: '操作',
      key: 'action',
      render: (_, record) => (
        <Space>
          <Button
            type="link"
            icon={<EditOutlined />}
            onClick={() => handleEdit(record)}
          >
            编辑
          </Button>
          <Button
            type="link"
            danger
            icon={<DeleteOutlined />}
            onClick={() => handleDelete(record.id)}
          >
            删除
          </Button>
        </Space>
      ),
    },
  ];

  return (
    <div style={{ padding: '24px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '24px' }}>
        <h1 style={{ fontSize: '24px', margin: 0 }}>服饰管理</h1>
        <Button
          type="primary"
          icon={<PlusOutlined />}
          onClick={() => {
            setEditingItem(null);
            form.resetFields();
            setModalVisible(true);
          }}
        >
          添加服饰
        </Button>
      </div>

      <Space style={{ marginBottom: '16px' }}>
        <Input.Search placeholder="搜索名称" style={{ width: 200 }} />
        <Select
          placeholder="分类筛选"
          style={{ width: 120 }}
          allowClear
          onChange={(value) => setFilters({ ...filters, category: value })}
          options={[
            { value: 'top', label: '上衣' },
            { value: 'bottom', label: '下装' },
            { value: 'dress', label: '连衣裙' },
            { value: 'outerwear', label: '外套' },
            { value: 'shoes', label: '鞋子' },
            { value: 'accessories', label: '配饰' },
          ]}
        />
        <Select
          placeholder="风格筛选"
          style={{ width: 120 }}
          allowClear
          onChange={(value) => setFilters({ ...filters, style: value })}
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
      </Space>

      <Table
        columns={columns}
        dataSource={clothing}
        rowKey="id"
        loading={loading}
        pagination={{
          ...pagination,
          onChange: (page, pageSize) => setPagination({ ...pagination, current: page, pageSize }),
        }}
      />

      <Modal
        title={editingItem ? '编辑服饰' : '添加服饰'}
        open={modalVisible}
        onOk={handleSave}
        onCancel={() => setModalVisible(false)}
        width={600}
      >
        <Form form={form} layout="vertical">
          <Form.Item name="name" label="名称" rules={[{ required: true }]}>
            <Input />
          </Form.Item>
          <Form.Item name="category" label="分类" rules={[{ required: true }]}>
            <Select options={[
              { value: 'top', label: '上衣' },
              { value: 'bottom', label: '下装' },
              { value: 'dress', label: '连衣裙' },
              { value: 'outerwear', label: '外套' },
              { value: 'shoes', label: '鞋子' },
              { value: 'accessories', label: '配饰' },
            ]} />
          </Form.Item>
          <Form.Item name="brand" label="品牌">
            <Input />
          </Form.Item>
          <Form.Item name="color" label="颜色">
            <Input />
          </Form.Item>
          <Form.Item name="material" label="材质">
            <Input />
          </Form.Item>
          <Form.Item name="style" label="风格">
            <Select options={[
              { value: 'minimalist', label: '简约' },
              { value: 'casual', label: '休闲' },
              { value: 'bohemian', label: '波西米亚' },
              { value: 'vintage', label: '复古' },
              { value: 'streetwear', label: '街头' },
              { value: 'business', label: '商务' },
              { value: 'sporty', label: '运动' },
              { value: 'romantic', label: '浪漫' },
            ]} />
          </Form.Item>
          <Form.Item name="price" label="价格">
            <Input type="number" />
          </Form.Item>
        </Form>
      </Modal>
    </div>
  );
};

export default ClothingManagement;