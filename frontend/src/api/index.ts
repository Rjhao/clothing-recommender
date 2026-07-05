import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'
const TRYON_API_URL = import.meta.env.VITE_TRYON_API_URL || 'http://localhost:8001'

// Create axios instances
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Request interceptor: attach JWT Bearer token from localStorage
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token')
    if (token) {
      config.headers = config.headers || {}
      config.headers.Authorization = `Bearer ${token}`
    }
    return config
  },
  (error) => Promise.reject(error)
)

const tryonApi = axios.create({
  baseURL: TRYON_API_URL,
  timeout: 60000,
  headers: {
    'Content-Type': 'application/json',
  },
})

// ============ Types ============

export interface ClothingItem {
  id: number
  name: string
  category: string
  brand?: string
  color?: string
  price: number
  image_url?: string
  source?: string
}

export interface User {
  id: number
  username: string
  email?: string
  gender?: string
  age?: number
  preferred_style?: string
}

export interface Recommendation {
  clothing_id: number
  score: number
  reason: string
  item: ClothingItem
}

export interface Interaction {
  id: number
  user_id: number
  clothing_id: number
  type: 'view' | 'buy' | 'rate'
  timestamp: string
  score?: number
}

export interface TryOnResult {
  result_id: string
  result_url: string
  explanation: string
}

export interface UploadedImage {
  image_id: string
  filename: string
  url: string
}

// ============ Auth API ============

export const authAPI = {
  login: async (username: string, password: string) => {
    const params = new URLSearchParams()
    params.append('username', username)
    params.append('password', password)
    const response = await api.post('/auth/login', params, {
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    })
    return response.data
  },

  register: async (username: string, password: string, email: string) => {
    const response = await api.post('/auth/register', { username, password, email })
    return response.data
  },

  getMe: async (token: string) => {
    const response = await api.get('/auth/me', {
      headers: { Authorization: `Bearer ${token}` },
    })
    return response.data
  },
}

// ============ User API ============

export const userAPI = {
  getProfile: async (userId: number): Promise<User> => {
    const response = await api.get(`/users/${userId}`)
    return response.data
  },

  updateProfile: async (userId: number, data: Partial<User>) => {
    const response = await api.put(`/users/${userId}`, data)
    return response.data
  },

  getRecommendations: async (userId: number, category?: string, limit = 10) => {
    const params = new URLSearchParams()
    params.append('limit', limit.toString())
    if (category) params.append('category', category)

    const response = await api.get(`/recommend/${userId}`, { params })
    return response.data
  },

  getSimilarItems: async (userId: number, clothingId: number, limit = 10) => {
    const response = await api.get(`/recommend/${userId}/similar/${clothingId}`, {
      params: { limit },
    })
    return response.data
  },

  getOutfitRecommendations: async (userId: number, baseItemId: number, limit = 5) => {
    const response = await api.get(`/recommend/${userId}/outfit/${baseItemId}`, {
      params: { limit },
    })
    return response.data
  },
}

// ============ Clothing API ============

export const clothingAPI = {
  getList: async (params?: {
    category?: string
    style?: string
    limit?: number
    offset?: number
  }): Promise<ClothingItem[]> => {
    const response = await api.get('/clothing/', { params })
    return response.data
  },

  getById: async (clothingId: number): Promise<ClothingItem> => {
    const response = await api.get(`/clothing/${clothingId}`)
    return response.data
  },

  getSimilar: async (clothingId: number, limit = 10) => {
    const response = await api.get(`/clothing/${clothingId}/similar`, { params: { limit } })
    return response.data
  },
}

// ============ Interaction API ============

export const interactionAPI = {
  recordView: async (userId: number, clothingId: number, duration = 0) => {
    const response = await api.post('/interactions/view', {
      user_id: userId,
      clothing_id: clothingId,
      duration,
    })
    return response.data
  },

  recordRating: async (userId: number, clothingId: number, score: number) => {
    const response = await api.post('/interactions/rate', {
      user_id: userId,
      clothing_id: clothingId,
      score,
    })
    return response.data
  },

  recordPurchase: async (userId: number, clothingId: number) => {
    const response = await api.post('/interactions/buy', {
      user_id: userId,
      clothing_id: clothingId,
    })
    return response.data
  },
}

// ============ Knowledge Graph API ============

export const kgAPI = {
  getGraphData: async (userId?: number) => {
    const params = userId ? { user_id: userId } : {}
    const response = await api.get('/knowledge-graph/', { params })
    return response.data
  },

  searchEntities: async (query: string) => {
    const response = await api.get('/knowledge-graph/search', {
      params: { q: query },
    })
    return response.data
  },

  getEntityDetails: async (entityId: string, entityType: string) => {
    const response = await api.get(`/knowledge-graph/entity/${entityType}/${entityId}`)
    return response.data
  },
}

// ============ Virtual Try-On API ============

export const tryonAPI = {
  getCategories: async (): Promise<string[]> => {
    const response = await tryonApi.get('/tryon/categories')
    return response.data
  },

  getClothing: async (category?: string) => {
    const params = category ? { category } : {}
    const response = await tryonApi.get('/tryon/clothing', { params })
    return response.data
  },

  uploadPersonImage: async (file: File): Promise<UploadedImage> => {
    const formData = new FormData()
    formData.append('file', file)

    const response = await tryonApi.post('/tryon/upload/person', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })
    return response.data
  },

  uploadClothImage: async (file: File): Promise<UploadedImage> => {
    const formData = new FormData()
    formData.append('file', file)

    const response = await tryonApi.post('/tryon/upload/cloth', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })
    return response.data
  },

  generateTryOn: async (
    personImageId: string,
    clothImageId: string
  ): Promise<TryOnResult> => {
    const response = await tryonApi.post('/tryon', {
      person_image_id: personImageId,
      cloth_image_id: clothImageId,
    })
    return response.data
  },

  getResult: async (resultId: string) => {
    const response = await tryonApi.get(`/tryon/result/${resultId}`, {
      responseType: 'blob',
    })
    return URL.createObjectURL(response.data)
  },
}

// ============ Health Check ============

export const healthAPI = {
  check: async () => {
    const response = await api.get('/health')
    return response.data
  },

  checkTryOn: async () => {
    const response = await tryonApi.get('/')
    return response.data
  },
}

export { api, tryonApi }
