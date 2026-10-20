import { createContext, useContext, useState, useEffect, ReactNode } from 'react'
import { authAPI } from '../api'

interface User {
  id: number
  username: string
  email?: string
  gender?: string
  age?: number
  height?: number
  weight?: number
  body_shape?: string
  skin_tone?: string
}

type RegisterExtra = Partial<
  Pick<User, 'gender' | 'age' | 'height' | 'weight' | 'body_shape' | 'skin_tone'>
>

interface AuthContextType {
  user: User | null
  token: string | null
  isAuthenticated: boolean
  isLoading: boolean
  login: (username: string, password: string) => Promise<void>
  register: (
    username: string,
    email: string,
    password: string,
    extra?: RegisterExtra
  ) => Promise<void>
  refreshUser: () => Promise<void>
  logout: () => void
}

const AuthContext = createContext<AuthContextType | undefined>(undefined)

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null)
  const [token, setToken] = useState<string | null>(localStorage.getItem('token'))
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    // Check if we have a token and fetch user info
    const initAuth = async () => {
      const storedToken = localStorage.getItem('token')
      if (storedToken) {
        try {
          const userData = await authAPI.getMe(storedToken)
          setUser(userData)
          setToken(storedToken)
        } catch (error) {
          // Token invalid, clear it
          localStorage.removeItem('token')
          setToken(null)
          setUser(null)
        }
      }
      setIsLoading(false)
    }
    initAuth()
  }, [])

  const login = async (username: string, password: string) => {
    const response = await authAPI.login(username, password)
    const { access_token } = response
    const userData = await authAPI.getMe(access_token)
    // Only persist the token after getMe succeeds, so it is not orphaned on failure
    localStorage.setItem('token', access_token)
    setToken(access_token)
    setUser(userData)
  }

  const register = async (
    username: string,
    email: string,
    password: string,
    extra?: RegisterExtra
  ) => {
    const response = await authAPI.register(username, password, email, extra)
    const { access_token } = response
    const userData = await authAPI.getMe(access_token)
    // Only persist the token after getMe succeeds, so it is not orphaned on failure
    localStorage.setItem('token', access_token)
    setToken(access_token)
    setUser(userData)
  }

  // Re-pull /auth/me so edits made elsewhere (e.g. the profile page) propagate.
  const refreshUser = async () => {
    const storedToken = localStorage.getItem('token')
    if (!storedToken) return
    const userData = await authAPI.getMe(storedToken)
    setUser(userData)
  }

  const logout = () => {
    localStorage.removeItem('token')
    setToken(null)
    setUser(null)
  }

  return (
    <AuthContext.Provider
      value={{
        user,
        token,
        isAuthenticated: !!token && !!user,
        isLoading,
        login,
        register,
        refreshUser,
        logout,
      }}
    >
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  const context = useContext(AuthContext)
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider')
  }
  return context
}
