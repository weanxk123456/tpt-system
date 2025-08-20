import axios from 'axios'

// 配置基础URL，开发环境下指向Flask后端
const API_BASE_URL = process.env.NODE_ENV === 'production' 
  ? '' // 生产环境使用相对路径
  : 'http://localhost:5000' // 开发环境指向Flask服务器

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 120000, // 2分钟超时
  withCredentials: true // 支持会话cookie
})

// 请求拦截器
api.interceptors.request.use(
  config => {
    return config
  },
  error => {
    return Promise.reject(error)
  }
)

// 响应拦截器
api.interceptors.response.use(
  response => {
    return response.data
  },
  error => {
    if (error.response) {
      // 服务器返回错误状态码
      throw new Error(`服务器错误: ${error.response.status}`)
    } else if (error.request) {
      // 请求发出但没有收到响应
      throw new Error('网络连接错误，请检查后端服务是否启动')
    } else {
      // 其他错误
      throw new Error(error.message)
    }
  }
)

const apiService = {
  // 计算评估指标
  async calculate(formData) {
    return await api.post('/calculate', formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    })
  },

  // 生成图表数据
  async generatePlot(data) {
    return await api.post('/generate_plot', data, {
      headers: {
        'Content-Type': 'application/json'
      }
    })
  },

  // 生成多样本图表数据
  async generateMultiSamplePlot(data) {
    return await api.post('/generate_multi_sample_plot', data, {
      headers: {
        'Content-Type': 'application/json'
      }
    })
  },

  // 生成静态图表
  async generateStaticPlot(data) {
    return await api.post('/generate_static_plot', data, {
      headers: {
        'Content-Type': 'application/json'
      }
    })
  },

  // 评估异常样本
  async evaluateAnomalies(data = {}) {
    return await api.post('/evaluate_anomalies', data, {
      headers: {
        'Content-Type': 'application/json'
      }
    })
  },

  // 清理会话数据
  async cleanup() {
    return await api.post('/cleanup')
  },

  // 生成回看数据对比图表
  async generateLookbackPlot(formData) {
    return await api.post('/generate_lookback_plot', formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    })
  }
}

export default apiService

