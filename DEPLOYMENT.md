# 部署指南

## 🚀 生产环境部署

### 方案一：传统服务器部署

#### 1. 前端部署
```bash
# 构建生产版本
npm run build

# 将dist目录部署到Web服务器
# 例如：Nginx配置
server {
    listen 80;
    server_name your-domain.com;
    root /path/to/vue-tpt-system/dist;
    index index.html;
    
    location / {
        try_files $uri $uri/ /index.html;
    }
    
    # API代理
    location /api/ {
        proxy_pass http://localhost:5000/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

#### 2. 后端部署
```bash
# 安装生产依赖
pip install gunicorn

# 使用Gunicorn启动
cd backend
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### 方案二：Docker部署

#### 1. 创建Dockerfile（前端）
```dockerfile
# 前端Dockerfile
FROM node:18-alpine as build
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/nginx.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

#### 2. 创建Dockerfile（后端）
```dockerfile
# 后端Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY backend/requirements.txt .
RUN pip install -r requirements.txt
COPY backend/ .
COPY upload/ ./upload/
EXPOSE 5000
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

#### 3. Docker Compose
```yaml
version: '3.8'
services:
  frontend:
    build:
      context: .
      dockerfile: Dockerfile.frontend
    ports:
      - "80:80"
    depends_on:
      - backend
      
  backend:
    build:
      context: .
      dockerfile: Dockerfile.backend
    ports:
      - "5000:5000"
    volumes:
      - ./data:/app/data
```

### 方案三：云平台部署

#### Vercel（前端）
```bash
# 安装Vercel CLI
npm i -g vercel

# 部署
vercel --prod
```

#### Heroku（后端）
```bash
# 创建Procfile
echo "web: gunicorn app:app" > backend/Procfile

# 部署到Heroku
heroku create your-app-name
git subtree push --prefix=backend heroku main
```

## 🔧 环境配置

### 环境变量
```bash
# 生产环境变量
NODE_ENV=production
VITE_API_BASE_URL=https://your-api-domain.com

# 后端环境变量
FLASK_ENV=production
SECRET_KEY=your-secret-key
```

### 安全配置
1. **HTTPS**: 生产环境必须使用HTTPS
2. **CORS**: 限制允许的域名
3. **文件上传**: 限制文件大小和类型
4. **会话安全**: 使用安全的会话配置

## 📊 性能优化

### 前端优化
- 启用Gzip压缩
- 配置CDN加速
- 图片懒加载
- 代码分割

### 后端优化
- 使用Redis缓存
- 数据库连接池
- 异步处理
- 负载均衡

## 🔍 监控和日志

### 前端监控
- 错误追踪（Sentry）
- 性能监控
- 用户行为分析

### 后端监控
- 应用性能监控
- 日志聚合
- 健康检查
- 资源使用监控

## 🛠️ 维护

### 备份策略
- 定期备份用户数据
- 代码版本控制
- 配置文件备份

### 更新流程
1. 测试环境验证
2. 灰度发布
3. 全量发布
4. 回滚预案

