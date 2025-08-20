# TPT-时序数据预测评估系统（Vue版）

基于Vue 3 + Vite重构的现代化时序数据预测评估系统，提供直观的用户界面和强大的数据分析功能。

## 🚀 功能特性

### 核心功能
- **数据上传**: 支持拖拽上传真实值和预测值的npy文件
- **指标计算**: 自动计算多种评估指标（MSE、RMSE、MAE、MAPE、SMAPE、R²、TDA、TID、TIPD等）
- **可视化图表**: 支持多种图表类型（时间序列、散点图、残差图、误差直方图）
- **多样本分析**: 支持多样本对比和分析
- **异常检测**: 自动识别异常样本
- **响应式设计**: 支持桌面和移动设备

### 技术栈
- **前端**: Vue 3 + Vite + Tailwind CSS + Chart.js
- **后端**: Flask + NumPy + SciPy + Matplotlib
- **数据处理**: Pandas + Scikit-learn

## 📦 项目结构

```
vue-tpt-system/
├── src/                    # Vue前端源码
│   ├── components/         # Vue组件
│   │   ├── UploadSection.vue      # 文件上传组件
│   │   ├── ResultsSection.vue     # 结果展示组件
│   │   ├── MetricsCard.vue        # 指标卡片组件
│   │   └── DataChart.vue          # 图表组件
│   ├── services/           # API服务
│   │   └── apiService.js          # 后端API封装
│   ├── assets/             # 静态资源
│   │   └── main.css               # 主样式文件
│   ├── App.vue             # 根组件
│   └── main.js             # 应用入口
├── backend/                # Flask后端
│   ├── app.py              # 主应用文件
│   ├── start_backend.py    # 后端启动脚本
│   └── requirements.txt    # Python依赖
├── public/                 # 公共资源
├── package.json            # Node.js依赖配置
├── vite.config.js          # Vite配置
├── tailwind.config.js      # Tailwind CSS配置
└── README.md               # 项目说明
```

## 🛠️ 安装和运行

### 环境要求
- Node.js 18+ 
- Python 3.8+
- npm 或 yarn

### 1. 安装前端依赖
```bash
cd vue-tpt-system
npm install
```

### 2. 安装后端依赖
```bash
cd backend
pip3 install -r requirements.txt
```

### 3. 启动开发服务器

#### 启动后端服务器（终端1）
```bash
cd backend
python3 start_backend.py
```
后端将运行在: http://localhost:5000

#### 启动前端开发服务器（终端2）
```bash
npm run dev
```
前端将运行在: http://localhost:3000

### 4. 访问应用
打开浏览器访问 http://localhost:3000

## 📖 使用指南

### 1. 数据准备
- 准备真实值和预测值的numpy数组文件（.npy格式）
- 数据格式：三维数组 `(patterns, pred_len, num_targets)`
- 确保真实值和预测值的形状完全一致

### 2. 上传数据
1. 在"数据上传"区域选择或拖拽真实值文件
2. 选择或拖拽预测值文件
3. 点击"计算评估指标"按钮

### 3. 查看结果
- **数据信息**: 显示样本数、目标变量数、预测长度
- **选择控制**: 选择样本、目标变量、图表类型
- **详细指标**: 点击"显示详细指标"查看所有评估指标
- **可视化图表**: 支持多种图表类型的动态切换

### 4. 图表类型说明
- **时间序列**: 对比真实值和预测值的时间序列曲线
- **多样本散点图**: 多个样本的散点图对比
- **多样本误差直方图**: 多个样本的误差分布对比
- **回看数据对比**: 单样本的训练数据与预测数据拼接的时间序列曲线（待修复）

### 5. 异常检测
点击"检测异常样本"按钮，系统将自动识别MAE和TDA指标异常的样本。

## 🔧 开发说明

### 组件架构
- **App.vue**: 根组件，管理全局状态和路由
- **UploadSection.vue**: 处理文件上传和表单提交
- **ResultsSection.vue**: 管理结果展示和用户交互
- **MetricsCard.vue**: 展示评估指标的卡片组件
- **DataChart.vue**: 封装Chart.js的图表组件

### API接口
- `POST /calculate`: 计算评估指标
- `POST /generate_plot`: 生成单样本图表数据
- `POST /generate_multi_sample_plot`: 生成多样本图表数据
- `POST /evaluate_anomalies`: 异常样本检测
- `POST /cleanup`: 清理会话数据
- `GET /health`: 健康检查

### 状态管理
使用Vue 3的响应式系统进行状态管理，主要状态包括：
- 上传状态和文件信息
- 计算结果和指标数据
- 图表配置和显示状态
- 错误信息和加载状态

## 🎨 样式设计

### 设计系统
- **主色调**: 蓝色系 (#3b82f6)
- **辅助色**: 绿色 (#10b981)、紫色 (#6366f1)
- **中性色**: 灰色系
- **字体**: Inter, system-ui, sans-serif

### 响应式设计
- 移动优先的响应式布局
- 支持触摸操作
- 自适应不同屏幕尺寸

## 🚀 部署

### 生产构建
```bash
npm run build
```

### 部署建议
- 前端：可部署到Nginx、Apache或CDN
- 后端：建议使用Gunicorn + Nginx部署Flask应用
- 数据库：如需持久化，可集成Redis或数据库

## 🤝 与原版对比

### 优势
1. **现代化框架**: 使用Vue 3最新特性
2. **组件化架构**: 更好的代码组织和复用
3. **响应式设计**: 支持移动设备
4. **类型安全**: 更好的开发体验
5. **性能优化**: Vite构建工具，热重载
6. **维护性**: 清晰的项目结构和文档

### 功能保持
- 完全保留原版所有功能
- API接口完全兼容
- 计算逻辑完全一致
- 支持所有原有的图表类型和指标

## 📝 更新日志

### v1.0.0 (2025-08-15)
- 初始版本发布
- 完整的Vue 3重构
- 响应式布局支持

### v1.1.0 (2025-08-20)
- 多样本散点图分页显示
- 构建回看数据显示窗口

## 📄 许可证

本项目基于原始TPT系统重构，保留所有原有功能和计算逻辑。

## 🙋‍♂️ 支持

如有问题或建议，请通过以下方式处理或联系：
- 重启电脑
- 重装系统
