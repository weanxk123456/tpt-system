<template>
  <section id="results-section" class="bg-white rounded-xl shadow-soft p-6 mb-8">
    <div class="flex justify-between items-center mb-6">
      <h2 class="text-2xl font-bold text-neutral flex items-center">
        <i class="fa fa-bar-chart text-primary mr-2"></i>评估结果
      </h2>
      <div class="flex space-x-3">
        <button 
          @click="handleNewCalculation"
          class="bg-white border border-gray-300 text-neutral hover:bg-gray-50 font-medium py-2 px-4 rounded-lg shadow-md transition-custom flex items-center"
        >
          <i class="fa fa-refresh mr-2"></i>重新计算
        </button>
      </div>
    </div>

    <!-- 数据信息 -->
    <div class="bg-neutral-light rounded-lg p-4 mb-6">
      <div class="grid md:grid-cols-3 gap-4">
        <div class="bg-white p-4 rounded-lg shadow-sm">
          <p class="text-gray-500 text-sm">总样本数</p>
          <p class="text-2xl font-bold text-neutral">{{ resultsData.num_patterns }}</p>
        </div>
        <div class="bg-white p-4 rounded-lg shadow-sm">
          <p class="text-gray-500 text-sm">目标变量数</p>
          <p class="text-2xl font-bold text-neutral">{{ resultsData.num_targets }}</p>
        </div>
        <div class="bg-white p-4 rounded-lg shadow-sm">
          <p class="text-gray-500 text-sm">预测长度</p>
          <p class="text-2xl font-bold text-neutral">{{ resultsData.predict_len }}</p>
        </div>
      </div>
    </div>

    <!-- 选择样本和目标 -->
    <div class="mb-6 flex flex-wrap gap-4 items-center">
      <div class="w-full md:w-auto">
        <label for="pattern-select" class="block text-sm font-medium text-gray-700 mb-1">选择样本</label>
        <select 
          id="pattern-select" 
          v-model="selectedPattern"
          @change="updateChart"
          class="w-full md:w-48 bg-white border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary"
        >
          <option v-for="i in resultsData.num_patterns" :key="i-1" :value="i-1">
            样本 {{ i-1 }}
          </option>
        </select>
      </div>

      <div class="w-full md:w-auto">
        <label for="target-select" class="block text-sm font-medium text-gray-700 mb-1">选择目标变量</label>
        <select 
          id="target-select" 
          v-model="selectedTarget"
          @change="updateChart"
          class="w-full md:w-48 bg-white border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary"
        >
          <option v-for="i in resultsData.num_targets" :key="i-1" :value="i-1">
            目标 {{ i-1 }}
          </option>
        </select>
      </div>

      <div class="w-full md:w-auto">
        <label for="plot-type-select" class="block text-sm font-medium text-gray-700 mb-1">图表类型</label>
        <select 
          id="plot-type-select" 
          v-model="selectedPlotType"
          @change="handlePlotTypeChange"
          class="w-full md:w-48 bg-white border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary"
        >
          <option value="time_series">时间序列</option>
<!--          <option value="scatter">散点图</option>-->
<!--          <option value="residual">残差图</option>-->
<!--          <option value="histogram">误差直方图</option>-->
          <option value="multi_scatter">多样本散点图</option>
          <option value="multi_histogram">多样本误差直方图</option>
          <option value="lookback_comparison">回看数据对比</option>
        </select>
      </div>

      <div class="w-full md:w-auto">
        <button 
          @click="showMetrics = !showMetrics"
          class="w-full md:w-auto bg-accent hover:bg-accent/90 text-white font-medium py-2 px-4 rounded-lg shadow-md transition-custom flex items-center"
        >
          <i class="fa fa-eye mr-2"></i>{{ showMetrics ? '隐藏' : '显示' }}详细指标
        </button>
      </div>

      <div class="w-full md:w-auto">
        <button 
          @click="evaluateAnomalies"
          :disabled="isEvaluatingAnomalies"
          class="w-full md:w-auto bg-red-500 hover:bg-red-600 text-white font-medium py-2 px-4 rounded-lg shadow-md transition-custom flex items-center disabled:opacity-50"
        >
          <i :class="isEvaluatingAnomalies ? 'fa fa-spinner fa-spin mr-2' : 'fa fa-exclamation-triangle mr-2'"></i>
          {{ isEvaluatingAnomalies ? '检测中...' : '检测异常样本' }}
        </button>
      </div>
    </div>

    <!-- 范围选择（仅对时间序列图有效） -->
    <div v-if="selectedPlotType === 'time_series'" class="mb-6 flex flex-wrap gap-4 items-center">
      <div class="w-full md:w-auto">
        <label for="start-idx" class="block text-sm font-medium text-gray-700 mb-1">起始索引</label>
        <input 
          type="number" 
          id="start-idx" 
          v-model.number="startIdx"
          :min="0" 
          :max="resultsData.predict_len - 1"
          class="w-full md:w-32 bg-white border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary"
        >
      </div>

      <div class="w-full md:w-auto">
        <label for="end-idx" class="block text-sm font-medium text-gray-700 mb-1">结束索引</label>
        <input 
          type="number" 
          id="end-idx" 
          v-model.number="endIdx"
          :min="1" 
          :max="resultsData.predict_len"
          class="w-full md:w-32 bg-white border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary"
        >
      </div>

      <div class="w-full md:w-auto">
        <button 
          @click="updateChart"
          class="w-full md:w-auto bg-secondary hover:bg-secondary/90 text-white font-medium py-2 px-4 rounded-lg shadow-md transition-custom flex items-center"
        >
          <i class="fa fa-refresh mr-2"></i>更新图表
        </button>
      </div>
    </div>

    <!-- 回看数据控制（仅对回看数据对比图有效） -->
    <div v-if="selectedPlotType === 'lookback_comparison'" class="mb-6">
      <div class="flex flex-wrap gap-4 items-center">
        <div class="w-full md:w-auto">
          <label for="lookback-steps" class="block text-sm font-medium text-gray-700 mb-1">回看步数</label>
          <input 
            type="number" 
            id="lookback-steps" 
            v-model.number="lookbackSteps"
            :min="1" 
            :max="100"
            class="w-full md:w-32 bg-white border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary"
          >
        </div>

        <div class="w-full md:w-auto">
          <label for="training-data-file" class="block text-sm font-medium text-gray-700 mb-1">训练数据文件</label>
          <input 
            type="file" 
            id="training-data-file" 
            @change="handleTrainingDataUpload"
            accept=".csv,.json,.npy"
            class="w-full md:w-64 bg-white border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary"
          >
        </div>

        <div class="w-full md:w-auto">
          <button 
            @click="updateLookbackChart"
            :disabled="!trainingDataUploaded"
            class="w-full md:w-auto bg-green-500 hover:bg-green-600 text-white font-medium py-2 px-4 rounded-lg shadow-md transition-custom flex items-center disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <i class="fa fa-refresh mr-2"></i>生成回看对比图
          </button>
        </div>
      </div>

      <div v-if="trainingDataUploaded" class="mt-2 text-sm text-green-600">
        <i class="fa fa-check-circle mr-1"></i>训练数据已上传
      </div>
    </div>
    <!-- 多样本选择控制（仅对多样本图表有效） -->
    <div v-if="isMultiSamplePlot" class="mb-6">
      <!-- 第一行：基本控制 -->
      <div class="flex flex-wrap gap-4 items-center mb-4">
        <div class="w-full md:w-auto">
          <label for="sample-count" class="block text-sm font-medium text-gray-700 mb-1">每页样本数</label>
          <select 
            id="sample-count" 
            v-model.number="samplesPerPage"
            @change="handleSamplesPerPageChange"
            class="w-full md:w-32 bg-white border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary"
          >
            <option :value="500">500</option>
            <option :value="1000">1000</option>
          </select>
        </div>

        <div class="w-full md:w-auto">
          <label for="sample-selection-mode" class="block text-sm font-medium text-gray-700 mb-1">选择模式</label>
          <select 
            id="sample-selection-mode" 
            v-model="sampleSelectionMode"
            @change="handleSampleSelectionModeChange"
            class="w-full md:w-48 bg-white border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary"
          >
            <option value="sequential">顺序分页</option>
            <option value="random">随机选择</option>
            <option value="custom">自定义选择</option>
          </select>
        </div>

        <div class="w-full md:w-auto">
          <label for="render-mode" class="block text-sm font-medium text-gray-700 mb-1">渲染模式</label>
          <select 
            id="render-mode" 
            v-model="renderMode"
            @change="handleRenderModeChange"
            class="w-full md:w-32 bg-white border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary"
          >
            <option value="dynamic">动态图表</option>
<!--            <option value="static">静态图片</option>-->
          </select>
        </div>

        <div v-if="sampleSelectionMode === 'custom'" class="w-full md:w-auto">
          <label for="custom-indices" class="block text-sm font-medium text-gray-700 mb-1">自定义索引（逗号分隔）</label>
          <input 
            type="text" 
            id="custom-indices" 
            v-model="customIndices"
            placeholder="0,1,2,5,10" 
            class="w-full md:w-64 bg-white border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary"
          >
        </div>

        <div class="w-full md:w-auto">
          <button 
            @click="updateMultiSampleChart"
            class="w-full md:w-auto bg-purple-500 hover:bg-purple-600 text-white font-medium py-2 px-4 rounded-lg shadow-md transition-custom flex items-center"
          >
            <i class="fa fa-refresh mr-2"></i>更新图表
          </button>
        </div>
      </div>

      <!-- 第二行：分页控制（仅在顺序分页模式下显示） -->
      <div v-if="sampleSelectionMode === 'sequential'" class="flex flex-wrap gap-4 items-center">
        <div class="flex items-center space-x-2">
          <span class="text-sm text-gray-600">总样本数: {{ resultsData.num_patterns }}</span>
          <span class="text-sm text-gray-600">|</span>
          <span class="text-sm text-gray-600">总页数: {{ totalPages }}</span>
          <span class="text-sm text-gray-600">|</span>
          <span class="text-sm text-gray-600">当前页: {{ currentPage }}</span>
        </div>

        <div class="flex items-center space-x-1">
          <button 
            @click="goToFirstPage"
            :disabled="currentPage === 1"
            class="px-2 py-1 text-sm bg-white border border-gray-300 rounded hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <i class="fa fa-angle-double-left"></i>
          </button>
          <button 
            @click="goToPreviousPage"
            :disabled="currentPage === 1"
            class="px-2 py-1 text-sm bg-white border border-gray-300 rounded hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <i class="fa fa-angle-left"></i>
          </button>
          
          <div class="flex items-center space-x-1">
            <span class="text-sm text-gray-600">跳转到</span>
            <input 
              type="number" 
              v-model.number="pageInput"
              :min="1" 
              :max="totalPages"
              @keyup.enter="goToPage"
              class="w-16 px-2 py-1 text-sm border border-gray-300 rounded focus:outline-none focus:ring-1 focus:ring-primary/50"
            >
            <span class="text-sm text-gray-600">页</span>
            <button 
              @click="goToPage"
              class="px-2 py-1 text-sm bg-primary text-white rounded hover:bg-primary/90"
            >
              跳转
            </button>
          </div>

          <button 
            @click="goToNextPage"
            :disabled="currentPage === totalPages"
            class="px-2 py-1 text-sm bg-white border border-gray-300 rounded hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <i class="fa fa-angle-right"></i>
          </button>
          <button 
            @click="goToLastPage"
            :disabled="currentPage === totalPages"
            class="px-2 py-1 text-sm bg-white border border-gray-300 rounded hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <i class="fa fa-angle-double-right"></i>
          </button>
        </div>
      </div>
    </div>

    <!-- 指标卡片 -->
    <MetricsCard 
      v-if="showMetrics" 
      :metrics="currentMetrics"
      class="mb-8"
    />

    <!-- 异常样本检测结果 -->
    <div v-if="anomalyResults" class="bg-red-50 border border-red-200 rounded-lg p-4 mb-6">
      <h3 class="text-lg font-semibold text-red-800 mb-3 flex items-center">
        <i class="fa fa-exclamation-triangle text-red-600 mr-2"></i>异常样本检测结果
      </h3>
      <div class="text-sm text-red-700">
        <p class="mb-2">检测到 <strong>{{ anomalyResults.length }}</strong> 个异常样本：</p>
        <div class="max-h-40 overflow-y-auto">
          <div v-for="(anomaly, index) in anomalyResults" :key="index" class="mb-1">
            样本 {{ anomaly.样本号 }}, 目标 {{ anomaly.目标号 }} - 
            MAE: {{ anomaly.MAE.toFixed(4) }}, 
            TDA: {{ anomaly.TDA.toFixed(2) }}%
          </div>
        </div>
      </div>
    </div>

    <!-- 图表展示区域 -->
    <div v-if="chartData || staticPlotData" class="mb-8">
      <!-- 动态图表 -->
      <DataChart 
        v-if="chartData && renderMode === 'dynamic'"
        :chartData="chartData"
        :chartType="selectedPlotType"
        :isLoading="isLoadingChart"
        :showBackButton="showBackButton"
        @point-click="handleScatterPointClick"
        @back-to-scatter="handleBackToScatter"
      />
      
      <!-- 静态图表 -->
      <StaticChart 
        v-if="staticPlotData && renderMode === 'static'"
        :staticData="staticPlotData"
        :isLoading="isLoadingChart"
        @region-click="handleStaticChartClick"
      />
    </div>
  </section>
</template>

<script>
import MetricsCard from './MetricsCard.vue'
import DataChart from './DataChart.vue'
import StaticChart from './StaticChart.vue'
import apiService from '../services/apiService.js'

export default {
  name: 'ResultsSection',
  components: {
    MetricsCard,
    DataChart,
    StaticChart
  },
  props: {
    resultsData: {
      type: Object,
      required: true
    }
  },
  data() {
    return {
      selectedPattern: 0,
      selectedTarget: 0,
      selectedPlotType: 'time_series',
      showMetrics: false,
      startIdx: 0,
      endIdx: 100,
      samplesPerPage: 1000,
      sampleSelectionMode: 'sequential',
      customIndices: '',
      currentPage: 1,
      pageInput: 1,
      renderMode: 'dynamic',
      chartData: null,
      staticPlotData: null,
      isLoadingChart: false,
      isEvaluatingAnomalies: false,
      anomalyResults: null,
      // 散点图跳转相关状态
      previousPlotType: null,
      previousSelectedPattern: null,
      previousSelectedTarget: null,
      showBackButton: false,
      // 防抖相关
      updateChartTimeout: null,
      updateMultiSampleChartTimeout: null,
      // 回看数据相关
      lookbackSteps: 10,
      trainingDataUploaded: false,
      trainingDataFile: null
    }
  },
  computed: {
    isMultiSamplePlot() {
      return this.selectedPlotType === 'multi_scatter' || this.selectedPlotType === 'multi_histogram'
    },
    isLookbackPlot() {
      return this.selectedPlotType === 'lookback_comparison'
    },
    currentMetrics() {
      if (!this.resultsData.all_metrics) return null
      return this.resultsData.all_metrics[this.selectedPattern]?.[this.selectedTarget]
    },
    totalPages() {
      if (!this.resultsData || this.sampleSelectionMode !== 'sequential') return 1
      return Math.ceil(this.resultsData.num_patterns / this.samplesPerPage)
    },
    currentPageSamples() {
      if (!this.resultsData || this.sampleSelectionMode !== 'sequential') return []
      const startIdx = (this.currentPage - 1) * this.samplesPerPage
      const endIdx = Math.min(startIdx + this.samplesPerPage, this.resultsData.num_patterns)
      return Array.from({length: endIdx - startIdx}, (_, i) => startIdx + i)
    }
  },
  mounted() {
    this.initializeData()
    this.updateChart()
  },
  methods: {
    initializeData() {
      // 设置默认值
      this.endIdx = Math.min(100, this.resultsData.predict_len)
      this.samplesPerPage = Math.min(1000, this.resultsData.num_patterns)
      this.currentPage = 1
      this.pageInput = 1
    },
    handleNewCalculation() {
      this.$emit('new-calculation')
    },
    handlePlotTypeChange() {
      this.updateControlsVisibility()
      if (this.isMultiSamplePlot) {
        this.updateMultiSampleChart()
      } else if (this.isLookbackPlot) {
        // 回看数据图表需要用户上传训练数据后才能生成
        if (this.trainingDataUploaded) {
          this.updateLookbackChart()
        }
      } else {
        this.updateChart()
      }
    },
    handleRenderModeChange() {
      // 当渲染模式改变时，清空当前图表数据并重新生成
      this.chartData = null
      this.staticPlotData = null
      if (this.isMultiSamplePlot) {
        this.updateMultiSampleChart()
      } else {
        this.updateChart()
      }
    },
    handleSampleSelectionModeChange() {
      // 当选择模式改变时，重置分页状态
      this.currentPage = 1
      this.pageInput = 1
    },
    handleSamplesPerPageChange() {
      // 当每页样本数改变时，重新计算当前页
      this.currentPage = 1
      this.pageInput = 1
    },
    goToFirstPage() {
      this.currentPage = 1
      this.pageInput = 1
      this.updateMultiSampleChart()
    },
    goToPreviousPage() {
      if (this.currentPage > 1) {
        this.currentPage--
        this.pageInput = this.currentPage
        this.updateMultiSampleChart()
      }
    },
    goToNextPage() {
      if (this.currentPage < this.totalPages) {
        this.currentPage++
        this.pageInput = this.currentPage
        this.updateMultiSampleChart()
      }
    },
    goToLastPage() {
      this.currentPage = this.totalPages
      this.pageInput = this.currentPage
      this.updateMultiSampleChart()
    },
    goToPage() {
      const page = parseInt(this.pageInput)
      if (page >= 1 && page <= this.totalPages) {
        this.currentPage = page
        this.updateMultiSampleChart()
      } else {
        // 重置输入框为当前页
        this.pageInput = this.currentPage
      }
    },
    updateControlsVisibility() {
      // Vue的响应式系统会自动处理显示/隐藏逻辑
    },
    async updateChart() {
      // 清除之前的防抖定时器
      if (this.updateChartTimeout) {
        clearTimeout(this.updateChartTimeout)
      }

      // 使用防抖机制，避免快速切换时重复请求
      this.updateChartTimeout = setTimeout(() => {
        this.doUpdateChart()
      }, 200)
    },
    async doUpdateChart() {
      if (this.isMultiSamplePlot) {
        await this.updateMultiSampleChart()
        return
      }

      if (this.isLookbackPlot) {
        await this.updateLookbackChart()
        return
      }

      // 验证范围
      if (this.selectedPlotType === 'time_series' && this.startIdx >= this.endIdx) {
        this.$emit('error', '起始索引必须小于结束索引')
        return
      }

      if (this.isLoadingChart) return

      this.isLoadingChart = true

      try {
        const requestData = {
          pattern_idx: this.selectedPattern,
          target_idx: this.selectedTarget,
          plot_type: this.selectedPlotType
        }

        if (this.selectedPlotType === 'time_series') {
          requestData.start_idx = this.startIdx
          requestData.end_idx = this.endIdx
        }

        const response = await apiService.generatePlot(requestData)
        
        if (response.success) {
          this.chartData = response.plot_data
        } else {
          this.$emit('error', response.error || '生成图表失败')
        }
      } catch (error) {
        this.$emit('error', '生成图表时发生网络错误: ' + error.message)
      } finally {
        this.isLoadingChart = false
      }
    },
    async updateMultiSampleChart() {
      // 清除之前的防抖定时器
      if (this.updateMultiSampleChartTimeout) {
        clearTimeout(this.updateMultiSampleChartTimeout)
      }

      // 使用防抖机制，避免快速切换时重复请求
      this.updateMultiSampleChartTimeout = setTimeout(() => {
        this.doUpdateMultiSampleChart()
      }, 200)
    },
    async doUpdateMultiSampleChart() {
      // 验证图表类型
      if (!this.isMultiSamplePlot) {
        this.$emit('error', '当前图表类型不支持多样本模式')
        return
      }

      if (this.isLoadingChart) return

      this.isLoadingChart = true

      try {
        const requestData = {
          target_idx: this.selectedTarget,
          plot_type: this.selectedPlotType === 'multi_scatter' ? 'scatter' : 'histogram'
        }

        // 根据选择模式设置参数
        if (this.sampleSelectionMode === 'sequential') {
          // 分页模式：使用当前页的样本
          requestData.sample_indices = this.currentPageSamples
        } else if (this.sampleSelectionMode === 'random') {
          // 随机模式：生成随机样本索引
          const randomIndices = []
          const availableIndices = Array.from({length: this.resultsData.num_patterns}, (_, i) => i)
          const actualCount = Math.min(this.samplesPerPage, this.resultsData.num_patterns)

          for (let i = 0; i < actualCount; i++) {
            const randomIndex = Math.floor(Math.random() * availableIndices.length)
            randomIndices.push(availableIndices.splice(randomIndex, 1)[0])
          }
          requestData.sample_indices = randomIndices
        } else if (this.sampleSelectionMode === 'custom') {
          // 自定义模式：解析自定义索引
          const customIndicesStr = this.customIndices.trim()
          if (!customIndicesStr) {
            this.$emit('error', '请输入自定义样本索引')
            return
          }

          try {
            const customIndices = customIndicesStr.split(',').map(s => {
              const idx = parseInt(s.trim())
              if (isNaN(idx) || idx < 0 || idx >= this.resultsData.num_patterns) {
                throw new Error(`无效的样本索引: ${s.trim()}`)
              }
              return idx
            })
            requestData.sample_indices = customIndices
          } catch (parseError) {
            this.$emit('error', parseError.message)
            return
          }
        }

        // 根据渲染模式选择不同的API
        if (this.renderMode === 'static' && this.selectedPlotType === 'multi_scatter') {
          // 生成静态图表
          const response = await apiService.generateStaticPlot(requestData)
          
          if (response.success) {
            this.staticPlotData = response.static_plot
            this.chartData = null // 清空动态图表数据
          } else {
            this.$emit('error', response.error || '生成静态图表失败')
          }
        } else {
          // 生成动态图表
          const response = await apiService.generateMultiSamplePlot(requestData)
          
          if (response.success) {
            this.chartData = response.plot_data
            this.staticPlotData = null // 清空静态图表数据
          } else {
            this.$emit('error', response.error || '生成多样本图表失败')
          }
        }
      } catch (error) {
        this.$emit('error', '生成图表时发生网络错误: ' + error.message)
      } finally {
        this.isLoadingChart = false
      }
    },
    async evaluateAnomalies() {
      this.isEvaluatingAnomalies = true

      try {
        const response = await apiService.evaluateAnomalies()
        
        if (response.success) {
          this.anomalyResults = response.anomalies
        } else {
          this.$emit('error', response.error || '异常样本检测失败')
        }
      } catch (error) {
        this.$emit('error', '评估异常样本时发生网络错误: ' + error.message)
      } finally {
        this.isEvaluatingAnomalies = false
      }
    },
    handleStaticChartClick(clickData) {
      // 处理静态图表点击事件
      console.log('静态图表点击:', clickData)
      
      // 可以在这里添加更多的交互逻辑，比如：
      // 1. 跳转到特定样本的详细视图
      // 2. 高亮显示相关样本
      // 3. 显示样本的详细指标
      
      if (clickData.samples && clickData.samples.length > 0) {
        // 示例：自动选择第一个样本并切换到单样本视图
        const firstSample = clickData.samples[0]
        this.selectedPattern = firstSample
        this.selectedPlotType = 'time_series'
        this.updateChart()
        
        // 显示提示信息
        this.$emit('info', `已切换到样本 ${firstSample} 的时间序列视图`)
      }
    },
    handleScatterPointClick(clickData) {
      // 处理散点图点击事件，跳转到对应样本的时间序列图
      console.log('散点图点击:', clickData)
      
      // 保存当前状态以便返回
      this.previousPlotType = this.selectedPlotType
      this.previousSelectedPattern = this.selectedPattern
      this.previousSelectedTarget = this.selectedTarget
      
      // 切换到点击的样本的时间序列图
      this.selectedPattern = clickData.sampleIndex
      this.selectedPlotType = 'time_series'
      this.showBackButton = true
      
      // 更新图表
      this.updateChart()
    },
    handleBackToScatter() {
      // 返回到之前的散点图
      if (this.previousPlotType) {
        this.selectedPlotType = this.previousPlotType
        this.selectedPattern = this.previousSelectedPattern
        this.selectedTarget = this.previousSelectedTarget
        this.showBackButton = false
        
        // 清除保存的状态
        this.previousPlotType = null
        this.previousSelectedPattern = null
        this.previousSelectedTarget = null
        
        // 更新图表
        if (this.isMultiSamplePlot) {
          this.updateMultiSampleChart()
        } else {
          this.updateChart()
        }
      }
    },
    handleTrainingDataUpload(event) {
      const file = event.target.files[0]
      if (file) {
        this.trainingDataFile = file
        this.trainingDataUploaded = true
      } else {
        this.trainingDataFile = null
        this.trainingDataUploaded = false
      }
    },
    async updateLookbackChart() {
      if (!this.trainingDataUploaded) {
        this.$emit('error', '请先上传训练数据文件')
        return
      }

      if (this.isLoadingChart) return
      this.isLoadingChart = true

      try {
        // 构建上传 FormData
        const formData = new FormData()
        formData.append('training_data', this.trainingDataFile)
        formData.append('pattern_idx', this.selectedPattern)
        formData.append('target_idx', this.selectedTarget)
        formData.append('lookback_steps', this.lookbackSteps)

        // 调用后端接口
        const response = await apiService.generateLookbackPlot(formData)
        // ✅ 正确处理平级 success

        if (response.success) {
          this.chartData = response.plot_data
          this.staticPlotData = null // 清空静态图表数据
        }
        // 后端返回 success=false 且带 error
        else if (response.error) {
          this.$emit('error', '生成回看对比图失败1: ' + response.error)
        }
        // 后端返回 success=false 无 error
        else {
          this.$emit('error', '生成回看对比图失败: 后端返回 success=false，但未提供错误信息')
        }
      } catch (error) {
        // 网络错误或 HTTP 错误
        if (error.response && error.response && error.response.error) {
          this.$emit('error', '生成回看对比图失败2: ' + error.response.error)
        } else {
          this.$emit('error', '生成回看对比图时发生网络错误: ' + error.message)
        }
      } finally {
        this.isLoadingChart = false
      }
    },
  }
}
</script>

