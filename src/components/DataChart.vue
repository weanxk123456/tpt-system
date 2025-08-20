<template>
  <div class="grid grid-cols-1 gap-6">
    <div class="bg-white border border-gray-200 rounded-lg shadow-sm p-4 hover:shadow-md transition-custom">
      <h3 class="text-lg font-semibold text-neutral mb-3 flex items-center">
        <i class="fa fa-line-chart text-primary mr-2"></i>
        <span>{{ chartTitle }}</span>
        <!-- 返回按钮（仅在单样本时间序列视图显示） -->
        <button 
          v-if="showBackButton"
          @click="handleBackToScatter"
          class="ml-auto bg-gray-500 hover:bg-gray-600 text-white text-sm px-3 py-1 rounded-md transition-custom flex items-center"
        >
          <i class="fa fa-arrow-left mr-1"></i>返回散点图
        </button>
      </h3>
      
      <!-- 样本信息（仅多样本图表显示） -->
      <div v-if="showSampleInfo" class="text-sm text-gray-600 mb-3">
        <button 
          @click="toggleSampleIndices"
          class="text-primary hover:underline focus:outline-none"
        >
          <strong>样本信息：</strong>
          使用了 <span>{{ sampleCount }}</span> 个样本
          <i :class="sampleIndicesExpanded ? 'fa fa-chevron-up ml-1' : 'fa fa-chevron-down ml-1'"></i>
        </button>
        <div 
          :class="['collapsible-content mt-1 p-2 bg-gray-100 rounded-md', { expanded: sampleIndicesExpanded }]"
        >
          索引：<span>{{ sampleIndicesText }}</span>
        </div>
      </div>

      <!-- 图表容器 -->
      <div class="relative h-80">
        <div v-if="isLoading" class="absolute inset-0 flex items-center justify-center bg-gray-50 rounded">
          <div class="text-center">
            <i class="fa fa-spinner fa-spin text-2xl text-gray-400 mb-2"></i>
            <p class="text-gray-500">加载图表中...</p>
          </div>
        </div>
        <canvas v-show="!isLoading" ref="chartCanvas" class="w-full h-full"></canvas>
      </div>
    </div>
  </div>
</template>

<script>
import { Chart, registerables } from 'chart.js'

// 注册Chart.js组件
Chart.register(...registerables)

export default {
  name: 'DataChart',
  props: {
    chartData: {
      type: Object,
      default: null
    },
    chartType: {
      type: String,
      default: 'time_series'
    },
    isLoading: {
      type: Boolean,
      default: false
    },
    showBackButton: {
      type: Boolean,
      default: false
    }
  },
  data() {
    return {
      chart: null,
      sampleIndicesExpanded: false,
      isCreatingChart: false, // 防止重复创建图表
      createChartTimeout: null // 防抖定时器
    }
  },
  computed: {
    chartTitle() {
      if (!this.chartData) return '数据可视化'
      return this.chartData.title || '数据可视化'
    },
    showSampleInfo() {
      return this.chartData && (this.chartData.sample_count > 1 || this.chartType.startsWith('multi_'))
    },
    sampleCount() {
      return this.chartData?.sample_count || 1
    },
    sampleIndicesText() {
      if (!this.chartData?.sample_indices) return '-'
      return this.chartData.sample_indices.join(', ')
    }
  },
  watch: {
    chartData: {
      handler(newData) {
        if (newData) {
          this.$nextTick(() => {
            this.createChart()
          })
        }
      },
      immediate: true
    }
  },
  beforeUnmount() {
    // 清除防抖定时器
    if (this.createChartTimeout) {
      clearTimeout(this.createChartTimeout)
    }
    this.destroyChart()
  },
  methods: {
    toggleSampleIndices() {
      this.sampleIndicesExpanded = !this.sampleIndicesExpanded
    },
    handleBackToScatter() {
      this.$emit('back-to-scatter')
    },
    destroyChart() {
      if (this.chart) {
        try {
          this.chart.destroy()
        } catch (error) {
          console.warn('销毁图表时出错:', error)
        }
        this.chart = null
      }
      this.isCreatingChart = false
    },
    createChart() {
      // 清除之前的防抖定时器
      if (this.createChartTimeout) {
        clearTimeout(this.createChartTimeout)
      }

      // 使用防抖机制，避免快速切换时重复创建图表
      this.createChartTimeout = setTimeout(() => {
        this.doCreateChart()
      }, 100)
    },
    doCreateChart() {
      if (!this.chartData || !this.$refs.chartCanvas || this.isCreatingChart) {
        return
      }

      this.isCreatingChart = true

      try {
        this.destroyChart()

        const ctx = this.$refs.chartCanvas.getContext('2d')
        if (!ctx) {
          console.error('无法获取canvas上下文')
          this.isCreatingChart = false
          return
        }

        const config = this.getChartConfig()
        if (!config) {
          console.error('无法获取图表配置')
          this.isCreatingChart = false
          return
        }

        this.chart = new Chart(ctx, config)
      } catch (error) {
        console.error('创建图表时出错:', error)
        this.destroyChart()
      } finally {
        this.isCreatingChart = false
      }
    },
    getChartConfig() {
      const { chartData, chartType } = this

      switch (chartType) {
        case 'time_series':
          return this.getTimeSeriesConfig(chartData)
        case 'scatter':
        case 'multi_scatter':
          return this.getScatterConfig(chartData)
        case 'residual':
          return this.getResidualConfig(chartData)
        case 'histogram':
        case 'multi_histogram':
          return this.getHistogramConfig(chartData)
        case 'lookback_comparison':
          return this.getLookbackComparisonConfig(chartData)
        default:
          return this.getTimeSeriesConfig(chartData)
      }
    },
    getTimeSeriesConfig(data) {
      const primaryColor = 'rgb(59, 130, 246)'
      const primaryColorAlpha = 'rgba(59, 130, 246, 0.1)'
      
      return {
        type: 'line',
        data: {
          labels: data.x || [],
          datasets: [
            {
              label: '真实值',
              data: data.true || [],
              borderColor: primaryColor,
              backgroundColor: primaryColorAlpha,
              borderWidth: 2,
              fill: false,
              tension: 0.1
            },
            {
              label: '预测值',
              data: data.pred || [],
              borderColor: 'rgb(239, 68, 68)',
              backgroundColor: 'rgba(239, 68, 68, 0.1)',
              borderWidth: 2,
              fill: false,
              tension: 0.1,
              borderDash: [5, 5] // 虚线区分预测值
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            title: {
              display: true,
              text: data.title || '时间序列对比'
            },
            legend: {
              display: true
            }
          },
          scales: {
            x: {
              display: true,
              title: {
                display: true,
                text: '时间步'
              }
            },
            y: {
              display: true,
              title: {
                display: true,
                text: '数值'
              }
            }
          }
        }
      }
    },
    getScatterConfig(data) {
      const datasets = []
      const primaryColor = 'rgba(59, 130, 246, 0.6)'
      const primaryColorBorder = 'rgba(59, 130, 246, 1)'
      
      if (data.sample_indices_per_point) {
        // 多样本散点图，统一使用相同颜色
        const sampleGroups = {}
        data.true.forEach((trueVal, index) => {
          const sampleIdx = data.sample_indices_per_point[index]
          if (!sampleGroups[sampleIdx]) {
            sampleGroups[sampleIdx] = { x: [], y: [] }
          }
          sampleGroups[sampleIdx].x.push(trueVal)
          sampleGroups[sampleIdx].y.push(data.pred[index])
        })

        // 为每个样本创建数据集，但使用统一颜色
        Object.keys(sampleGroups).forEach((sampleIdx) => {
          datasets.push({
            label: `样本 ${sampleIdx}`,
            data: sampleGroups[sampleIdx].x.map((x, i) => ({
              x: x,
              y: sampleGroups[sampleIdx].y[i],
              sampleIndex: parseInt(sampleIdx) // 添加样本索引信息
            })),
            backgroundColor: primaryColor,
            borderColor: primaryColorBorder,
            borderWidth: 1,
            pointRadius: 2 // 减小点的大小以提高性能
          })
        })
      } else {
        // 单样本散点图
        datasets.push({
          label: '预测 vs 真实',
          data: data.true.map((x, i) => ({ x: x, y: data.pred[i] })),
          backgroundColor: primaryColor,
          borderColor: primaryColorBorder,
          borderWidth: 1,
          pointRadius: 3
        })
      }

      // 计算y=x辅助线的范围
      const allValues = [...(data.true || []), ...(data.pred || [])]
      const minVal = Math.min(...allValues)
      const maxVal = Math.max(...allValues)
      const padding = (maxVal - minVal) * 0.1
      const lineMin = minVal - padding
      const lineMax = maxVal + padding

      // 添加y=x辅助线
      datasets.push({
        label: 'y = x',
        data: [
          { x: lineMin, y: lineMin },
          { x: lineMax, y: lineMax }
        ],
        type: 'line',
        borderColor: 'rgba(156, 163, 175, 0.8)',
        borderWidth: 1,
        borderDash: [3, 3],
        fill: false,
        pointRadius: 0,
        pointHoverRadius: 0,
        showLine: true
      })

      return {
        type: 'scatter',
        data: { datasets },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            title: {
              display: true,
              text: data.title || '散点图'
            },
            legend: {
              display: false // 移除图例
            }
          },
          scales: {
            x: {
              display: true,
              title: {
                display: true,
                text: '真实值'
              }
            },
            y: {
              display: true,
              title: {
                display: true,
                text: '预测值'
              }
            }
          },
          elements: {
            point: {
              radius: 2 // 全局设置点的大小
            }
          },
          onClick: (event, elements) => {
            // 处理散点图点击事件
            if (elements.length > 0 && data.sample_indices_per_point) {
              const element = elements[0]
              const datasetIndex = element.datasetIndex
              const dataIndex = element.index
              
              // 获取点击的数据点
              const clickedData = datasets[datasetIndex].data[dataIndex]
              if (clickedData && clickedData.sampleIndex !== undefined) {
                // 发射事件，传递样本索引
                this.$emit('point-click', {
                  sampleIndex: clickedData.sampleIndex,
                  trueValue: clickedData.x,
                  predValue: clickedData.y
                })
              }
            }
          }
        }
      }
    },
    getResidualConfig(data) {
      const primaryColor = 'rgba(59, 130, 246, 0.6)'
      const primaryColorBorder = 'rgba(59, 130, 246, 1)'
      
      return {
        type: 'scatter',
        data: {
          datasets: [{
            label: '残差',
            data: data.true.map((x, i) => ({ x: x, y: data.residuals[i] })),
            backgroundColor: primaryColor,
            borderColor: primaryColorBorder,
            borderWidth: 1,
            pointRadius: 2
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            title: {
              display: true,
              text: data.title || '残差图'
            },
            legend: {
              display: false // 移除图例
            }
          },
          scales: {
            x: {
              display: true,
              title: {
                display: true,
                text: '真实值'
              }
            },
            y: {
              display: true,
              title: {
                display: true,
                text: '残差'
              }
            }
          },
          elements: {
            point: {
              radius: 2
            }
          }
        }
      }
    },
    getHistogramConfig(data) {
      const primaryColor = 'rgba(59, 130, 246, 0.6)'
      const primaryColorBorder = 'rgba(59, 130, 246, 1)'
      
      return {
        type: 'bar',
        data: {
          labels: data.bins || [],
          datasets: [{
            label: '频率密度',
            data: data.hist || [],
            backgroundColor: primaryColor,
            borderColor: primaryColorBorder,
            borderWidth: 1
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            title: {
              display: true,
              text: data.title || '误差直方图'
            },
            legend: {
              display: false // 移除图例
            }
          },
          scales: {
            x: {
              display: true,
              title: {
                display: true,
                text: '误差值'
              }
            },
            y: {
              display: true,
              title: {
                display: true,
                text: '频率密度'
              }
            }
          }
        }
      }
    },
    getLookbackComparisonConfig(data) {
      const primaryColor = 'rgb(59, 130, 246)'
      const primaryColorAlpha = 'rgba(59, 130, 246, 0.1)'
      const secondaryColor = 'rgb(239, 68, 68)'
      const secondaryColorAlpha = 'rgba(239, 68, 68, 0.1)'
      const tertiaryColor = 'rgb(34, 197, 94)'
      const tertiaryColorAlpha = 'rgba(34, 197, 94, 0.1)'
      
      const datasets = []
      
      // 训练集数据（如果有）
      if (data.training_true && data.training_pred) {
        datasets.push({
          label: '训练集-真实值',
          data: data.training_true,
          borderColor: tertiaryColor,
          backgroundColor: tertiaryColorAlpha,
          borderWidth: 2,
          fill: false,
          tension: 0.1,
          pointRadius: 1
        })
        datasets.push({
          label: '训练集-预测值',
          data: data.training_pred,
          borderColor: tertiaryColor,
          backgroundColor: tertiaryColorAlpha,
          borderWidth: 2,
          fill: false,
          tension: 0.1,
          borderDash: [3, 3],
          pointRadius: 1
        })
      }
      
      // 测试集真实值
      if (data.true) {
        datasets.push({
          label: '测试集-真实值',
          data: data.true,
          borderColor: primaryColor,
          backgroundColor: primaryColorAlpha,
          borderWidth: 2,
          fill: false,
          tension: 0.1,
          pointRadius: 2
        })
      }
      
      // 测试集预测值
      if (data.pred) {
        datasets.push({
          label: '测试集-预测值',
          data: data.pred,
          borderColor: secondaryColor,
          backgroundColor: secondaryColorAlpha,
          borderWidth: 2,
          fill: false,
          tension: 0.1,
          borderDash: [5, 5],
          pointRadius: 2
        })
      }
      
      return {
        type: 'line',
        data: {
          labels: data.x || [],
          datasets: datasets
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            title: {
              display: true,
              text: data.title || '回看数据对比'
            },
            legend: {
              display: true,
              position: 'top'
            }
          },
          scales: {
            x: {
              display: true,
              title: {
                display: true,
                text: '时间步'
              }
            },
            y: {
              display: true,
              title: {
                display: true,
                text: '数值'
              }
            }
          },
          interaction: {
            intersect: false,
            mode: 'index'
          }
        }
      }
    }
  }
}
</script>

