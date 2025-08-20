<template>
  <div class="grid grid-cols-1 gap-6">
    <div class="bg-white border border-gray-200 rounded-lg shadow-sm p-4 hover:shadow-md transition-custom">
      <h3 class="text-lg font-semibold text-neutral mb-3 flex items-center">
        <i class="fa fa-image text-primary mr-2"></i>
        <span>{{ staticData.title }}</span>
      </h3>
      
      <!-- 静态图表信息 -->
      <div class="text-sm text-gray-600 mb-3">
        <div class="flex flex-wrap gap-4">
          <span><strong>样本数量：</strong>{{ staticData.sample_count }}</span>
          <span><strong>数据点：</strong>{{ staticData.data_points }}</span>
          <span><strong>渲染模式：</strong>静态图片</span>
        </div>
        <div class="mt-1">
          <button 
            @click="toggleSampleIndices"
            class="text-primary hover:underline focus:outline-none"
          >
            <strong>样本索引：</strong>
            <span>{{ sampleIndicesText }}</span>
            <i :class="sampleIndicesExpanded ? 'fa fa-chevron-up ml-1' : 'fa fa-chevron-down ml-1'"></i>
          </button>
          <div 
            :class="['collapsible-content mt-1 p-2 bg-gray-100 rounded-md', { expanded: sampleIndicesExpanded }]"
          >
            {{ staticData.sample_indices.join(', ') }}
          </div>
        </div>
      </div>

      <!-- 静态图片容器 -->
      <div class="relative">
        <div v-if="isLoading" class="absolute inset-0 flex items-center justify-center bg-gray-50 rounded">
          <div class="text-center">
            <i class="fa fa-spinner fa-spin text-2xl text-gray-400 mb-2"></i>
            <p class="text-gray-500">加载图表中...</p>
          </div>
        </div>
        <div 
          v-show="!isLoading" 
          class="relative cursor-pointer"
          @click="handleImageClick"
          @mousemove="handleMouseMove"
          ref="imageContainer"
        >
          <img 
            :src="`data:image/png;base64,${staticData.image_base64}`"
            :alt="staticData.title"
            class="w-full h-auto rounded-lg shadow-sm"
            @load="handleImageLoad"
            ref="staticImage"
          >
          <!-- 点击提示 -->
          <div 
            v-if="hoveredRegion"
            class="absolute bg-black bg-opacity-75 text-white text-xs px-2 py-1 rounded pointer-events-none z-10"
            :style="tooltipStyle"
          >
            样本: {{ hoveredRegion.samples.join(', ') }}
          </div>
        </div>
      </div>

      <!-- 点击说明 -->
      <div class="mt-3 text-sm text-gray-500 flex items-center">
        <i class="fa fa-info-circle mr-2"></i>
        <span>点击图表区域可查看该区域包含的样本信息</span>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'StaticChart',
  props: {
    staticData: {
      type: Object,
      required: true
    },
    isLoading: {
      type: Boolean,
      default: false
    }
  },
  data() {
    return {
      sampleIndicesExpanded: false,
      hoveredRegion: null,
      tooltipStyle: {},
      imageLoaded: false
    }
  },
  computed: {
    sampleIndicesText() {
      if (!this.staticData.sample_indices || this.staticData.sample_indices.length === 0) {
        return '无'
      }
      const indices = this.staticData.sample_indices
      if (indices.length <= 10) {
        return indices.join(', ')
      }
      return `${indices.slice(0, 10).join(', ')}... (共${indices.length}个)`
    }
  },
  methods: {
    toggleSampleIndices() {
      this.sampleIndicesExpanded = !this.sampleIndicesExpanded
    },
    handleImageLoad() {
      this.imageLoaded = true
    },
    handleMouseMove(event) {
      if (!this.imageLoaded || !this.staticData.clickable_regions) return

      const rect = this.$refs.staticImage.getBoundingClientRect()
      const x = event.clientX - rect.left
      const y = event.clientY - rect.top
      
      // 将像素坐标转换为数据坐标
      const imageWidth = this.$refs.staticImage.width
      const imageHeight = this.$refs.staticImage.height
      
      if (imageWidth === 0 || imageHeight === 0) return

      // 假设图表区域占图片的80%，并且有边距
      const chartLeft = imageWidth * 0.1
      const chartRight = imageWidth * 0.9
      const chartTop = imageHeight * 0.1
      const chartBottom = imageHeight * 0.9

      if (x < chartLeft || x > chartRight || y < chartTop || y > chartBottom) {
        this.hoveredRegion = null
        return
      }

      // 查找对应的区域
      const region = this.findRegionAtPosition(x, y, imageWidth, imageHeight)
      if (region) {
        this.hoveredRegion = region
        this.tooltipStyle = {
          left: `${event.clientX - rect.left + 10}px`,
          top: `${event.clientY - rect.top - 30}px`
        }
      } else {
        this.hoveredRegion = null
      }
    },
    handleImageClick(event) {
      if (!this.imageLoaded || !this.staticData.clickable_regions) return

      const rect = this.$refs.staticImage.getBoundingClientRect()
      const x = event.clientX - rect.left
      const y = event.clientY - rect.top
      
      const region = this.findRegionAtPosition(x, y, this.$refs.staticImage.width, this.$refs.staticImage.height)
      if (region && region.samples.length > 0) {
        // 发出点击事件，传递样本信息
        this.$emit('region-click', {
          samples: region.samples,
          region: region
        })
        
        // 显示样本信息
        const sampleText = region.samples.length === 1 
          ? `样本 ${region.samples[0]}` 
          : `${region.samples.length} 个样本: ${region.samples.join(', ')}`
        
        alert(`点击区域包含 ${sampleText}`)
      }
    },
    findRegionAtPosition(x, y, imageWidth, imageHeight) {
      if (!this.staticData.clickable_regions) return null

      // 简化的区域匹配逻辑
      // 这里需要根据后端返回的区域坐标进行匹配
      for (const region of this.staticData.clickable_regions) {
        // 将数据坐标转换为像素坐标（这里需要根据实际的坐标系统调整）
        const pixelRegion = this.dataToPixelCoords(region, imageWidth, imageHeight)
        
        if (x >= pixelRegion.left && x <= pixelRegion.right && 
            y >= pixelRegion.top && y <= pixelRegion.bottom) {
          return region
        }
      }
      return null
    },
    dataToPixelCoords(region, imageWidth, imageHeight) {
      // 这是一个简化的坐标转换，实际应用中需要根据具体的图表坐标系统调整
      const chartLeft = imageWidth * 0.1
      const chartRight = imageWidth * 0.9
      const chartTop = imageHeight * 0.1
      const chartBottom = imageHeight * 0.9
      
      const chartWidth = chartRight - chartLeft
      const chartHeight = chartBottom - chartTop
      
      // 假设数据坐标已经标准化到0-1范围
      return {
        left: chartLeft + region.x_start * chartWidth,
        right: chartLeft + region.x_end * chartWidth,
        top: chartTop + region.y_start * chartHeight,
        bottom: chartTop + region.y_end * chartHeight
      }
    }
  }
}
</script>

<style scoped>
.collapsible-content {
  max-height: 0;
  overflow: hidden;
  transition: max-height 0.3s ease;
}

.collapsible-content.expanded {
  max-height: 200px;
}

.transition-custom {
  transition: all 0.2s ease-in-out;
}
</style>

