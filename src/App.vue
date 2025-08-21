<template>
  <div class="bg-gray-50 font-sans min-h-screen">
    <!-- 导航栏 -->
    <nav class="bg-white shadow-md fixed w-full z-50 transition-all duration-300">
      <div class="container mx-auto px-4 py-3 flex justify-between items-center">
        <div class="flex items-center space-x-2">
          <i class="fa fa-line-chart text-primary text-2xl"></i>
          <h1 class="text-xl font-bold text-neutral">TPT-时序数据预测评估系统（Vue版）</h1>
        </div>
        <button class="md:hidden text-neutral text-xl">
          <i class="fa fa-bars"></i>
        </button>
      </div>
    </nav>

    <!-- 主要内容 -->
    <main class="container mx-auto px-4 pt-24 pb-12">
      <!-- 上传区域 -->
      <UploadSection @upload-success="handleUploadSuccess" @error="handleError" />

      <!-- 结果区域 -->
      <ResultsSection
        v-if="showResults"
        :results-data="resultsData"
        @new-calculation="handleNewCalculation"
        @error="handleError"
      />

      <!-- 错误消息 -->
      <div v-if="errorMessage" class="bg-red-50 border border-red-200 rounded-lg p-4 mb-6">
        <div class="flex items-start">
          <div class="flex-shrink-0 pt-0.5">
            <i class="fa fa-exclamation-circle text-red-500"></i>
          </div>
          <div class="ml-3">
            <h3 class="text-sm font-medium text-red-800">错误</h3>
            <div class="mt-1 text-sm text-red-700">
              <p>{{ errorMessage }}</p>
            </div>
          </div>
        </div>
      </div>
    </main>

    <!-- 页脚 -->
    <footer class="bg-white border-t border-gray-200">
      <div class="container mx-auto px-4 py-6">
        <div class="flex flex-col md:flex-row justify-between items-center">
          <div class="mb-4 md:mb-0">
            <p class="text-gray-600 text-sm">© 2025 TPT-时序数据预测评估系统（Vue版）. 保留所有权利.</p>
          </div>
        </div>
      </div>
    </footer>
  </div>
</template>

<script>
import UploadSection from './components/UploadSection.vue'
import ResultsSection from './components/ResultsSection.vue'

export default {
  name: 'App',
  components: {
    UploadSection,
    ResultsSection
  },
  data() {
    return {
      showResults: false,
      resultsData: null,
      errorMessage: '',
    }
  },
  methods: {
    handleUploadSuccess(data) {
      this.resultsData = data
      this.showResults = true
      this.errorMessage = ''
      // 滚动到结果区域
      this.$nextTick(() => {
        const resultsElement = document.querySelector('#results-section')
        if (resultsElement) {
          resultsElement.scrollIntoView({ behavior: 'smooth' })
        }
      })
    },
    handleNewCalculation() {
      this.showResults = false
      this.resultsData = null
      this.errorMessage = ''
    },
    handleError(message) {
      this.errorMessage = message
    }
  }
}
</script>

