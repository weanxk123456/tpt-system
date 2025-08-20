<template>
  <section class="bg-white rounded-xl shadow-soft p-6 mb-8 transform hover:scale-[1.01] transition-custom">
    <h2 class="text-2xl font-bold text-neutral mb-4 flex items-center">
      <i class="fa fa-upload text-primary mr-2"></i>数据上传
    </h2>
    <div class="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-primary transition-custom">
      <i class="fa fa-cloud-upload text-5xl text-gray-400 mb-4"></i>
      <p class="text-gray-600 mb-4">拖放或点击上传真实值和预测值的npy文件</p>
      <p class="text-gray-500 text-sm mb-6">文件格式：三维数组 (patterns, pred_len, num_targets)</p>

      <form @submit.prevent="handleSubmit" class="space-y-4">
        <div class="grid md:grid-cols-2 gap-4">
          <!-- 真实值文件上传 -->
          <div class="relative">
            <label for="actual-file" class="block text-sm font-medium text-gray-700 mb-1">真实值文件</label>
            <div class="mt-1 flex justify-center px-6 pt-5 pb-6 border-2 border-gray-300 border-dashed rounded-md hover:border-primary transition-custom">
              <div class="space-y-1 text-center">
                <i class="fa fa-file-text-o text-2xl text-gray-400 mb-2"></i>
                <div class="flex text-sm text-gray-600">
                  <label for="actual-file" class="relative cursor-pointer bg-white rounded-md font-medium text-primary hover:text-primary-dark">
                    <span>选择文件</span>
                    <input 
                      id="actual-file" 
                      ref="actualFileInput"
                      type="file" 
                      accept=".npy" 
                      class="sr-only"
                      @change="handleActualFileChange"
                    >
                  </label>
                  <p class="pl-1">{{ actualFileName || '未选择文件' }}</p>
                </div>
                <p class="text-xs text-gray-500">支持 .npy 格式</p>
              </div>
            </div>
          </div>

          <!-- 预测值文件上传 -->
          <div class="relative">
            <label for="predicted-file" class="block text-sm font-medium text-gray-700 mb-1">预测值文件</label>
            <div class="mt-1 flex justify-center px-6 pt-5 pb-6 border-2 border-gray-300 border-dashed rounded-md hover:border-primary transition-custom">
              <div class="space-y-1 text-center">
                <i class="fa fa-file-text-o text-2xl text-gray-400 mb-2"></i>
                <div class="flex text-sm text-gray-600">
                  <label for="predicted-file" class="relative cursor-pointer bg-white rounded-md font-medium text-primary hover:text-primary-dark">
                    <span>选择文件</span>
                    <input 
                      id="predicted-file" 
                      ref="predictedFileInput"
                      type="file" 
                      accept=".npy" 
                      class="sr-only"
                      @change="handlePredictedFileChange"
                    >
                  </label>
                  <p class="pl-1">{{ predictedFileName || '未选择文件' }}</p>
                </div>
                <p class="text-xs text-gray-500">支持 .npy 格式</p>
              </div>
            </div>
          </div>
        </div>

        <button 
          type="submit" 
          :disabled="isCalculating"
          class="w-full bg-primary hover:bg-primary/90 text-white font-medium py-3 px-4 rounded-lg shadow-md transition-custom flex items-center justify-center disabled:opacity-50"
        >
          <i :class="isCalculating ? 'fa fa-spinner fa-spin mr-2' : 'fa fa-calculator mr-2'"></i>
          {{ isCalculating ? '计算中...' : '计算评估指标' }}
        </button>
      </form>
    </div>
  </section>
</template>

<script>
import apiService from '../services/apiService.js'

export default {
  name: 'UploadSection',
  data() {
    return {
      actualFileName: '',
      predictedFileName: '',
      isCalculating: false
    }
  },
  methods: {
    handleActualFileChange(event) {
      const file = event.target.files[0]
      this.actualFileName = file ? file.name : ''
    },
    handlePredictedFileChange(event) {
      const file = event.target.files[0]
      this.predictedFileName = file ? file.name : ''
    },
    async handleSubmit() {
      const actualFile = this.$refs.actualFileInput.files[0]
      const predictedFile = this.$refs.predictedFileInput.files[0]

      if (!actualFile || !predictedFile) {
        this.$emit('error', '请选择真实值和预测值文件')
        return
      }

      this.isCalculating = true

      try {
        const formData = new FormData()
        formData.append('actual_file', actualFile)
        formData.append('predicted_file', predictedFile)

        const response = await apiService.calculate(formData)
        
        if (response.success) {
          this.$emit('upload-success', response)
        } else {
          this.$emit('error', response.error || '计算失败')
        }
      } catch (error) {
        this.$emit('error', '发生网络错误: ' + error.message)
      } finally {
        this.isCalculating = false
      }
    }
  }
}
</script>

