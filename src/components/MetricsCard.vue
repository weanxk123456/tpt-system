<template>
  <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
    <!-- 误差指标 -->
    <div class="bg-white border border-gray-200 rounded-lg shadow-sm p-4 hover:shadow-md transition-custom">
      <h3 class="text-lg font-semibold text-neutral mb-2 flex items-center">
        <i class="fa fa-area-chart text-primary mr-2"></i>误差指标
      </h3>
      <div class="space-y-3">
        <div class="flex justify-between">
          <span class="text-gray-600">MSE</span>
          <span class="font-medium">
            {{ formatValue(metrics?.mse) }}
            <span v-if="avgMetrics?.mse" class="text-gray-500"> / {{ formatValue(avgMetrics.mse) }}</span>
          </span>
        </div>
        <p class="metric-desc">均方误差 [越小越好]</p>

        <div class="flex justify-between">
          <span class="text-gray-600">RMSE</span>
          <span class="font-medium">
            {{ formatValue(metrics?.rmse) }}
            <span v-if="avgMetrics?.rmse" class="text-gray-500"> / {{ formatValue(avgMetrics.rmse) }}</span>
          </span>
        </div>
        <p class="metric-desc">均方根误差 [越小越好]</p>

        <div class="flex justify-between">
          <span class="text-gray-600">MAE</span>
          <span class="font-medium">
            {{ formatValue(metrics?.mae) }}
            <span v-if="avgMetrics?.mae" class="text-gray-500"> / {{ formatValue(avgMetrics.mae) }}</span>
          </span>
        </div>
        <p class="metric-desc">平均绝对误差 [越小越好]</p>

        <div class="flex justify-between">
          <span class="text-gray-600">MAPE (%)</span>
          <span class="font-medium">
            {{ formatValue(metrics?.mape, '%') }}
            <span v-if="avgMetrics?.mape" class="text-gray-500"> / {{ formatValue(avgMetrics.mape, '%') }}</span>
          </span>
        </div>
        <p class="metric-desc">平均绝对百分比误差 [越小越好]</p>

        <div class="flex justify-between">
          <span class="text-gray-600">SMAPE (%)</span>
          <span class="font-medium">
            {{ formatValue(metrics?.smape, '%') }}
            <span v-if="avgMetrics?.smape" class="text-gray-500"> / {{ formatValue(avgMetrics.smape, '%') }}</span>
          </span>
        </div>
        <p class="metric-desc">对称平均绝对百分比误差 [越小越好]</p>

        <div class="flex justify-between">
          <span class="text-gray-600">R²</span>
          <span class="font-medium">
            {{ formatValue(metrics?.r2) }}
            <span v-if="avgMetrics?.r2" class="text-gray-500"> / {{ formatValue(avgMetrics.r2) }}</span>
          </span>
        </div>
        <p class="metric-desc">决定系数 [越接近1越好]</p>
      </div>
    </div>

    <!-- 趋势指标 -->
    <div class="bg-white border border-gray-200 rounded-lg shadow-sm p-4 hover:shadow-md transition-custom">
      <h3 class="text-lg font-semibold text-neutral mb-2 flex items-center">
        <i class="fa fa-line-chart text-secondary mr-2"></i>趋势指标
      </h3>
      <div class="space-y-3">
        <div class="flex justify-between">
          <span class="text-gray-600">TDA</span>
          <span class="font-medium">
            {{ formatValue(metrics?.tda) }}
            <span v-if="avgMetrics?.tda" class="text-gray-500"> / {{ formatValue(avgMetrics.tda) }}</span>
          </span>
        </div>
        <p class="metric-desc">趋势方向准确率 [越高越好]</p>

        <div class="flex justify-between">
          <span class="text-gray-600">TID</span>
          <span class="font-medium">
            {{ formatValue(metrics?.tid) }}
            <span v-if="avgMetrics?.tid" class="text-gray-500"> / {{ formatValue(avgMetrics.tid) }}</span>
          </span>
        </div>
        <p class="metric-desc">趋势强度偏差率 [越小越好]</p>

        <div class="flex justify-between">
          <span class="text-gray-600">TIPD</span>
          <span class="font-medium">
            {{ formatValue(metrics?.tipd) }}
            <span v-if="avgMetrics?.tipd" class="text-gray-500"> / {{ formatValue(avgMetrics.tipd) }}</span>
          </span>
        </div>
        <p class="metric-desc">趋势转折点检测率 [越高越好]</p>
      </div>
    </div>

    <!-- 综合指标 -->
    <div class="bg-white border border-gray-200 rounded-lg shadow-sm p-4 hover:shadow-md transition-custom">
      <h3 class="text-lg font-semibold text-neutral mb-2 flex items-center">
        <i class="fa fa-balance-scale text-accent mr-2"></i>综合指标
      </h3>
      <div class="space-y-3">
        <div class="flex justify-between">
          <span class="text-gray-600">趋势匹配度</span>
          <span class="font-medium">
            {{ formatValue(metrics?.trend) }}
            <span v-if="avgMetrics?.trend" class="text-gray-500"> / {{ formatValue(avgMetrics.trend) }}</span>
          </span>
        </div>
        <p class="metric-desc">趋势形状匹配程度 [越高越好]</p>

        <div class="flex justify-between">
          <span class="text-gray-600">幅值匹配度</span>
          <span class="font-medium">
            {{ formatValue(metrics?.amplitude) }}
            <span v-if="avgMetrics?.amplitude" class="text-gray-500"> / {{ formatValue(avgMetrics.amplitude) }}</span>
          </span>
        </div>
        <p class="metric-desc">幅值大小匹配程度 [越高越好]</p>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'MetricsCard',
  props: {
    metrics: {
      type: Object,
      default: null
    },
    avgMetrics: {
      type: Object,
      default: () => ({})
    }
  },
  methods: {
    formatValue(value, suffix = '') {
      if (value === null || value === undefined || isNaN(value)) {
        return '-'
      }
      const formatted = typeof value === 'number' ? value.toFixed(4) : value
      return formatted + suffix
    }
  }
}
</script>