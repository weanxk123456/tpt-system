# -*- coding: utf-8 -*-
# @Time: 2025/7/9  15:30
# @Author: luwangxing
# @FileName: main.py
# @Software: PyCharm
"""
    Description: 增强版时序数据预测评估系统，支持选择样本数量绘制散点图和误差直方图

"""
import os
import json
import uuid
import numpy as np
from scipy.integrate import simpson
from typing import Dict, Any, List, Tuple
from flask import Flask, request, jsonify, send_file, render_template, session
import tempfile
import zipfile
import matplotlib.pyplot as plt
import io
from werkzeug.utils import secure_filename
from flask_session import Session
import shutil
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import base64
from datetime import datetime
from collections import defaultdict  # NEW FEATURE: Import defaultdict for average calculation
import pickle

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题


class EvaluationMetricsCalculator:
    def __init__(self, pred_data: np.ndarray, true_data: np.ndarray, session_id: str,
                 lookback_data: np.ndarray = None):  # MODIFIED: Add optional lookback_data
        """
        初始化评估指标计算器

        参数:
            pred_data: 预测值数据 (patterns, pred_len, num_targets)
            true_data: 真实值数据 (patterns, pred_len, num_targets)
            session_id: 用户会话ID，用于存储临时文件
            lookback_data: (可选) 回看窗口数据 (patterns, lookback_len, num_targets)
        """
        # 将数据转换为float64以避免JSON序列化问题
        self.pred = pred_data.astype(np.float64)
        self.true = true_data.astype(np.float64)
        self.num_patterns, self.pred_len, self.num_targets = self.pred.shape
        self.session_id = session_id

        # NEW FEATURE: Handle lookback data
        self.lookback = lookback_data.astype(np.float64) if lookback_data is not None else None
        self.lookback_len = self.lookback.shape[1] if self.lookback is not None else 0
        self.has_lookback = self.lookback is not None

        # 创建会话专用临时目录
        self.session_dir = os.path.join(tempfile.gettempdir(), f"ts_evaluation_{session_id}")
        os.makedirs(self.session_dir, exist_ok=True)

        # 验证两个数组形状是否相同
        assert self.true.shape == self.pred.shape, "预测值和真实值形状不匹配"
        # NEW FEATURE: Validate lookback data shape
        if self.has_lookback:
            assert self.lookback.shape[0] == self.num_patterns, "回看数据和主数据样本数不匹配"
            assert self.lookback.shape[2] == self.num_targets, "回看数据和主数据目标数不匹配"

    # ... (从 trend_direction_accuracy 到 cal_amplitude 的所有方法保持不变) ...
    def trend_direction_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """趋势方向准确率 TDA"""
        if len(y_true) < 2:
            return float('nan')
        true_dir = np.sign(np.diff(y_true))
        pred_dir = np.sign(np.diff(y_pred))
        valid_indices = ~(np.isnan(true_dir) | np.isnan(pred_dir))
        if not np.any(valid_indices):
            return float('nan')
        correct = np.sum(true_dir[valid_indices] == pred_dir[valid_indices])
        return correct / np.sum(valid_indices)

    def trend_intensity_deviation(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """趋势强度偏差率 TID"""
        if len(y_true) < 2:
            return float('nan')
        delta_true = np.diff(y_true)
        delta_pred = np.diff(y_pred)
        mask_nonzero = (delta_true != 0) & ~np.isnan(delta_true) & ~np.isnan(delta_pred)
        if np.any(mask_nonzero):
            deviations = np.abs(delta_pred[mask_nonzero] / delta_true[mask_nonzero] - 1)
            return np.mean(deviations)
        else:
            return float('nan')

    def detect_inflection_points(self, y: np.ndarray) -> np.ndarray:
        """检测趋势转折点"""
        if len(y) < 3:
            return np.array([])
        first_diff = np.diff(y)
        second_diff = np.diff(first_diff)
        valid_indices = ~(np.isnan(second_diff[:-1]) | np.isnan(second_diff[1:]))
        inflection_indices = np.where((second_diff[:-1] * second_diff[1:] < 0) & valid_indices)[0] + 1
        return inflection_indices

    def trend_inflection_point_detection_rate(self, y_true: np.ndarray, y_pred: np.ndarray,
                                              tolerance: int = 2) -> float:
        """趋势转折点检测率 TIPD"""
        true_ips = self.detect_inflection_points(y_true)
        pred_ips = self.detect_inflection_points(y_pred)

        matched = 0
        for t in true_ips:
            if any(abs(t - p) <= tolerance for p in pred_ips):
                matched += 1

        total_true = len(true_ips)
        return matched / total_true if total_true > 0 else float('nan')

    def get_score(self, angle_deg, k):
        """根据角度求解分数"""
        numerator = 1 - np.exp(-k * angle_deg)
        denominator = 1 - np.exp(-k * 180)
        score = 1 - numerator / denominator
        return score

    def get_angle(self, arr1_diff, arr2_diff):
        """计算两个序列的角度"""
        dot = arr1_diff * arr2_diff + 1
        mag1 = np.sqrt(arr1_diff ** 2 + 1)
        mag2 = np.sqrt(arr2_diff ** 2 + 1)

        cos_theta = dot / (mag1 * mag2 + 1e-8)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)

        # 求解角度
        angle_rad = np.arccos(cos_theta)
        angle_deg = np.degrees(angle_rad)
        return angle_deg

    def cal_trend(self, arr1, arr2, k):
        """计算趋势匹配度"""
        arr1_diff = np.diff(arr1, axis=1)
        arr2_diff = np.diff(arr2, axis=1)

        angle_deg = self.get_angle(arr1_diff, arr2_diff)

        # 根据角度求解分数
        score = self.get_score(angle_deg, k)

        # 根据方向求解掩码，方向相同分数不变，方向不同分数打折
        mask = np.sign(arr1_diff * arr2_diff)
        score[mask == -1] *= 0.7

        return np.mean(score, axis=1)

    def total_segment_max_period(self, data):
        """获取片段数据的最大纵坐标范围"""
        delta = np.max(data, axis=1) - np.min(data, axis=1)
        return np.percentile(delta, 99.9, axis=0)

    def normalization(self, data, periods):
        """数据归一化处理"""
        min_datas = np.min(data, axis=1, keepdims=True)
        max_datas = np.max(data, axis=1, keepdims=True)
        deltas = max_datas.squeeze(1) - min_datas.squeeze(1)

        masks = deltas < periods
        masks_expanded = masks[:, None, :]

        normal_period_data = (data - min_datas) / (periods + 1e-8)
        normal_minmax_data = (data - min_datas) / (max_datas - min_datas + 1e-8)

        normal_data = np.where(masks_expanded, normal_period_data, normal_minmax_data)
        return normal_data

    def cal_amplitude(self, true_data, pred_data, periods_true, periods_pred):
        """计算幅值匹配度"""
        normal_true = self.normalization(true_data, periods_true)
        normal_pred = self.normalization(pred_data, periods_pred)

        diff = np.abs(normal_true - normal_pred)
        area = simpson(diff, axis=1)
        amplitude = 1 - area / (diff.shape[1] - 1)
        return amplitude, normal_true, normal_pred

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, trend, amplitude) -> Dict[str, float]:
        """计算单个样本和目标的评估指标"""
        # 过滤掉NaN值（如果有）
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        # 误差指标
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_true - y_pred))

        mask_nonzero = y_true != 0
        if np.any(mask_nonzero):
            mape = np.mean(np.abs((y_true[mask_nonzero] - y_pred[mask_nonzero]) / y_true[mask_nonzero])) * 100
        else:
            mape = float('nan')

        denominator = (np.abs(y_true) + np.abs(y_pred))
        mask_nonzero = denominator != 0
        if np.any(mask_nonzero):
            smape = 100 * np.mean(2 * np.abs(y_true[mask_nonzero] - y_pred[mask_nonzero]) / denominator[mask_nonzero])
        else:
            smape = float('nan')

        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        if ss_tot > 0:
            r2 = 1 - (ss_res / ss_tot)
        else:
            r2 = float('nan')

        # 趋势指标
        tda = self.trend_direction_accuracy(y_true, y_pred)
        tid = self.trend_intensity_deviation(y_true, y_pred)
        tipd = self.trend_inflection_point_detection_rate(y_true, y_pred)

        return {
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "mape": float(mape),
            "smape": float(smape),
            "r2": float(r2),
            "tda": float(tda),
            "tid": float(tid),
            "tipd": float(tipd),
            "trend": float(trend),
            "amplitude": float(amplitude)
        }

    def compute_all_metrics(self) -> Dict[str, Any]:
        """计算所有样本和目标的评估指标，使用并行处理加速"""
        from functools import partial

        results = {}

        # 趋势指标和幅值指标
        periods_true = self.total_segment_max_period(self.true)
        periods_pred = self.total_segment_max_period(self.pred)
        trend = self.cal_trend(self.true, self.pred, k=0.03)
        amplitude, normal_true, normal_pred = self.cal_amplitude(self.true, self.pred, periods_true, periods_pred)

        # 向量化计算所有样本和目标的指标，大幅提升性能
        # 重塑数据为 (num_samples*num_targets, pred_len)
        true_reshaped = self.true.reshape(-1, self.pred_len)
        pred_reshaped = self.pred.reshape(-1, self.pred_len)
        trends_reshaped = trend.reshape(-1)
        amplitudes_reshaped = amplitude.reshape(-1)

        # 批量计算所有样本和目标的指标
        all_metrics = np.array(
            [self.calculate_metrics(true_reshaped[i], pred_reshaped[i], trends_reshaped[i], amplitudes_reshaped[i]) for
             i in range(true_reshaped.shape[0])])

        # 重塑结果并转换为字典格式
        for i in range(self.num_patterns):
            sample_results = {}
            for j in range(self.num_targets):
                idx = i * self.num_targets + j
                sample_results[str(j)] = all_metrics[idx]
            results[str(i)] = sample_results

        return results

    # MODIFIED: This method is now a specific case of compute_average_metrics, but we keep it for potential direct use.
    def compute_average_metrics(self) -> Dict[str, float]:
        """计算所有样本和目标的平均评估指标"""
        all_metrics = self.compute_all_metrics()
        metrics_list = []

        for pattern in all_metrics.values():
            for target in pattern.values():
                metrics_list.append(target)

        if not metrics_list:
            return {}

        # 计算平均值
        avg_metrics = {}
        for metric in metrics_list[0].keys():
            valid_values = [m[metric] for m in metrics_list if m.get(metric) is not None and not np.isnan(m[metric])]
            if valid_values:
                avg_metrics[metric] = float(np.mean(valid_values))
            else:
                avg_metrics[metric] = None  # 将NaN转换为None，以便JSON序列化

        return avg_metrics

    # ... (get_chart_data, total_directional_accuracy, total_inverted_direction, evaluate_all_samples 保持不变) ...
    def get_chart_data(self, pattern_idx, target_idx):
        """获取图表所需的原始数据"""
        y_true = self.true[pattern_idx, :, target_idx]
        y_pred = self.pred[pattern_idx, :, target_idx]
        error = y_true - y_pred

        # 计算误差分布
        hist, bin_edges = np.histogram(error, bins=30, density=True)
        bins = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(bin_edges) - 1)]

        # 确保所有数据都是Python原生类型
        return {
            'y_true': convert_numpy_array(y_true),
            'y_pred': convert_numpy_array(y_pred),
            'error': convert_numpy_array(error),
            'error_distribution': convert_numpy_array(hist),
            'bins': convert_numpy_array(bins)
        }

    def generate_lookback_plot_data(self, training_data: np.ndarray, pattern_idx: int, target_idx: int, lookback_steps: int):
        """生成回看数据对比图表所需的数据"""
        # 获取测试集的真实值和预测值
        test_true = self.true[pattern_idx, :, target_idx]
        test_pred = self.pred[pattern_idx, :, target_idx]

        # 获取训练集的真实值
        # 训练数据是前n步的数据，其长度为lookback_steps
        training_true = training_data[pattern_idx, -lookback_steps:, target_idx]  # 取最后lookback_steps步
        
        # 对于训练数据，我们没有预测值，所以使用真实值作为预测值
        # 或者可以设置为NaN，表示没有预测值
        training_pred = np.full_like(training_true, np.nan)  # 训练期没有预测值

        # 构建x轴标签
        # 训练数据从-lookback_steps到-1
        # 测试数据从0到pred_len-1
        x_labels = list(range(-lookback_steps, self.pred_len))

        # 填充训练数据和测试数据，使其在x轴上对齐
        full_training_true = np.concatenate([training_true, np.full(self.pred_len, np.nan)])
        full_training_pred = np.concatenate([training_pred, np.full(self.pred_len, np.nan)])

        full_test_true = np.concatenate([np.full(lookback_steps, np.nan), test_true])
        full_test_pred = np.concatenate([np.full(lookback_steps, np.nan), test_pred])

        return {
            'x': convert_numpy_array(x_labels),
            'training_true': convert_numpy_array(full_training_true),
            'training_pred': convert_numpy_array(full_training_pred),
            'true': convert_numpy_array(full_test_true),
            'pred': convert_numpy_array(full_test_pred),
            'title': f'样本 {pattern_idx} 目标 {target_idx} 的回看数据对比图'
        }

    def total_directional_accuracy(self, y_true, y_pred):
        """总体方向准确率（从test.py整合）"""
        delta_true = np.diff(y_true, axis=1)
        delta_pred = np.diff(y_pred, axis=1)
        direction_true = np.sign(delta_true)
        direction_pred = np.sign(delta_pred)
        matches = direction_true == direction_pred
        return 100 * np.mean(matches)

    def total_inverted_direction(self, y_true, y_pred):
        """总体反向方向率（从test.py整合）"""
        delta_true = np.diff(y_true, axis=1)
        delta_pred = np.diff(y_pred, axis=1)
        direction_true = np.sign(delta_true)
        direction_pred = np.sign(delta_pred)
        inversions = direction_true != direction_pred
        return 100 * np.mean(inversions)

    def evaluate_all_samples(self, mae_thresh=0.5, tda_thresh=50):
        """评估所有样本，检测异常样本（从test.py整合）"""
        results = []
        for i in range(self.num_patterns):
            for j in range(self.num_targets):
                pred_i = self.pred[i, :, j]
                true_i = self.true[i, :, j]
                mae = mean_absolute_error(true_i, pred_i)
                mse = mean_squared_error(true_i, pred_i)
                r2 = r2_score(true_i, pred_i)
                tda = self.total_directional_accuracy(true_i[np.newaxis, :], pred_i[np.newaxis, :])
                tid = self.total_inverted_direction(true_i[np.newaxis, :], pred_i[np.newaxis, :])

                results.append({
                    "样本号": i,
                    "目标号": j,
                    "MAE": mae,
                    "MSE": mse,
                    "R2": r2,
                    "TDA": tda,
                    "TID": tid
                })

        df = pd.DataFrame(results)
        abnormal_df = df[(df["MAE"] > mae_thresh) | (df["TDA"] < tda_thresh)]

        return abnormal_df

    # MODIFIED: generate_plot_data now supports lookback window
    def generate_plot_data(self, pattern_idx, target_idx, plot_type, start_idx=0, end_idx=None):
        """生成不同类型的图表数据（从test.py整合）"""
        if end_idx is None:
            end_idx = self.pred_len

        # NEW FEATURE: Handle lookback window for time series plot
        if plot_type == "time_series" and self.has_lookback:
            lookback_series = self.lookback[pattern_idx, :, target_idx]
            true_series = self.true[pattern_idx, :, target_idx]
            pred_series = self.pred[pattern_idx, :, target_idx]

            # Concatenate lookback data with the actual series
            full_true_series = np.concatenate([lookback_series, true_series])

            # Pad prediction series with NaNs to align with the true series on the plot
            full_pred_series = np.concatenate([
                np.full(self.lookback_len, np.nan),
                pred_series
            ])

            # Create x-axis labels from negative (lookback) to positive (prediction)
            x_labels = list(range(-self.lookback_len, self.pred_len))

            return {
                'x': x_labels,
                'pred': convert_numpy_array(full_pred_series),
                'true': convert_numpy_array(full_true_series),
                'title': f'样本 {pattern_idx} 目标 {target_idx} 的时间序列对比 (含回看窗口)'
            }

        # Original logic for other plot types or when no lookback data is available
        pred = self.pred[pattern_idx, start_idx:end_idx, target_idx]
        true = self.true[pattern_idx, start_idx:end_idx, target_idx]

        if plot_type == "time_series":
            return {
                'x': list(range(start_idx, end_idx)),
                'pred': convert_numpy_array(pred),
                'true': convert_numpy_array(true),
                'title': f'样本 {pattern_idx} 目标 {target_idx} 的时间序列对比'
            }
        elif plot_type == "scatter":
            return {
                'true': convert_numpy_array(true),
                'pred': convert_numpy_array(pred),
                'title': f'样本 {pattern_idx} 目标 {target_idx} 的散点图'
            }
        elif plot_type == "residual":
            residuals = true - pred  # Corrected: Residual is typically true - pred
            return {
                'true': convert_numpy_array(true),
                'residuals': convert_numpy_array(residuals),
                'title': f'样本 {pattern_idx} 目标 {target_idx} 的残差图'
            }
        elif plot_type == "histogram":
            errors = true - pred  # Corrected: Error is typically true - pred
            hist, bin_edges = np.histogram(errors, bins=50, density=True)
            bins = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(bin_edges) - 1)]
            return {
                'bins': convert_numpy_array(bins),
                'hist': convert_numpy_array(hist),
                'title': f'样本 {pattern_idx} 目标 {target_idx} 的误差直方图'
            }
        elif plot_type == "lookback_comparison":
            # Call the new method to generate lookback comparison data
            return self.generate_lookback_plot_data(self.lookback, pattern_idx, target_idx, self.lookback_len)
        else:
            return {"error": "不支持的图表类型"}

    def generate_multi_sample_plot_data(self, target_idx, plot_type, sample_indices=None, num_samples=None):
        """
        生成多样本的散点图和误差直方图数据

        参数:
            target_idx: 目标变量索引
            plot_type: 图表类型 ('scatter' 或 'histogram')
            sample_indices: 指定的样本索引列表，如果为None则使用num_samples
            num_samples: 要使用的样本数量，从0开始选择
        """
        if sample_indices is None:
            if num_samples is None:
                sample_indices = list(range(self.num_patterns))
            else:
                sample_indices = list(range(min(num_samples, self.num_patterns)))
        
        if plot_type == "scatter":
            all_true = []
            all_pred = []
            sample_indices_per_point = []  # 新增：记录每个点对应的样本索引
            
            for sample_idx in sample_indices:
                true_values = self.true[sample_idx, :, target_idx].flatten()
                pred_values = self.pred[sample_idx, :, target_idx].flatten()
                
                all_true.extend(true_values)
                all_pred.extend(pred_values)
                sample_indices_per_point.extend([sample_idx] * len(true_values))
            
            return {
                'true': all_true,
                'pred': all_pred,
                'sample_indices_per_point': sample_indices_per_point,  # 新增：每个点的样本索引
                'title': f'目标 {target_idx} 的散点图 (样本数: {len(sample_indices)})',
                'sample_count': len(sample_indices),
                'sample_indices': sample_indices
            }
        else:
            raise ValueError(f"不支持的图表类型: {plot_type}")

    def save_results(self):
        """保存计算结果到文件"""
        all_metrics = self.compute_all_metrics()
        avg_metrics = self.compute_average_metrics()
        
        results = {
            'all_metrics': all_metrics,
            'avg_metrics': avg_metrics,
            'num_patterns': self.num_patterns,
            'num_targets': self.num_targets,
            'pred_len': self.pred_len
        }
        
        results_path = os.path.join(self.session_dir, 'results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, cls=NpEncoder)
        return results_path

    def cleanup(self):
        """清理临时文件"""
        if os.path.exists(self.session_dir):
            try:
                shutil.rmtree(self.session_dir)
            except Exception as e:
                print(f"清理会话文件时出错: {str(e)}")


# ... (Flask App setup, helper functions, NpEncoder class 保持不变) ...


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def convert_numpy_types(obj):
    """递归转换numpy类型为Python原生类型"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


def convert_numpy_array(arr):
    """转换numpy数组为Python列表"""
    if isinstance(arr, np.ndarray):
        return arr.tolist()
    return arr


def convert_nan_to_null(obj):
    """将NaN值转换为null"""
    if isinstance(obj, float) and np.isnan(obj):
        return None
    elif isinstance(obj, dict):
        return {key: convert_nan_to_null(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_nan_to_null(item) for item in obj]
    else:
        return obj


def convert_numpy_array(arr):
    if isinstance(arr, np.ndarray):
        return arr.tolist()
    elif isinstance(arr, (np.float32, np.float64, np.int32, np.int64)):
        return arr.item()
    return arr


def convert_numpy_types(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.generic, np.number)):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    return obj


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.generic, np.number)):
            return obj.item()
        elif isinstance(obj, float) and np.isnan(obj):
            return None
        return super().default(obj)


def convert_nan_to_null(data):
    if isinstance(data, dict):
        return {k: convert_nan_to_null(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_nan_to_null(v) for v in data]
    elif isinstance(data, float) and np.isnan(data):
        return None
    return data
