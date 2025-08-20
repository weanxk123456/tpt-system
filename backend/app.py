# -*- coding: utf-8 -*-
"""
Vue版本的TPT时序数据预测评估系统后端API
基于原始run_0710.py修改，添加CORS支持和Vue前端适配
"""

import os
import sys
import json
import uuid
import numpy as np
import pickle
import tempfile
import zipfile
import matplotlib.pyplot as plt
import io
from scipy.integrate import simpson
from typing import Dict, Any, List, Tuple
from flask import Flask, request, jsonify, send_file, render_template, session, Response
from werkzeug.utils import secure_filename
from flask_session import Session
from flask_cors import CORS
import shutil
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import base64
from datetime import datetime
import traceback

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False

# 导入原始的EvaluationMetricsCalculator类
from run_0710 import EvaluationMetricsCalculator, convert_numpy_types, convert_numpy_array, convert_nan_to_null

app = Flask(__name__)

# 配置CORS，允许Vue前端访问
CORS(app, supports_credentials=True, origins=['http://localhost:3000', 'http://127.0.0.1:3000'])

# 配置会话
app.config['SECRET_KEY'] = 'vue-tpt-system-secret-key-2025'
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_USE_SIGNER'] = True
app.config['SESSION_FILE_DIR'] = tempfile.mkdtemp()
Session(app)

# 全局变量存储计算器实例
calculators = {}

@app.before_request
def before_request():
    """在每个请求前执行，确保会话ID存在"""
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())

@app.route('/calculate', methods=['POST'])
def calculate():
    """计算评估指标的主要接口"""
    try:
        # 检查文件是否上传
        if 'actual_file' not in request.files or 'predicted_file' not in request.files:
            return jsonify({"success": False, "error": "请上传真实值和预测值文件"}), 400

        actual_file = request.files['actual_file']
        predicted_file = request.files['predicted_file']

        if actual_file.filename == '' or predicted_file.filename == '':
            return jsonify({"success": False, "error": "请选择有效的文件"}), 400

        # 获取会话ID
        session_id = session['session_id']

        # 创建临时目录
        session_dir = os.path.join(tempfile.gettempdir(), f"ts_evaluation_{session_id}")
        os.makedirs(session_dir, exist_ok=True)

        # 保存上传的文件
        actual_path = os.path.join(session_dir, secure_filename(actual_file.filename))
        predicted_path = os.path.join(session_dir, secure_filename(predicted_file.filename))
        
        actual_file.save(actual_path)
        predicted_file.save(predicted_path)

        # 加载numpy数据
        try:
            actual_data = np.load(actual_path)
            predicted_data = np.load(predicted_path)
        except Exception as e:
            return jsonify({"success": False, "error": f"无法加载numpy文件: {str(e)}"}), 400

        # 验证数据形状
        if actual_data.shape != predicted_data.shape:
            return jsonify({"success": False, "error": "真实值和预测值的形状不匹配"}), 400

        if len(actual_data.shape) != 3:
            return jsonify({"success": False, "error": "数据必须是三维数组 (patterns, pred_len, num_targets)"}), 400

        # 创建评估计算器
        calculator = EvaluationMetricsCalculator(predicted_data, actual_data, session_id)

        # 计算所有指标
        all_metrics = calculator.compute_all_metrics()
        avg_metrics = calculator.compute_average_metrics()

        # 保存结果
        results_path = calculator.save_results()

        # 获取第一个样本和目标的图表数据作为默认显示
        chart_data = calculator.get_chart_data(0, 0)

        # 转换数据类型
        converted_metrics = convert_numpy_types(all_metrics)
        converted_avg = convert_numpy_types(avg_metrics)
        converted_chart_data = convert_numpy_types(chart_data)

        # 保存计算器实例
        calculator_path = os.path.join(session_dir, 'calculator.pkl')
        with open(calculator_path, 'wb') as f:
            pickle.dump(calculator, f)

        calculators[session_id] = calculator

        return jsonify({
            "success": True,
            "all_metrics": convert_nan_to_null(converted_metrics),
            "avg_metrics": converted_avg,
            "data": {
                "comparison": converted_chart_data
            },
            "num_patterns": calculator.num_patterns,
            "num_targets": calculator.num_targets,
            "results_path": results_path,
            "predict_len": calculator.pred_len
        })

    except Exception as e:
        return jsonify({"success": False, "error": f"处理请求时发生错误: {str(e)}"}), 500

@app.route('/generate_plot', methods=['POST'])
def generate_plot():
    """生成不同类型的图表数据"""
    try:
        # 检查会话
        if 'session_id' not in session:
            return jsonify({"success": False, "error": "请先刷新页面建立会话"}), 400
        session_id = session['session_id']

        # 加载计算器
        calculator = get_calculator(session_id)
        if not calculator:
            return jsonify({"success": False, "error": "请先上传并计算指标"}), 400

        data = request.json
        pattern_idx = int(data.get('pattern_idx', 0))
        target_idx = int(data.get('target_idx', 0))
        plot_type = data.get('plot_type', 'time_series')
        start_idx = int(data.get('start_idx', 0))
        end_idx = data.get('end_idx')
        if end_idx is not None:
            end_idx = int(end_idx)

        # 验证索引
        if pattern_idx < 0 or pattern_idx >= calculator.num_patterns:
            return jsonify({"success": False, "error": f"无效的样本索引: {pattern_idx}"}), 400
        if target_idx < 0 or target_idx >= calculator.num_targets:
            return jsonify({"success": False, "error": f"无效的目标索引: {target_idx}"}), 400

        # 生成图表数据
        plot_data = calculator.generate_plot_data(pattern_idx, target_idx, plot_type, start_idx, end_idx)

        return jsonify({
            "success": True,
            "plot_data": plot_data
        })

    except Exception as e:
        return jsonify({"success": False, "error": f"生成图表数据时发生错误: {str(e)}"}), 500

@app.route('/generate_multi_sample_plot', methods=['POST'])
def generate_multi_sample_plot():
    """生成多样本的散点图和误差直方图数据"""
    try:
        # 检查会话
        if 'session_id' not in session:
            return jsonify({"success": False, "error": "请先刷新页面建立会话"}), 400
        session_id = session['session_id']

        # 加载计算器
        calculator = get_calculator(session_id)
        if not calculator:
            return jsonify({"success": False, "error": "请先上传并计算指标"}), 400

        data = request.json
        target_idx = int(data.get('target_idx', 0))
        plot_type = data.get('plot_type', 'scatter')
        num_samples = data.get('num_samples')
        sample_indices = data.get('sample_indices')

        # 验证目标索引
        if target_idx < 0 or target_idx >= calculator.num_targets:
            return jsonify({"success": False, "error": f"无效的目标索引: {target_idx}"}), 400

        # 验证图表类型
        if plot_type not in ['scatter', 'histogram']:
            return jsonify({"success": False, "error": f"不支持的图表类型: {plot_type}"}), 400

        # 处理样本数量参数
        if num_samples is not None:
            num_samples = int(num_samples)
            if num_samples <= 0:
                return jsonify({"success": False, "error": "样本数量必须大于0"}), 400
            if num_samples > calculator.num_patterns:
                num_samples = calculator.num_patterns

        # 处理样本索引参数
        if sample_indices is not None:
            if not isinstance(sample_indices, list):
                return jsonify({"success": False, "error": "样本索引必须是列表格式"}), 400
            # 验证所有索引都在有效范围内
            invalid_indices = [idx for idx in sample_indices if idx < 0 or idx >= calculator.num_patterns]
            if invalid_indices:
                return jsonify({"success": False, "error": f"无效的样本索引: {invalid_indices}"}), 400

        # 生成多样本图表数据
        plot_data = calculator.generate_multi_sample_plot_data(
            target_idx=target_idx,
            plot_type=plot_type,
            sample_indices=sample_indices,
            num_samples=num_samples
        )

        return jsonify({
            "success": True,
            "plot_data": plot_data
        })

    except Exception as e:
        return jsonify({"success": False, "error": f"生成多样本图表数据时发生错误: {str(e)}"}), 500

@app.route('/evaluate_anomalies', methods=['POST'])
def evaluate_anomalies():
    """评估异常样本"""
    try:
        # 检查会话
        if 'session_id' not in session:
            return jsonify({"success": False, "error": "请先刷新页面建立会话"}), 400
        session_id = session['session_id']

        # 加载计算器
        calculator = get_calculator(session_id)
        if not calculator:
            return jsonify({"success": False, "error": "请先上传并计算指标"}), 400

        data = request.json or {}
        mae_thresh = float(data.get('mae_thresh', 0.5))
        tda_thresh = float(data.get('tda_thresh', 50))

        # 评估异常样本
        anomaly_df = calculator.evaluate_all_samples(mae_thresh, tda_thresh)
        
        # 转换为字典列表
        anomalies = anomaly_df.to_dict('records')

        return jsonify({
            "success": True,
            "anomalies": anomalies,
            "total_anomalies": len(anomalies)
        })

    except Exception as e:
        return jsonify({"success": False, "error": f"评估异常样本时发生错误: {str(e)}"}), 500

@app.route('/generate_static_plot', methods=['POST'])
def generate_static_plot():
    """生成静态图表图片"""
    try:
        # 检查会话
        if 'session_id' not in session:
            return jsonify({"success": False, "error": "请先刷新页面建立会话"}), 400
        session_id = session['session_id']

        # 加载计算器
        calculator = get_calculator(session_id)
        if not calculator:
            return jsonify({"success": False, "error": "请先上传并计算指标"}), 400

        data = request.json
        target_idx = int(data.get('target_idx', 0))
        plot_type = data.get('plot_type', 'scatter')
        sample_indices = data.get('sample_indices', [])

        # 验证目标索引
        if target_idx < 0 or target_idx >= calculator.num_targets:
            return jsonify({"success": False, "error": f"无效的目标索引: {target_idx}"}), 400

        # 验证样本索引
        if not sample_indices:
            return jsonify({"success": False, "error": "请提供样本索引"}), 400

        invalid_indices = [idx for idx in sample_indices if idx < 0 or idx >= calculator.num_patterns]
        if invalid_indices:
            return jsonify({"success": False, "error": f"无效的样本索引: {invalid_indices}"}), 400

        # 生成静态图表
        if plot_type == 'scatter':
            # 收集所有样本的数据
            all_true = []
            all_pred = []
            sample_info = []
            
            for sample_idx in sample_indices:
                true_values = calculator.true_data[sample_idx, :, target_idx].flatten()
                pred_values = calculator.pred_data[sample_idx, :, target_idx].flatten()
                
                all_true.extend(true_values)
                all_pred.extend(pred_values)
                sample_info.extend([sample_idx] * len(true_values))

            # 创建matplotlib图表
            plt.figure(figsize=(10, 8))
            plt.scatter(all_true, all_pred, alpha=0.6, s=20, c='#3b82f6', edgecolors='none')
            
            # 添加对角线
            min_val = min(min(all_true), min(all_pred))
            max_val = max(max(all_true), max(all_pred))
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
            
            plt.xlabel('真实值', fontsize=12)
            plt.ylabel('预测值', fontsize=12)
            plt.title(f'多样本散点图 - 目标{target_idx} ({len(sample_indices)}个样本)', fontsize=14)
            plt.grid(True, alpha=0.3)
            
            # 保存为base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()

            return jsonify({
                "success": True,
                "static_plot": {
                    "image_base64": image_base64,
                    "title": f'多样本散点图 - 目标{target_idx}',
                    "sample_count": len(sample_indices),
                    "sample_indices": sample_indices,
                    "data_points": len(all_true),
                    "clickable_regions": generate_clickable_regions(all_true, all_pred, sample_info)
                }
            })
        else:
            return jsonify({"success": False, "error": f"暂不支持的静态图表类型: {plot_type}"}), 400

    except Exception as e:
        return jsonify({"success": False, "error": f"生成静态图表时发生错误: {str(e)}"}), 500

@app.route("/generate_lookback_plot", methods=["POST"])
def generate_lookback_plot():
    """生成回看数据对比图表"""
    try:
        # 检查会话
        if "session_id" not in session:
            return jsonify({"success": False, "error": "请先刷新页面建立会话"}), 400
        session_id = session["session_id"]
        # 加载计算器
        calculator = get_calculator(session_id)
        if not calculator:
            return jsonify({"success": False, "error": "请先上传并计算指标"}), 400
        # 检查上传的训练数据
        if "training_data" not in request.files:
            return jsonify({"success": False, "error": "请上传训练数据文件"}), 400
        training_file = request.files["training_data"]
        # 获取参数
        pattern_idx = int(request.form.get("pattern_idx", 0))
        target_idx = int(request.form.get("target_idx", 0))
        lookback_steps = int(request.form.get("lookback_steps", 10))
        # 验证索引参数
        if pattern_idx < 0 or pattern_idx >= calculator.num_patterns:
            return jsonify({"success": False, "error": f"无效的样本索引: {pattern_idx}"}), 400
        if target_idx < 0 or target_idx >= calculator.num_targets:
            return jsonify({"success": False, "error": f"无效的目标索引: {target_idx}"}), 400
        if lookback_steps <= 0:
            return jsonify({"success": False, "error": "回看步数必须大于0"}), 400
        # 加载训练数据
        try:
            training_data = np.load(training_file.stream)
        except Exception as e:
            return jsonify({"success": False, "error": f"无法加载训练数据文件: {str(e)}"}), 400
        # 验证训练数据形状
        if training_data.ndim != 3:
            return jsonify({"success": False, "error": "训练数据必须是三维数组 (patterns, pred_len, num_targets)"}), 400
        if training_data.shape[0] != calculator.num_patterns or training_data.shape[2] != calculator.num_targets:
            return jsonify({"success": False, "error": "训练数据与预测/真实数据样本数或目标数不匹配"}), 400
        # 生成回看图表数据
        plot_data = calculator.generate_lookback_plot_data(
            training_data=training_data,
            pattern_idx=pattern_idx,
            target_idx=target_idx,
            lookback_steps=lookback_steps
        )

        # ---- 递归处理 NaN 和 numpy 类型 ----
        def nan_to_none(obj):
            if isinstance(obj, dict):
                return {k: nan_to_none(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [nan_to_none(v) for v in obj]
            elif isinstance(obj, np.ndarray):
                return nan_to_none(obj.tolist())
            elif isinstance(obj, (np.float32, np.float64, float)):
                if np.isnan(obj):
                    return None
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64, int)):
                return int(obj)
            elif obj is None:
                return None
            else:
                return obj

        clean_plot_data = nan_to_none(plot_data)
        print(type(clean_plot_data))

        return jsonify({"success": True, "plot_data": clean_plot_data})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": f"生成回看对比图时发生错误: {str(e)}"}), 500


def generate_clickable_regions(true_values, pred_values, sample_indices):
    """生成可点击区域的坐标信息"""
    # 简化版本：将图表分成网格区域，每个区域包含对应的样本信息
    regions = []
    
    # 计算数据范围
    min_true = min(true_values)
    max_true = max(true_values)
    min_pred = min(pred_values)
    max_pred = max(pred_values)
    
    # 创建10x10网格
    grid_size = 10
    true_step = (max_true - min_true) / grid_size
    pred_step = (max_pred - min_pred) / grid_size
    
    for i in range(grid_size):
        for j in range(grid_size):
            true_start = min_true + i * true_step
            true_end = min_true + (i + 1) * true_step
            pred_start = min_pred + j * pred_step
            pred_end = min_pred + (j + 1) * pred_step
            
            # 找到在这个区域内的样本
            samples_in_region = set()
            for idx, (true_val, pred_val, sample_idx) in enumerate(zip(true_values, pred_values, sample_indices)):
                if true_start <= true_val <= true_end and pred_start <= pred_val <= pred_end:
                    samples_in_region.add(sample_idx)
            
            if samples_in_region:
                regions.append({
                    "x_start": true_start,
                    "x_end": true_end,
                    "y_start": pred_start,
                    "y_end": pred_end,
                    "samples": list(samples_in_region)
                })
    
    return regions

@app.route('/cleanup', methods=['POST'])
def cleanup():
    """清理会话数据"""
    try:
        if 'session_id' in session:
            session_id = session['session_id']
            
            # 从内存中删除计算器
            if session_id in calculators:
                del calculators[session_id]
            
            # 删除临时文件
            session_dir = os.path.join(tempfile.gettempdir(), f"ts_evaluation_{session_id}")
            if os.path.exists(session_dir):
                shutil.rmtree(session_dir)
            
            # 清除会话
            session.clear()

        return jsonify({"success": True})

    except Exception as e:
        return jsonify({"success": False, "error": f"清理数据时发生错误: {str(e)}"}), 500

def get_calculator(session_id):
    """获取计算器实例"""
    # 先从内存中查找
    if session_id in calculators:
        return calculators[session_id]
    
    # 尝试从文件加载
    session_dir = os.path.join(tempfile.gettempdir(), f"ts_evaluation_{session_id}")
    calculator_path = os.path.join(session_dir, 'calculator.pkl')
    if os.path.exists(calculator_path):
        try:
            with open(calculator_path, 'rb') as f:
                calculator = pickle.load(f)
            # 保存到内存中
            calculators[session_id] = calculator
            return calculator
        except Exception:
            pass
    
    return None

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({"status": "ok", "message": "Vue TPT System Backend is running"})

if __name__ == '__main__':
    # 开发环境运行
    app.run(host='0.0.0.0', port=5000, debug=True)

