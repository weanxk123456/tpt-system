#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vue版本TPT系统后端启动脚本
"""

import os
import sys

# 添加路径以便导入原始模块

from app import app

if __name__ == '__main__':
    print("启动Vue版TPT系统后端服务器...")
    print("前端开发服务器: http://localhost:3000")
    print("后端API服务器: http://localhost:5000")
    print("按 Ctrl+C 停止服务器")

    # 启动Flask开发服务器
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )

