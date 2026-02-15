#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
绘制 world_metrics.csv 中的变量折线图
支持选择变量和添加阈值线

新增功能：
- 风险指标计算（基于Engle (1982)和Bollerslev (1986)）
  - 支持两种预期规则：滚动平均（Rolling Mean）和朴素预期（Naive Forecast）
  - 风险指标公式：h_t = λ * h_{t-1} + (1-λ) * e_{t-1}^2，其中λ=0.94（RiskMetrics标准）
"""

import os
import sys
import argparse
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 使用无图形后端
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# 定义可用的变量
AVAILABLE_VARIABLES = [
    "price",
    "price_inflation_rate",
    "interest_rate",
    "total_wealth",
    "total_income",
    "total_consumption",
    "avg_wealth",
    "avg_income",
    "avg_consumption",
    "gini_wealth",
    "gini_income",
    "unemployment_rate",
    "risk_indicator_rolling",
    "risk_indicator_naive",
    "risk_indicator_comparison"
]

def calculate_risk_indicator(df, method='rolling', window=10, lambda_param=0.94):
    """
    计算风险指标（基于Engle (1982)和Bollerslev (1986)）
    
    Args:
        df: 包含price列的DataFrame
        method: 'rolling' 或 'naive'，表示预期规则
        window: rolling mean的窗口大小（仅用于rolling方法）
        lambda_param: RiskMetrics参数λ，默认0.94
    
    Returns:
        (timestep, risk_values) 元组
    """
    if 'price' not in df.columns:
        print(f"❌ 错误：CSV文件中缺少 'price' 列，无法计算风险指标")
        return None
    
    # 计算通胀率 π_t = log P_t - log P_{t-1}
    prices = df['price'].values
    log_prices = np.log(prices)
    pi_t = np.diff(log_prices)  # π_t = log P_t - log P_{t-1}
    
    # 在开头插入NaN以保持长度一致（第一个时间步没有前一个价格）
    pi_t = np.insert(pi_t, 0, np.nan)
    
    n = len(pi_t)
    
    # 初始化数组
    E_pi = np.full(n, np.nan)  # 预期通胀率
    e_t = np.full(n, np.nan)   # 预测误差
    h_t = np.full(n, np.nan)   # 风险指标
    
    # 计算预期和误差
    for t in range(1, n):
        if method == 'rolling':
            # 方式1：rolling mean E_{t-1}[π_t] = (1/k) * sum_{i=1}^k π_{t-i}
            if t <= window:
                # 如果数据不足，使用所有可用数据
                available_data = pi_t[1:t]  # 跳过第一个NaN
                if len(available_data) > 0:
                    E_pi[t] = np.mean(available_data)
                else:
                    E_pi[t] = 0.0
            else:
                E_pi[t] = np.mean(pi_t[t-window:t])
        elif method == 'naive':
            # 方式2：naive forecast E_{t-1}[π_t] = π_{t-1}
            E_pi[t] = pi_t[t-1]
        
        # 计算预测误差 e_t = π_t - E_{t-1}[π_t]
        if not np.isnan(E_pi[t]) and not np.isnan(pi_t[t]):
            e_t[t] = pi_t[t] - E_pi[t]
    
    # 计算风险指标 h_t = λ * h_{t-1} + (1-λ) * e_{t-1}^2
    # 根据RiskMetrics标准：h_t 使用 e_{t-1}^2，所以需要找到第一个有效的 e_t
    first_valid_idx = None
    for i in range(1, n):
        if not np.isnan(e_t[i]):
            first_valid_idx = i
            break
    
    if first_valid_idx is not None:
        # 初始化：h_t[first_valid_idx] = e_t[first_valid_idx]^2
        # 这是第一个可用的值，作为初始条件
        h_t[first_valid_idx] = e_t[first_valid_idx] ** 2
        
        # 递归计算：h_t = λ * h_{t-1} + (1-λ) * e_{t-1}^2
        # 注意：h_t[t] 使用 e_t[t-1]^2，所以从 first_valid_idx+1 开始
        for t in range(first_valid_idx + 1, n):
            if not np.isnan(e_t[t-1]) and not np.isnan(h_t[t-1]):
                h_t[t] = lambda_param * h_t[t-1] + (1 - lambda_param) * (e_t[t-1] ** 2)
    
    timestep = df['timestep'].values
    return (timestep, h_t)


def read_csv_data(csv_path, variable):
    """
    读取CSV文件并提取指定变量的数据
    
    Args:
        csv_path: CSV文件路径
        variable: 变量名
    
    Returns:
        (timestep, values) 或 None（如果出错）
    """
    if not os.path.exists(csv_path):
        print(f"❌ 错误：找不到文件 {csv_path}")
        return None
    
    try:
        df = pd.read_csv(csv_path)
        
        if df.empty:
            print(f"❌ 错误：CSV文件为空: {csv_path}")
            return None
        
        # 处理风险指标（需要计算）
        if variable == 'risk_indicator_rolling':
            return calculate_risk_indicator(df, method='rolling')
        elif variable == 'risk_indicator_naive':
            return calculate_risk_indicator(df, method='naive')
        elif variable == 'risk_indicator_comparison':
            # 对比图由专门的函数处理，这里不需要返回数据
            return None
        
        # 检查变量是否存在
        if variable not in df.columns:
            print(f"❌ 错误：变量 '{variable}' 不存在于CSV文件中: {csv_path}")
            print(f"   可用的变量：{', '.join(df.columns.tolist())}")
            return None
        
        # 提取数据
        timestep = df['timestep'].values
        values = df[variable].values
        
        return (timestep, values)
        
    except Exception as e:
        print(f"❌ 错误：无法读取CSV文件 {csv_path}: {e}")
        return None


def plot_variable(csv_path, variable, threshold=None, output_dir=None, data_folder=None, 
                  baseline_csv=None, real_csv=None):
    """
    绘制指定变量的折线图
    
    Args:
        csv_path: CSV文件路径（单文件模式）
        variable: 要绘制的变量名
        threshold: 阈值（可选），如果提供会画红色虚线
        output_dir: 输出目录（可选），默认为 data_folder/plot 或 csv_path 所在目录
        data_folder: 数据文件夹路径（可选，用于确定默认输出目录）
        baseline_csv: baseline CSV文件路径（对比模式）
        real_csv: real CSV文件路径（对比模式）
    """
    # 确定输出目录
    if output_dir is None:
        if data_folder:
            # 使用 data_folder 参数时，默认输出到 data_folder/plot
            output_dir = os.path.join(data_folder, "plot")
        elif csv_path:
            # 如果使用 --csv-file，尝试智能推断 plot 目录
            csv_dir = os.path.dirname(csv_path)
            # 如果 CSV 文件在 metrics_csv 目录下，输出到父目录的 plot 目录
            if os.path.basename(csv_dir) == "metrics_csv":
                output_dir = os.path.join(os.path.dirname(csv_dir), "plot")
            else:
                # 否则输出到 CSV 文件所在目录
                output_dir = csv_dir
        elif baseline_csv:
            # 对比模式：如果 baseline CSV 在某个子目录下，尝试使用父目录的 plot
            baseline_dir = os.path.dirname(baseline_csv)
            if os.path.basename(baseline_dir) in ["baseline", "real"]:
                output_dir = os.path.join(os.path.dirname(baseline_dir), "plot")
            else:
                output_dir = baseline_dir
        else:
            output_dir = "."
    os.makedirs(output_dir, exist_ok=True)
    
    # 特殊处理：如果绘制风险指标
    if variable == 'risk_indicator_comparison':
        # 生成包含两种方法的对比图
        return plot_risk_indicator_comparison(csv_path, output_dir, data_folder, 
                                             baseline_csv, real_csv, threshold)
    elif variable in ['risk_indicator_rolling', 'risk_indicator_naive']:
        # 根据指定的变量决定绘制哪种方法（单一方法）
        method = 'rolling' if variable == 'risk_indicator_rolling' else 'naive'
        return plot_risk_indicator_single(csv_path, output_dir, data_folder, 
                                         baseline_csv, real_csv, threshold, method)
    
    # 创建图形
    plt.figure(figsize=(10, 6))
    
    # 判断是单文件模式还是对比模式
    if baseline_csv and real_csv:
        # 对比模式：绘制两条线
        baseline_data = read_csv_data(baseline_csv, variable)
        real_data = read_csv_data(real_csv, variable)
        
        if baseline_data is None or real_data is None:
            return False
        
        timestep_baseline, values_baseline = baseline_data
        timestep_real, values_real = real_data
        
        # 绘制两条折线
        plt.plot(timestep_baseline, values_baseline, marker='o', markersize=4, 
                linewidth=2, label='Baseline', color='blue', alpha=0.7)
        plt.plot(timestep_real, values_real, marker='s', markersize=4, 
                linewidth=2, label='Real', color='red', alpha=0.7)
        
    else:
        # 单文件模式：只绘制一条线
        data = read_csv_data(csv_path, variable)
        if data is None:
            return False
        
        timestep, values = data
        plt.plot(timestep, values, marker='o', markersize=4, linewidth=2, label=variable)
    
    # 添加阈值线（如果提供）
    if threshold is not None:
        try:
            threshold_value = float(threshold)
            plt.axhline(y=threshold_value, color='red', linestyle='--', linewidth=2, 
                       label=f'Threshold: {threshold_value}')
        except ValueError:
            print(f"⚠️  警告：阈值 '{threshold}' 不是有效数字，跳过阈值线")
    
    # 设置标签和标题
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel(variable.replace('_', ' ').title(), fontsize=12)
    title = f'{variable.replace("_", " ").title()} Over Time'
    if baseline_csv and real_csv:
        title += ' (Baseline vs Real)'
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    # 保存图片
    output_path = os.path.join(output_dir, f"{variable}.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 已生成图像: {output_path}")
    return True


def plot_risk_indicator_single(csv_path, output_dir, data_folder=None,
                                baseline_csv=None, real_csv=None, threshold=None, method='rolling'):
    """
    绘制单一风险指标图（只显示一种预期方法）
    
    Args:
        csv_path: CSV文件路径（单文件模式）
        output_dir: 输出目录
        data_folder: 数据文件夹路径
        baseline_csv: baseline CSV文件路径（对比模式）
        real_csv: real CSV文件路径（对比模式）
        threshold: 阈值（可选）
        method: 'rolling' 或 'naive'，表示预期规则
    """
    # 判断是单文件模式还是对比模式
    if baseline_csv and real_csv:
        # 对比模式：需要读取两个文件
        baseline_df = pd.read_csv(baseline_csv)
        real_df = pd.read_csv(real_csv)
        
        baseline_data = calculate_risk_indicator(baseline_df, method=method)
        real_data = calculate_risk_indicator(real_df, method=method)
        
        if baseline_data is None or real_data is None:
            return False
        
        # 创建图形
        plt.figure(figsize=(10, 6))
        
        timestep_b, values_b = baseline_data
        timestep_r, values_r = real_data
        
        plt.plot(timestep_b, values_b, marker='o', markersize=4, linewidth=2, 
                label='Baseline', color='blue', alpha=0.7)
        plt.plot(timestep_r, values_r, marker='s', markersize=4, linewidth=2, 
                label='Real', color='red', alpha=0.7)
        
        method_name = 'Rolling Mean' if method == 'rolling' else 'Naive Forecast'
        method_formula = 'E_{t-1}[π_t] = (1/k)Σπ_{t-i}' if method == 'rolling' else 'E_{t-1}[π_t] = π_{t-1}'
        
        plt.xlabel('Time Step', fontsize=12)
        plt.ylabel('Risk Indicator', fontsize=12)
        plt.title(f'Risk Indicator ({method_name} Forecast: {method_formula}) (Baseline vs Real)', 
                 fontsize=14, fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(fontsize=10)
        # 使用科学计数法格式化y轴
        plt.gca().yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        plt.gca().ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        
        # 添加阈值线（如果提供）
        if threshold is not None:
            try:
                threshold_value = float(threshold)
                plt.axhline(y=threshold_value, color='red', linestyle='--', linewidth=2, 
                           label=f'Threshold: {threshold_value}')
                plt.legend(fontsize=10)
            except ValueError:
                print(f"⚠️  警告：阈值 '{threshold}' 不是有效数字，跳过阈值线")
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, f"risk_indicator_{method}.png")
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 已生成图像: {output_path}")
        return True
        
    else:
        # 单文件模式：只绘制一种方法
        df = pd.read_csv(csv_path)
        data = calculate_risk_indicator(df, method=method)
        
        if data is None:
            return False
        
        timestep, values = data
        
        # 创建图形
        plt.figure(figsize=(10, 6))
        
        method_name = 'Rolling Mean' if method == 'rolling' else 'Naive Forecast'
        method_formula = 'E_{t-1}[π_t] = (1/k)Σπ_{t-i}' if method == 'rolling' else 'E_{t-1}[π_t] = π_{t-1}'
        
        plt.plot(timestep, values, marker='o', markersize=4, linewidth=2, 
                label=f'Risk Indicator ({method_name})', color='blue', alpha=0.7)
        
        plt.xlabel('Time Step', fontsize=12)
        plt.ylabel('Risk Indicator', fontsize=12)
        plt.title(f'Risk Indicator ({method_name} Forecast: {method_formula})', 
                 fontsize=14, fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(fontsize=10)
        # 使用科学计数法格式化y轴
        plt.gca().yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        plt.gca().ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        
        # 添加阈值线（如果提供）
        if threshold is not None:
            try:
                threshold_value = float(threshold)
                plt.axhline(y=threshold_value, color='red', linestyle='--', linewidth=2, 
                           label=f'Threshold: {threshold_value}')
                plt.legend(fontsize=10)
            except ValueError:
                print(f"⚠️  警告：阈值 '{threshold}' 不是有效数字，跳过阈值线")
        
        plt.tight_layout()
        
        # 保存图片
        output_path = os.path.join(output_dir, f"risk_indicator_{method}.png")
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 已生成图像: {output_path}")
        return True


def plot_risk_indicator_comparison(csv_path, output_dir, data_folder=None,
                                   baseline_csv=None, real_csv=None, threshold=None):
    """
    绘制风险指标对比图（同时显示rolling和naive两种方法）
    
    Args:
        csv_path: CSV文件路径（单文件模式）
        output_dir: 输出目录
        data_folder: 数据文件夹路径
        baseline_csv: baseline CSV文件路径（对比模式）
        real_csv: real CSV文件路径（对比模式）
        threshold: 阈值（可选）
    """
    # 判断是单文件模式还是对比模式
    if baseline_csv and real_csv:
        # 对比模式：需要读取两个文件
        baseline_df = pd.read_csv(baseline_csv)
        real_df = pd.read_csv(real_csv)
        
        baseline_rolling = calculate_risk_indicator(baseline_df, method='rolling')
        baseline_naive = calculate_risk_indicator(baseline_df, method='naive')
        real_rolling = calculate_risk_indicator(real_df, method='rolling')
        real_naive = calculate_risk_indicator(real_df, method='naive')
        
        if any(x is None for x in [baseline_rolling, baseline_naive, real_rolling, real_naive]):
            return False
        
        # 创建图形：两个子图
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # 子图1：Rolling Mean方法
        timestep_b, values_b = baseline_rolling
        timestep_r, values_r = real_rolling
        ax1.plot(timestep_b, values_b, marker='o', markersize=3, linewidth=2, 
                label='Baseline (Rolling)', color='blue', alpha=0.7)
        ax1.plot(timestep_r, values_r, marker='s', markersize=3, linewidth=2, 
                label='Real (Rolling)', color='red', alpha=0.7)
        ax1.set_xlabel('Time Step', fontsize=11)
        ax1.set_ylabel('Risk Indicator', fontsize=11)
        ax1.set_title('Risk Indicator (Rolling Mean Forecast)', fontsize=12, fontweight='bold')
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.legend(fontsize=9)
        # 使用科学计数法格式化y轴
        ax1.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        ax1.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        
        # 子图2：Naive Forecast方法
        timestep_b, values_b = baseline_naive
        timestep_r, values_r = real_naive
        ax2.plot(timestep_b, values_b, marker='o', markersize=3, linewidth=2, 
                label='Baseline (Naive)', color='blue', alpha=0.7)
        ax2.plot(timestep_r, values_r, marker='s', markersize=3, linewidth=2, 
                label='Real (Naive)', color='red', alpha=0.7)
        ax2.set_xlabel('Time Step', fontsize=11)
        ax2.set_ylabel('Risk Indicator', fontsize=11)
        ax2.set_title('Risk Indicator (Naive Forecast)', fontsize=12, fontweight='bold')
        ax2.grid(True, linestyle='--', alpha=0.5)
        ax2.legend(fontsize=9)
        # 使用科学计数法格式化y轴
        ax2.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        ax2.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, "risk_indicator_comparison.png")
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 已生成图像: {output_path}")
        return True
        
    else:
        # 单文件模式：绘制两种方法在同一图上
        df = pd.read_csv(csv_path)
        rolling_data = calculate_risk_indicator(df, method='rolling')
        naive_data = calculate_risk_indicator(df, method='naive')
        
        if rolling_data is None or naive_data is None:
            return False
        
        timestep_rolling, values_rolling = rolling_data
        timestep_naive, values_naive = naive_data
        
        # 创建图形：两个子图
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # 子图1：Rolling Mean方法
        ax1.plot(timestep_rolling, values_rolling, marker='o', markersize=3, 
                linewidth=2, label='Risk Indicator (Rolling Mean)', color='blue', alpha=0.7)
        ax1.set_xlabel('Time Step', fontsize=11)
        ax1.set_ylabel('Risk Indicator', fontsize=11)
        ax1.set_title('Risk Indicator (Rolling Mean Forecast: E_{t-1}[π_t] = (1/k)Σπ_{t-i})', 
                     fontsize=12, fontweight='bold')
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.legend(fontsize=9)
        # 使用科学计数法格式化y轴
        ax1.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        ax1.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        
        # 子图2：Naive Forecast方法
        ax2.plot(timestep_naive, values_naive, marker='s', markersize=3, 
                linewidth=2, label='Risk Indicator (Naive Forecast)', color='green', alpha=0.7)
        ax2.set_xlabel('Time Step', fontsize=11)
        ax2.set_ylabel('Risk Indicator', fontsize=11)
        ax2.set_title('Risk Indicator (Naive Forecast: E_{t-1}[π_t] = π_{t-1})', 
                     fontsize=12, fontweight='bold')
        ax2.grid(True, linestyle='--', alpha=0.5)
        ax2.legend(fontsize=9)
        # 使用科学计数法格式化y轴
        ax2.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        ax2.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        
        # 添加阈值线（如果提供）
        if threshold is not None:
            try:
                threshold_value = float(threshold)
                ax1.axhline(y=threshold_value, color='red', linestyle='--', linewidth=1.5, 
                           alpha=0.7, label=f'Threshold: {threshold_value}')
                ax2.axhline(y=threshold_value, color='red', linestyle='--', linewidth=1.5, 
                           alpha=0.7, label=f'Threshold: {threshold_value}')
                ax1.legend(fontsize=9)
                ax2.legend(fontsize=9)
            except ValueError:
                print(f"⚠️  警告：阈值 '{threshold}' 不是有效数字，跳过阈值线")
        
        plt.tight_layout()
        
        # 保存图片
        output_path = os.path.join(output_dir, "risk_indicator_comparison.png")
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 已生成图像: {output_path}")
        return True


def main():
    parser = argparse.ArgumentParser(
        description='绘制 world_metrics.csv 中的变量折线图',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
示例：
  # 使用 data_folder 参数（传统方式）
  # 绘制 price 变量（无阈值）
  python {sys.argv[0]} complex-20agents-20months price
  
  # 绘制 interest_rate 变量，添加阈值 0.03
  python {sys.argv[0]} complex-20agents-20months interest_rate --threshold 0.03
  
  # 绘制多个变量
  python {sys.argv[0]} complex-20agents-20months price interest_rate unemployment_rate
  
  # 使用 --csv-file 直接指定 CSV 文件路径
  python {sys.argv[0]} --csv-file /path/to/world_metrics.csv price_inflation_rate --output-dir /path/to/output
  
  # 对比模式：同时绘制 baseline 和 real 两条线
  python {sys.argv[0]} --baseline-csv /path/to/baseline/world_metrics.csv --real-csv /path/to/real/world_metrics.csv price_inflation_rate --output-dir /path/to/output
  
  # 对比模式：使用 --compare-dir（自动查找 baseline 和 real 子目录）
  python {sys.argv[0]} --compare-dir /path/to/shapley_2 price_inflation_rate --output-dir /path/to/output

可用变量：
  {', '.join(AVAILABLE_VARIABLES)}
        """
    )
    
    parser.add_argument(
        'data_folder',
        nargs='?',
        help='数据文件夹名称（相对于 data 目录），例如：complex-20agents-20months。如果指定了 --csv-file，则不需要此参数'
    )
    
    parser.add_argument(
        'variables',
        nargs='+',
        help='要绘制的变量名，可以是多个'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        help='阈值（红色虚线），适用于所有变量'
    )
    
    parser.add_argument(
        '--thresholds',
        nargs='+',
        type=float,
        help='每个变量对应的阈值（数量需与变量数量相同）'
    )
    
    parser.add_argument(
        '--data-root',
        default='/mnt/shared-storage-user/meijilin/Economic_System_Attribution/ACL24-EconAgent/data',
        help='数据根目录路径（默认：/mnt/shared-storage-user/meijilin/ACL24-EconAgent/data）'
    )
    
    parser.add_argument(
        '--output-dir',
        help='输出目录（默认：{data_folder}/plot 或 CSV 文件所在目录）'
    )
    
    parser.add_argument(
        '--csv-file',
        help='直接指定 CSV 文件路径（如果指定，则不需要 data_folder 参数）'
    )
    
    parser.add_argument(
        '--baseline-csv',
        help='Baseline CSV 文件路径（用于对比模式，需同时指定 --real-csv）'
    )
    
    parser.add_argument(
        '--real-csv',
        help='Real CSV 文件路径（用于对比模式，需同时指定 --baseline-csv）'
    )
    
    parser.add_argument(
        '--compare-dir',
        help='包含 baseline 和 real 子目录的父目录路径（会自动查找 baseline/world_metrics.csv 和 real/world_metrics.csv）'
    )
    
    args = parser.parse_args()
    
    # 确定 CSV 文件路径
    baseline_csv = None
    real_csv = None
    csv_path = None
    data_folder = None
    
    # 检查对比模式
    if args.compare_dir:
        # 使用 compare-dir 模式
        baseline_csv = os.path.join(args.compare_dir, "baseline", "world_metrics.csv")
        real_csv = os.path.join(args.compare_dir, "real", "world_metrics.csv")
        
        if not os.path.exists(baseline_csv):
            print(f"❌ 错误：找不到 baseline CSV 文件: {baseline_csv}")
            return 1
        if not os.path.exists(real_csv):
            print(f"❌ 错误：找不到 real CSV 文件: {real_csv}")
            return 1
        
        data_folder = args.compare_dir
        
    elif args.baseline_csv or args.real_csv:
        # 使用 --baseline-csv 和 --real-csv 模式
        if not args.baseline_csv or not args.real_csv:
            print(f"❌ 错误：对比模式需要同时指定 --baseline-csv 和 --real-csv")
            return 1
        
        baseline_csv = args.baseline_csv
        real_csv = args.real_csv
        
        if not os.path.exists(baseline_csv):
            print(f"❌ 错误：Baseline CSV 文件不存在: {baseline_csv}")
            return 1
        if not os.path.exists(real_csv):
            print(f"❌ 错误：Real CSV 文件不存在: {real_csv}")
            return 1
        
        data_folder = os.path.dirname(baseline_csv)
        
    elif args.csv_file:
        # 如果指定了 CSV 文件，直接使用
        csv_path = args.csv_file
        data_folder = None
        if not os.path.exists(csv_path):
            print(f"❌ 错误：CSV 文件不存在: {csv_path}")
            return 1
    else:
        # 否则使用 data_folder 参数
        if not args.data_folder:
            print(f"❌ 错误：必须指定 data_folder、--csv-file、--compare-dir 或 --baseline-csv/--real-csv")
            return 1
        # 构建完整的数据文件夹路径
        full_data_folder = os.path.join(args.data_root, args.data_folder)
        
        if not os.path.exists(full_data_folder):
            print(f"❌ 错误：数据文件夹不存在: {full_data_folder}")
            return 1
        
        # 构建CSV文件路径
        csv_path = os.path.join(full_data_folder, "metrics_csv", "world_metrics.csv")
        data_folder = full_data_folder
    
    # 检查阈值参数
    if args.thresholds and len(args.thresholds) != len(args.variables):
        print(f"❌ 错误：--thresholds 的数量 ({len(args.thresholds)}) 必须与变量数量 ({len(args.variables)}) 相同")
        return 1
    
    # 验证变量是否可用
    invalid_vars = [v for v in args.variables if v not in AVAILABLE_VARIABLES]
    if invalid_vars:
        print(f"⚠️  警告：以下变量不在可用列表中（将尝试绘制）：{', '.join(invalid_vars)}")
    
    # 绘制每个变量
    success_count = 0
    for i, variable in enumerate(args.variables):
        # 确定阈值
        threshold = None
        if args.thresholds:
            threshold = args.thresholds[i]
        elif args.threshold:
            threshold = args.threshold
        
        if plot_variable(csv_path, variable, threshold, args.output_dir, data_folder, 
                        baseline_csv, real_csv):
            success_count += 1
    
    print(f"\n🎯 完成！成功绘制 {success_count}/{len(args.variables)} 个变量")
    return 0 if success_count == len(args.variables) else 1


if __name__ == "__main__":
    sys.exit(main())

