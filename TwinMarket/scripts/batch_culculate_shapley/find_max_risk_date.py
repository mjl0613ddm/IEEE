#!/usr/bin/env python3
"""
辅助脚本：从market_metrics.csv中查找risk_indicator_simple的最大值对应的日期

用法:
    python find_max_risk_date.py <result_dir>

返回:
    JSON格式: {"start_date": "YYYY-MM-DD", "target_date": "YYYY-MM-DD", "max_risk_value": float}
    错误时输出错误信息到stderr并返回非零退出码
"""

import sys
import json
import pandas as pd
from pathlib import Path


def find_max_risk_date(result_dir):
    """
    从market_metrics.csv中找到risk_indicator_simple的最大值对应的日期
    
    Args:
        result_dir: 结果目录路径（包含analysis/market_metrics.csv）
        
    Returns:
        dict: {"start_date": str, "target_date": str, "max_risk_value": float}
        
    Raises:
        FileNotFoundError: 如果market_metrics.csv不存在
        ValueError: 如果没有有效的risk_indicator_simple数据
    """
    result_path = Path(result_dir)
    csv_file = result_path / "analysis" / "market_metrics.csv"
    
    if not csv_file.exists():
        raise FileNotFoundError(f"找不到文件: {csv_file}")
    
    # 读取CSV文件
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        raise ValueError(f"无法读取CSV文件 {csv_file}: {e}")
    
    # 检查必需的列
    if 'date' not in df.columns:
        raise ValueError(f"CSV文件缺少'date'列: {csv_file}")
    
    if 'risk_indicator_simple' not in df.columns:
        raise ValueError(f"CSV文件缺少'risk_indicator_simple'列: {csv_file}")
    
    # 过滤掉risk_indicator_simple为空值的行
    df_valid = df[df['risk_indicator_simple'].notna()].copy()
    
    if df_valid.empty:
        raise ValueError(f"CSV文件中没有有效的risk_indicator_simple数据: {csv_file}")
    
    # 找到最大风险值对应的日期
    max_idx = df_valid['risk_indicator_simple'].idxmax()
    max_row = df_valid.loc[max_idx]
    max_risk_value = float(max_row['risk_indicator_simple'])
    target_date = str(max_row['date'])
    
    # 第一个有效日期作为开始日期
    first_valid_row = df_valid.iloc[0]
    start_date = str(first_valid_row['date'])
    
    return {
        "start_date": start_date,
        "target_date": target_date,
        "max_risk_value": max_risk_value
    }


def main():
    """主函数"""
    if len(sys.argv) != 2:
        print("用法: python find_max_risk_date.py <result_dir>", file=sys.stderr)
        sys.exit(1)
    
    result_dir = sys.argv[1]
    
    try:
        result = find_max_risk_date(result_dir)
        # 输出JSON格式结果到stdout
        print(json.dumps(result, indent=2))
        sys.exit(0)
    except FileNotFoundError as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"未知错误: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
